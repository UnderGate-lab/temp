"""
デッドロック対策を強化した frame_restorer.py
VRAM最適化版
"""

import logging
import queue
import threading
import time
import textwrap
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import heapq

import cv2
import numpy as np
import torch

from lada import LOG_LEVEL
from lada.lib import image_utils, video_utils, threading_utils, mask_utils
from lada.lib import visualization_utils
from lada.lib.mosaic_detector import MosaicDetector
from lada.lib.mosaic_detection_model import MosaicDetectionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

# ====================================
# 診断クラス群
# ====================================

class ParallelDiagnostics:
    """並列処理の稼働状況を診断"""
    
    def __init__(self):
        self.worker_stats = defaultdict(lambda: {
            'clips_processed': 0,
            'total_time': 0,
            'wait_time': 0,
            'gpu_time': 0,
            'last_activity': 0
        })
        self.queue_stats = []
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_worker_start(self, worker_id):
        with self.lock:
            self.worker_stats[worker_id]['last_activity'] = time.time()
    
    def record_worker_processing(self, worker_id, clip_frames, processing_time):
        with self.lock:
            stats = self.worker_stats[worker_id]
            stats['clips_processed'] += 1
            stats['gpu_time'] += processing_time
            stats['last_activity'] = time.time()
    
    def record_worker_wait(self, worker_id, wait_time):
        with self.lock:
            self.worker_stats[worker_id]['wait_time'] += wait_time
    
    def record_queue_size(self, queue_name, size):
        self.queue_stats.append({
            'time': time.time() - self.start_time,
            'queue': queue_name,
            'size': size
        })
    
    def get_report(self):
        with self.lock:
            total_elapsed = time.time() - self.start_time
            
            report = ["=" * 60]
            report.append("並列処理診断レポート")
            report.append("=" * 60)
            report.append(f"経過時間: {total_elapsed:.2f}秒")
            report.append("")
            
            report.append("ワーカー統計:")
            report.append("-" * 60)
            
            active_workers = 0
            total_clips = 0
            for worker_id, stats in sorted(self.worker_stats.items()):
                idle_time = time.time() - stats['last_activity']
                is_active = idle_time < 5.0
                
                if is_active:
                    active_workers += 1
                
                total_clips += stats['clips_processed']
                
                status = "🟢 稼働中" if is_active else "🔴 待機"
                report.append(
                    f"{worker_id}: {status} | "
                    f"処理: {stats['clips_processed']} | "
                    f"GPU: {stats['gpu_time']:.1f}s | "
                    f"待機: {stats['wait_time']:.1f}s"
                )
            
            report.append("")
            report.append(f"アクティブワーカー: {active_workers}/{len(self.worker_stats)}")
            report.append(f"総処理クリップ数: {total_clips}")
            
            total_gpu_time = sum(s['gpu_time'] for s in self.worker_stats.values())
            total_wait_time = sum(s['wait_time'] for s in self.worker_stats.values())
            
            if total_elapsed > 0 and len(self.worker_stats) > 0:
                gpu_utilization = (total_gpu_time / (total_elapsed * len(self.worker_stats))) * 100
                wait_ratio = (total_wait_time / (total_elapsed * len(self.worker_stats))) * 100
                
                report.append("")
                report.append(f"並列効率:")
                report.append(f"  GPU利用率: {gpu_utilization:.1f}%")
                report.append(f"  待機時間: {wait_ratio:.1f}%")
                
                if gpu_utilization < 20:
                    report.append("  ⚠️ GPU利用率が低い - 並列処理が機能していない可能性")
                if active_workers < len(self.worker_stats) * 0.5:
                    report.append("  ⚠️ 稼働ワーカーが少ない - キューが空の可能性")
            
            report.append("=" * 60)
            
            return "\n".join(report)


class QueueMonitor:
    """キューの詰まりを監視"""
    
    def __init__(self, queue_obj, name, warning_threshold=0.8):
        self.queue = queue_obj
        self.name = name
        self.warning_threshold = warning_threshold
        self.last_check = time.time()
        self.stall_count = 0
    
    def check(self):
        now = time.time()
        if now - self.last_check > 5.0:
            if self.queue.maxsize > 0:
                usage = self.queue.qsize() / self.queue.maxsize
                
                if usage > self.warning_threshold:
                    self.stall_count += 1
                    logger.warning(
                        f"⚠️ {self.name}が{usage*100:.0f}%満杯 "
                        f"(停滞: {self.stall_count})"
                    )
            
            self.last_check = now

def load_models(device, mosaic_restoration_model_name, mosaic_restoration_model_path, mosaic_restoration_config_path, mosaic_detection_model_path):
    if mosaic_restoration_model_name.startswith("deepmosaics"):
        from lada.deepmosaics.models import loadmodel, model_util
        mosaic_restoration_model = loadmodel.video(model_util.device_to_gpu_id(device), mosaic_restoration_model_path)
        pad_mode = 'reflect'
    elif mosaic_restoration_model_name.startswith("basicvsrpp"):
        from lada.basicvsrpp.inference import load_model, get_default_gan_inference_config
        if mosaic_restoration_config_path:
            config = mosaic_restoration_config_path
        else:
            config = get_default_gan_inference_config()
        mosaic_restoration_model = load_model(config, mosaic_restoration_model_path, device)
        pad_mode = 'zero'
    else:
        raise NotImplementedError()
    mosaic_detection_model = MosaicDetectionModel(mosaic_detection_model_path, device, classes=[0], conf=0.2)
    return mosaic_detection_model, mosaic_restoration_model, pad_mode

# ====================================
# 基本FrameRestorerクラス
# ====================================

class FrameRestorer:
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode,
                 mosaic_detection=False):
        """
        基本FrameRestorerクラス - 旧インターフェースを維持
        """
        logger.info(f"🎯 FrameRestorer初期化 (preserve_relative_scale={preserve_relative_scale} - 無視されます)")
        
        self.device = device
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.video_file = video_file
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False

        # キューの初期化
        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        self.frame_restoration_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        max_clips_in_mosaic_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4))
        logger.debug(f"Set queue size of queue mosaic_clip_queue to {max_clips_in_mosaic_clips_queue}")
        self.mosaic_clip_queue = queue.Queue(maxsize=max_clips_in_mosaic_clips_queue)

        max_clips_in_restored_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4))
        logger.debug(f"Set queue size of queue restored_clip_queue to {max_clips_in_restored_clips_queue}")
        self.restored_clip_queue = queue.Queue(maxsize=max_clips_in_restored_clips_queue)

        self.frame_detection_queue = queue.Queue()

        # 新しいMosaicDetectorインターフェース
        logger.info("🔄 MosaicDetectorを初期化中...")
        try:
            self.mosaic_detector = MosaicDetector(
                self.mosaic_detection_model, 
                self.video_file,
                frame_detection_queue=self.frame_detection_queue,
                mosaic_clip_queue=self.mosaic_clip_queue,
                device=self.device,
                max_clip_length=self.max_clip_length,
                pad_mode=self.preferred_pad_mode
            )
            logger.info("✅ MosaicDetector初期化成功")
        except Exception as e:
            logger.error(f"❌ MosaicDetector初期化エラー: {e}")
            raise

        self.clip_restoration_thread = None
        self.frame_restoration_thread = None
        self.clip_restoration_thread_should_be_running = False
        self.frame_restoration_thread_should_be_running = False

        self.queue_stats = {}
        self.queue_stats["restored_clip_queue_max_size"] = 0
        self.queue_stats["restored_clip_queue_wait_time_put"] = 0
        self.queue_stats["restored_clip_queue_wait_time_get"] = 0
        self.queue_stats["mosaic_clip_queue_wait_time_get"] = 0
        self.queue_stats["frame_restoration_queue_max_size"] = 0
        self.queue_stats["frame_restoration_queue_wait_time_get"] = 0
        self.queue_stats["frame_restoration_queue_wait_time_put"] = 0
        self.queue_stats["frame_detection_queue_wait_time_get"] = 0

        logger.info(f"✅ FrameRestorer初期化完了:")
        logger.info(f"  - ビデオ: {self.video_file}")
        logger.info(f"  - 解像度: {self.video_meta_data.video_width}x{self.video_meta_data.video_height}")
        logger.info(f"  - 最大クリップ長: {self.max_clip_length}")

    def start(self, start_ns=0):
        logger.info(f"🚀 FrameRestorer開始: start_ns={start_ns}")
        
        assert self.frame_restoration_thread is None and self.clip_restoration_thread is None
        assert self.mosaic_clip_queue.empty()
        assert self.restored_clip_queue.empty()
        assert self.frame_detection_queue.empty()
        assert self.frame_restoration_queue.empty()

        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
        self.stop_requested = False
        self.frame_restoration_thread_should_be_running = True
        self.clip_restoration_thread_should_be_running = True

        self.frame_restoration_thread = threading.Thread(target=self._frame_restoration_worker)
        self.clip_restoration_thread = threading.Thread(target=self._clip_restoration_worker)

        logger.info("🔄 MosaicDetectorを開始中...")
        self.mosaic_detector.start(start_ns=start_ns)
        self.clip_restoration_thread.start()
        self.frame_restoration_thread.start()

        logger.info("✅ FrameRestorer開始完了")

    def stop(self):
        logger.debug("FrameRestorer: 停止中...")
        start = time.time()
        self.stop_requested = True
        self.clip_restoration_thread_should_be_running = False
        self.frame_restoration_thread_should_be_running = False

        self.mosaic_detector.stop()

        threading_utils.put_closing_queue_marker(self.mosaic_clip_queue, "mosaic_clip_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")
        if self.clip_restoration_thread:
            self.clip_restoration_thread.join()
            logger.debug("clip restoration worker: stopped")
        self.clip_restoration_thread = None

        threading_utils.put_closing_queue_marker(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.put_closing_queue_marker(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")
        if self.frame_restoration_thread:
            self.frame_restoration_thread.join()
            logger.debug("frame restoration worker: stopped")
        self.frame_restoration_thread = None

        threading_utils.empty_out_queue(self.mosaic_clip_queue, "mosaic_clip_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")

        assert self.mosaic_clip_queue.empty()
        assert self.restored_clip_queue.empty()
        assert self.frame_detection_queue.empty()
        assert self.frame_restoration_queue.empty()

        logger.debug(f"FrameRestorer: 停止完了, 所要時間: {time.time() - start}")

    def _restore_clip_frames(self, images):
        if self.mosaic_restoration_model_name.startswith("deepmosaics"):
            from lada.deepmosaics.inference import restore_video_frames
            from lada.deepmosaics.models import model_util
            restored_clip_images = restore_video_frames(model_util.device_to_gpu_id(self.device), self.mosaic_restoration_model, images)
        elif self.mosaic_restoration_model_name.startswith("basicvsrpp"):
            from lada.basicvsrpp.inference import inference
            restored_clip_images = inference(self.mosaic_restoration_model, images, self.device)
        else:
            raise NotImplementedError()
        return restored_clip_images

    def _restore_frame(self, frame, frame_num, restored_clips):
        for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = buffered_clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(clip_mask, orig_crop_shape[:2],interpolation=cv2.INTER_NEAREST)
            t, l, b, r = orig_clip_box
            blend_mask = mask_utils.create_blend_mask(clip_mask)
            blended_img = (frame[t:b + 1, l:r + 1, :] * (1 - blend_mask[..., None]) + clip_img * (blend_mask[..., None])).clip(0, 255).astype(np.uint8)
            frame[t:b + 1, l:r + 1, :] = blended_img

    def _restore_clip(self, clip):
        if self.mosaic_detection:
            restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
        else:
            images = clip.get_clip_images()
            restored_clip_images = self._restore_clip_frames(images)
        assert len(restored_clip_images) == len(clip.get_clip_images())

        for i in range(len(restored_clip_images)):
            assert clip.data[i][0].shape == restored_clip_images[i].shape
            clip.data[i] = restored_clip_images[i], clip.data[i][1], clip.data[i][2], clip.data[i][3], clip.data[i][4]

    def _collect_garbage(self, clip_buffer):
        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
        for processed_clip in processed_clips:
            clip_buffer.remove(processed_clip)

    def _contains_at_least_one_clip_starting_after_frame_num(self, frame_num, clip_buffer):
        return len(clip_buffer) > 0 and frame_num < max(clip_buffer, key=lambda c: c.frame_start).frame_start

    def _clip_restoration_worker(self):
        logger.debug("clip restoration worker: started")
        eof = False
        while self.clip_restoration_thread_should_be_running:
            s = time.time()
            clip = self.mosaic_clip_queue.get()
            self.queue_stats["mosaic_clip_queue_wait_time_get"] += time.time() - s
            if self.stop_requested:
                logger.debug("clip restoration worker: mosaic_clip_queue consumer unblocked")
            if clip is None:
                if not self.stop_requested:
                    eof = True
                    self.clip_restoration_thread_should_be_running = False
                    self.queue_stats["restored_clip_queue_max_size"] = max(self.restored_clip_queue.qsize()+1, self.queue_stats["restored_clip_queue_max_size"])
                    s = time.time()
                    self.restored_clip_queue.put(None)
                    self.queue_stats["restored_clip_queue_wait_time_put"] += time.time() -s
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
            else:
                self._restore_clip(clip)
                self.queue_stats["restored_clip_queue_max_size"] = max(self.restored_clip_queue.qsize()+1, self.queue_stats["restored_clip_queue_max_size"])
                s = time.time()
                self.restored_clip_queue.put(clip)
                self.queue_stats["restored_clip_queue_wait_time_put"] += time.time() - s
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
        if eof:
            logger.debug("clip restoration worker: stopped itself, EOF")

    def _read_next_frame(self, video_frames_generator, expected_frame_num):
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            s = time.time()
            elem = self.frame_detection_queue.get()
            self.queue_stats["frame_detection_queue_wait_time_get"] += time.time() - s
            if self.stop_requested:
                logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
            assert elem is None
            return None
        s = time.time()
        elem = self.frame_detection_queue.get()
        self.queue_stats["frame_detection_queue_wait_time_get"] += time.time() - s
        if self.stop_requested:
            logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
            return None
        assert elem is not None
        detection_frame_num, mosaic_detected = elem
        assert self.stop_requested or detection_frame_num == expected_frame_num
        return mosaic_detected, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer):
        s = time.time()
        clip = self.restored_clip_queue.get()
        self.queue_stats["restored_clip_queue_wait_time_get"] += time.time() - s
        if self.stop_requested:
            logger.debug("frame restoration worker: restored_clip_queue consumer unblocked")
        if clip is None:
            return False
        assert self.stop_requested or clip.frame_start >= current_frame_num
        clip_buffer.append(clip)
        return True

    def _frame_restoration_worker(self):
        logger.debug("frame restoration worker: started")
        with video_utils.VideoReader(self.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()

            frame_num = self.start_frame
            clips_remaining = True
            clip_buffer = []

            while self.frame_restoration_thread_should_be_running:
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if _frame_result is None:
                    if not self.stop_requested:
                        self.eof = True
                        self.frame_restoration_thread_should_be_running = False
                        self.frame_restoration_queue.put(None)
                    break
                else:
                    mosaic_detected, frame, frame_pts = _frame_result
                if mosaic_detected:
                    while clips_remaining and not self._contains_at_least_one_clip_starting_after_frame_num(frame_num, clip_buffer):
                        clips_remaining = self._read_next_clip(frame_num, clip_buffer)

                    self._restore_frame(frame, frame_num, clip_buffer)
                    self.queue_stats["frame_restoration_queue_max_size"] = max(self.frame_restoration_queue.qsize()+1, self.queue_stats["frame_restoration_queue_max_size"])
                    s = time.time()
                    self.frame_restoration_queue.put((frame, frame_pts))
                    self.queue_stats["frame_restoration_queue_wait_time_put"] += time.time() -s
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                    self._collect_garbage(clip_buffer)
                else:
                    self.queue_stats["frame_restoration_queue_max_size"] = max(self.frame_restoration_queue.qsize()+1, self.queue_stats["frame_restoration_queue_max_size"])
                    s = time.time()
                    self.frame_restoration_queue.put((frame, frame_pts))
                    self.queue_stats["frame_restoration_queue_wait_time_put"] += time.time() - s
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                frame_num += 1
            if self.eof:
                logger.debug("frame restoration worker: stopped itself, EOF")

    def __iter__(self):
        return self

    def __next__(self):
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration
        else:
            while not self.stop_requested:
                s = time.time()
                elem = self.frame_restoration_queue.get()
                self.queue_stats["frame_restoration_queue_wait_time_get"] += time.time() -s
                if self.stop_requested:
                    logger.debug("frame_restoration_queue consumer unblocked")
                if elem is None and not self.stop_requested:
                    raise StopIteration
                return elem

    def get_frame_restoration_queue(self):
        return self.frame_restoration_queue

# ====================================
# VRAM最適化版FrameRestorer
# ====================================

class OptimizedFrameRestorer:
    """
    VRAM最適化版FrameRestorer - 並列処理とVRAM管理を強化
    """
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length,
                 mosaic_restoration_model_name, mosaic_detection_model, mosaic_restoration_model,
                 preferred_pad_mode, mosaic_detection=False, batch_size=16, parallel_clips=4,
                 enable_vram_direct=True, dynamic_batch_size=True, auto_worker_count=False,
                 enable_diagnostics=True):
        """
        VRAM最適化版 - 並列処理とメモリ管理を強化
        """
        logger.info(f"🎯 OptimizedFrameRestorer初期化 (VRAM最適化版):")
        logger.info(f"  - batch_size: {batch_size}")
        logger.info(f"  - parallel_clips: {parallel_clips}")
        logger.info(f"  - enable_vram_direct: {enable_vram_direct}")
        
        self.device = device
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.video_file = video_file
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False

        # VRAM最適化設定
        self.batch_size = batch_size
        self.parallel_clips = parallel_clips
        self.enable_vram_direct = enable_vram_direct
        self.dynamic_batch_size = dynamic_batch_size
        self.auto_worker_count = auto_worker_count
        self.enable_diagnostics = enable_diagnostics

        if auto_worker_count:
            self.parallel_clips = self._calculate_optimal_workers()

        # キューの初期化（VRAM最適化）
        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (
            self.video_meta_data.video_width * self.video_meta_data.video_height * 3
        )
        self.frame_restoration_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        max_clips_in_mosaic_clips_queue = max(
            self.parallel_clips * 2,
            (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)
        )
        self.mosaic_clip_queue = queue.Queue(maxsize=max_clips_in_mosaic_clips_queue)

        max_clips_in_restored_clips_queue = max(
            self.parallel_clips * 2,
            (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)
        )
        self.restored_clip_queue = queue.Queue(maxsize=max_clips_in_restored_clips_queue)

        self.frame_detection_queue = queue.Queue()

        # 並列処理用キュー
        self.unordered_clips_queue = queue.Queue(maxsize=max_clips_in_restored_clips_queue * 2)
        self.next_expected_clip_id = 0
        self.clip_counter = 0
        self.clip_counter_lock = threading.Lock()

        # 新しいMosaicDetectorインターフェース
        logger.info("🔄 MosaicDetectorを初期化中...")
        try:
            self.mosaic_detector = MosaicDetector(
                self.mosaic_detection_model, 
                self.video_file,
                frame_detection_queue=self.frame_detection_queue,
                mosaic_clip_queue=self.mosaic_clip_queue,
                device=self.device,
                max_clip_length=self.max_clip_length,
                pad_mode=self.preferred_pad_mode
            )
            logger.info("✅ MosaicDetector初期化成功")
        except Exception as e:
            logger.error(f"❌ MosaicDetector初期化エラー: {e}")
            raise

        # 並列処理スレッド
        self.clip_executor = None
        self.clip_restoration_threads = []
        self.frame_restoration_thread = None
        self.clip_ordering_thread = None

        self.clip_restoration_threads_should_be_running = False
        self.frame_restoration_thread_should_be_running = False
        self.clip_ordering_thread_should_be_running = False

        # VRAM管理
        self.workers_finished_count = 0
        self.workers_finished_lock = threading.Lock()

        # 診断機能
        if enable_diagnostics:
            self.diagnostics = ParallelDiagnostics()
            self.queue_monitors = {
                'mosaic_clip': QueueMonitor(self.mosaic_clip_queue, 'mosaic_clip_queue'),
                'restored_clip': QueueMonitor(self.restored_clip_queue, 'restored_clip_queue'),
            }
            logger.info("✅ 診断機能を有効化")
        else:
            self.diagnostics = None
            self.queue_monitors = {}

        self.queue_stats = {
            "restored_clip_queue_max_size": 0,
            "restored_clip_queue_wait_time_put": 0,
            "restored_clip_queue_wait_time_get": 0,
            "mosaic_clip_queue_wait_time_get": 0,
            "frame_restoration_queue_max_size": 0,
            "frame_restoration_queue_wait_time_get": 0,
            "frame_restoration_queue_wait_time_put": 0,
            "frame_detection_queue_wait_time_get": 0,
            "parallel_clips_processed": 0,
            "clip_wait_for_order": 0,
            "clip_timeout_count": 0,
            "clip_skipped_count": 0,
            "deadlock_recovery_count": 0,
        }

        logger.info(f"✅ OptimizedFrameRestorer初期化完了:")
        logger.info(f"  - 並列ワーカー数: {self.parallel_clips}")
        logger.info(f"  - バッチサイズ: {self.batch_size}")
        logger.info(f"  - VRAM直接処理: {'有効' if enable_vram_direct else '無効'}")

    def _calculate_optimal_workers(self):
        """最適なワーカー数を計算"""
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory
            optimal_workers = max(1, int((total_vram * 0.7) / (1024**3)))
            return min(optimal_workers, 8)
        return 4

    def _get_next_clip_id(self):
        with self.clip_counter_lock:
            clip_id = self.clip_counter
            self.clip_counter += 1
            return clip_id

    def start(self, start_ns=0):
        logger.info(f"🚀 OptimizedFrameRestorer開始: start_ns={start_ns}")
        
        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(
            self.start_ns, self.video_meta_data.video_fps_exact
        )
        self.stop_requested = False
        self.next_expected_clip_id = 0
        self.clip_counter = 0
        self.workers_finished_count = 0

        self.frame_restoration_thread_should_be_running = True
        self.clip_restoration_threads_should_be_running = True
        self.clip_ordering_thread_should_be_running = True

        # スレッドプールの初期化
        self.clip_executor = ThreadPoolExecutor(
            max_workers=self.parallel_clips,
            thread_name_prefix="ClipRestorer"
        )

        # MosaicDetector開始
        self.mosaic_detector.start(start_ns=start_ns)

        # 並列クリップ処理ワーカー起動
        for i in range(self.parallel_clips):
            thread = threading.Thread(
                target=self._clip_restoration_worker_vram,
                name=f"ClipWorker-{i}"
            )
            thread.start()
            self.clip_restoration_threads.append(thread)

        # クリップ順序付けワーカー起動
        self.clip_ordering_thread = threading.Thread(
            target=self._clip_ordering_worker_with_recovery,
            name="ClipOrderingWorker"
        )
        self.clip_ordering_thread.start()

        # フレーム復元ワーカー起動
        self.frame_restoration_thread = threading.Thread(
            target=self._frame_restoration_worker_vram,
            name="FrameRestorationWorker"
        )
        self.frame_restoration_thread.start()

        logger.info(f"✅ 起動完了: {self.parallel_clips} ワーカー")

    def _clip_restoration_worker_vram(self):
        """VRAM最適化版クリップ復元ワーカー"""
        worker_name = threading.current_thread().name
        logger.debug(f"{worker_name}: VRAM最適化ワーカー起動")
        
        error_count = 0
        max_errors = 10
        clips_processed = 0

        while self.clip_restoration_threads_should_be_running:
            if self.diagnostics:
                self.diagnostics.record_queue_size('mosaic_clip_queue', self.mosaic_clip_queue.qsize())
            
            try:
                wait_start = time.time()
                clip = self.mosaic_clip_queue.get(timeout=0.5)
                wait_time = time.time() - wait_start
                
                if self.diagnostics:
                    self.diagnostics.record_worker_wait(worker_name, wait_time)

                if clip is None:
                    logger.info(f"{worker_name}: EOFマーカー受信 (処理数: {clips_processed})")
                    
                    # EOFマーカーを他のワーカーのために戻す
                    self.mosaic_clip_queue.put(None)
                    
                    # 全ワーカー終了を待ってからEOFマーカー送信
                    with self.workers_finished_lock:
                        self.workers_finished_count += 1
                        finished_count = self.workers_finished_count
                        logger.info(f"{worker_name}: 終了 ({finished_count}/{self.parallel_clips})")
                        
                        # 最後のワーカーだけがEOFマーカーを送信
                        if finished_count == self.parallel_clips:
                            logger.info(f"{worker_name}: 全ワーカー終了 - EOFマーカー送信")
                            self.unordered_clips_queue.put(None)
                    
                    break

                clip_id = self._get_next_clip_id()
                clip_length = len(clip.get_clip_images())
                
                if self.diagnostics:
                    self.diagnostics.record_worker_start(worker_name)
                
                gpu_start = time.time()
                
                try:
                    # VRAM最適化処理
                    processed_clip = self._restore_clip_vram(clip)
                    
                    gpu_time = time.time() - gpu_start
                    
                    if self.diagnostics:
                        self.diagnostics.record_worker_processing(worker_name, clip_length, gpu_time)
                    
                    self.unordered_clips_queue.put((clip_id, processed_clip))
                    self.queue_stats['parallel_clips_processed'] += 1
                    clips_processed += 1
                    error_count = 0
                    
                except Exception as e:
                    logger.error(f"{worker_name}: エラー (clip={clip_id}): {e}")
                    error_count += 1
                    if error_count >= max_errors:
                        break

            except queue.Empty:
                if self.stop_requested:
                    break
                continue

        logger.debug(f"{worker_name}: 終了 (合計: {clips_processed})")

    def _restore_clip_vram(self, clip):
        """VRAM最適化版クリップ復元"""
        if self.mosaic_detection:
            restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
        else:
            images = clip.get_clip_images()
            # VRAM最適化処理
            restored_clip_images = self._restore_clip_frames_vram(images)

        assert len(restored_clip_images) == len(clip.get_clip_images())

        for i in range(len(restored_clip_images)):
            clip.data[i] = (
                restored_clip_images[i],
                clip.data[i][1],
                clip.data[i][2],
                clip.data[i][3],
                clip.data[i][4]
            )
        
        return clip

    def _restore_clip_frames_vram(self, images):
        """VRAM最適化版フレーム復元"""
        if self.mosaic_restoration_model_name.startswith("deepmosaics"):
            from lada.deepmosaics.inference import restore_video_frames
            from lada.deepmosaics.models import model_util
            restored_clip_images = restore_video_frames(
                model_util.device_to_gpu_id(self.device), 
                self.mosaic_restoration_model, 
                images
            )
            
        elif self.mosaic_restoration_model_name.startswith("basicvsrpp"):
            from lada.basicvsrpp.inference import inference
            
            # VRAM最適化: テンソルを適切に処理
            numpy_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    if img.is_cuda:
                        img = img.cpu()
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = img.permute(1, 2, 0)
                    img = img.numpy()
                    if img.dtype == np.float32 and img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    elif img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                numpy_images.append(img)
            
            restored_clip_images = inference(
                self.mosaic_restoration_model, 
                numpy_images,
                self.device
            )
        else:
            raise NotImplementedError()
        
        return restored_clip_images

    def _clip_ordering_worker_with_recovery(self):
        """クリップ順序付けワーカー（デッドロックリカバリー付き）"""
        pending_clips = []
        eof = False
        last_progress_time = time.time()
        stall_threshold = 10.0
        last_log_time = time.time()

        while self.clip_ordering_thread_should_be_running:
            try:
                item = self.unordered_clips_queue.get(timeout=1.0)

                if item is None:
                    if not self.stop_requested:
                        eof = True
                        logger.info(
                            f"clip_ordering_worker: EOF受信 (pending={len(pending_clips)}, "
                            f"next_expected={self.next_expected_clip_id})"
                        )
                        break
                    else:
                        break

                clip_id, clip = item
                heapq.heappush(pending_clips, (clip_id, clip))

                # 進捗があったら時刻更新
                made_progress = False
                while pending_clips and pending_clips[0][0] == self.next_expected_clip_id:
                    _, ordered_clip = heapq.heappop(pending_clips)
                    self.restored_clip_queue.put(ordered_clip)
                    self.next_expected_clip_id += 1
                    made_progress = True
                
                if made_progress:
                    last_progress_time = time.time()

            except queue.Empty:
                # デッドロック検出とリカバリー
                current_time = time.time()
                stall_duration = current_time - last_progress_time
                
                if stall_duration > stall_threshold:
                    if pending_clips:
                        expected_id = self.next_expected_clip_id
                        actual_id = pending_clips[0][0]
                        
                        if actual_id > expected_id:
                            logger.error(
                                f"🚨 デッドロック検知! (停滞: {stall_duration:.1f}秒)\n"
                                f"   期待ID: {expected_id}\n"
                                f"   次のID: {actual_id}\n"
                                f"   保留: {len(pending_clips)}\n"
                                f"   → ID {expected_id}~{actual_id-1} をスキップして続行"
                            )
                            
                            # 欠落したクリップをスキップ
                            skipped_count = actual_id - expected_id
                            self.queue_stats['clip_skipped_count'] += skipped_count
                            self.queue_stats['deadlock_recovery_count'] += 1
                            self.next_expected_clip_id = actual_id
                            last_progress_time = current_time
                
                # 定期的にステータスログ出力
                if current_time - last_log_time > 5.0:
                    logger.debug(
                        f"[ORDER] 待機中 - 期待ID:{self.next_expected_clip_id}, "
                        f"保留:{len(pending_clips)}, "
                        f"Qサイズ:{self.unordered_clips_queue.qsize()}, "
                        f"停滞:{stall_duration:.1f}s"
                    )
                    last_log_time = current_time
                
                if self.stop_requested:
                    break
                continue

        if eof:
            logger.info(f"clip_ordering_worker: EOF処理開始 (pending={len(pending_clips)})")
            
            # EOFでも残りのクリップを全て処理
            while pending_clips:
                expected_id = self.next_expected_clip_id
                actual_id = pending_clips[0][0]
                
                if actual_id == expected_id:
                    _, ordered_clip = heapq.heappop(pending_clips)
                    self.restored_clip_queue.put(ordered_clip)
                    logger.debug(f"clip_ordering_worker: クリップ {expected_id} 送出(EOF処理中)")
                    self.next_expected_clip_id += 1
                
                elif actual_id > expected_id:
                    # クリップ欠落を警告
                    logger.warning(
                        f"⚠️ クリップ順序エラー(EOF時): ID {expected_id} が欠落\n"
                        f"   次のクリップID: {actual_id}\n"
                        f"   保留中: {len(pending_clips)}\n"
                        f"   リアルタイム再生のため続行します"
                    )
                    # 欠落したIDをスキップして続行
                    skipped_count = actual_id - expected_id
                    self.queue_stats['clip_skipped_count'] += skipped_count
                    self.next_expected_clip_id = actual_id
                
                else:
                    logger.error(f"❌ 異常: 既に送出済みのID {actual_id} が残っています")
                    heapq.heappop(pending_clips)
            
            self.restored_clip_queue.put(None)
            logger.info("clip_ordering_worker: EOFマーカー送出完了")

    def _frame_restoration_worker_vram(self):
        """VRAM最適化版フレーム復元ワーカー"""
        logger.debug("frame restoration worker: started (VRAM最適化版)")
        with video_utils.VideoReader(self.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()
            frame_num = self.start_frame
            clips_remaining = True
            clip_buffer = []

            while self.frame_restoration_thread_should_be_running:
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if _frame_result is None:
                    if not self.stop_requested:
                        self.eof = True
                        self.frame_restoration_queue.put(None)
                    break

                mosaic_detected, frame, frame_pts = _frame_result

                if mosaic_detected:
                    while clips_remaining and not self._contains_at_least_one_clip_starting_after_frame_num(
                        frame_num, clip_buffer
                    ):
                        clips_remaining = self._read_next_clip(frame_num, clip_buffer)

                    self._restore_frame(frame, frame_num, clip_buffer)
                    self._collect_garbage(clip_buffer)

                self.frame_restoration_queue.put((frame, frame_pts))
                frame_num += 1

    def _read_next_frame(self, video_frames_generator, expected_frame_num):
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            elem = self.frame_detection_queue.get()
            assert elem is None
            return None
        
        elem = self.frame_detection_queue.get()
        if self.stop_requested:
            return None
        assert elem is not None
        detection_frame_num, mosaic_detected = elem
        return mosaic_detected, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer):
        """タイムアウト時の自動スキップ機能"""
        try:
            clip = self.restored_clip_queue.get(timeout=5.0)
            if self.stop_requested or clip is None:
                return False
            clip_buffer.append(clip)
            return True
        except queue.Empty:
            # タイムアウト時の処理
            timeout_count = self.queue_stats["clip_timeout_count"]
            self.queue_stats["clip_timeout_count"] += 1
            
            logger.warning(
                f"⚠️ クリップ取得タイムアウト (frame={current_frame_num}, count={timeout_count+1})\n"
                f"   バッファ内: {len(clip_buffer)}, 次期待ID: {self.next_expected_clip_id}"
            )
            
            # 連続タイムアウトでスキップ
            if timeout_count >= 3:
                logger.error(
                    f"🚨 連続タイムアウト検知 (count={timeout_count+1})\n"
                    f"   → クリップ待ちをスキップして続行"
                )
                return False  # クリップ待ちを諦める
            
            return True  # まだリトライ

    def _restore_frame(self, frame, frame_num, restored_clips):
        for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = buffered_clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(clip_mask, orig_crop_shape[:2], interpolation=cv2.INTER_NEAREST)
            t, l, b, r = orig_clip_box
            blend_mask = mask_utils.create_blend_mask(clip_mask)
            blended_img = (
                frame[t:b + 1, l:r + 1, :] * (1 - blend_mask[..., None]) +
                clip_img * (blend_mask[..., None])
            ).clip(0, 255).astype(np.uint8)
            frame[t:b + 1, l:r + 1, :] = blended_img

    def _collect_garbage(self, clip_buffer):
        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
        for processed_clip in processed_clips:
            clip_buffer.remove(processed_clip)

    def _contains_at_least_one_clip_starting_after_frame_num(self, frame_num, clip_buffer):
        return len(clip_buffer) > 0 and frame_num < max(clip_buffer, key=lambda c: c.frame_start).frame_start

    def stop(self):
        """VRAM最適化版停止処理"""
        logger.debug("OptimizedFrameRestorer: 停止中...")
        start = time.time()
        
        if hasattr(self, 'diagnostics') and self.diagnostics is not None:
            import sys
            print("\n" + "="*70, file=sys.stdout)
            print("📊 並列処理診断レポート", file=sys.stdout)
            print("="*70, file=sys.stdout)
            try:
                report = self.diagnostics.get_report()
                print(report, file=sys.stdout)
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ レポート生成エラー: {e}", file=sys.stderr)
            print("="*70 + "\n", file=sys.stdout)
            sys.stdout.flush()
        
        self.stop_requested = True
        self.clip_restoration_threads_should_be_running = False
        self.frame_restoration_thread_should_be_running = False
        self.clip_ordering_thread_should_be_running = False

        self.mosaic_detector.stop()

        # 全ワーカーに停止シグナルを送信
        for _ in range(self.parallel_clips):
            threading_utils.put_closing_queue_marker(self.mosaic_clip_queue, "mosaic_clip_queue")

        threading_utils.empty_out_queue(self.unordered_clips_queue, "unordered_clips_queue")

        # ワーカースレッドの終了を待機
        for thread in self.clip_restoration_threads:
            if thread:
                thread.join(timeout=2.0)
        self.clip_restoration_threads.clear()
        logger.debug("clip restoration workers: stopped")

        # スレッドプールのシャットダウン
        if self.clip_executor:
            self.clip_executor.shutdown(wait=True, cancel_futures=True)
            self.clip_executor = None

        # 順序付けワーカーの終了
        threading_utils.put_closing_queue_marker(self.unordered_clips_queue, "unordered_clips_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")

        if self.clip_ordering_thread:
            self.clip_ordering_thread.join(timeout=2.0)
            self.clip_ordering_thread = None
        logger.debug("clip ordering worker: stopped")

        # フレーム復元ワーカーの終了
        threading_utils.put_closing_queue_marker(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.put_closing_queue_marker(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")

        if self.frame_restoration_thread:
            self.frame_restoration_thread.join(timeout=2.0)
            self.frame_restoration_thread = None
        logger.debug("frame restoration worker: stopped")

        # VRAM解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 全キューのクリア
        threading_utils.empty_out_queue(self.mosaic_clip_queue, "mosaic_clip_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")
        threading_utils.empty_out_queue(self.unordered_clips_queue, "unordered_clips_queue")

        logger.debug(f"OptimizedFrameRestorer: 停止完了, 所要時間: {time.time() - start:.2f}s")
        
        # 統計出力
        logger.info(f"📈 処理統計:")
        logger.info(f"   並列クリップ処理数: {self.queue_stats.get('parallel_clips_processed', 0)}")
        logger.info(f"   タイムアウト回数: {self.queue_stats.get('clip_timeout_count', 0)}")
        logger.info(f"   スキップ回数: {self.queue_stats.get('clip_skipped_count', 0)}")
        logger.info(f"   リカバリー回数: {self.queue_stats.get('deadlock_recovery_count', 0)}")

    def __iter__(self):
        return self

    def __next__(self):
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration
        while not self.stop_requested:
            elem = self.frame_restoration_queue.get()
            if self.stop_requested:
                return None
            if elem is None and not self.stop_requested:
                raise StopIteration
            return elem

    def get_frame_restoration_queue(self):
        return self.frame_restoration_queue
