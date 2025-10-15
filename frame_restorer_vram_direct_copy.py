import logging
import queue
import threading
import time
import textwrap
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
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


class GPUMemoryOptimizer:
    """GPUメモリ使用効率を最適化するクラス"""
    
    def __init__(self, device="cuda:0"):
        self.device = device
        self.allocated_tensors = {}
        self.max_vram_usage = 0.8  # VRAMの80%まで使用
        self._calculate_vram_limit()
    
    def _calculate_vram_limit(self):
        """利用可能なVRAMを計算"""
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory
            self.vram_limit = int(total_vram * self.max_vram_usage)
            print(f"[GPU-MEM] VRAM制限: {self.vram_limit/(1024**3):.1f}GB")
    
    def allocate_tensor_pool(self, shape_dtypes):
        """テンソルプールを事前割り当て"""
        for name, (shape, dtype) in shape_dtypes.items():
            try:
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                self.allocated_tensors[name] = tensor
                print(f"[GPU-MEM] テンソルプール割当: {name} {shape}")
            except RuntimeError as e:
                print(f"[GPU-MEM] テンソル割当失敗 {name}: {e}")
    
    def get_tensor(self, name, shape, dtype):
        """再利用可能なテンソルを取得"""
        if name in self.allocated_tensors:
            tensor = self.allocated_tensors[name]
            if tensor.shape == shape and tensor.dtype == dtype:
                return tensor
        # 新規作成
        return torch.zeros(shape, dtype=dtype, device=self.device)


class DynamicBatchProcessor:
    """動的バッチ処理の最適化"""
    
    def __init__(self, initial_batch_size=16, max_batch_size=32):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.processing_times = []
        self.throughput_history = []
        
    def adjust_batch_size(self, processing_time, frame_count):
        """処理時間に基づいてバッチサイズを動的に調整"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)
        
        avg_time = np.mean(self.processing_times)
        current_throughput = frame_count / avg_time if avg_time > 0 else 0
        self.throughput_history.append(current_throughput)
        
        # スループットが安定または向上している場合、バッチサイズを増加
        if len(self.throughput_history) >= 3:
            recent_avg = np.mean(self.throughput_history[-3:])
            if len(self.throughput_history) >= 6:
                previous_avg = np.mean(self.throughput_history[-6:-3])
                if recent_avg >= previous_avg * 0.95:  # スループットが5%以上低下していない
                    new_batch_size = min(self.current_batch_size * 2, self.max_batch_size)
                    if new_batch_size != self.current_batch_size:
                        print(f"[BATCH] バッチサイズ調整: {self.current_batch_size} -> {new_batch_size}")
                        self.current_batch_size = new_batch_size
                else:
                    # スループット低下時はバッチサイズを減少
                    new_batch_size = max(self.current_batch_size // 2, self.initial_batch_size)
                    if new_batch_size != self.current_batch_size:
                        print(f"[BATCH] スループット低下のためバッチサイズ調整: {self.current_batch_size} -> {new_batch_size}")
                        self.current_batch_size = new_batch_size
        
        return self.current_batch_size


class DirectVRAMWriter:
    """VRAM直接書き込みによる高速化"""
    
    def __init__(self, device="cuda:0"):
        self.device = device
        self.frame_buffers = {}
        
    def numpy_to_vram_direct(self, numpy_frame):
        """NumPy配列をVRAMに直接転送 - 安全版"""
        try:
            # NumPy配列が連続メモリか確認
            if not numpy_frame.flags['C_CONTIGUOUS']:
                numpy_frame = np.ascontiguousarray(numpy_frame)
            
            # PyTorchテンソルに変換してGPUに転送
            tensor_frame = torch.from_numpy(numpy_frame).to(self.device, non_blocking=True)
            return tensor_frame
        except Exception as e:
            print(f"[VRAM-WRITER] NumPy to VRAM転送エラー: {e}")
            # エラー時はCPU上で処理を続行
            return torch.from_numpy(numpy_frame)
    
    def vram_to_numpy_direct(self, vram_tensor):
        """VRAMからNumPy配列に直接転送 - 安全版"""
        try:
            # GPUテンソルをCPUに移動してNumPyに変換
            if hasattr(vram_tensor, 'device') and vram_tensor.device.type == 'cuda':
                cpu_tensor = vram_tensor.cpu()
                numpy_array = cpu_tensor.numpy()
            else:
                # すでにCPU上にある場合
                numpy_array = vram_tensor.numpy()
            return numpy_array
        except Exception as e:
            print(f"[VRAM-WRITER] VRAM to NumPy転送エラー: {e}")
            # エラー時は単純にnumpy()を試す
            try:
                return vram_tensor.numpy()
            except:
                # 最終手段として新規配列を作成
                return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def create_frame_buffer(self, frame_id, width, height, channels=3):
        """VRAM上にフレームバッファを作成"""
        buffer_key = f"frame_{frame_id}"
        if buffer_key not in self.frame_buffers:
            try:
                # GPUメモリ上に直接バッファを作成
                frame_buffer = torch.zeros((height, width, channels), 
                                         dtype=torch.uint8, 
                                         device=self.device)
                self.frame_buffers[buffer_key] = frame_buffer
            except Exception as e:
                print(f"[VRAM-WRITER] フレームバッファ作成エラー: {e}")
                return None
        return self.frame_buffers[buffer_key]
    
    def cleanup_buffers(self, keep_last_n=10):
        """古いバッファをクリーンアップ"""
        if len(self.frame_buffers) > keep_last_n:
            keys_to_remove = list(self.frame_buffers.keys())[:-keep_last_n]
            for key in keys_to_remove:
                try:
                    del self.frame_buffers[key]
                except:
                    pass
            if keys_to_remove:
                torch.cuda.empty_cache()


class EnhancedParallelProcessor:
    """強化された並列処理"""
    
    def __init__(self, num_workers=None, device="cuda:0"):
        self.device = device
        if num_workers is None:
            # 利用可能なGPUメモリに基づいてワーカー数を自動決定
            num_workers = self._calculate_optimal_workers()
        
        self.num_workers = num_workers
        self.worker_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.gpu_memory_optimizer = GPUMemoryOptimizer(device)
        self.batch_processor = DynamicBatchProcessor()
        
        print(f"[PARALLEL] 並列ワーカー数: {num_workers}")
    
    def _calculate_optimal_workers(self):
        """最適なワーカー数を計算"""
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory
            # 1ワーカーあたり約1GBを想定
            optimal_workers = max(1, int((total_vram * 0.7) / (1024**3)))
            return min(optimal_workers, 8)  # 最大8ワーカー
        return 4  # デフォルト


class FrameRestorer:
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode,
                 mosaic_detection=False):
        self.device = device
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.preserve_relative_scale = preserve_relative_scale
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False

        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        self.frame_restoration_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        max_clips_in_mosaic_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4))
        logger.debug(f"Set queue size of queue mosaic_clip_queue to {max_clips_in_mosaic_clips_queue}")
        self.mosaic_clip_queue = queue.Queue(maxsize=max_clips_in_mosaic_clips_queue)

        max_clips_in_restored_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4))
        logger.debug(f"Set queue size of queue restored_clip_queue to {max_clips_in_restored_clips_queue}")
        self.restored_clip_queue = queue.Queue(maxsize=max_clips_in_restored_clips_queue)

        self.frame_detection_queue = queue.Queue()

        self.mosaic_detector = MosaicDetector(self.mosaic_detection_model, self.video_meta_data.video_file,
                                              frame_detection_queue=self.frame_detection_queue,
                                              mosaic_clip_queue=self.mosaic_clip_queue,
                                              device=self.device,
                                              max_clip_length=self.max_clip_length,
                                              pad_mode=self.preferred_pad_mode,
                                              preserve_relative_scale=self.preserve_relative_scale,
                                              dont_preserve_relative_scale=(not self.preserve_relative_scale))

        self.clip_restoration_thread = None
        self.frame_restoration_thread = None
        self.clip_restoration_thread_should_be_running = False
        self.frame_restoration_thread_should_be_running = False
        self.stop_requested = False

        self.queue_stats = {}
        self.queue_stats["restored_clip_queue_max_size"] = 0
        self.queue_stats["restored_clip_queue_wait_time_put"] = 0
        self.queue_stats["restored_clip_queue_wait_time_get"] = 0
        self.queue_stats["mosaic_clip_queue_wait_time_get"] = 0
        self.queue_stats["frame_restoration_queue_max_size"] = 0
        self.queue_stats["frame_restoration_queue_wait_time_get"] = 0
        self.queue_stats["frame_restoration_queue_wait_time_put"] = 0
        self.queue_stats["frame_detection_queue_wait_time_get"] = 0

    def start(self, start_ns=0):
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

        self.mosaic_detector.start(start_ns=start_ns)
        self.clip_restoration_thread.start()
        self.frame_restoration_thread.start()

    def stop(self):
        logger.debug("FrameRestorer: stopping...")
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

        logger.debug(f"FrameRestorer: stopped, took {time.time() - start}")

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
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
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


class HighlyOptimizedFrameRestorer:
    """
    高度に最適化されたFrameRestorer - VRAM直接書き込み版
    """

    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length,
                 mosaic_restoration_model_name, mosaic_detection_model, mosaic_restoration_model,
                 preferred_pad_mode, mosaic_detection=False, batch_size=16, parallel_clips=4,
                 enable_vram_direct=True, dynamic_batch_size=True, auto_worker_count=True):

        self.device = device
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.preserve_relative_scale = preserve_relative_scale
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False

        self.parallel_clips = parallel_clips
        self.batch_size = batch_size
        self.enable_vram_direct = enable_vram_direct
        self.dynamic_batch_size = dynamic_batch_size
        self.auto_worker_count = auto_worker_count

        # VRAM最適化コンポーネントの初期化
        self.vram_writer = DirectVRAMWriter(self.device)
        self.gpu_optimizer = GPUMemoryOptimizer(self.device)
        
        if auto_worker_count:
            self.parallel_clips = self._calculate_optimal_workers()
        
        self.batch_processor = DynamicBatchProcessor(initial_batch_size=batch_size) if dynamic_batch_size else None

        # テンソルプールの事前割り当て
        self._preallocate_tensor_pool()

        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (
            self.video_meta_data.video_width * self.video_meta_data.video_height * 3
        )
        self.frame_restoration_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        max_clips_in_mosaic_clips_queue = max(
            self.parallel_clips * 2,
            (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)
        )
        logger.debug(f"mosaic_clip_queue size: {max_clips_in_mosaic_clips_queue}")
        self.mosaic_clip_queue = queue.Queue(maxsize=max_clips_in_mosaic_clips_queue)

        max_clips_in_restored_clips_queue = max(
            self.parallel_clips * 2,
            (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)
        )
        logger.debug(f"restored_clip_queue size: {max_clips_in_restored_clips_queue}")
        self.restored_clip_queue = queue.Queue(maxsize=max_clips_in_restored_clips_queue)

        self.frame_detection_queue = queue.Queue()

        self.mosaic_detector = MosaicDetector(
            self.mosaic_detection_model, self.video_meta_data.video_file,
            frame_detection_queue=self.frame_detection_queue,
            mosaic_clip_queue=self.mosaic_clip_queue,
            device=self.device,
            max_clip_length=self.max_clip_length,
            pad_mode=self.preferred_pad_mode,
            preserve_relative_scale=self.preserve_relative_scale,
            dont_preserve_relative_scale=(not self.preserve_relative_scale)
        )

        self.clip_executor = None
        self.clip_restoration_threads = []
        self.frame_restoration_thread = None
        self.clip_ordering_thread = None

        self.clip_restoration_threads_should_be_running = False
        self.frame_restoration_thread_should_be_running = False
        self.clip_ordering_thread_should_be_running = False

        self.unordered_clips_queue = queue.Queue(maxsize=max_clips_in_restored_clips_queue * 2)
        self.next_expected_clip_id = 0
        self.clip_counter = 0
        self.clip_counter_lock = threading.Lock()

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
            "vram_usage_mb": 0,
            "gpu_utilization": 0
        }

        logger.info(f"HighlyOptimizedFrameRestorer VRAM版 initialized: parallel={self.parallel_clips}, batch={batch_size}, VRAM直接={enable_vram_direct}")

    def _calculate_optimal_workers(self):
        """最適なワーカー数を計算"""
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory
            # 1ワーカーあたり約1GBを想定
            optimal_workers = max(1, int((total_vram * 0.7) / (1024**3)))
            return min(optimal_workers, 8)  # 最大8ワーカー
        return 4  # デフォルト

    def _preallocate_tensor_pool(self):
        """よく使用するテンソル形状を事前割り当て"""
        common_shapes = {
            "clip_frame_256": ((256, 256, 3), torch.uint8),
            "clip_frame_512": ((512, 512, 3), torch.uint8),
            "blend_mask": ((256, 256), torch.float32),
        }
        self.gpu_optimizer.allocate_tensor_pool(common_shapes)

    def _get_next_clip_id(self):
        with self.clip_counter_lock:
            clip_id = self.clip_counter
            self.clip_counter += 1
            return clip_id

    def start(self, start_ns=0):
        assert self.frame_restoration_thread is None
        assert self.mosaic_clip_queue.empty()
        assert self.restored_clip_queue.empty()
        assert self.frame_detection_queue.empty()
        assert self.frame_restoration_queue.empty()

        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(
            self.start_ns, self.video_meta_data.video_fps_exact
        )
        self.stop_requested = False
        self.next_expected_clip_id = 0
        self.clip_counter = 0

        self.frame_restoration_thread_should_be_running = True
        self.clip_restoration_threads_should_be_running = True
        self.clip_ordering_thread_should_be_running = True

        self.clip_executor = ThreadPoolExecutor(
            max_workers=self.parallel_clips,
            thread_name_prefix="ClipRestorer"
        )

        self.mosaic_detector.start(start_ns=start_ns)

        for i in range(self.parallel_clips):
            thread = threading.Thread(
                target=self._clip_restoration_worker_vram,
                name=f"ClipRestorationWorker-{i}"
            )
            thread.start()
            self.clip_restoration_threads.append(thread)

        self.clip_ordering_thread = threading.Thread(
            target=self._clip_ordering_worker,
            name="ClipOrderingWorker"
        )
        self.clip_ordering_thread.start()

        self.frame_restoration_thread = threading.Thread(
            target=self._frame_restoration_worker_vram,
            name="FrameRestorationWorker"
        )
        self.frame_restoration_thread.start()

        logger.info(f"HighlyOptimizedFrameRestorer VRAM版 started: {self.parallel_clips} workers")

    def stop(self):
        logger.debug("HighlyOptimizedFrameRestorer: stopping...")
        start = time.time()
        self.stop_requested = True

        self.clip_restoration_threads_should_be_running = False
        self.frame_restoration_thread_should_be_running = False
        self.clip_ordering_thread_should_be_running = False

        self.mosaic_detector.stop()

        for _ in range(self.parallel_clips):
            threading_utils.put_closing_queue_marker(self.mosaic_clip_queue, "mosaic_clip_queue")

        threading_utils.empty_out_queue(self.unordered_clips_queue, "unordered_clips_queue")

        for thread in self.clip_restoration_threads:
            if thread:
                thread.join(timeout=2.0)
        self.clip_restoration_threads.clear()
        logger.debug("clip restoration workers: stopped")

        if self.clip_executor:
            self.clip_executor.shutdown(wait=True, cancel_futures=True)
            self.clip_executor = None

        threading_utils.put_closing_queue_marker(self.unordered_clips_queue, "unordered_clips_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")

        if self.clip_ordering_thread:
            self.clip_ordering_thread.join(timeout=2.0)
            self.clip_ordering_thread = None
        logger.debug("clip ordering worker: stopped")

        threading_utils.put_closing_queue_marker(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.put_closing_queue_marker(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")

        if self.frame_restoration_thread:
            self.frame_restoration_thread.join(timeout=2.0)
            self.frame_restoration_thread = None
        logger.debug("frame restoration worker: stopped")

        # VRAMバッファのクリーンアップ
        if self.vram_writer:
            self.vram_writer.cleanup_buffers(0)
        torch.cuda.empty_cache()

        threading_utils.empty_out_queue(self.mosaic_clip_queue, "mosaic_clip_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")
        threading_utils.empty_out_queue(self.unordered_clips_queue, "unordered_clips_queue")

        logger.debug(f"HighlyOptimizedFrameRestorer: stopped, took {time.time() - start:.2f}s")
        logger.info(f"Parallel clips: {self.queue_stats['parallel_clips_processed']}")
        logger.info(f"Timeouts: {self.queue_stats['clip_timeout_count']}")

    def _restore_clip_frames_vram(self, images):
        """VRAM直接書き込みを使用したクリップフレーム復元 - 修正版"""
        try:
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
                
                # 重要: imagesはNumPy配列のリストとして渡す必要がある
                # inference()内部でimg2tensor()が呼ばれ、適切に変換される
                
                # PyTorchテンソルに変換されている場合はNumPyに戻す
                numpy_images = []
                for img in images:
                    if isinstance(img, torch.Tensor):
                        # GPUテンソルをCPUに移動してNumPyに変換
                        if img.is_cuda:
                            img = img.cpu()
                        # CHW -> HWC形式に変換(必要に応じて)
                        if img.ndim == 3 and img.shape[0] == 3:  # (C, H, W)
                            img = img.permute(1, 2, 0)
                        img = img.numpy()
                        # float32 -> uint8変換(必要に応じて)
                        if img.dtype == np.float32 and img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        elif img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                    numpy_images.append(img)
                
                # NumPy配列としてinference()に渡す
                restored_clip_images = inference(
                    self.mosaic_restoration_model, 
                    numpy_images,  # NumPy配列のリスト
                    self.device
                )
            else:
                raise NotImplementedError()
            
            return restored_clip_images
            
        except Exception as e:
            logger.error(f"VRAMクリップ復元エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # エラー時は安全な方法にフォールバック
            return self._restore_clip_frames_safe(images)

    def _restore_clip_frames_safe(self, images):
        """安全なクリップフレーム復元 - 従来の方法"""
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

    def _restore_clip_vram(self, clip):
        """VRAM直接書き込みを使用したクリップ復元"""
        try:
            if self.mosaic_detection:
                restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
            else:
                images = clip.get_clip_images()
                restored_clip_images = self._restore_clip_frames_vram(images)

            assert len(restored_clip_images) == len(clip.get_clip_images())

            for i in range(len(restored_clip_images)):
                assert clip.data[i][0].shape == restored_clip_images[i].shape
                clip.data[i] = (
                    restored_clip_images[i],
                    clip.data[i][1],
                    clip.data[i][2],
                    clip.data[i][3],
                    clip.data[i][4]
                )
            
            return clip
        except Exception as e:
            logger.error(f"VRAMクリップ復元エラー: {e}")
            # エラー時は安全な方法にフォールバック
            return self._restore_clip_safe(clip)

    def _restore_clip_safe(self, clip):
        """安全なクリップ復元"""
        try:
            if self.mosaic_detection:
                restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
            else:
                images = clip.get_clip_images()
                restored_clip_images = self._restore_clip_frames_safe(images)

            assert len(restored_clip_images) == len(clip.get_clip_images())

            for i in range(len(restored_clip_images)):
                assert clip.data[i][0].shape == restored_clip_images[i].shape
                clip.data[i] = (
                    restored_clip_images[i],
                    clip.data[i][1],
                    clip.data[i][2],
                    clip.data[i][3],
                    clip.data[i][4]
                )
            
            return clip
        except Exception as e:
            logger.error(f"安全なクリップ復元エラー: {e}")
            # 最終手段として元のクリップを返す
            return clip

    def _clip_restoration_worker_vram(self):
        """VRAM直接書き込みを使用したクリップ復元ワーカー"""
        worker_name = threading.current_thread().name
        logger.debug(f"{worker_name}: VRAM版開始")

        while self.clip_restoration_threads_should_be_running:
            try:
                s = time.time()
                clip = self.mosaic_clip_queue.get(timeout=0.1)
                self.queue_stats["mosaic_clip_queue_wait_time_get"] += time.time() - s

                if clip is None:
                    logger.debug(f"{worker_name}: 停止マーカー受信")
                    break

                clip_id = self._get_next_clip_id()
                
                # VRAM直接書き込みを使用した復元処理
                processed_clip = self._restore_clip_vram(clip)
                
                self.unordered_clips_queue.put((clip_id, processed_clip))
                self.queue_stats['parallel_clips_processed'] += 1

            except queue.Empty:
                if self.stop_requested:
                    break
                continue
            except Exception as e:
                logger.error(f"{worker_name}: VRAMクリップ処理エラー: {e}")
                if self.stop_requested:
                    break

        logger.debug(f"{worker_name}: VRAM版停止")

    def _clip_ordering_worker(self):
        logger.debug("clip ordering worker: started")

        pending_clips = []
        eof = False

        while self.clip_ordering_thread_should_be_running:
            try:
                s = time.time()
                item = self.unordered_clips_queue.get(timeout=0.1)

                if item is None:
                    if not self.stop_requested:
                        eof = True
                        break
                    else:
                        break

                clip_id, clip = item
                heapq.heappush(pending_clips, (clip_id, clip))

                while pending_clips and pending_clips[0][0] == self.next_expected_clip_id:
                    expected_id, ordered_clip = heapq.heappop(pending_clips)

                    s = time.time()
                    self.restored_clip_queue.put(ordered_clip)
                    self.queue_stats["restored_clip_queue_wait_time_put"] += time.time() - s
                    self.queue_stats["restored_clip_queue_max_size"] = max(
                        self.restored_clip_queue.qsize(),
                        self.queue_stats["restored_clip_queue_max_size"]
                    )

                    self.next_expected_clip_id += 1

                if len(pending_clips) > 0:
                    self.queue_stats['clip_wait_for_order'] += 1

            except queue.Empty:
                if self.stop_requested:
                    break
                continue
            except Exception as e:
                logger.error(f"clip ordering worker: error: {e}")
                if self.stop_requested:
                    break

        if eof:
            while pending_clips:
                if pending_clips[0][0] == self.next_expected_clip_id:
                    _, ordered_clip = heapq.heappop(pending_clips)
                    self.restored_clip_queue.put(ordered_clip)
                    self.next_expected_clip_id += 1
                else:
                    logger.warning(f"clip ordering: missing clip {self.next_expected_clip_id}")
                    break

            self.restored_clip_queue.put(None)
            logger.debug("clip ordering worker: sent EOF marker")

        logger.debug("clip ordering worker: stopped")

    def _restore_frame_vram(self, frame, frame_num, restored_clips):
        """VRAM直接書き込みを使用したフレーム復元"""
        try:
            # フレームをVRAMに転送
            vram_frame = self.vram_writer.numpy_to_vram_direct(frame)

            for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
                clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = buffered_clip.pop()
                
                # クリップ画像とマスクをVRAMに転送
                vram_clip_img = self.vram_writer.numpy_to_vram_direct(clip_img)
                vram_clip_mask = self.vram_writer.numpy_to_vram_direct(clip_mask)

                # パディング解除とリサイズをCPUで実行（安全性のため）
                clip_img_unpadded = image_utils.unpad_image(clip_img, pad_after_resize)
                clip_mask_unpadded = image_utils.unpad_image(clip_mask, pad_after_resize)
                clip_img_resized = image_utils.resize(clip_img_unpadded, orig_crop_shape[:2])
                clip_mask_resized = image_utils.resize(clip_mask_unpadded, orig_crop_shape[:2], interpolation=cv2.INTER_NEAREST)

                # リサイズ後の画像をVRAMに転送
                vram_clip_img_final = self.vram_writer.numpy_to_vram_direct(clip_img_resized)
                vram_clip_mask_final = self.vram_writer.numpy_to_vram_direct(clip_mask_resized)

                # VRAM上でブレンド処理
                t, l, b, r = orig_clip_box
                bg_region = vram_frame[t:b+1, l:r+1, :]
                
                # ブレンドマスクの作成
                blend_mask_numpy = mask_utils.create_blend_mask(clip_mask_resized)
                vram_blend_mask = self.vram_writer.numpy_to_vram_direct(blend_mask_numpy)
                
                # ブレンド実行
                blended = bg_region * (1 - vram_blend_mask[..., None]) + vram_clip_img_final * vram_blend_mask[..., None]
                blended = torch.clamp(blended, 0, 255).byte()
                
                # 元のフレームにブレンド結果を書き込み
                vram_frame[t:b+1, l:r+1, :] = blended

            # 結果をCPUに戻す
            return self.vram_writer.vram_to_numpy_direct(vram_frame)
            
        except Exception as e:
            logger.error(f"VRAMフレーム復元エラー: {e}")
            # エラー時は従来の方法にフォールバック
            return self._restore_frame_safe(frame, frame_num, restored_clips)

    def _restore_frame_safe(self, frame, frame_num, restored_clips):
        """安全なフレーム復元 - 従来の方法"""
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
        
        return frame

    def _collect_garbage(self, clip_buffer):
        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
        for processed_clip in processed_clips:
            clip_buffer.remove(processed_clip)

    def _contains_at_least_one_clip_starting_after_frame_num(self, frame_num, clip_buffer):
        return len(clip_buffer) > 0 and frame_num < max(clip_buffer, key=lambda c: c.frame_start).frame_start

    def _read_next_frame(self, video_frames_generator, expected_frame_num):
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            s = time.time()
            elem = self.frame_detection_queue.get()
            self.queue_stats["frame_detection_queue_wait_time_get"] += time.time() - s
            if self.stop_requested:
                logger.debug("frame restoration worker: frame_detection_queue EOF")
            assert elem is None, f"Expected None but got {elem}"
            return None

        s = time.time()
        elem = self.frame_detection_queue.get()
        self.queue_stats["frame_detection_queue_wait_time_get"] += time.time() - s
        if self.stop_requested:
            return None

        assert elem is not None, "Expected detection result but got None"
        detection_frame_num, mosaic_detected = elem
        assert self.stop_requested or detection_frame_num == expected_frame_num, \
            f"Frame detection queue out of sync: {detection_frame_num} != {expected_frame_num}"
        return mosaic_detected, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer):
        """次のクリップを読み込み - タイムアウトリカバリ強化版"""
        s = time.time()
        try:
            clip = self.restored_clip_queue.get(timeout=5.0)
            self.queue_stats["restored_clip_queue_wait_time_get"] += time.time() - s

            if self.stop_requested:
                return False
            if clip is None:
                return False

            assert self.stop_requested or clip.frame_start >= current_frame_num, \
                f"Clip queue out of sync: {clip.frame_start} < {current_frame_num}"
            clip_buffer.append(clip)
            return True

        except queue.Empty:
            logger.warning(f"Clip queue timeout at frame {current_frame_num} - continuing without restoration")
            self.queue_stats["restored_clip_queue_wait_time_get"] += time.time() - s
            self.queue_stats["clip_timeout_count"] += 1
            return False

    def _frame_restoration_worker_vram(self):
        """VRAM直接書き込みを使用したフレーム復元ワーカー"""
        logger.debug("frame restoration worker: VRAM版開始")

        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()
            frame_num = self.start_frame
            clips_remaining = True
            clip_buffer = []
            timeout_count = 0
            max_consecutive_timeouts = 3

            while self.frame_restoration_thread_should_be_running:
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if _frame_result is None:
                    if not self.stop_requested:
                        self.eof = True
                        self.frame_restoration_thread_should_be_running = False
                        self.frame_restoration_queue.put(None)
                    break

                mosaic_detected, frame, frame_pts = _frame_result

                if mosaic_detected:
                    clip_wait_start = time.time()
                    clip_wait_timeout = 10.0

                    while clips_remaining and not self._contains_at_least_one_clip_starting_after_frame_num(
                        frame_num, clip_buffer
                    ):
                        if time.time() - clip_wait_start > clip_wait_timeout:
                            logger.warning(f"Frame {frame_num}: Clip collection timeout after {clip_wait_timeout}s")
                            timeout_count += 1

                            if timeout_count >= max_consecutive_timeouts:
                                logger.error(f"Frame {frame_num}: Too many timeouts ({timeout_count}) - GPU overload?")
                            break

                        clips_remaining = self._read_next_clip(frame_num, clip_buffer)
                        if not clips_remaining:
                            break
                    else:
                        timeout_count = 0

                    if len([c for c in clip_buffer if c.frame_start == frame_num]) > 0:
                        frame = self._restore_frame_vram(frame, frame_num, clip_buffer)
                    else:
                        if timeout_count > 0:
                            logger.debug(f"Frame {frame_num}: No clips available, outputting original")

                    self._collect_garbage(clip_buffer)

                self.queue_stats["frame_restoration_queue_max_size"] = max(
                    self.frame_restoration_queue.qsize() + 1,
                    self.queue_stats["frame_restoration_queue_max_size"]
                )
                s = time.time()
                self.frame_restoration_queue.put((frame, frame_pts))
                self.queue_stats["frame_restoration_queue_wait_time_put"] += time.time() - s

                if self.stop_requested:
                    break

                frame_num += 1

            if self.eof:
                logger.debug("frame restoration worker: VRAM版 EOF")

        logger.debug("frame restoration worker: VRAM版停止")

    def __iter__(self):
        return self

    def __next__(self):
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration

        while not self.stop_requested:
            s = time.time()
            elem = self.frame_restoration_queue.get()
            self.queue_stats["frame_restoration_queue_wait_time_get"] += time.time() - s

            if self.stop_requested:
                return None
            if elem is None and not self.stop_requested:
                raise StopIteration
            return elem

    def get_frame_restoration_queue(self):
        return self.frame_restoration_queue


# 後方互換性のためのエイリアス
OptimizedFrameRestorer = HighlyOptimizedFrameRestorer
