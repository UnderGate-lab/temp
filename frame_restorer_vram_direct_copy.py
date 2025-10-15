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


class OptimizedFrameRestorer:
    """
    並列化されたFrameRestorer - タイムアウトリカバリ機能強化版
    複数のクリップを同時に処理しながら、フレーム順序を厳守
    """
    
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length, 
                 mosaic_restoration_model_name, mosaic_detection_model, mosaic_restoration_model, 
                 preferred_pad_mode, mosaic_detection=False, batch_size=16, parallel_clips=4):
        
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
        
        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (
            self.video_meta_data.video_width * self.video_meta_data.video_height * 3
        )
        self.frame_restoration_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)
        
        max_clips_in_mosaic_clips_queue = max(
            parallel_clips * 2,
            (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)
        )
        logger.debug(f"mosaic_clip_queue size: {max_clips_in_mosaic_clips_queue}")
        self.mosaic_clip_queue = queue.Queue(maxsize=max_clips_in_mosaic_clips_queue)
        
        max_clips_in_restored_clips_queue = max(
            parallel_clips * 2,
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
            "clip_timeout_count": 0
        }
        
        logger.info(f"OptimizedFrameRestorer initialized: parallel={parallel_clips}, batch={batch_size}")
    
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
                target=self._clip_restoration_worker,
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
            target=self._frame_restoration_worker,
            name="FrameRestorationWorker"
        )
        self.frame_restoration_thread.start()
        
        logger.info(f"OptimizedFrameRestorer started: {self.parallel_clips} workers")
    
    def stop(self):
        logger.debug("OptimizedFrameRestorer: stopping...")
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
        
        threading_utils.empty_out_queue(self.mosaic_clip_queue, "mosaic_clip_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")
        threading_utils.empty_out_queue(self.unordered_clips_queue, "unordered_clips_queue")
        
        logger.debug(f"OptimizedFrameRestorer: stopped, took {time.time() - start:.2f}s")
        logger.info(f"Parallel clips: {self.queue_stats['parallel_clips_processed']}")
        logger.info(f"Timeouts: {self.queue_stats['clip_timeout_count']}")
    
    def _restore_clip_frames(self, images):
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
            restored_clip_images = inference(
                self.mosaic_restoration_model,
                images,
                self.device
            )
        else:
            raise NotImplementedError()
        return restored_clip_images
    
    def _restore_clip(self, clip):
        if self.mosaic_detection:
            restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
        else:
            images = clip.get_clip_images()
            restored_clip_images = self._restore_clip_frames(images)
        
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
    
    def _clip_restoration_worker(self):
        worker_name = threading.current_thread().name
        logger.debug(f"{worker_name}: started")
        
        while self.clip_restoration_threads_should_be_running:
            try:
                s = time.time()
                clip = self.mosaic_clip_queue.get(timeout=0.1)
                self.queue_stats["mosaic_clip_queue_wait_time_get"] += time.time() - s
                
                if clip is None:
                    logger.debug(f"{worker_name}: received stop marker")
                    break
                
                clip_id = self._get_next_clip_id()
                self._restore_clip(clip)
                
                self.unordered_clips_queue.put((clip_id, clip))
                self.queue_stats['parallel_clips_processed'] += 1
                
            except queue.Empty:
                if self.stop_requested:
                    break
                continue
            except Exception as e:
                logger.error(f"{worker_name}: error processing clip: {e}")
                if self.stop_requested:
                    break
        
        logger.debug(f"{worker_name}: stopped")
    
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
            clip = self.restored_clip_queue.get(timeout=10.0)
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
    
    def _frame_restoration_worker(self):
        """フレーム復元ワーカー - タイムアウトリカバリ強化版"""
        logger.debug("frame restoration worker: started")
        
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
                    clip_wait_timeout = 15.0
                    
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
                        self._restore_frame(frame, frame_num, clip_buffer)
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
                logger.debug("frame restoration worker: EOF")
        
        logger.debug("frame restoration worker: stopped")
    
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
