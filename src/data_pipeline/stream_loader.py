import asyncio
import cv2
import queue
import threading
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StreamLoader:
    def __init__(self, source_path, max_queue_size=100):
        self.source_path = source_path
        self.max_queue_size = max_queue_size
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=self.max_queue_size)
        self.running = False
        self.thread = None

    def start(self):
        logger.info(f"Opening stream source: {self.source_path}")
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open stream source: {self.source_path}")
            raise ValueError(f"Cannot open video source: {self.source_path}")
        
        self.running = True
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()
        logger.info("Stream reader producer thread started.")

    def _producer(self):
        target_interval = 1.0 / 30.0  # 30 FPS
        while self.running:
            if not self.cap or not self.cap.isOpened():
                break
            
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                break
            
            # Put frame in queue (blocks if queue is full to prevent memory bloat)
            try:
                self.frame_queue.put(frame, timeout=1.0)
            except queue.Full:
                logger.warning("Frame queue is full, waiting...")
                continue
            
            # Compute remaining time to sleep to maintain 30 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        self.running = False
        if self.cap:
            self.cap.release()

    async def read_frames_async(self):
        """Asynchronously yield frames from the queue without blocking the event loop."""
        if not self.running:
            self.start()
            
        while self.running or not self.frame_queue.empty():
            try:
                # Retrieve from queue in a non-blocking thread execution
                frame = await asyncio.to_thread(self.frame_queue.get, True, 0.1)
                yield frame
                self.frame_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Released stream loader resources.")
