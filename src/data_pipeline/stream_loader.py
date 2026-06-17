import cv2
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StreamLoader:
    def __init__(self, source_path):
        self.source_path = source_path
        self.cap = None

    def start(self):
        logger.info(f"Opening stream source: {self.source_path}")
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open stream source: {self.source_path}")
            raise ValueError(f"Cannot open video source: {self.source_path}")

    def read_frames(self):
        if not self.cap or not self.cap.isOpened():
            self.start()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.info("End of stream or read error.")
                break
            yield frame
            # Simulate real-time frame rate if video file
            time.sleep(0.033)  # ~30 FPS

    def release(self):
        if self.cap:
            self.cap.release()
            logger.info("Released stream source.")
