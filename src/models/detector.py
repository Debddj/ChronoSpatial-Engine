import os
import cv2
import numpy as np
import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ObjectDetector:
    def __init__(self, config):
        model_cfg = config.get("model", {})
        spatial_cfg = model_cfg.get("spatial_hyperparameters", {})
        if not spatial_cfg:
            spatial_cfg = config.get("spatial_hyperparameters", {})
            
        self.detector_type = spatial_cfg.get("detector_type", "yolov8").lower()
        self.confidence_threshold = spatial_cfg.get("confidence_threshold", 0.25)
        self.bounding_box_limit = spatial_cfg.get("bounding_box_limit", 50)
        
        self.model = None
        
        logger.info(f"Initializing ObjectDetector of type '{self.detector_type}'...")
        
        if self.detector_type == "yolov8":
            try:
                import importlib
                ultralytics = importlib.import_module("ultralytics")
                YOLO = ultralytics.YOLO
                # Load YOLOv8n model
                self.model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 detector successfully loaded.")
            except Exception as e:
                logger.warning(f"Failed to load YOLOv8 detector: {e}. Falling back to 'torchvision' detector.")
                self.detector_type = "torchvision"
                
        if self.detector_type == "torchvision":
            try:
                from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
                weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                self.model = ssdlite320_mobilenet_v3_large(weights=weights)
                self.model.eval()
                # Run a dry run to verify
                dummy_input = torch.zeros((3, 320, 320), dtype=torch.float32)
                with torch.no_grad():
                    self.model([dummy_input])
                logger.info("Torchvision SSDLite detector successfully loaded.")
            except Exception as e:
                logger.warning(f"Failed to load Torchvision detector: {e}. Falling back to 'contours' detector.")
                self.detector_type = "contours"
                
        if self.detector_type == "contours":
            logger.info("Contours detector successfully loaded (using OpenCV contour analysis).")

    def detect(self, frame):
        """
        Runs object detection on the input BGR frame.
        Returns a list of dicts:
        [
            {
                "bbox": [xmin, ymin, xmax, ymax],
                "confidence": float,
                "class_id": int
            },
            ...
        ]
        Sorted by confidence descending and capped by bounding_box_limit.
        """
        detections = []
        
        if frame is None or frame.size == 0:
            return detections
            
        h, w, _ = frame.shape
        
        if self.detector_type == "yolov8" and self.model is not None:
            try:
                # Use YOLOv8n to detect objects
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                for r in results:
                    for box in r.boxes:
                        coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        detections.append({
                            "bbox": coords,
                            "confidence": conf,
                            "class_id": cls
                        })
            except Exception as e:
                logger.error(f"YOLOv8 detection failed: {e}. Falling back to contours.")
                # Temporary fallback if model runtime fails
                self.detector_type = "contours"
                
        if self.detector_type == "torchvision" and self.model is not None:
            try:
                # Convert BGR to RGB and format to [0, 1] torch tensor
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_input = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                
                with torch.no_grad():
                    predictions = self.model([tensor_input])[0]
                    
                boxes = predictions["boxes"].cpu().numpy()
                scores = predictions["scores"].cpu().numpy()
                labels = predictions["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score >= self.confidence_threshold:
                        detections.append({
                            "bbox": box.tolist(),
                            "confidence": float(score),
                            "class_id": int(label)
                        })
            except Exception as e:
                logger.error(f"Torchvision detection failed: {e}. Falling back to contours.")
                self.detector_type = "contours"
                
        if self.detector_type == "contours":
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # Apply adaptive thresholding to detect shapes/contours
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Filter very small/noisy contours, and contours that occupy the whole image
                    if 150 < area < (w * h * 0.9):
                        x, y, w_box, h_box = cv2.boundingRect(cnt)
                        xmin, ymin, xmax, ymax = float(x), float(y), float(x + w_box), float(y + h_box)
                        
                        # Compute pseudo-confidence based on area size relative to image
                        conf = float(np.clip(area / 10000.0, 0.3, 0.95))
                        detections.append({
                            "bbox": [xmin, ymin, xmax, ymax],
                            "confidence": conf,
                            "class_id": 0  # default object label
                        })
            except Exception as e:
                logger.error(f"Contours detection failed: {e}")
                
        # Sort by confidence descending and apply bounding_box_limit
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        return detections[:self.bounding_box_limit]
