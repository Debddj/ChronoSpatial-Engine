import time
import os
import numpy as np
import cv2
import onnxruntime as ort
from src.data_pipeline.transforms import preprocess_frame
from src.models.ann_regressor import create_temporal_features
from src.models.detector import ObjectDetector
from src.models.tracker import SimpleTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    def __init__(self, config):
        self.config = config
        model_cfg = config.get("model", {})
        ann_cfg = model_cfg.get("ann_regressor", {})
        
        # Resolve ONNX model path robustly
        onnx_path = model_cfg.get("onnx_model_path", "models/chronospatial_unified_quantized.onnx")
        if not os.path.exists(onnx_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            onnx_path = os.path.join(base_dir, onnx_path)
            
        logger.info(f"Loading ONNX model from: {onnx_path}")
        
        # Initialize ONNX Runtime Session (CPU Execution Provider)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        self.risk_threshold = ann_cfg.get("risk_threshold", 0.80)
        self.max_history = ann_cfg.get("max_history", 5)
        self.temporal_buffer = []
        
        # Initialize Detector and Tracker
        self.detector = ObjectDetector(config)
        self.tracker = SimpleTracker(config)
        
        logger.info("InferenceEngine successfully compiled and initialized with ONNX Runtime, Detector, and Tracker.")

    def run_inference(self, frame, velocity_vectors=None, asset_distances=None):
        start_time = time.perf_counter()
        
        h_f, w_f, _ = frame.shape
        
        # Always run detector and tracker to update tracking states
        detections = self.detector.detect(frame)
        active_tracks = self.tracker.update(detections, frame.shape)
        
        tracked_objects = []
        
        # If legacy/custom features are provided directly by the caller, run full-frame inference
        if velocity_vectors is not None or asset_distances is not None:
            preprocessed = preprocess_frame(frame)
            temporal_features = create_temporal_features(
                velocity_vectors=velocity_vectors,
                asset_distances=asset_distances,
                max_history=self.max_history
            )
            temporal_features_batch = np.expand_dims(temporal_features, axis=0)
            
            inputs = {
                "image": preprocessed.astype(np.float32),
                "temporal_features": temporal_features_batch.astype(np.float32)
            }
            outputs = self.session.run(None, inputs)
            risk_score = float(outputs[0][0][0])
            risk_score = max(0.0, min(1.0, risk_score))
            
            # Map any active tracks to output format (but don't run separate inference for them)
            for tid, tr in active_tracks.items():
                vel_hist, dist_hist = tr.get_temporal_history()
                cx = (tr.bbox[0] + tr.bbox[2]) / (2.0 * w_f)
                cy = (tr.bbox[1] + tr.bbox[3]) / (2.0 * h_f)
                grid_x = int(np.clip(cx * self.tracker.grid_size, 0, self.tracker.grid_size - 1))
                grid_y = int(np.clip(cy * self.tracker.grid_size, 0, self.tracker.grid_size - 1))
                
                tracked_objects.append({
                    "track_id": tr.track_id,
                    "bbox": tr.bbox,
                    "grid_cell": [grid_x, grid_y],
                    "velocity": list(vel_hist[-1]),
                    "distance": float(dist_hist[-1]),
                    "risk_score": risk_score # share overall frame risk score
                })
        else:
            # Automatic Object-Level Inference
            if len(active_tracks) > 0:
                for tid, tr in active_tracks.items():
                    xmin, ymin, xmax, ymax = tr.bbox
                    
                    # 1. Clamp bounding box coordinates to frame boundaries
                    xmin_c = int(max(0, min(w_f - 1, xmin)))
                    ymin_c = int(max(0, min(h_f - 1, ymin)))
                    xmax_c = int(max(xmin_c + 1, min(w_f, xmax)))
                    ymax_c = int(max(ymin_c + 1, min(h_f, ymax)))
                    
                    # 2. Crop object and preprocess
                    crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]
                    preprocessed_crop = preprocess_frame(crop)
                    
                    # 3. Retrieve temporal history
                    vel_hist, dist_hist = tr.get_temporal_history()
                    temporal_features = create_temporal_features(
                        velocity_vectors=vel_hist,
                        asset_distances=dist_hist,
                        max_history=self.max_history
                    )
                    temporal_features_batch = np.expand_dims(temporal_features, axis=0)
                    
                    # 4. Run ONNX session on object crop
                    inputs = {
                        "image": preprocessed_crop.astype(np.float32),
                        "temporal_features": temporal_features_batch.astype(np.float32)
                    }
                    outputs = self.session.run(None, inputs)
                    obj_risk_score = float(outputs[0][0][0])
                    obj_risk_score = max(0.0, min(1.0, obj_risk_score))
                    
                    # 5. Compute grid cell position
                    cx = (xmin + xmax) / (2.0 * w_f)
                    cy = (ymin + ymax) / (2.0 * h_f)
                    grid_x = int(np.clip(cx * self.tracker.grid_size, 0, self.tracker.grid_size - 1))
                    grid_y = int(np.clip(cy * self.tracker.grid_size, 0, self.tracker.grid_size - 1))
                    
                    tracked_objects.append({
                        "track_id": tr.track_id,
                        "bbox": tr.bbox,
                        "grid_cell": [grid_x, grid_y],
                        "velocity": list(vel_hist[-1]),
                        "distance": float(dist_hist[-1]),
                        "risk_score": obj_risk_score
                    })
                
                # Overall risk is the maximum risk score of tracked objects
                risk_score = max([obj["risk_score"] for obj in tracked_objects])
            else:
                # Fallback: run on full frame with zero temporal features
                preprocessed = preprocess_frame(frame)
                temporal_features = create_temporal_features(
                    velocity_vectors=None,
                    asset_distances=None,
                    max_history=self.max_history
                )
                temporal_features_batch = np.expand_dims(temporal_features, axis=0)
                
                inputs = {
                    "image": preprocessed.astype(np.float32),
                    "temporal_features": temporal_features_batch.astype(np.float32)
                }
                outputs = self.session.run(None, inputs)
                risk_score = float(outputs[0][0][0])
                risk_score = max(0.0, min(1.0, risk_score))
                
        is_anomaly = risk_score > self.risk_threshold
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "risk_score": risk_score,
            "is_anomaly": is_anomaly,
            "inference_time_ms": elapsed_ms,
            "tracked_objects": tracked_objects
        }

    def reset_temporal_buffer(self):
        self.temporal_buffer = []