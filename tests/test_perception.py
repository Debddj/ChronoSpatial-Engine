import numpy as np
import pytest
import cv2
from src.models.detector import ObjectDetector
from src.models.tracker import SimpleTracker
from src.models.inference_engine import InferenceEngine


def test_detector_modes():
    config = {
        "spatial_hyperparameters": {
            "detector_type": "contours",
            "confidence_threshold": 0.25,
            "bounding_box_limit": 50
        }
    }
    detector = ObjectDetector(config)
    assert detector.detector_type == "contours"
    
    # Run detector on dummy frame with a drawn rectangle
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 50), (180, 150), (255, 255, 255), -1)
    
    detections = detector.detect(frame)
    assert len(detections) > 0
    assert "bbox" in detections[0]
    assert "confidence" in detections[0]
    assert "class_id" in detections[0]


def test_tracker_association():
    config = {
        "spatial_hyperparameters": {
            "grid_size": 16,
            "max_objects": 10,
            "iou_threshold": 0.3,
            "distance_scale_factor": 1.0
        },
        "model": {
            "ann_regressor": {
                "max_history": 5
            }
        }
    }
    tracker = SimpleTracker(config)
    
    # 1. Update with one detection
    detections_1 = [{"bbox": [10, 20, 30, 40], "confidence": 0.9, "class_id": 0}]
    tracks_1 = tracker.update(detections_1, (100, 100, 3))
    assert len(tracks_1) == 1
    tr1 = next(iter(tracks_1.values()))
    assert tr1.track_id == 1
    
    # 2. Update with slightly moved detection (IoU should match)
    detections_2 = [{"bbox": [12, 22, 32, 42], "confidence": 0.95, "class_id": 0}]
    tracks_2 = tracker.update(detections_2, (100, 100, 3))
    assert len(tracks_2) == 1
    tr2 = next(iter(tracks_2.values()))
    assert tr2.track_id == 1
    assert tr2.age == 2
    
    # Verify velocity and distance
    vel_hist, dist_hist = tr2.get_temporal_history()
    assert vel_hist.shape == (5, 2)
    assert dist_hist.shape == (5,)
    # Last velocity should be positive due to movement
    assert vel_hist[-1][0] > 0
    assert vel_hist[-1][1] > 0


def test_inference_engine_with_perception():
    config = {
        "model": {
            "onnx_model_path": "models/chronospatial_unified_quantized.onnx",
            "ann_regressor": {
                "risk_threshold": 0.8,
                "max_history": 5
            },
            "spatial_hyperparameters": {
                "detector_type": "contours",
                "confidence_threshold": 0.25,
                "grid_size": 16,
                "max_objects": 10,
                "bounding_box_limit": 10,
                "iou_threshold": 0.3,
                "distance_scale_factor": 1.0
            }
        }
    }
    
    engine = InferenceEngine(config)
    
    # Generate frame with an object (drawn white rectangle)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    
    # Run inference without custom temporal inputs
    results = engine.run_inference(frame)
    assert "risk_score" in results
    assert "is_anomaly" in results
    assert "tracked_objects" in results
    assert len(results["tracked_objects"]) > 0
    
    obj = results["tracked_objects"][0]
    assert "track_id" in obj
    assert "bbox" in obj
    assert "grid_cell" in obj
    assert "velocity" in obj
    assert "distance" in obj
    assert "risk_score" in obj
