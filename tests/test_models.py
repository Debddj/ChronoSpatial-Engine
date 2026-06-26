import numpy as np
from src.models.cnn_extractor import CNNExtractor
from src.models.ann_regressor import ANNRegressor
from src.models.inference_engine import InferenceEngine

import torch

def test_cnn_extractor():
    extractor = CNNExtractor(feature_dim=128)
    
    # 1. Test PyTorch forward pass: (1, 3, 224, 224) input tensor -> (1, 128) output tensor
    dummy_tensor = torch.randn(1, 3, 224, 224).float()
    embeddings = extractor(dummy_tensor)
    assert embeddings.shape == (1, 128)
    assert isinstance(embeddings, torch.Tensor)
    
    # 2. Test NumPy compatibility wrapper: (1, 3, 224, 224) input array -> (128,) output array
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    features = extractor.extract_features(dummy_input)
    assert features.shape == (128,)
    assert isinstance(features, np.ndarray)

def test_ann_regressor():
    regressor = ANNRegressor(input_dim=64, risk_threshold=0.5)
    # Target features with positive elements to produce higher risk
    features = np.ones(64, dtype=np.float32) * 0.5
    results = regressor.predict_risk(features)
    assert "risk_score" in results
    assert "is_anomaly" in results
    assert 0.0 <= results["risk_score"] <= 1.0
    assert isinstance(results["is_anomaly"], bool)

def test_inference_engine():
    # Test with default (quantized INT8) ONNX model
    config = {
        "model": {
            "onnx_model_path": "models/chronospatial_unified_quantized.onnx",
            "ann_regressor": {"risk_threshold": 0.8}
        }
    }
    engine = InferenceEngine(config)
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    results = engine.run_inference(dummy_frame)
    assert "risk_score" in results
    assert "is_anomaly" in results
    assert "inference_time_ms" in results
    assert 0.0 <= results["risk_score"] <= 1.0
    assert isinstance(results["is_anomaly"], bool)
    
    # Test with FP32 ONNX model
    config_fp32 = {
        "model": {
            "onnx_model_path": "models/chronospatial_unified.onnx",
            "ann_regressor": {"risk_threshold": 0.8}
        }
    }
    engine_fp32 = InferenceEngine(config_fp32)
    results_fp32 = engine_fp32.run_inference(dummy_frame)
    assert "risk_score" in results_fp32
    assert "is_anomaly" in results_fp32
    assert "inference_time_ms" in results_fp32
    assert 0.0 <= results_fp32["risk_score"] <= 1.0
    assert isinstance(results_fp32["is_anomaly"], bool)

