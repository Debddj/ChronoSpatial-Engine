import time
import os
import numpy as np
import onnxruntime as ort
from src.data_pipeline.transforms import preprocess_frame
from src.models.ann_regressor import create_temporal_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    def __init__(self, config):
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
        
        logger.info("InferenceEngine successfully compiled and initialized with ONNX Runtime.")

    def run_inference(self, frame, velocity_vectors=None, asset_distances=None):
        start_time = time.perf_counter()
        
        # 1. Preprocess frame: (1, 3, 224, 224)
        preprocessed = preprocess_frame(frame)
        
        # 2. Build temporal features: (15,)
        temporal_features = create_temporal_features(
            velocity_vectors=velocity_vectors,
            asset_distances=asset_distances,
            max_history=self.max_history
        )
        
        # 3. Add batch dimension: (1, 15)
        temporal_features_batch = np.expand_dims(temporal_features, axis=0)
        
        # 4. Execute ONNX session inference
        inputs = {
            "image": preprocessed.astype(np.float32),
            "temporal_features": temporal_features_batch.astype(np.float32)
        }
        
        outputs = self.session.run(None, inputs)
        
        # 5. Extract results
        risk_score = float(outputs[0][0][0])
        risk_score = max(0.0, min(1.0, risk_score))
        is_anomaly = risk_score > self.risk_threshold
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "risk_score": risk_score,
            "is_anomaly": is_anomaly,
            "inference_time_ms": elapsed_ms
        }

    def reset_temporal_buffer(self):
        self.temporal_buffer = []