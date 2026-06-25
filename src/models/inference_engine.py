import time
import numpy as np
import torch
from src.data_pipeline.transforms import preprocess_frame
from src.models.cnn_extractor import CNNExtractor
from src.models.ann_regressor import ANNRegressor, create_temporal_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    def __init__(self, config):
        cnn_cfg = config.get("model", {}).get("cnn_extractor", {})
        ann_cfg = config.get("model", {}).get("ann_regressor", {})
        
        self.extractor = CNNExtractor(feature_dim=cnn_cfg.get("feature_dim", 128))
        self.regressor = ANNRegressor(
            input_dim=ann_cfg.get("input_dim", 128 + 12),
            hidden_dims=tuple(ann_cfg.get("hidden_dims", [256, 128, 64])),
            dropout_rate=ann_cfg.get("dropout_rate", 0.3),
            risk_threshold=ann_cfg.get("risk_threshold", 0.80)
        )
        
        self.max_history = ann_cfg.get("max_history", 5)
        self.temporal_buffer = []
        
        logger.info("InferenceEngine successfully compiled and initialized.")

    def run_inference(self, frame, velocity_vectors=None, asset_distances=None):
        start_time = time.perf_counter()
        
        preprocessed = preprocess_frame(frame)
        spatial_features = self.extractor.extract_features(preprocessed)
        
        temporal_features = create_temporal_features(
            velocity_vectors=velocity_vectors,
            asset_distances=asset_distances,
            max_history=self.max_history
        )
        
        combined_features = np.concatenate([spatial_features, temporal_features])
        
        results = self.regressor.predict_risk(combined_features)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        results["inference_time_ms"] = elapsed_ms
        
        return results

    def reset_temporal_buffer(self):
        self.temporal_buffer = []