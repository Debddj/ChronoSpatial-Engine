import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CNNExtractor:
    def __init__(self, feature_dim=128):
        self.feature_dim = feature_dim
        logger.info(f"Initialized CNNExtractor with feature dimension {self.feature_dim}")

    def extract_features(self, preprocessed_frame):
        # Mock feature extraction: generate deterministic features based on frame mean
        # to simulate spatial features.
        mean_val = np.mean(preprocessed_frame)
        np.random.seed(int(mean_val * 1000) % 2**32)
        features = np.random.randn(self.feature_dim).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
