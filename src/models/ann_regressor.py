import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ANNRegressor:
    def __init__(self, input_dim=128, risk_threshold=0.80):
        self.input_dim = input_dim
        self.risk_threshold = risk_threshold
        logger.info(f"Initialized ANNRegressor with threshold {self.risk_threshold}")

    def predict_risk(self, features, temporal_history=None):
        # Mock regression: compute a risk score from features
        base_score = np.dot(features, np.ones_like(features) * 0.1)
        risk_score = float(1.0 / (1.0 + np.exp(-base_score)))  # Sigmoid to [0, 1]
        
        # Clip to [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))
        
        is_anomaly = risk_score > self.risk_threshold
        return {
            "risk_score": risk_score,
            "is_anomaly": is_anomaly
        }
