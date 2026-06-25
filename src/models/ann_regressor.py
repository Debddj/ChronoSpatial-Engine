import numpy as np
import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ANNRegressor(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=(256, 128, 64), dropout_rate=0.3, risk_threshold=0.80):
        super(ANNRegressor, self).__init__()
        self.input_dim = input_dim
        self.risk_threshold = risk_threshold
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"ANNRegressor initialized: input_dim={input_dim}, hidden_dims={hidden_dims}, dropout={dropout_rate}")

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        return self.network(x)

    def predict_risk(self, features, temporal_history=None):
        self.eval()
        with torch.no_grad():
            if temporal_history is not None:
                if not isinstance(temporal_history, torch.Tensor):
                    temporal_history = torch.from_numpy(temporal_history).float()
                if len(temporal_history.shape) == 1:
                    temporal_history = temporal_history.unsqueeze(0)
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)
                x = torch.cat([features, temporal_history], dim=-1)
            else:
                if not isinstance(features, torch.Tensor):
                    features = torch.from_numpy(features).float()
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)
                x = features
            
            risk_score = self.forward(x)
            risk_score = risk_score.squeeze().item()
            risk_score = max(0.0, min(1.0, risk_score))
            
            is_anomaly = risk_score > self.risk_threshold
            return {
                "risk_score": risk_score,
                "is_anomaly": is_anomaly
            }


def create_temporal_features(velocity_vectors=None, asset_distances=None, max_history=5):
    if velocity_vectors is None:
        velocity_vectors = np.zeros((max_history, 2), dtype=np.float32)
    if asset_distances is None:
        asset_distances = np.zeros(max_history, dtype=np.float32)
    
    velocity_features = velocity_vectors.flatten()
    distance_features = asset_distances.flatten()
    
    return np.concatenate([velocity_features, distance_features]).astype(np.float32)