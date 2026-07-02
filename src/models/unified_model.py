import torch
import torch.nn as nn
import torchvision.models as models
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedChronoSpatialModel(nn.Module):
    """Unified model combining CNN feature extractor and ANN risk regressor for ONNX export."""
    
    def __init__(self, cnn_feature_dim=128, ann_input_dim=143, ann_hidden_dims=(256, 128, 64), 
                 ann_dropout_rate=0.3, risk_threshold=0.80):
        super(UnifiedChronoSpatialModel, self).__init__()
        
        # CNN Backbone (MobileNetV3 Small)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.cnn_backbone = models.mobilenet_v3_small(weights=weights)
        
        # Freeze backbone
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
            
        # Get CNN output features
        cnn_out_features = self.cnn_backbone.classifier[0].in_features
        self.cnn_backbone.classifier = nn.Identity()
        
        # CNN projection
        self.cnn_projection = nn.Linear(cnn_out_features, cnn_feature_dim)
        
        # ANN Regressor
        layers = []
        prev_dim = ann_input_dim
        
        for hidden_dim in ann_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(ann_dropout_rate))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.ann_network = nn.Sequential(*layers)
        self.risk_threshold = risk_threshold
        
        logger.info(f"UnifiedChronoSpatialModel initialized: CNN={cnn_feature_dim}, ANN input={ann_input_dim}")

    def forward(self, image, temporal_features):
        """
        Forward pass for unified model.
        Args:
            image: [batch, 3, 224, 224] preprocessed image tensor
            temporal_features: [batch, 15] temporal features (velocity + distances)
        Returns:
            risk_score: [batch, 1] collision risk index in [0, 1]
        """
        # CNN feature extraction
        cnn_features = self.cnn_backbone(image)
        cnn_embedding = self.cnn_projection(cnn_features)
        
        # Concatenate CNN embedding with temporal features
        combined = torch.cat([cnn_embedding, temporal_features], dim=1)
        
        # ANN risk prediction
        risk_score = self.ann_network(combined)
        return risk_score


def create_unified_model(config=None):
    """Factory function to create unified model from config."""
    if config is None:
        config = {}
    
    cnn_cfg = config.get("cnn_extractor", {})
    ann_cfg = config.get("ann_regressor", {})
    
    model = UnifiedChronoSpatialModel(
        cnn_feature_dim=cnn_cfg.get("feature_dim", 128),
        ann_input_dim=ann_cfg.get("input_dim", 143),
        ann_hidden_dims=tuple(ann_cfg.get("hidden_dims", [256, 128, 64])),
        ann_dropout_rate=ann_cfg.get("dropout_rate", 0.3),
        risk_threshold=ann_cfg.get("risk_threshold", 0.80)
    )
    
    # Check if a trained PyTorch model path is provided or exists at standard location
    pt_path = config.get("pytorch_model_path", "models/chronospatial_unified.pt")
    import os
    if not os.path.exists(pt_path):
        # Resolve path relative to project root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        pt_path = os.path.join(base_dir, pt_path)
        
    if os.path.exists(pt_path):
        import torch
        logger.info(f"Loading trained PyTorch model weights from: {pt_path}")
        try:
            model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        except Exception as e:
            logger.warning(f"Could not load state dict from {pt_path} (this is expected if running tests with custom configurations): {e}")
        
    return model