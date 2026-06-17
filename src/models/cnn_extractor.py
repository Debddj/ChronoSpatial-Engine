import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CNNExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super(CNNExtractor, self).__init__()
        self.feature_dim = feature_dim
        
        # Load pre-trained MobileNetV3 Small backbone
        logger.info("Loading pre-trained MobileNetV3 Small backbone...")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get output dimensions of features before classification
        in_features = self.backbone.classifier[0].in_features
        
        # Strip the default classification head
        self.backbone.classifier = nn.Identity()
        
        # Custom projection layer to project to target tracking embedding dimension
        self.projection = nn.Linear(in_features, self.feature_dim)
        
        logger.info(f"CNNExtractor initialized with feature projection: {in_features} -> {self.feature_dim}")

    def forward(self, x):
        # Convert numpy array to torch tensor if necessary
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
            
        # Extract features (returns shape [batch_size, in_features] due to nn.Identity classifier)
        latent = self.backbone(x)
        
        # Project to target feature embedding dimension [batch_size, feature_dim]
        embeddings = self.projection(latent)
        return embeddings

    def extract_features(self, preprocessed_frame):
        """Extract spatial features. Accepts a preprocessed NumPy array and returns a NumPy array."""
        self.eval()
        with torch.no_grad():
            if len(preprocessed_frame.shape) == 3:
                preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
            
            tensor_input = torch.from_numpy(preprocessed_frame).float()
            embeddings = self.forward(tensor_input)
            
            output = embeddings.cpu().numpy()
            if output.shape[0] == 1:
                return output[0]
            return output
