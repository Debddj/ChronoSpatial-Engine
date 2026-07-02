import os
import sys
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from src.models.unified_model import create_unified_model
from src.data_pipeline.transforms import preprocess_frame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChronoSpatialExplainer:
    def __init__(self, config):
        self.config = config
        model_cfg = config.get("model", {})
        
        # Create the PyTorch model
        self.model = create_unified_model(model_cfg)
        self.model.eval()
        
        # target layer for Grad-CAM
        # MobileNetV3 Small last feature extraction block
        self.target_layer = self.model.cnn_backbone.features[-1]
        
        self.gradients = None
        self.activations = None
        
        logger.info("ChronoSpatialExplainer initialized with PyTorch model.")

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def explain(self, crop_frame, temporal_features_np):
        """
        Generates Grad-CAM heatmap overlaid on cropped frame and Integrated Gradients attributions.
        
        Args:
            crop_frame: numpy BGR image of the cropped tracked object.
            temporal_features_np: numpy array of shape (15,) containing motion history.
            
        Returns:
            dict containing:
                "risk_score": float,
                "spatial_heatmap_b64": base64 string of the overlaid image,
                "temporal_attributions": dict of feature attribution scores.
        """
        # Ensure we have correct dimensions
        if len(temporal_features_np.shape) == 1:
            temporal_features_np = np.expand_dims(temporal_features_np, axis=0)
            
        # Preprocess crop to match model input shape
        h_crop, w_crop, _ = crop_frame.shape
        preprocessed = preprocess_frame(crop_frame, target_size=(224, 224))
        
        # Convert to torch tensors
        image_tensor = torch.from_numpy(preprocessed).float()
        temporal_tensor = torch.from_numpy(temporal_features_np).float()
        
        # Enable gradients for explanation
        image_tensor.requires_grad = True
        temporal_tensor.requires_grad = True
        
        # Register hooks for Grad-CAM
        h_forward = self.target_layer.register_forward_hook(self._save_activation)
        h_backward = self.target_layer.register_backward_hook(self._save_gradient)
        
        # Run forward pass
        self.model.zero_grad()
        output = self.model(image_tensor, temporal_tensor)
        risk_score = float(output.squeeze().item())
        
        # Backward pass to calculate gradients
        output.backward(retain_graph=True)
        
        # Remove hooks
        h_forward.remove()
        h_backward.remove()
        
        # --- 1. Compute Grad-CAM ---
        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)  # ReLU
        
        # Resize to original crop size
        cam_resized = torch.nn.functional.interpolate(
            cam, size=(h_crop, w_crop), mode="bilinear", align_corners=False
        )
        cam_np = cam_resized.squeeze().cpu().numpy()
        
        # Normalize heatmap to [0, 1]
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
            
        # Draw overlay: color map jet on original BGR crop
        heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(crop_frame, 0.6, heatmap_color, 0.4, 0)
        
        # Encode overlay to base64 string
        _, buffer = cv2.imencode(".jpg", overlay)
        overlay_b64 = base64.b64encode(buffer).decode("utf-8")
        
        # --- 2. Compute Integrated Gradients (Path-Integrated SHAP) ---
        # Using a baseline of zero temporal features (represents a static asset)
        baseline_temp = torch.zeros_like(temporal_tensor)
        steps = 30
        accumulated_grads = np.zeros_like(temporal_features_np)
        
        for step in range(steps + 1):
            alpha = step / steps
            temp_step = baseline_temp + alpha * (temporal_tensor - baseline_temp)
            temp_step = temp_step.clone().detach().requires_grad_(True)
            
            # Re-run forward pass with current step
            out_step = self.model(image_tensor, temp_step)
            self.model.zero_grad()
            out_step.backward()
            
            accumulated_grads += temp_step.grad.cpu().numpy()
            
        # Average the gradients
        avg_grads = accumulated_grads / (steps + 1)
        
        # Integrated Gradients = (input - baseline) * avg_grads
        attributions = (temporal_features_np - baseline_temp.cpu().numpy()) * avg_grads
        attributions = attributions.squeeze()
        
        # Format temporal attributions for readability
        # Temporal features map: 10 velocities (5 steps * 2) + 5 distances (5 steps * 1)
        velocities_att = attributions[:10].reshape(5, 2)
        distances_att = attributions[10:]
        
        formatted_attributions = {
            "velocity_history": [
                {
                    "step_t": int(-4 + i),
                    "vx_attribution": float(velocities_att[i][0]),
                    "vy_attribution": float(velocities_att[i][1])
                }
                for i in range(5)
            ],
            "distance_history": [
                {
                    "step_t": int(-4 + i),
                    "attribution": float(distances_att[i])
                }
                for i in range(5)
            ]
        }
        
        return {
            "risk_score": risk_score,
            "spatial_heatmap_b64": overlay_b64,
            "temporal_attributions": formatted_attributions
        }
