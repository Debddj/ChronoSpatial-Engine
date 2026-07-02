#!/usr/bin/env python3
"""
Synthetic data generator and PyTorch training pipeline for ChronoSpatial Engine.
Simulates near-miss and collision scenarios using a physics-based model (TTC),
renders synthetic training frames, trains the unified PyTorch model,
and exports/quantizes the model for production.
"""
import os
import sys
import time
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.unified_model import create_unified_model
from src.data_pipeline.transforms import preprocess_frame
from src.models.ann_regressor import create_temporal_features
from src.utils.logger import get_logger
from scripts.export_onnx import export_onnx
from scripts.quantize_onnx import quantize_model

logger = get_logger(__name__)


class SyntheticTrajectoryDataset(Dataset):
    """
    Generates a synthetic dataset of perspective-correct obstacle frames
    and associated motion history labeled with physics-based collision risk.
    """
    def __init__(self, num_samples=1000, distance_scale_factor=1.0, max_history=5):
        self.num_samples = num_samples
        self.distance_scale_factor = distance_scale_factor
        self.max_history = max_history
        self.images = []
        self.temporal_features = []
        self.labels = []
        
        self._generate_data()

    def _generate_data(self):
        logger.info(f"Generating {self.num_samples} synthetic collision/near-miss trajectory samples...")
        np.random.seed(42)
        
        for idx in range(self.num_samples):
            # 1. Randomly initialize trajectory parameters
            # Longitudinal starting distance (meters)
            d0 = np.random.uniform(5.0, 50.0)
            # Longitudinal speed (m/s) -> positive is approaching, negative is moving away
            v_lon = np.random.uniform(-5.0, 15.0)
            # Lateral starting offset (meters) and speed (m/s)
            x0 = np.random.uniform(-4.0, 4.0)
            v_lat = np.random.uniform(-1.5, 1.5)
            
            # 2. Simulate 5 steps at 10 Hz (dt = 0.1s)
            dt = 0.1
            distances = []
            velocities = []
            x_coords = []
            
            for step in range(self.max_history):
                t = step * dt
                d_t = max(1.0, d0 - v_lon * t)  # clamp distance to 1m min
                x_t = x0 + v_lat * t
                
                distances.append(d_t)
                x_coords.append(x_t)
                
                if step > 0:
                    vx = (x_t - x_coords[step - 1]) / dt
                    vy = (d_t - distances[step - 1]) / dt
                    velocities.append([vx, vy])
                else:
                    velocities.append([0.0, 0.0])
                    
            distances = np.array(distances, dtype=np.float32)
            velocities = np.array(velocities, dtype=np.float32)
            
            # 3. Calculate ground-truth collision risk score
            # Time-to-collision (TTC) based on final state
            d_final = distances[-1]
            if v_lon > 0:
                ttc = d_final / v_lon
                if ttc <= 0.5:
                    risk = 1.0
                elif ttc <= 5.0:
                    # Exponential decay risk based on TTC
                    risk = float(np.exp(-0.5 * (ttc - 0.5)))
                else:
                    risk = 0.0
            else:
                # Proximity risk for very close objects even if they are static/moving away
                if d_final <= 3.0:
                    risk = float(0.5 * (1.0 - (d_final - 1.0) / 2.0))
                else:
                    risk = 0.0
            
            # Add safety margin if distance is critically close
            if d_final <= 1.5:
                risk = max(risk, 0.95)
                
            # 4. Generate perspective-correct visual frame
            # 224x224 grayscale-ish scene
            frame = np.ones((224, 224, 3), dtype=np.uint8) * 60  # dark road background
            # Draw a horizon line
            cv2.line(frame, (0, 112), (224, 112), (90, 90, 90), 1)
            
            # Render the obstacle box based on perspective projection
            # Height is inversely proportional to distance
            box_h = int(224.0 * (self.distance_scale_factor / d_final))
            box_h = np.clip(box_h, 4, 180)
            box_w = int(box_h * 1.3)  # vehicles are wider than tall
            
            # Project lateral coordinate x_final to horizontal image center
            cx = 112 + int(x_coords[-1] * (80.0 / d_final))
            cx = np.clip(cx, 0, 223)
            
            # Vertical center rests on the "ground" below the horizon
            cy = 112 + int(60.0 / d_final)
            cy = np.clip(cy, 112, 223)
            
            # Compute top-left and bottom-right points
            x1 = max(0, cx - box_w // 2)
            y1 = max(0, cy - box_h // 2)
            x2 = min(223, cx + box_w // 2)
            y2 = min(223, cy + box_h // 2)
            
            # Draw obstacle (red car box with yellow license plate/lights detail)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 200), -1)  # red body
            # Draw taillights
            light_w = max(2, (x2 - x1) // 8)
            cv2.rectangle(frame, (x1, y2 - light_w * 2), (x1 + light_w, y2 - light_w), (0, 0, 255), -1)
            cv2.rectangle(frame, (x2 - light_w, y2 - light_w * 2), (x2, y2 - light_w), (0, 0, 255), -1)
            
            # Preprocess the image
            preprocessed_img = preprocess_frame(frame, target_size=(224, 224))
            # preprocessed_img has shape (1, 3, 224, 224) -> remove batch dimension for Dataset
            preprocessed_img = np.squeeze(preprocessed_img, axis=0)
            
            # Flatten velocities & distances into 15-d temporal features vector
            temp_feats = create_temporal_features(
                velocity_vectors=velocities,
                asset_distances=distances,
                max_history=self.max_history
            )
            
            self.images.append(preprocessed_img)
            self.temporal_features.append(temp_feats)
            self.labels.append(risk)
            
        self.images = np.array(self.images, dtype=np.float32)
        self.temporal_features = np.array(self.temporal_features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        logger.info("Dataset generation completed.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx]),
            torch.tensor(self.temporal_features[idx]),
            torch.tensor(self.labels[idx]).unsqueeze(0)
        )


def train_model(config_path="config/model_config.yaml", num_epochs=10, batch_size=32, lr=1e-3):
    """Executes the training loop to fine-tune the model on synthetic risk data."""
    logger.info("Starting ChronoSpatial model training pipeline...")
    
    # 1. Load model config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})
    
    # 2. Generate datasets
    train_dataset = SyntheticTrajectoryDataset(num_samples=1000)
    val_dataset = SyntheticTrajectoryDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Create Model
    model = create_unified_model(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Freeze CNN backbone completely, only train projection layer & ANN head
    for param in model.cnn_backbone.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if "cnn_projection" in name or "ann_network" in name:
            param.requires_grad = True
            
    # Optimizer and Loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 4. Training Loop
    best_val_loss = float("inf")
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    pt_model_path = os.path.join(checkpoint_dir, "chronospatial_unified.pt")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, temp_feats, labels in train_loader:
            images = images.to(device)
            temp_feats = temp_feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(images, temp_feats)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, temp_feats, labels in val_loader:
                images = images.to(device)
                temp_feats = temp_feats.to(device)
                labels = labels.to(device)
                
                predictions = model(images, temp_feats)
                loss = criterion(predictions, labels)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        logger.info(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save PyTorch state dict
            torch.save(model.state_dict(), pt_model_path)
            logger.info(f"Saved new best model checkpoint to {pt_model_path}")
            
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    # 5. Export PyTorch model to ONNX format
    logger.info("Exporting trained model to ONNX...")
    export_onnx(model_config_path=config_path, output_path="models/chronospatial_unified.onnx")
    
    # 6. Quantize the ONNX model to INT8
    logger.info("Running post-training quantization to regenerate quantized INT8 production model...")
    quantize_model(
        input_path="models/chronospatial_unified.onnx",
        output_path="models/chronospatial_unified_quantized.onnx",
        calibration_video="data/sample_feed.mp4"
    )
    
    logger.info("ChronoSpatial model compilation pipeline successfully completed!")


if __name__ == "__main__":
    train_model(num_epochs=10)
