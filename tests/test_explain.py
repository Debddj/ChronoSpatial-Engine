import base64
import numpy as np
import pytest
import cv2
from fastapi.testclient import TestClient
from src.models.explain import ChronoSpatialExplainer
from src.serving.app import app
from scripts.train import SyntheticTrajectoryDataset


def test_synthetic_dataset():
    dataset = SyntheticTrajectoryDataset(num_samples=10)
    assert len(dataset) == 10
    image, temporal, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert temporal.shape == (15,)
    assert label.shape == (1,)
    assert 0.0 <= label.item() <= 1.0


def test_explainer():
    config = {
        "model": {
            "cnn_extractor": {"feature_dim": 128},
            "ann_regressor": {
                "input_dim": 143,
                "hidden_dims": [64, 32],
                "dropout_rate": 0.1,
                "risk_threshold": 0.80,
                "max_history": 5
            }
        }
    }
    explainer = ChronoSpatialExplainer(config)
    
    # Run explanation on dummy crop frame
    crop = np.zeros((100, 120, 3), dtype=np.uint8)
    crop[20:80, 30:90] = (0, 0, 255)  # red block
    temp_feats = np.random.randn(15).astype(np.float32)
    
    explanation = explainer.explain(crop, temp_feats)
    assert "risk_score" in explanation
    assert "spatial_heatmap_b64" in explanation
    assert "temporal_attributions" in explanation
    
    # Check base64 string
    assert len(explanation["spatial_heatmap_b64"]) > 0
    # Decode base64
    img_data = base64.b64decode(explanation["spatial_heatmap_b64"])
    assert len(img_data) > 0
    
    # Check attributions
    att = explanation["temporal_attributions"]
    assert "velocity_history" in att
    assert "distance_history" in att
    assert len(att["velocity_history"]) == 5
    assert len(att["distance_history"]) == 5


def test_explain_endpoint():
    client = TestClient(app)
    
    # Generate dummy frame file
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    _, encoded = cv2.imencode('.jpg', frame)
    frame_bytes = encoded.tobytes()
    
    response = client.post(
        "/explain",
        files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
        data={
            "bbox": "100,100,200,200",
            "temporal_features": ",".join([str(0.1 * i) for i in range(15)])
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "spatial_heatmap_b64" in data
    assert "temporal_attributions" in data
