import cv2
import numpy as np
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_config_loaded" in data
    assert "server_config_loaded" in data

def test_websocket_telemetry():
    # Generate a mock JPEG image bytes to send over WebSocket
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(frame, (112, 112), 30, (255, 0, 0), -1)
    _, encoded = cv2.imencode('.jpg', frame)
    frame_bytes = encoded.tobytes()
    
    with client.websocket_connect("/ws/telemetry") as websocket:
        websocket.send_bytes(frame_bytes)
        response_data = websocket.receive_json()
        
        assert "risk_score" in response_data
        assert "is_anomaly" in response_data
        assert 0.0 <= response_data["risk_score"] <= 1.0
        assert isinstance(response_data["is_anomaly"], bool)
