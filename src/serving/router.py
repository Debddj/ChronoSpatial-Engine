import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.models.inference_engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)
websocket_router = APIRouter()

@websocket_router.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    # Initialize inference engine using app state config
    config = getattr(websocket.app.state, "model_config", {})
    engine = InferenceEngine(config)
    
    try:
        while True:
            # Expecting binary frame data (JPEG/PNG bytes) or JSON message
            data = await websocket.receive_bytes()
            
            # Decode image from bytes
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid image binary data"})
                continue
                
            # Run inference
            results = engine.run_inference(frame)
            
            # Send back the results
            await websocket.send_json(results)
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
