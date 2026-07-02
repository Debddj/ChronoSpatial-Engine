import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from src.models.inference_engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)
websocket_router = APIRouter()

_explainer = None


def get_explainer(config):
    global _explainer
    if _explainer is None:
        from src.models.explain import ChronoSpatialExplainer
        _explainer = ChronoSpatialExplainer(config)
    return _explainer


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


@websocket_router.post("/explain")
async def explain_risk(
    request: Request,
    file: UploadFile = File(...),
    bbox: str = Form(None),  # "ymin,xmin,ymax,xmax" in pixel coords
    temporal_features: str = Form(None)  # "v1_x,v1_y,...,d5" comma separated
):
    """
    HTTP POST endpoint to explain collision/anomaly risk for a specific object/frame.
    Returns Grad-CAM visual overlay and SHAP temporal attributions.
    """
    # 1. Read and decode the uploaded image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image binary data"}
        
    h_f, w_f, _ = frame.shape
    
    # 2. Parse crop bounding box if provided
    if bbox:
        try:
            coords = [float(x) for x in bbox.split(",")]
            ymin, xmin, ymax, xmax = coords
            ymin_c = int(max(0, min(h_f - 1, ymin)))
            xmin_c = int(max(0, min(w_f - 1, xmin)))
            ymax_c = int(max(ymin_c + 1, min(h_f, ymax)))
            xmax_c = int(max(xmin_c + 1, min(w_f, xmax)))
            crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]
        except Exception as e:
            logger.warning(f"Failed to parse bbox '{bbox}': {e}. Using full frame.")
            crop = frame
    else:
        crop = frame
        
    # 3. Parse temporal features
    if temporal_features:
        try:
            temp_feats = np.array([float(x) for x in temporal_features.split(",")], dtype=np.float32)
            if len(temp_feats) != 15:
                # Pad/trim to 15 elements
                temp_feats = np.pad(temp_feats, (0, max(0, 15 - len(temp_feats))))[:15]
        except Exception as e:
            logger.warning(f"Failed to parse temporal features '{temporal_features}': {e}. Using default zeros.")
            temp_feats = np.zeros(15, dtype=np.float32)
    else:
        temp_feats = np.zeros(15, dtype=np.float32)
        
    # 4. Initialize or fetch cached explainer
    config = getattr(request.app.state, "model_config", {})
    explainer = get_explainer(config)
    
    # 5. Run explainability algorithms
    explanation = explainer.explain(crop, temp_feats)
    return explanation
