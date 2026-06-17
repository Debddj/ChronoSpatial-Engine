import os
import yaml
from fastapi import FastAPI
from src.serving.router import websocket_router
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_app(model_config_path="config/model_config.yaml", server_config_path="config/server_config.yaml"):
    app = FastAPI(
        title="ChronoSpatial Engine API",
        description="Real-time telemetry frame processing & risk assessment service.",
        version="1.0.0"
    )
    
    # Resolve paths robustly
    resolved_model_path = model_config_path
    resolved_server_path = server_config_path
    
    if not os.path.exists(resolved_model_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        resolved_model_path = os.path.join(base_dir, model_config_path)
        resolved_server_path = os.path.join(base_dir, server_config_path)
        
    # Load configurations
    try:
        with open(resolved_model_path, "r") as f:
            app.state.model_config = yaml.safe_load(f)
        with open(resolved_server_path, "r") as f:
            app.state.server_config = yaml.safe_load(f)
        logger.info("Configurations loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        app.state.model_config = {}
        app.state.server_config = {}

    app.include_router(websocket_router)
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_config_loaded": bool(app.state.model_config),
            "server_config_loaded": bool(app.state.server_config)
        }
        
    return app

app = create_app()
