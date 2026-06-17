# ChronoSpatial Engine

A real-time telemetry frame processing & temporal risk assessment service using FastAPI WebSockets, OpenCV, and a CNN-ANN hybrid framework.

## Architecture
- **Data Pipeline**: Handles OpenCV frame extraction, real-time telemetry loading, and normalization.
- **Model Pipeline**: Performs spatial feature extraction (CNN) and computes a temporal risk assessment (ANN).
- **API Serving**: Implements an asynchronous WebSocket route for ultra-low latency telemetry and frame evaluation.

## Repository Layout
- `src/data_pipeline/`: Stream decoder (`stream_loader.py`) and tensor transformation (`transforms.py`).
- `src/models/`: CNN extractor, ANN regressor, and coordinates compiler.
- `src/serving/`: FastAPI application (`app.py`) and websocket endpoints (`router.py`).
- `config/`: Configuration parameters for spatial features & server tuning.

## Getting Started

### Prerequisites
- Python 3.10+
- FFmpeg (for video decoding support in OpenCV)

### Installation
```bash
pip install -r requirements.txt
```

### Running Server
```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### Running Tests
```bash
pytest tests/
```
