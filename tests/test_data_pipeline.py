import numpy as np
import pytest
from src.data_pipeline.transforms import preprocess_frame
from src.data_pipeline.stream_loader import StreamLoader

def test_preprocess_frame():
    # Generate a dummy HWC frame: 480x640x3
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    preprocessed = preprocess_frame(dummy_frame, target_size=(224, 224))
    
    # Assert BCHW shape: (1, 3, 224, 224)
    assert preprocessed.shape == (1, 3, 224, 224)
    assert preprocessed.dtype == np.float32
    assert np.max(preprocessed) <= 1.0
    assert np.min(preprocessed) >= 0.0

def test_stream_loader_invalid_path():
    loader = StreamLoader("nonexistent_video.mp4")
    with pytest.raises(ValueError):
        loader.start()
