import numpy as np
import pytest
from src.data_pipeline.transforms import preprocess_frame
from src.data_pipeline.stream_loader import StreamLoader

def test_preprocess_frame():
    # Generate a dummy HWC frame: 480x640x3
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Test with zero-mean unit-variance parameters (should reside in [0, 1])
    preprocessed_unnorm = preprocess_frame(
        dummy_frame, 
        target_size=(224, 224), 
        mean=[0.0, 0.0, 0.0], 
        std=[1.0, 1.0, 1.0]
    )
    assert preprocessed_unnorm.shape == (1, 3, 224, 224)
    assert preprocessed_unnorm.dtype == np.float32
    assert np.max(preprocessed_unnorm) <= 1.0
    assert np.min(preprocessed_unnorm) >= 0.0

    # Test with standard ImageNet parameters (should scale to [-2.2, 2.7])
    preprocessed_norm = preprocess_frame(dummy_frame, target_size=(224, 224))
    assert preprocessed_norm.shape == (1, 3, 224, 224)
    assert np.min(preprocessed_norm) >= -2.2
    assert np.max(preprocessed_norm) <= 2.7

def test_stream_loader_invalid_path():
    loader = StreamLoader("nonexistent_video.mp4")
    with pytest.raises(ValueError):
        loader.start()
