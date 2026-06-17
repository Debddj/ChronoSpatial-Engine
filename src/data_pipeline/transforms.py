import cv2
import numpy as np

def preprocess_frame(frame, target_size=(224, 224), mean=None, std=None):
    """
    Preprocess raw frame:
    1. Resize to target size.
    2. Convert channel from BGR to RGB.
    3. Normalize pixel values using vectorized NumPy scaling.
    4. Transpose HWC layout to CHW and add batch dimension (BCHW).
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    else:
        mean = np.array(mean, dtype=np.float32)
        
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    else:
        std = np.array(std, dtype=np.float32)

    # 1. Resize frame
    resized = cv2.resize(frame, target_size)
    
    # 2. Convert channel from BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 3. Vectorized normalization: scale to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Subtract mean and divide by std (vectorized channel broadcast)
    normalized = (normalized - mean) / std
    
    # 4. Transpose from HWC to CHW
    chw = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension (BCHW)
    batch = np.expand_dims(chw, axis=0)
    return batch
