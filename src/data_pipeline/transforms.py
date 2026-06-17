import cv2
import numpy as np

def preprocess_frame(frame, target_size=(224, 224)):
    # Resize frame
    resized = cv2.resize(frame, target_size)
    # Convert to float and normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    # Transpose dimensions from HWC to CHW
    chw = np.transpose(normalized, (2, 0, 1))
    # Add batch dimension: BCHW
    batch = np.expand_dims(chw, axis=0)
    return batch
