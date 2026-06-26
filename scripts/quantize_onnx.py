#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) script using onnxruntime.
Converts the unified FP32 ONNX model to Static INT8.
"""
import os
import sys
import cv2
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.logger import get_logger
from src.data_pipeline.transforms import preprocess_frame

logger = get_logger(__name__)


class VideoCalibrationDataReader(CalibrationDataReader):
    """Generates calibration data from a sample video file."""
    def __init__(self, video_path, limit=30):
        self.cap = cv2.VideoCapture(video_path)
        self.limit = limit
        self.count = 0
        
    def get_next(self):
        if self.count >= self.limit:
            self.cap.release()
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return None
            
        preprocessed = preprocess_frame(frame)
        temporal = np.zeros((1, 15), dtype=np.float32)
        
        self.count += 1
        return {
            "image": preprocessed.astype(np.float32),
            "temporal_features": temporal
        }


def quantize_model(input_path="models/chronospatial_unified.onnx", 
                   output_path="models/chronospatial_unified_quantized.onnx",
                   calibration_video="data/sample_feed.mp4"):
    """Apply static INT8 Post-Training Quantization to the ONNX model."""
    logger.info(f"Checking for FP32 model at: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"FP32 model not found at {input_path}. Please run export_onnx.py first.")
        sys.exit(1)
        
    logger.info(f"Checking for calibration video at: {calibration_video}")
    if not os.path.exists(calibration_video):
        logger.error(f"Calibration video not found at {calibration_video}.")
        sys.exit(1)
        
    logger.info("Starting Post-Training Static Quantization (INT8)...")
    
    try:
        # Load the model and clean value_info to avoid shape inference conflicts
        model = onnx.load(input_path)
        initializer_names = {init.name for init in model.graph.initializer}
        new_value_info = [vi for vi in model.graph.value_info if vi.name not in initializer_names]
        
        del model.graph.value_info[:]
        model.graph.value_info.extend(new_value_info)
        logger.info("Filtered graph.value_info: removed initializer entries.")
        
        # Save a temporary model file for the calibrator
        temp_filtered_path = input_path + ".temp"
        onnx.save(model, temp_filtered_path)
        
        # Create CalibrationDataReader
        dr = VideoCalibrationDataReader(calibration_video, limit=30)
        
        # Run static quantization
        quantize_static(
            model_input=temp_filtered_path,
            model_output=output_path,
            calibration_data_reader=dr,
            quant_format=ort.quantization.QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8
        )
        
        # Clean up temp file
        if os.path.exists(temp_filtered_path):
            os.remove(temp_filtered_path)
            
        logger.info(f"Quantized model saved to: {output_path}")
        
        # Verify the quantized model
        quantized_model = onnx.load(output_path)
        onnx.checker.check_model(quantized_model)
        logger.info("Quantized ONNX model verified successfully.")
        
        # Compare file sizes
        original_size = os.path.getsize(input_path) / 1024 / 1024
        quantized_size = os.path.getsize(output_path) / 1024 / 1024
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Original model size: {original_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Size reduction: {reduction:.2f}%")
        
    except Exception as e:
        logger.exception(f"Quantization failed with error: {e}")
        # Make sure temp file is cleaned up in case of failure
        temp_filtered_path = input_path + ".temp"
        if os.path.exists(temp_filtered_path):
            os.remove(temp_filtered_path)
        sys.exit(1)


if __name__ == "__main__":
    quantize_model()
