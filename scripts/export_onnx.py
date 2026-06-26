#!/usr/bin/env python3
"""
Export unified PyTorch model to ONNX format.
"""
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.unified_model import create_unified_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_onnx(model_config_path="config/model_config.yaml", 
                output_path="models/chronospatial_unified.onnx",
                opset_version=18):
    """Export unified model to ONNX format."""
    
    # Load config
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("model", {})
    
    # Create model
    model = create_unified_model(model_config)
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    dummy_temporal = torch.randn(batch_size, 15, dtype=torch.float32)
    
    # Export to ONNX using new API
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        (dummy_image, dummy_temporal),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image", "temporal_features"],
        output_names=["risk_score"],
        dynamic_shapes={
            "image": {0: "batch_size"},
            "temporal_features": {0: "batch_size"}
        },
        verbose=False
    )
    
    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path)
    logger.info("Embedded model weights into the single ONNX file.")
    
    # Remove external data file if it exists
    external_data_path = output_path + ".data"
    if os.path.exists(external_data_path):
        os.remove(external_data_path)
        logger.info(f"Removed external data file: {external_data_path}")
        
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")
    
    # Print model info
    logger.info(f"Model inputs: {[inp.name for inp in onnx_model.graph.input]}")
    logger.info(f"Model outputs: {[out.name for out in onnx_model.graph.output]}")
    
    return output_path


if __name__ == "__main__":
    export_onnx()