#!/usr/bin/env python3
"""
Convert YOLOv8s PyTorch model to ONNX format for GPU-accelerated inference.
This script downloads the YOLOv8s model and exports it to ONNX format.
"""

import os
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir_exists(directory):
    """Ensure the directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def download_and_convert_to_onnx(model_name="yolov8s", output_dir="models", dynamic=True):
    """
    Download YOLOv8 model and convert it to ONNX format
    
    Args:
        model_name (str): YOLOv8 model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        output_dir (str): Directory to save the ONNX model
        dynamic (bool): Whether to use dynamic axes for the ONNX model
        
    Returns:
        str: Path to the converted ONNX model
    """
    # Ensure output directory exists
    output_dir = ensure_dir_exists(output_dir)
    
    # Download and load the model
    logger.info(f"Loading {model_name} model...")
    try:
        model = YOLO(model_name)
        logger.info(f"Successfully loaded {model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Export to ONNX
    try:
        logger.info(f"Exporting {model_name} to ONNX format...")
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        # Export with dynamic axes if requested
        if dynamic:
            model.export(format="onnx", 
                         dynamic=True, 
                         simplify=True,
                         opset=12)
        else:
            model.export(format="onnx",
                         simplify=True,
                         opset=12)
        
        # Move the file if it's not in the right place
        default_export_path = f"{model_name}.onnx"
        if os.path.exists(default_export_path) and default_export_path != onnx_path:
            os.rename(default_export_path, onnx_path)
            logger.info(f"Moved model to {onnx_path}")
        
        if os.path.exists(onnx_path):
            logger.info(f"Model successfully exported to {onnx_path}")
            return onnx_path
        else:
            logger.error(f"Expected ONNX file at {onnx_path} not found")
            raise FileNotFoundError(f"Expected ONNX file at {onnx_path} not found")
            
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert YOLOv8 model to ONNX format')
    parser.add_argument('--model', type=str, default='yolov8s', 
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 model to download and convert')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save the ONNX model')
    parser.add_argument('--dynamic', action='store_true', default=True,
                        help='Use dynamic axes for variable input size')
    
    args = parser.parse_args()
    
    try:
        onnx_path = download_and_convert_to_onnx(args.model, args.output_dir, args.dynamic)
        logger.info(f"Conversion completed successfully. ONNX model saved to: {onnx_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 