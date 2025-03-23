#!/usr/bin/env python3
"""
Export weights from the inference package model cache.
This script copies the weights from the model cache to a specified destination directory.
It can also optionally convert the ONNX weights to PyTorch format.
"""

import os
import shutil
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_cache_dir(model_id):
    """
    Get the cache directory for a specific model.
    
    Args:
        model_id (str): Model ID in the format "model-name/version"
    
    Returns:
        Path: Path to the model cache directory
    """
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/tmp/cache")
    return Path(cache_dir) / model_id

def export_weights(model_id, destination_dir, include_metadata=True, convert_to_pytorch=False):
    """
    Export weights and optionally metadata from the model cache.
    
    Args:
        model_id (str): Model ID in the format "model-name/version"
        destination_dir (str): Destination directory where weights will be copied
        include_metadata (bool): Whether to include metadata files like environment.json
        convert_to_pytorch (bool): Whether to convert ONNX weights to PyTorch format
        
    Returns:
        bool: True if export was successful, False otherwise
    """
    # Get the model cache directory
    cache_dir = get_model_cache_dir(model_id)
    
    # Check if the cache directory exists
    if not cache_dir.exists():
        logger.error(f"Model cache directory not found: {cache_dir}")
        return False
    
    # Create the destination directory if it doesn't exist
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # First, copy the weights file
    weights_file = cache_dir / "weights.onnx"
    if not weights_file.exists():
        logger.error(f"Weights file not found: {weights_file}")
        return False
    
    logger.info(f"Copying weights from {weights_file} to {dest_path / 'weights.onnx'}")
    shutil.copy2(weights_file, dest_path / "weights.onnx")
    
    # If requested, copy metadata files
    if include_metadata:
        metadata_files = ["environment.json", "labels.txt", "class_mapping.csv"]
        for file_name in metadata_files:
            file_path = cache_dir / file_name
            if file_path.exists():
                logger.info(f"Copying metadata file {file_name} to {dest_path / file_name}")
                shutil.copy2(file_path, dest_path / file_name)
            else:
                logger.warning(f"Metadata file not found: {file_path}")
    
    # If requested, convert ONNX to PyTorch format
    if convert_to_pytorch:
        try:
            import torch
            import onnx
            from onnx2torch import convert
            
            logger.info("Converting ONNX weights to PyTorch format...")
            onnx_model = onnx.load(str(weights_file))
            pytorch_model = convert(onnx_model)
            
            torch_file = dest_path / "weights.pt"
            torch.save(pytorch_model.state_dict(), torch_file)
            logger.info(f"PyTorch weights saved to {torch_file}")
            
        except ImportError as e:
            logger.error(f"Could not convert to PyTorch format: missing dependencies. {e}")
            logger.error("Install dependencies with: pip install torch onnx onnx2torch")
            # Continue without conversion but still return success
            
        except Exception as e:
            logger.error(f"Error converting to PyTorch format: {e}")
            # Continue without conversion but still return success
    
    logger.info(f"Successfully exported model files to {destination_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Export model weights from the inference cache')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID in format "model-name/version"')
    parser.add_argument('--destination', type=str, required=True, help='Destination directory to copy weights to')
    parser.add_argument('--metadata', action='store_true', help='Include metadata files in export')
    parser.add_argument('--pytorch', action='store_true', help='Convert ONNX weights to PyTorch format')
    
    args = parser.parse_args()
    
    success = export_weights(args.model_id, args.destination, args.metadata, args.pytorch)
    if success:
        logger.info("Export completed successfully")
    else:
        logger.error("Export failed")
        exit(1)

if __name__ == "__main__":
    main() 