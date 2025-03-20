# model.py
import inference
import logging
import numpy as np
from PIL import Image

import os
import onnxruntime as ort

# Set thread options explicitly to avoid affinity issues
os.environ["OMP_NUM_THREADS"] = "4"  # Adjust based on available cores
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

# Configure session options (compatible approach)
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load the model during module import
model = inference.get_model("yolo-nas-hi-res/6")
logger = logging.getLogger(__name__)

def get_image_dimensions(image_path):
    """Get actual image dimensions"""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Error getting image dimensions: {str(e)}")
        return (3840, 2160)  # Default to 4K if can't read image

def normalize_coordinates(x, y, width, height, img_width, img_height):
    """
    Convert coordinates and dimensions to normalized [0,1] range
    Ensures non-zero dimensions and handles edge cases
    """
    # The input x,y appears to be the center point, so we need to convert to top-left
    x1 = x - (width / 2)
    y1 = y - (height / 2)
    x2 = x + (width / 2)
    y2 = y + (height / 2)
    
    # Clip to image bounds
    x1 = max(0, min(img_width - 1, x1))
    x2 = max(0, min(img_width - 1, x2))
    y1 = max(0, min(img_height - 1, y1))
    y2 = max(0, min(img_height - 1, y2))
    
    # Convert to normalized coordinates
    x1 = x1 / img_width
    y1 = y1 / img_height
    x2 = x2 / img_width
    y2 = y2 / img_height
    
    # Ensure proper ordering
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Ensure minimum size
    MIN_SIZE = 0.001
    if x2 - x1 < MIN_SIZE:
        center = (x1 + x2) / 2
        x1 = center - MIN_SIZE/2
        x2 = center + MIN_SIZE/2
    
    if y2 - y1 < MIN_SIZE:
        center = (y1 + y2) / 2
        y1 = center - MIN_SIZE/2
        y2 = center + MIN_SIZE/2
    
    # Final clipping to ensure we're in [0,1]
    x1 = max(0.0, min(1.0, x1))
    x2 = max(0.0, min(1.0, x2))
    y1 = max(0.0, min(1.0, y1))
    y2 = max(0.0, min(1.0, y2))
    
    return [x1, y1, x2, y2]

def process_image(image_path):
    """
    Processes an image to detect wood imperfections.
    :param image_path: Path to the image file.
    :return: JSON-serializable inference results
    """
    logger.info(f"Running inference on image {image_path}")
    
    # Get actual image dimensions
    img_width, img_height = get_image_dimensions(image_path)
    logger.info(f"Image dimensions: {img_width}x{img_height}")
    
    # Run inference
    results = model.infer(image=image_path)
    logger.info(f"Inference complete for {image_path}")

    try:
        serialized_results = {
            "predictions": []
        }

        # Process each detection
        for result in results:
            if hasattr(result, 'predictions'):
                for prediction in result.predictions:
                    try:
                        # Get base prediction attributes
                        x = float(getattr(prediction, 'x', 0))
                        y = float(getattr(prediction, 'y', 0))
                        width = float(getattr(prediction, 'width', 0))
                        height = float(getattr(prediction, 'height', 0))
                        
                        # Skip invalid detections
                        if width <= 0 or height <= 0:
                            logger.warning(f"Skipping detection with invalid dimensions: {width}x{height}")
                            continue
                            
                        # Check if detection is within image bounds
                        if x + width/2 > img_width or y + height/2 > img_height:
                            logger.warning(
                                f"Detection at ({x},{y}) with size {width}x{height} "
                                f"extends beyond image bounds {img_width}x{img_height}"
                            )
                        
                        # Normalize coordinates with edge handling
                        bbox = normalize_coordinates(x, y, width, height, img_width, img_height)
                        
                        # Calculate normalized dimensions
                        bbox_width = bbox[2] - bbox[0]
                        bbox_height = bbox[3] - bbox[1]
                        bbox_area = bbox_width * bbox_height
                        
                        # Skip if normalization resulted in invalid box
                        if bbox_area <= 0:
                            logger.warning(f"Skipping detection with zero area after normalization")
                            continue
                        
                        serialized_prediction = {
                            'class_name': getattr(prediction, 'class_name', 'damage'),
                            'confidence': float(getattr(prediction, 'confidence', 0.0)),
                            'detection_id': getattr(prediction, 'detection_id', None),
                            'bbox': bbox
                        }
                        
                        logger.debug(f"Processed prediction: {serialized_prediction}")
                        serialized_results["predictions"].append(serialized_prediction)
                        
                    except Exception as e:
                        logger.error(f"Error processing individual prediction: {str(e)}", exc_info=True)
                        continue
            else:
                logger.warning("Result object does not contain 'predictions' attribute")

        return serialized_results

    except Exception as e:
        logger.error(f"Error serializing results: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process prediction results: {str(e)}")