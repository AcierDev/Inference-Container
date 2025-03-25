# model.py
import logging
import numpy as np
from PIL import Image
import os
import cv2
import torch
from ultralytics import YOLO

# Configure logger
logger = logging.getLogger(__name__)

# Set up PyTorch device with CUDA if available
def get_device():
    """
    Determine the best device for inference (CUDA or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU for inference")
    return device

# Model path - update this to point to your YOLOv8 PT file
MODEL_PATH = os.environ.get("YOLOV8_MODEL_PATH", "inferences/model/roboflow-2.pt")

# The class name to use for all detections
DEFAULT_CLASS_NAME = "damage"
CLASS_NAMES = ['corner', 'crack', 'damage', 'edge', 'knot', 'router', 'side', 'tearout']

# Initialize the PyTorch model
try:
    device = get_device()
    model = YOLO(MODEL_PATH)
    logger.info(f"Successfully loaded YOLOv8 model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLOv8 model: {str(e)}")
    raise RuntimeError(f"Model initialization failed: {str(e)}")

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
    Processes an image to detect wood imperfections using YOLOv8.
    :param image_path: Path to the image file.
    :return: JSON-serializable inference results
    """
    logger.info(f"Running inference on image {image_path}")
    
    img_width, img_height = get_image_dimensions(image_path)
    logger.info(f"Image dimensions: {img_width}x{img_height}")
    
    try:
        results = model(image_path, conf=0.1, iou=0.1)
        
        # Prepare serialized results
        serialized_results = {
            "predictions": []
        }
        
        # Process each detection
        detection_id = 0
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            
            for i, box in enumerate(boxes):
                try:
                    # Get coordinates (comes as xyxy format)
                    xyxy = box.xyxy[0].tolist()  # get box coordinates in (x1, y1, x2, y2) format
                    conf = float(box.conf[0])    # confidence
                    
                    # Convert to xywh format (center, width, height)
                    x1, y1, x2, y2 = xyxy
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Skip invalid detections
                    if width <= 0 or height <= 0:
                        logger.warning(f"Skipping detection with invalid dimensions: {width}x{height}")
                        continue
                    
                    # Use class name from the model if available
                    cls_id = int(box.cls[0]) if hasattr(box, 'cls') else -1
                    class_name = CLASS_NAMES[cls_id] if cls_id in range(len(CLASS_NAMES)) else DEFAULT_CLASS_NAME
                    
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
                        'class_name': class_name,
                        'confidence': conf,
                        'detection_id': detection_id,
                        'bbox': bbox
                    }
                    
                    serialized_results["predictions"].append(serialized_prediction)
                    detection_id += 1
                    
                except Exception as e:
                    logger.error(f"Error processing individual prediction: {str(e)}", exc_info=True)
                    continue
        
        return serialized_results

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process image: {str(e)}")