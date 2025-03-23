# model.py
import logging
import numpy as np
from PIL import Image
import os
import cv2
import onnxruntime as ort
import time

# Configure logger
logger = logging.getLogger(__name__)

# Set up ONNX Runtime session with GPU acceleration
def get_onnx_session(model_path):
    """
    Create an ONNX Runtime session with GPU acceleration if available
    """
    # Check if GPU is available and set providers accordingly
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] 
    
    try:
        # Create ONNX Runtime session
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Log which provider is being used
        provider_name = session.get_providers()[0]
        logger.info(f"Using {provider_name} for inference")
        return session
    except Exception as e:
        logger.error(f"Error creating ONNX session: {str(e)}")
        # Fall back to CPU if there's an error with GPU
        try:
            logger.warning("Falling back to CPU execution")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            return session
        except Exception as e2:
            logger.error(f"Error creating CPU ONNX session: {str(e2)}")
            raise RuntimeError(f"Failed to initialize ONNX Runtime: {str(e2)}")

# Model path - update this to point to your YOLOv8s ONNX file
MODEL_PATH = os.environ.get("YOLOV8_MODEL_PATH", "models/yolov8s.onnx")
# Path to custom class names file
CLASS_NAMES_FILE = os.environ.get("YOLOV8_CLASS_FILE", "models/classes.txt")

# YOLOv8 class names - COCO classes by default
# Can be customized based on your specific model's classes
YOLOV8_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Define classes to treat as "damage" for our use case
# This allows you to map certain detected objects to "damage" category
DAMAGE_CLASSES = ["crack", "hole", "stain", "dent", "scratch", "tear", "split"]

# The class name to use if we're using our own custom model with a single class
DEFAULT_CLASS_NAME = "damage"

# Try to load custom class names if file exists
def load_class_names(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Loaded {len(class_names)} custom class names from {file_path}")
                return class_names
    except Exception as e:
        logger.error(f"Error loading class names from {file_path}: {str(e)}")
    return None

# Load custom class names if available
custom_classes = load_class_names(CLASS_NAMES_FILE)
if custom_classes:
    YOLOV8_CLASSES = custom_classes
    logger.info(f"Using custom classes: {YOLOV8_CLASSES}")
else:
    logger.info(f"Using default COCO classes. Create {CLASS_NAMES_FILE} for custom classes.")

# Initialize the ONNX session
try:
    model = get_onnx_session(MODEL_PATH)
    logger.info(f"Successfully loaded YOLOv8s model from {MODEL_PATH}")
    
    # Get model metadata
    inputs = model.get_inputs()
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    logger.info(f"Model input shape: {input_shape}")
    
    outputs = model.get_outputs()
    output_names = [output.name for output in outputs]
    logger.info(f"Model output names: {output_names}")
    
except Exception as e:
    logger.error(f"Failed to load YOLOv8s model: {str(e)}")
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

def process_yolov8_output(outputs, img_width, img_height, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process YOLOv8 ONNX model output to get bounding boxes, classes and scores
    """
    predictions = []
    
    # YOLOv8 output is (batch, 84, num_boxes) where 84 is 4 box coordinates + 80 class scores
    # Extract the output data
    output = outputs[0]
    
    # Get number of detections
    boxes = []
    scores = []
    class_ids = []

    # Apply sigmoid activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Process each detection
    for i in range(output.shape[1]):
        # Extract box coordinates and dimensions
        x, y, w, h = output[0, i, 0:4]
        
        # Calculate confidence scores (objectness) - apply sigmoid!
        confidence = sigmoid(float(output[0, i, 4]))
        
        if confidence < conf_threshold:
            continue
            
        # Get class scores (5th element onwards) - apply sigmoid!
        class_scores = sigmoid(output[0, i, 5:])
        class_id = np.argmax(class_scores)
        class_score = float(class_scores[class_id])
        
        # Combine objectness with class confidence
        score = confidence * class_score
        
        if score < conf_threshold:
            continue
            
        # Add to our lists
        boxes.append([x, y, w, h])
        scores.append(score)
        class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    
    # Process final detections
    detection_id = 0
    for i in indices:
        # Get the box coordinates
        x, y, w, h = boxes[i]
        
        # Convert from normalized coordinates to actual coordinates if needed
        if x <= 1 and y <= 1 and w <= 1 and h <= 1:
            x *= img_width
            y *= img_height
            w *= img_width
            h *= img_height
        
        # Get class name based on class id
        class_id = class_ids[i]
        # Use COCO class names if available and within range
        if class_id < len(YOLOV8_CLASSES):
            class_name = YOLOV8_CLASSES[class_id]
            # Map specific classes to "damage" for this application
            if class_name in DAMAGE_CLASSES:
                class_name = "damage"
        else:
            class_name = DEFAULT_CLASS_NAME  # Default to "damage" for our use case
        
        # Create detection object
        detection = {
            'x': float(x),
            'y': float(y),
            'width': float(w),
            'height': float(h),
            'confidence': float(scores[i]),
            'class_id': int(class_ids[i]),
            'class_name': class_name,
            'detection_id': detection_id
        }
        
        detection_id += 1
        predictions.append(detection)
    
    return predictions

def process_image(image_path):
    """
    Processes an image to detect wood imperfections using YOLOv8s.
    :param image_path: Path to the image file.
    :return: JSON-serializable inference results
    """
    logger.info(f"Running inference on image {image_path}")
    start_time = time.time()
    
    # Get actual image dimensions
    img_width, img_height = get_image_dimensions(image_path)
    logger.info(f"Image dimensions: {img_width}x{img_height}")
    
    try:
        # Load and preprocess the image
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not load image {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get model input shape
        input_shape = model.get_inputs()[0].shape
        input_height, input_width = input_shape[2], input_shape[3]
        
        # Resize and normalize image
        resized = cv2.resize(img, (input_width, input_height))
        input_data = resized.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Change shape from HWC to NCHW format (batch, channels, height, width)
        input_data = input_data.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        input_name = model.get_inputs()[0].name
        output_names = [output.name for output in model.get_outputs()]
        
        logger.info(f"Running inference with input shape: {input_data.shape}")
        outputs = model.run(output_names, {input_name: input_data})
        
        # Process outputs to get detections - using appropriate confidence threshold
        predictions = process_yolov8_output(outputs, img_width, img_height, conf_threshold=0.4, iou_threshold=0.45)
        logger.info(f"Found {len(predictions)} detections")
        
        # Prepare serialized results
        serialized_results = {
            "predictions": []
        }
        
        # Process each detection
        for prediction in predictions:
            try:
                # Get base prediction attributes
                x = float(prediction['x'])
                y = float(prediction['y'])
                width = float(prediction['width']) 
                height = float(prediction['height'])
                
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
                    'class_name': prediction['class_name'],
                    'confidence': prediction['confidence'],
                    'detection_id': prediction['detection_id'],
                    'bbox': bbox
                }
                
                logger.debug(f"Processed prediction: {serialized_prediction}")
                serialized_results["predictions"].append(serialized_prediction)
                
            except Exception as e:
                logger.error(f"Error processing individual prediction: {str(e)}", exc_info=True)
                continue
        
        elapsed_time = time.time() - start_time
        logger.info(f"Inference completed in {elapsed_time:.2f} seconds")
        return serialized_results

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process image: {str(e)}")