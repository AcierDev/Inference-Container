# model.py
import logging
import os
import torch
from inference import get_model

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

# Classification model configuration - set via environment variables
MODEL_ID = os.environ.get("CLASSIFICATION_MODEL_ID", "first-classification-attempt/2")

# Initialize the classification model
try:
    device = get_device()
    model = get_model(model_id=MODEL_ID)
    logger.info(f"Successfully loaded classification model with ID: {MODEL_ID}")
except Exception as e:
    logger.error(f"Failed to load classification model: {str(e)}")
    raise RuntimeError(f"Model initialization failed: {str(e)}")

def process_image(image_path):
    """
    Processes an image using the classification model.
    :param image_path: Path to the image file.
    :return: JSON-serializable inference results
    """
    logger.info(f"Running classification inference on image {image_path}")
    
    try:
        # Run inference
        results = model.infer(image_path)
        
        logger.info(f"Raw results type: {type(results)}, content: {results}")
        
        # Format the results into a standardized response
        serialized_results = {
            "predictions": []
        }
        
        # Handle the result based on its type
        if isinstance(results, list) and len(results) > 0:
            # First item in list
            result_item = results[0]
            
            # Access attributes using dot notation for Pydantic models
            if hasattr(result_item, 'predictions') and result_item.predictions:
                # Get first prediction from the predictions list
                pred = result_item.predictions[0]
                
                # Extract class_name and confidence
                class_name = getattr(pred, 'class_name', None)
                # If not found, try 'class' attribute
                if class_name is None and hasattr(pred, 'class'):
                    class_name = getattr(pred, 'class', 'Unknown')
                    
                confidence = getattr(pred, 'confidence', 0.0)
                
            # If no predictions attribute, try direct top/confidence attributes
            elif hasattr(result_item, 'top') and hasattr(result_item, 'confidence'):
                class_name = result_item.top
                confidence = result_item.confidence
            else:
                logger.warning("Couldn't find class and confidence in response")
                class_name = "Unknown"
                confidence = 0.0
                
            # Convert to the format the system expects
            serialized_prediction = {
                'class_name': class_name,
                'confidence': confidence,
                'detection_id': 0,
                'is_classification': True
            }
            
            serialized_results["predictions"].append(serialized_prediction)
            logger.info(f"Classification result: {class_name} with confidence {confidence}")
        else:
            logger.warning("No classification results returned from model")
        
        return serialized_results

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process image: {str(e)}")