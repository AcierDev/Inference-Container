from flask import Flask, request, jsonify
import logging
from datetime import datetime
import threading
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
import colorsys
import concurrent.futures

# Local imports
from inferences.model import process_image

# Initialize Flask app
app = Flask(__name__)

# Configure base directories
BASE_STORAGE_DIR = Path('storage')
UPLOAD_DIR = BASE_STORAGE_DIR / 'uploads'
PROCESSED_DIR = BASE_STORAGE_DIR / 'processed'
ARCHIVE_DIR = BASE_STORAGE_DIR / 'archive'
ANNOTATED_DIR = PROCESSED_DIR / 'annotated'

# Define category directories for classification results
CLASSIFICATION_DIRS = {
    'Good': PROCESSED_DIR / 'classification' / 'good',
    'Bad': PROCESSED_DIR / 'classification' / 'bad',
    'Unknown': PROCESSED_DIR / 'classification' / 'unknown',
    'failed': PROCESSED_DIR / 'failed_processing'
}

# Ensure all directories exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, ARCHIVE_DIR, ANNOTATED_DIR, *CLASSIFICATION_DIRS.values()]:
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(log_dir / 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a thread pool for background tasks
background_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Dictionary to track background tasks
background_tasks = {}

# Define a maximum number of tasks to keep in memory
MAX_TASK_HISTORY = 100

def create_annotated_image(image_path: Path, result: dict) -> Path:
    """
    Create a copy of the image with classification label
    
    Args:
        image_path (Path): Path to the original image
        result (dict): Classification results
        
    Returns:
        Path: Path to the annotated image
    """
    # Load image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw classification result
    width, height = img.size
    
    # Check if we have any predictions
    predictions = result.get('predictions', [])
    if predictions:
        prediction = predictions[0]
        
        # Get classification result and confidence
        class_name = prediction.get('class_name', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        
        # Choose color based on classification result
        if class_name == 'Bad':
            color = (255, 0, 0)  # Red for bad
        elif class_name == 'Good':
            color = (0, 255, 0)  # Green for good
        else:
            color = (255, 255, 0)  # Yellow for unknown
        
        # Create label
        label = f"{class_name}: {confidence:.2f}"
        
        # Draw a background box at the bottom of the image
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        
        # Position the label at the top center
        label_x = (width - label_width) // 2
        label_y = 20
        
        # Draw label background
        draw.rectangle(
            [label_x - 10, label_y - 10, label_x + label_width + 10, label_y + label_height + 10],
            fill=color
        )
        
        # Draw label text
        draw.text((label_x, label_y), label, fill='white', font=font)
    
    # Save annotated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotated_path = ANNOTATED_DIR / f"annotated_{timestamp}{image_path.suffix}"
    img.save(annotated_path, quality=95)
    
    return annotated_path

def copy_to_directory(src_path: Path, dest_dir: Path, suffix: str = "") -> Path:
    """Copy file to destination directory with optional suffix in filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}{suffix}{src_path.suffix}"
    dest_path = dest_dir / new_filename
    shutil.copy2(str(src_path), str(dest_path))
    return dest_path

def organize_processed_image(file_path: Path, result: dict) -> dict:
    """
    Organize the processed image by copying to the relevant classification folder.
    Returns dict with paths to all copies.
    """
    organized_paths = {
        'classification': None,
        'annotated': None,
        'original': file_path
    }
    
    try:
        predictions = result.get('predictions', [])
        
        # Create annotated version with classification result
        organized_paths['annotated'] = create_annotated_image(file_path, result)
        
        if not predictions:
            # Handle images with no predictions
            unknown_path = copy_to_directory(
                file_path, 
                CLASSIFICATION_DIRS['Unknown'], 
                "_unknown"
            )
            organized_paths['classification'] = unknown_path
        else:
            # Get the first prediction (should be the only one for classification)
            pred = predictions[0]
            class_name = pred.get('class_name', 'Unknown')
            confidence = pred.get('confidence', 0.0)
            
            # Map class name to our known categories
            if class_name not in CLASSIFICATION_DIRS:
                class_name = 'Unknown'
            
            # Copy to appropriate classification directory
            class_path = copy_to_directory(
                file_path,
                CLASSIFICATION_DIRS[class_name],
                f"_{class_name}_{confidence:.2f}"
            )
            organized_paths['classification'] = class_path
        
        # Delete original upload after copying to all relevant directories
        file_path.unlink()
        
        logger.info(f"Organized image into classification directory")
        
        return organized_paths
    
    except Exception as e:
        logger.error(f"Error organizing processed image: {str(e)}", exc_info=True)
        
        # Try to save to failed processing directory if something goes wrong
        try:
            failed_path = copy_to_directory(
                file_path,
                CLASSIFICATION_DIRS['failed'],
                "_failed"
            )
            organized_paths['classification'] = failed_path
        except Exception as inner_e:
            logger.error(f"Error copying to failed directory: {str(inner_e)}")
        
        return organized_paths

def organize_processed_image_async(file_path: Path, result: dict, task_id: str):
    """
    Background task to organize processed images.
    This function will be called in a separate thread.
    """
    background_tasks[task_id] = {
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "file_path": str(file_path),
        "task_type": "file_organization"
    }
    
    try:
        organized_paths = organize_processed_image(file_path, result)
        
        background_tasks[task_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "success": True,
            "paths": {
                "classification": str(organized_paths['classification']) if organized_paths['classification'] else None,
                "annotated": str(organized_paths['annotated']) if organized_paths['annotated'] else None
            }
        })
        logger.info(f"Background task {task_id}: Completed file organization")
        return organized_paths
    except Exception as e:
        error_msg = f"Background task {task_id}: Error organizing file {file_path}: {str(e)}"
        logger.error(error_msg)
        
        background_tasks[task_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "success": False
        })
        
        # Move to failed directory if organization fails
        try:
            failed_path = CLASSIFICATION_DIRS['failed'] / file_path.name
            shutil.move(str(file_path), str(failed_path))
            background_tasks[task_id]["moved_to_failed"] = str(failed_path)
        except Exception as move_error:
            logger.error(f"Background task {task_id}: Error moving file to failed directory: {str(move_error)}")
            background_tasks[task_id]["move_error"] = str(move_error)
        
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    storage_info = {
        "classification": {k: str(v) for k, v in CLASSIFICATION_DIRS.items()},
        "base_dirs": {
            "upload": str(UPLOAD_DIR),
            "processed": str(PROCESSED_DIR),
            "archive": str(ARCHIVE_DIR)
        }
    }
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "storage": storage_info
    })

@app.route('/background-tasks', methods=['GET'])
def get_background_tasks():
    """Endpoint to check the status of background tasks"""
    return jsonify({
        "success": True,
        "task_count": len(background_tasks),
        "tasks": background_tasks
    })

@app.route('/detect-imperfection', methods=['POST'])
def detect_imperfection():
    """Main endpoint for wood imperfection detection"""
    if 'image' not in request.files:
        logger.warning("No image part in the request")
        return jsonify({
            "success": False,
            "error": "No image part"
        }), 400

    file = request.files['image']
    
    if file.filename == '':
        logger.warning("No file selected for uploading")
        return jsonify({
            "success": False,
            "error": "No selected file"
        }), 400
    
    def allowed_file(filename):
        """Check if the file extension is allowed"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    if file and allowed_file(file.filename):
        try:
            # Generate secure filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            secure_fname = secure_filename(file.filename)
            filename = f"{timestamp}_{secure_fname}"
            upload_path = UPLOAD_DIR / filename
            
            # Save file
            file.save(str(upload_path))
            logger.info(f"File saved at {upload_path}")
            
            try:
                # Process image
                result = process_image(str(upload_path))
                
                # Generate a unique task ID for the background task
                task_id = f"task_{timestamp}_{hash(filename) % 10000:04d}"
                
                # Schedule file organization to run in background
                background_executor.submit(organize_processed_image_async, upload_path, result, task_id)
                
                # Prepare the response with task ID
                response_data = {
                    "success": True,
                    "data": result,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Processing complete. File organization running in background.",
                    "background_task_id": task_id
                }
                
                logger.info(f"File organization for {filename} scheduled as background task {task_id}")
                
                return jsonify(response_data), 200
                
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Move to failed directory
                failed_path = CLASSIFICATION_DIRS['failed'] / filename
                shutil.move(str(upload_path), str(failed_path))
                
                return jsonify({
                    "success": False,
                    "error": "Failed to process image",
                    "detail": str(e)
                }), 500
                
        except Exception as e:
            error_msg = f"Error handling file upload: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({
                "success": False,
                "error": "Failed to handle file upload",
                "detail": str(e)
            }), 500
    else:
        logger.warning(f"File {file.filename} is not allowed")
        return jsonify({
            "success": False,
            "error": "Invalid file type",
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        }), 400
    
def cleanup_old_files(max_age_hours=24):
    """
    Clean up old files and maintain directory structure.
    Files older than max_age_hours are moved to archive while preserving their category structure.
    
    Args:
        max_age_hours (int): Maximum age of files in hours before they're archived
    """
    try:
        current_time = datetime.now()
        
        def archive_old_files(directory: Path):
            """Archive old files from a directory while maintaining structure"""
            for file_path in directory.glob('**/*'):
                if not file_path.is_file():
                    continue
                    
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    try:
                        # Create corresponding archive directory structure
                        relative_path = file_path.relative_to(PROCESSED_DIR)
                        archive_path = ARCHIVE_DIR / relative_path
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Move file to archive
                        shutil.move(str(file_path), str(archive_path))
                        logger.info(f"Archived old file: {file_path} -> {archive_path}")
                    except Exception as e:
                        logger.error(f"Error archiving file {file_path}: {str(e)}")
        
        # Clean up temporary upload directory completely
        for file_path in UPLOAD_DIR.glob('*'):
            if file_path.is_file():
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > 3600:  # Clean uploads older than 1 hour
                    try:
                        file_path.unlink()
                        logger.info(f"Removed old upload: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing old upload {file_path}: {str(e)}")
        
        # Archive old files from classification directory
        archive_old_files(PROCESSED_DIR / 'classification')
        
        # Archive old files from failed processing directory
        archive_old_files(CLASSIFICATION_DIRS['failed'])
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def cleanup_old_task_history():
    """Remove old tasks from the history to prevent memory leaks"""
    if len(background_tasks) > MAX_TASK_HISTORY:
        # Get tasks sorted by completion time (oldest first)
        sorted_tasks = sorted(
            background_tasks.items(),
            key=lambda x: x[1].get('end_time', x[1].get('start_time', '')),
            reverse=False
        )
        # Remove oldest tasks
        tasks_to_remove = len(background_tasks) - MAX_TASK_HISTORY
        for i in range(tasks_to_remove):
            if i < len(sorted_tasks):
                task_id = sorted_tasks[i][0]
                del background_tasks[task_id]
        logger.info(f"Cleaned up {tasks_to_remove} old background tasks from history")

def start_cleanup_scheduler():
    """Start the periodic cleanup of old files"""
    def cleanup_schedule():
        """Clean up old files and task history"""
        while True:
            try:
                logger.info("Running scheduled cleanup...")
                cleanup_old_files()
                cleanup_old_task_history()
                threading.Event().wait(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in cleanup schedule: {str(e)}")
                threading.Event().wait(3600)  # Wait and try again
    
    cleanup_thread = threading.Thread(target=cleanup_schedule, daemon=True)
    cleanup_thread.start()

# Add a function to gracefully shut down the thread pool
def shutdown_executor():
    """Shutdown the background thread pool gracefully"""
    logger.info("Shutting down background task executor...")
    background_executor.shutdown(wait=True)
    
    # Clear the background tasks dictionary to free memory
    background_tasks.clear()
    logger.info("Cleared background task history")
    
    logger.info("Background task executor shut down successfully")

# Register the shutdown function to run when Flask exits
import atexit
atexit.register(shutdown_executor)

if __name__ == '__main__':
    try:
        # Start cleanup scheduler
        logger.info("Starting cleanup scheduler...")
        start_cleanup_scheduler()
        
        # Log startup
        logger.info("Starting Flask application...")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise