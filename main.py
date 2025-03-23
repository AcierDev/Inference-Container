from flask import Flask, request, jsonify
import logging
from datetime import datetime
import threading
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
import colorsys

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

# Define category directories for defect types
DEFECT_DIRS = {
    'corner': PROCESSED_DIR / 'defect_types' / 'corner',
    'crack': PROCESSED_DIR / 'defect_types' / 'crack',
    'damage': PROCESSED_DIR / 'defect_types' / 'damage',
    'edge': PROCESSED_DIR / 'defect_types' / 'edge',
    'knot': PROCESSED_DIR / 'defect_types' / 'knot',
    'router': PROCESSED_DIR / 'defect_types' / 'router',
    'side': PROCESSED_DIR / 'defect_types' / 'side',
    'tearout': PROCESSED_DIR / 'defect_types' / 'tearout',
    'clean': PROCESSED_DIR / 'defect_types' / 'no_defects',
    'failed': PROCESSED_DIR / 'failed_processing'
}

# Define count-based directories
COUNT_DIRS = {
    '1-5': PROCESSED_DIR / 'count_based' / '01_to_05_defects',
    '5-10': PROCESSED_DIR / 'count_based' / '05_to_10_defects',
    '10-15': PROCESSED_DIR / 'count_based' / '10_to_15_defects',
    '15-20': PROCESSED_DIR / 'count_based' / '15_to_20_defects',
    '20+': PROCESSED_DIR / 'count_based' / '20_plus_defects'
}

# Ensure all directories exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, ARCHIVE_DIR, ANNOTATED_DIR, *DEFECT_DIRS.values(), *COUNT_DIRS.values()]:
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(log_dir / 'app.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_annotated_image(image_path: Path, result: dict) -> Path:
    """
    Create a copy of the image with bounding boxes and labels drawn.
    
    Args:
        image_path (Path): Path to the original image
        result (dict): Detection results containing predictions
        
    Returns:
        Path: Path to the annotated image
    """
    # Load image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Generate distinct colors for each defect type
    defect_types = {pred['class_name'] for pred in result.get('predictions', [])}
    num_colors = max(len(defect_types), 1)
    colors = {}
    for i, defect_type in enumerate(defect_types):
        # Generate evenly spaced hues
        hue = i / num_colors
        # Convert HSV to RGB (using full saturation and value)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # Convert to 8-bit RGB
        colors[defect_type] = tuple(int(x * 255) for x in rgb)
    
    # Draw predictions
    width, height = img.size
    for pred in result.get('predictions', []):
        # Get normalized coordinates
        bbox = pred['bbox']
        x1, y1, x2, y2 = bbox
        
        # Convert to pixel coordinates
        x1 = int(x1 * width)
        x2 = int(x2 * width)
        y1 = int(y1 * height)
        y2 = int(y2 * height)
        
        # Get defect type and confidence
        defect_type = pred['class_name']
        confidence = pred['confidence']
        
        # Get color for this defect type
        color = colors.get(defect_type, (255, 255, 0))  # Yellow as default
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Create label
        label = f"{defect_type}: {confidence:.2f}"
        
        # Calculate label background size
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        
        # Draw label background
        label_x = x1
        label_y = max(0, y1 - label_height - 5)
        draw.rectangle(
            [label_x, label_y, label_x + label_width, label_y + label_height],
            fill=color
        )
        
        # Draw label text
        draw.text((label_x, label_y), label, fill='white', font=font)
    
    # Add summary text at the bottom
    summary_text = f"Total defects: {len(result.get('predictions', []))}"
    summary_bbox = draw.textbbox((0, 0), summary_text, font=font)
    summary_width = summary_bbox[2] - summary_bbox[0]
    summary_y = height - 30  # 30 pixels from bottom
    
    # Draw summary background
    draw.rectangle(
        [10, summary_y, 10 + summary_width + 10, summary_y + 25],
        fill=(0, 0, 0)
    )
    
    # Draw summary text
    draw.text((15, summary_y), summary_text, fill='white', font=font)
    
    # Save annotated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotated_path = ANNOTATED_DIR / f"annotated_{timestamp}{image_path.suffix}"
    img.save(annotated_path, quality=95)
    
    return annotated_path

def get_count_category(num_defects: int) -> str:
    """Determine the count-based category for number of defects"""
    if num_defects == 0:
        return None
    elif num_defects <= 5:
        return '1-5'
    elif num_defects <= 10:
        return '5-10'
    elif num_defects <= 15:
        return '10-15'
    elif num_defects <= 20:
        return '15-20'
    else:
        return '20+'

def copy_to_directory(src_path: Path, dest_dir: Path, suffix: str = "") -> Path:
    """Copy file to destination directory with optional suffix in filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}{suffix}{src_path.suffix}"
    dest_path = dest_dir / new_filename
    shutil.copy2(str(src_path), str(dest_path))
    return dest_path

def organize_processed_image(file_path: Path, result: dict) -> dict:
    """
    Organize the processed image by copying to all relevant category folders.
    Returns dict with paths to all copies.
    """
    organized_paths = {
        'defect_types': [],
        'count_based': None,
        'annotated': None,
        'original': file_path
    }
    
    try:
        predictions = result.get('predictions', [])
        num_predictions = len(predictions)
        
        # If there are any predictions, create annotated version
        if num_predictions > 0:
            organized_paths['annotated'] = create_annotated_image(file_path, result)
        
        # Get unique defect types
        defect_types = set(pred['class_name'] for pred in predictions)
        
        if num_predictions == 0:
            # Handle clean images
            clean_path = copy_to_directory(
                file_path, 
                DEFECT_DIRS['clean'], 
                "_clean"
            )
            organized_paths['defect_types'].append(clean_path)
        else:
            # Copy to each defect type directory
            for defect_type in defect_types:
                if defect_type in DEFECT_DIRS:
                    type_specific_path = copy_to_directory(
                        file_path,
                        DEFECT_DIRS[defect_type],
                        f"_{defect_type}_{num_predictions}defects"
                    )
                    organized_paths['defect_types'].append(type_specific_path)
            
            # Copy to count-based directory
            count_category = get_count_category(num_predictions)
            if count_category:
                count_path = copy_to_directory(
                    file_path,
                    COUNT_DIRS[count_category],
                    f"_{num_predictions}defects"
                )
                organized_paths['count_based'] = count_path
        
        # Delete original upload after copying to all relevant directories
        file_path.unlink()
        logger.info(f"Organized image into {len(organized_paths['defect_types']) + (1 if organized_paths['count_based'] else 0)} directories")
        
        return organized_paths
        
    except Exception as e:
        logger.error(f"Error organizing file {file_path}: {str(e)}")
        # Move to failed directory if organization fails
        failed_path = DEFECT_DIRS['failed'] / file_path.name
        shutil.move(str(file_path), str(failed_path))
        return {
            'defect_types': [failed_path], 
            'count_based': None, 
            'annotated': None, 
            'original': file_path
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    storage_info = {
        "defect_types": {k: str(v) for k, v in DEFECT_DIRS.items()},
        "count_based": {k: str(v) for k, v in COUNT_DIRS.items()},
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
            
            # Save the file
            file.save(str(upload_path))
            logger.info(f"File saved at {upload_path}")
            
            try:
                # Process the image
                result = process_image(str(upload_path))
                
                # Organize the file based on results
                organized_paths = organize_processed_image(upload_path, result)
                
                # Add file location info to result
                result['file_info'] = {
                    'original_filename': file.filename,
                    'stored_locations': {
                        'defect_types': [str(p) for p in organized_paths['defect_types']],
                        'count_based': str(organized_paths['count_based']) if organized_paths['count_based'] else None
                    }
                }
                
                logger.info(f"Successfully processed and organized {filename}")
                
                return jsonify({
                    "success": True,
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Move to failed directory
                failed_path = DEFECT_DIRS['failed'] / filename
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
        
        # Archive old files from defect type directories
        archive_old_files(PROCESSED_DIR / 'defect_types')
        
        # Archive old files from count-based directories
        archive_old_files(PROCESSED_DIR / 'count_based')
        
        # Archive old files from failed processing directory
        archive_old_files(DEFECT_DIRS['failed'])
        
        # Optional: Clean up old archives
        # Uncomment and modify if you want to permanently delete very old archived files
        """
        for file_path in ARCHIVE_DIR.glob('**/*'):
            if not file_path.is_file():
                continue
                
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.total_seconds() > (max_age_hours * 2 * 3600):  # Double the retention for archives
                try:
                    file_path.unlink()
                    logger.info(f"Removed old archive: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing old archive {file_path}: {str(e)}")
        """
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def start_cleanup_scheduler():
    """Start the periodic cleanup of old files"""
    def cleanup_schedule():
        while True:
            cleanup_old_files()
            threading.Event().wait(3600)  # Run every hour
    
    cleanup_thread = threading.Thread(target=cleanup_schedule, daemon=True)
    cleanup_thread.start()

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