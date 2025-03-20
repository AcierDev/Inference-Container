import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from pathlib import Path
import json
from datetime import datetime
import time
import warnings

class WoodDetectionTester:
    def __init__(self, api_url="http://localhost:5000", retries=3, retry_delay=1):
        self.api_url = api_url
        self.retries = retries
        self.retry_delay = retry_delay
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Colors for different defect types
        self.colors = {
            'damage': 'red',
            'knot': 'blue',
            'crack': 'green',
            'default': 'yellow'
        }
        
        # Test API health
        self._check_api_health()
    
    def _check_api_health(self):
        """Check if the API is ready"""
        for i in range(self.retries):
            try:
                response = requests.get(f"{self.api_url}/health")
                if response.ok:
                    print("API is ready!")
                    return True
                else:
                    print(f"API not ready (attempt {i+1}/{self.retries})")
            except requests.RequestException as e:
                print(f"Error connecting to API (attempt {i+1}/{self.retries}): {str(e)}")
            
            if i < self.retries - 1:
                print(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        raise ConnectionError("Could not connect to API")
    
    def test_image(self, image_path: str, save_visualization: bool = True) -> dict:
        """Test a single image and return the detection results"""
        print(f"\nTesting image: {image_path}")
        
        # Verify image exists and is readable
        try:
            img = Image.open(image_path)
            img.verify()  # Verify it's actually an image
        except Exception as e:
            print(f"Error: Invalid image file - {str(e)}")
            return None
        
        # Send request to API with retries
        for i in range(self.retries):
            try:
                with open(image_path, 'rb') as f:
                    files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                    response = requests.post(
                        f"{self.api_url}/detect-imperfection",
                        files=files,
                        timeout=30  # 30 second timeout
                    )
                
                if response.ok:
                    break
                    
                print(f"Request failed (attempt {i+1}/{self.retries}): {response.text}")
                if i < self.retries - 1:
                    time.sleep(self.retry_delay)
                    
            except requests.RequestException as e:
                print(f"Request error (attempt {i+1}/{self.retries}): {str(e)}")
                if i < self.retries - 1:
                    time.sleep(self.retry_delay)
                continue
        else:
            print("All retry attempts failed")
            return None
            
        try:
            result = response.json()
        except json.JSONDecodeError:
            print("Error: Invalid JSON response from API")
            return None
        
        if result.get('success'):
            num_predictions = len(result['data']['predictions'])
            print(f"Successfully detected {num_predictions} imperfections")
            
            # Save raw results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.results_dir / f"result_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved raw results to {result_file}")
            
            # Create visualization
            if save_visualization:
                try:
                    self._create_visualization(image_path, result['data'], timestamp)
                except Exception as e:
                    print(f"Warning: Failed to create visualization - {str(e)}")
            
            return result
        else:
            print(f"Detection failed: {result.get('error')}")
            return None
    
    def _create_visualization(self, image_path: str, result: dict, timestamp: str) -> None:
        """Create and save visualization of detection results"""
        # Load image
        img = Image.open(image_path)
        
        # Create figure with size based on image dimensions
        dpi = 100
        width = img.width / dpi
        height = img.height / dpi
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        
        # Remove axes
        ax.axis('off')
        
        # Display image
        ax.imshow(img)
        
        # Draw bounding boxes
        for pred in result['predictions']:
            # Get normalized coordinates
            bbox = pred['bbox']
            x1, y1, x2, y2 = bbox
            
            # Convert normalized coordinates to pixel coordinates
            x1 *= img.width
            x2 *= img.width
            y1 *= img.height
            y2 *= img.height
            
            # Get defect type and confidence
            defect_type = pred['class_name']
            confidence = pred['confidence']
            
            # Create rectangle
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=self.colors.get(defect_type, self.colors['default']),
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with better positioning and visibility
            label = f"{defect_type}: {confidence:.2f}"
            ax.text(
                x1, y1-5, label,
                color='white',
                bbox=dict(facecolor='black', alpha=0.8, pad=2, edgecolor='none'),
                fontsize=8,
                ha='left',
                va='bottom'
            )
        
        # Add title
        plt.title(f"Wood Imperfection Detection Results\n{os.path.basename(image_path)}")
        
        # Save visualization with high quality
        viz_file = self.results_dir / f"visualization_{timestamp}.png"
        plt.savefig(viz_file, bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.close()
        print(f"Saved visualization to {viz_file}")

def main():
    # First check if we have test images
    test_dir = Path("test_images")
    if not test_dir.exists() or not list(test_dir.glob("*")):
        print("\nNo test images found. Running download script...")
    
    # Initialize tester
    try:
        tester = WoodDetectionTester()
    except ConnectionError as e:
        print(f"Failed to initialize tester: {str(e)}")
        return
    
    # Test all images in the test directory
    success_count = 0
    total_count = 0
    
    for image_path in test_dir.glob("*"):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            total_count += 1
            if tester.test_image(str(image_path)) is not None:
                success_count += 1
    
    print(f"\nTesting completed: {success_count}/{total_count} images processed successfully")

if __name__ == "__main__":
    # Suppress warnings about font family
    warnings.filterwarnings("ignore", category=UserWarning)
    main()