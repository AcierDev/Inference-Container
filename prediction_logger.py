# prediction_logger.py
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import csv
from dataclasses import dataclass
import threading
from collections import deque

@dataclass
class PredictionMetrics:
    """Stores metrics about predictions"""
    timestamp: str
    filename: str
    num_predictions: int
    processing_time: float  # in seconds
    confidence_scores: list[float]
    image_size: tuple[int, int]
    prediction_types: dict[str, int]  # e.g., {'knot': 5, 'crack': 2}
    total_defect_area: float  # normalized total area of defects

class PredictionLogger:
    def __init__(self, base_dir: str = 'logs'):
        self.base_dir = Path(base_dir)
        self.predictions_dir = self.base_dir / 'predictions'
        self.metrics_dir = self.base_dir / 'metrics'
        self._setup_directories()
        
        # Setup logging
        self.logger = logging.getLogger('prediction_logger')
        self._setup_logger()
        
        # Cache for quick stats (last 1000 predictions)
        self.recent_predictions = deque(maxlen=1000)
        self._lock = threading.Lock()

    def _setup_directories(self):
        """Create necessary directories"""
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Create/check CSV files
        self.daily_metrics_file = self.metrics_dir / 'daily_metrics.csv'
        if not self.daily_metrics_file.exists():
            with open(self.daily_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date', 'total_predictions', 'avg_predictions_per_image',
                    'avg_confidence', 'avg_processing_time', 'total_defect_area'
                ])

    def _setup_logger(self):
        """Setup dedicated logger for predictions"""
        handler = logging.FileHandler(self.base_dir / 'predictions.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def calculate_metrics(self, result: Dict[str, Any], 
                         filename: str, 
                         processing_time: float) -> PredictionMetrics:
        """Calculate metrics from prediction results"""
        predictions = result.get('predictions', [])
        
        # Get confidence scores
        confidence_scores = [
            pred.get('confidence', 0.0) 
            for pred in predictions
        ]
        
        # Count prediction types
        prediction_types = {}
        for pred in predictions:
            pred_type = pred.get('class_name', 'unknown')
            prediction_types[pred_type] = prediction_types.get(pred_type, 0) + 1
        
        # Calculate total defect area
        total_defect_area = sum(
            (pred['bbox'][2] - pred['bbox'][0]) * 
            (pred['bbox'][3] - pred['bbox'][1])
            for pred in predictions
        )
        
        # Get image size
        image_size = (
            result.get('metadata', {})
            .get('image_size', {'width': 0, 'height': 0})
        )
        
        return PredictionMetrics(
            timestamp=datetime.now().isoformat(),
            filename=filename,
            num_predictions=len(predictions),
            processing_time=processing_time,
            confidence_scores=confidence_scores,
            image_size=(image_size['width'], image_size['height']),
            prediction_types=prediction_types,
            total_defect_area=total_defect_area
        )

    def log_prediction(self, 
                      result: Dict[str, Any], 
                      filename: str, 
                      processing_time: float):
        """Log a prediction result with metrics"""
        try:
            # Calculate metrics
            metrics = self.calculate_metrics(result, filename, processing_time)
            
            # Store in recent predictions cache
            with self._lock:
                self.recent_predictions.append(metrics)
            
            # Log detailed prediction data
            prediction_file = self.predictions_dir / f"{datetime.now().strftime('%Y%m%d')}_{filename}.json"
            with open(prediction_file, 'w') as f:
                json.dump({
                    'timestamp': metrics.timestamp,
                    'filename': filename,
                    'processing_time': processing_time,
                    'metrics': {
                        'num_predictions': metrics.num_predictions,
                        'confidence_scores': metrics.confidence_scores,
                        'prediction_types': metrics.prediction_types,
                        'total_defect_area': metrics.total_defect_area,
                        'image_size': metrics.image_size
                    },
                    'result': result
                }, f, indent=2)
            
            # Log summary
            self.logger.info(
                f"Processed {filename}: {metrics.num_predictions} predictions, "
                f"types: {metrics.prediction_types}, "
                f"avg confidence: {sum(metrics.confidence_scores)/len(metrics.confidence_scores):.2f} "
                f"time: {metrics.processing_time:.2f}s"
            )
            
            # Update daily metrics
            self._update_daily_metrics()
            
        except Exception as e:
            self.logger.error(f"Error logging prediction: {str(e)}", exc_info=True)

    def _update_daily_metrics(self):
        """Update daily metrics file with latest statistics"""
        try:
            with self._lock:
                if not self.recent_predictions:
                    return
                
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Calculate daily averages
                total_predictions = sum(m.num_predictions for m in self.recent_predictions)
                avg_predictions = total_predictions / len(self.recent_predictions)
                avg_confidence = sum(
                    sum(m.confidence_scores) / len(m.confidence_scores) 
                    for m in self.recent_predictions if m.confidence_scores
                ) / len(self.recent_predictions)
                avg_processing_time = sum(
                    m.processing_time for m in self.recent_predictions
                ) / len(self.recent_predictions)
                total_defect_area = sum(
                    m.total_defect_area for m in self.recent_predictions
                )
                
                # Write to CSV
                with open(self.daily_metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        today,
                        total_predictions,
                        avg_predictions,
                        avg_confidence,
                        avg_processing_time,
                        total_defect_area
                    ])
                
        except Exception as e:
            self.logger.error(f"Error updating daily metrics: {str(e)}", exc_info=True)

    def get_recent_stats(self) -> Dict[str, Any]:
        """Get statistics from recent predictions"""
        with self._lock:
            if not self.recent_predictions:
                return {}
            
            total_images = len(self.recent_predictions)
            total_predictions = sum(m.num_predictions for m in self.recent_predictions)
            
            # Aggregate prediction types
            all_types = {}
            for m in self.recent_predictions:
                for pred_type, count in m.prediction_types.items():
                    all_types[pred_type] = all_types.get(pred_type, 0) + count
            
            return {
                'total_images_processed': total_images,
                'total_predictions': total_predictions,
                'avg_predictions_per_image': total_predictions / total_images,
                'prediction_types': all_types,
                'avg_processing_time': sum(
                    m.processing_time for m in self.recent_predictions
                ) / total_images,
                'avg_confidence': sum(
                    sum(m.confidence_scores) / len(m.confidence_scores) 
                    for m in self.recent_predictions if m.confidence_scores
                ) / total_images
            }

# Global instance
prediction_logger = PredictionLogger()