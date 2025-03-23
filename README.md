# Wood Imperfection Detection with YOLOv8s

This project implements wood imperfection detection using YOLOv8s with ONNX and GPU acceleration.

## Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended for faster inference)
- Required packages are listed in `requirements.txt`

## Setup

1. Clone the repository
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Convert the YOLOv8s model to ONNX format

```bash
python convert_to_onnx.py
```

This will:

- Download the YOLOv8s model
- Convert it to ONNX format
- Save it in the `models` directory

## Environment Variables

You can customize the model settings using environment variables:

- `YOLOV8_MODEL_PATH`: Path to your YOLOv8s ONNX model file (default: `models/yolov8s.onnx`)
- `YOLOV8_CLASS_FILE`: Path to a text file containing custom class names (default: `models/classes.txt`)

## Custom Class Names

To use custom class names for your model:

1. Create a text file with one class name per line
2. Set the `YOLOV8_CLASS_FILE` environment variable to point to this file
3. For wood imperfection detection, consider classes like:
   - crack
   - hole
   - stain
   - dent
   - scratch
   - split
   - tear

If a custom class file is not provided, the standard COCO class names will be used.

## Usage

Run the inference on an image:

```python
from inferences.model import process_image

# Run inference on an image
results = process_image("path/to/your/image.jpg")

# Process the results
for prediction in results["predictions"]:
    bbox = prediction["bbox"]
    confidence = prediction["confidence"]
    class_name = prediction["class_name"]
    # Do something with the detection...
```

## GPU Acceleration

The system automatically uses GPU acceleration if available through ONNX Runtime. If a GPU is not available or if there's an error with the GPU, it will gracefully fall back to CPU processing.

## Model Information

- YOLOv8s: A medium-sized model from the YOLOv8 family by Ultralytics
- ONNX: Provides cross-platform acceleration
- Detection format: Bounding boxes with normalized coordinates [x1, y1, x2, y2]

## Troubleshooting

### Confidence Scores

If you encounter extremely high confidence scores (like tens of thousands instead of values between 0 and 1), the issue has been fixed by:

1. Applying sigmoid activation to the raw model outputs
2. Adjusting confidence thresholds appropriately

This happens because YOLOv8 outputs raw logits that need to be converted to probabilities using the sigmoid function.
