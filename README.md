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

You can customize the model path using environment variables:

- `YOLOV8_MODEL_PATH`: Path to your YOLOv8s ONNX model file (default: `models/yolov8s.onnx`)

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
