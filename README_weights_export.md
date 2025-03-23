# Model Weights Export Utility

This utility allows you to export the model weights from the inference package's cache to a specified destination directory.

## Prerequisites

- Python 3.7+
- Access to the inference package with a loaded model

## Usage

The basic command to export weights is:

```bash
python export_weights.py --model-id MODEL_ID --destination DESTINATION [--metadata] [--pytorch]
```

### Arguments

- `--model-id`: Required. The ID of the model to export weights for, in the format "model-name/version". Example: "yolo-nas-hi-res/6"
- `--destination`: Required. Path to the directory where weights will be exported.
- `--metadata`: Optional. Include this flag to also export metadata files like environment.json that contain information about the model.
- `--pytorch`: Optional. Include this flag to attempt converting the ONNX weights to PyTorch format. Requires additional dependencies (torch, onnx, onnx2torch).

### Examples

1. Export just the weights file:

   ```bash
   python export_weights.py --model-id "yolo-nas-hi-res/6" --destination "./exported_weights"
   ```

2. Export weights and metadata:

   ```bash
   python export_weights.py --model-id "yolo-nas-hi-res/6" --destination "./exported_weights" --metadata
   ```

3. Export weights, metadata, and attempt conversion to PyTorch format:
   ```bash
   python export_weights.py --model-id "yolo-nas-hi-res/6" --destination "./exported_weights" --metadata --pytorch
   ```

## Exported Files

- `weights.onnx`: The ONNX model weights file
- `environment.json`: (if --metadata is specified) Contains model metadata including class names and configurations
- Other potential metadata files like labels.txt or class_mapping.csv, if available

## PyTorch Conversion

If you include the `--pytorch` flag, the script will attempt to convert the ONNX model to PyTorch format. This requires additional dependencies:

```bash
pip install torch onnx onnx2torch
```

If successful, a `weights.pt` file will be created alongside the ONNX file in the destination directory.

## Troubleshooting

- If the model cache directory is not found, make sure the model has been loaded at least once through the inference package.
- The default cache location is `/tmp/cache`, but it can be customized with the `MODEL_CACHE_DIR` environment variable.
- If PyTorch conversion fails, check the error message for specific dependencies or compatibility issues.

## License

This utility is provided as-is. Use in accordance with the license of the underlying model and inference package.
