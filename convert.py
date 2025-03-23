from ultralytics import YOLO

def convert_pt_to_onnx(pt_model_path, onnx_model_path):
    # Load the YOLOv8 model
    model = YOLO(pt_model_path)  # Load the YOLOv8 model

    # Export the model to ONNX format
    model.export(format='onnx')

    print(f"Model successfully converted to {onnx_model_path}")

# Example usage
pt_model_path = 'best.pt'  # Path to your YOLOv8 .pt model
onnx_model_path = 'best.onnx'  # Path to save the converted ONNX model

convert_pt_to_onnx(pt_model_path, onnx_model_path)
