import torch
import torch.onnx

# Load your PyTorch model
model = torch.load("F:/Project/face_detection_model.pth")  # Replace with your model's loading logic
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor with the correct input size
dummy_input = torch.randn(1, 3, 224, 224)  # Example for an image model (batch, channels, height, width)

# Export the model to ONNX
onnx_file_path = "F:/Project"
torch.onnx.export(
    model,                       # PyTorch model
    dummy_input,                 # Dummy input tensor
    onnx_file_path,              # Output ONNX file path
    export_params=True,          # Store the trained parameters
    opset_version=11,            # ONNX opset version
    do_constant_folding=True,    # Optimize the model
    input_names=['input'],       # Input tensor names
    output_names=['output'],     # Output tensor names
    dynamic_axes={               # For variable input sizes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model has been converted to ONNX and saved at {onnx_file_path}")
