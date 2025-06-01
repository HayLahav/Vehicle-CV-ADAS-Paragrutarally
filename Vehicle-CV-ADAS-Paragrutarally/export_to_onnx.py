
import sys
import torch

# Add the model directory to the system path
sys.path.append('model')  # Adjust if the TwinLite.py is in a different relative path

# Import the model class
from TwinLite import TwinLiteNet  # Adjust the class name if needed

# Initialize the model
model = TwinLiteNet()

# Load the checkpoint
checkpoint = torch.load('pretrained/best.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# Dummy input - adjust shape if needed
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'pretrained/best.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("✅ Exported best.onnx successfully.")
