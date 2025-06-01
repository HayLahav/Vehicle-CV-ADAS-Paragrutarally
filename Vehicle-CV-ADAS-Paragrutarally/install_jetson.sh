#!/bin/bash
# install_jetson.sh - Jetson Xavier installation script for Vehicle-CV-ADAS

set -e  # Exit on any error

echo "=========================================="
echo "Vehicle-CV-ADAS Jetson Xavier Setup"
echo "=========================================="

# Check if running on Jetson
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "Error: This script is designed for NVIDIA Jetson platforms"
    exit 1
fi

echo "Detected Jetson platform:"
cat /etc/nv_tegra_release

# Check JetPack version
if command -v jetson_release &> /dev/null; then
    echo "JetPack info:"
    jetson_release
else
    echo "jetson_release not found. Please ensure JetPack is properly installed."
fi

# Update system packages
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch for Jetson (if not already installed)
echo "Checking PyTorch installation..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch for Jetson..."
    
    # Detect Jetson Xavier (AGX/NX) and JetPack version
    JETPACK_VERSION="4.6"
    if grep -q "JETPACK_VERSION=5" /etc/nv_tegra_release; then
        JETPACK_VERSION="5.0"
    fi
    
    if [ "$JETPACK_VERSION" = "5.0" ]; then
        # JetPack 5.0+ PyTorch wheel
        wget -q --show-progress https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
        pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
    else
        # JetPack 4.6 PyTorch wheel
        wget -q --show-progress https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
        pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    fi
    
    echo "PyTorch installed successfully"
else
    echo "PyTorch already installed:"
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
fi

# Install TorchVision for Jetson (if not already installed)
echo "Checking TorchVision installation..."
if ! python3 -c "import torchvision" 2>/dev/null; then
    echo "Installing TorchVision for Jetson..."
    
    # Install dependencies for building torchvision
    sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev
    
    # Clone and build torchvision
    if [ -d "torchvision" ]; then
        rm -rf torchvision
    fi
    
    git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.11.1
    python3 setup.py install --user
    cd ..
    rm -rf torchvision
    
    echo "TorchVision installed successfully"
else
    echo "TorchVision already installed:"
    python3 -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
fi

# Install ONNX Runtime GPU for Jetson (if not already installed)
echo "Checking ONNX Runtime installation..."
if ! python3 -c "import onnxruntime" 2>/dev/null; then
    echo "Installing ONNX Runtime GPU for Jetson..."
    
    if [ "$JETPACK_VERSION" = "5.0" ]; then
        # JetPack 5.0+ ONNX Runtime wheel
        wget -q --show-progress https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.12.0-cp38-cp38-linux_aarch64.whl
        pip3 install onnxruntime_gpu-1.12.0-cp38-cp38-linux_aarch64.whl
    else
        # JetPack 4.6 ONNX Runtime wheel
        wget -q --show-progress https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
        pip3 install onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
    fi
    
    echo "ONNX Runtime GPU installed successfully"
else
    echo "ONNX Runtime already installed:"
    python3 -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"
fi

# Install other Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements_jetson.txt

# Set up Jetson performance optimization
echo "Setting up Jetson performance optimization..."
cat << 'EOF' > jetson_performance.py
#!/usr/bin/env python3
"""Jetson performance optimization script"""
import subprocess
import os

def optimize_jetson_performance():
    """Set Jetson to maximum performance mode"""
    try:
        # Set to maximum performance mode
        subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)
        print("✓ Set to maximum performance mode (nvpmodel -m 0)")
        
        # Enable jetson_clocks
        subprocess.run(['sudo', 'jetson_clocks'], check=True)
        print("✓ Enabled jetson_clocks")
        
        # Set GPU to maximum frequency (Xavier AGX)
        try:
            subprocess.run(['sudo', 'bash', '-c', 
                           'echo 1377000000 > /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq'], 
                           check=True)
            print("✓ Set GPU to maximum frequency")
        except:
            print("! Could not set GPU frequency (might be Xavier NX or different model)")
        
        print("Jetson performance optimization completed!")
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not optimize performance: {e}")
        print("You may need to run this script with sudo privileges")

if __name__ == "__main__":
    optimize_jetson_performance()
EOF

chmod +x jetson_performance.py

# Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except ImportError:
    print('✗ PyTorch not found')

try:
    import torchvision
    print(f'✓ TorchVision {torchvision.__version__}')
except ImportError:
    print('✗ TorchVision not found')

try:
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    print(f'✓ ONNX Runtime {onnxruntime.__version__}')
    print(f'  Available providers: {providers}')
except ImportError:
    print('✗ ONNX Runtime not found')

try:
    import cv2
    print(f'✓ OpenCV {cv2.__version__}')
except ImportError:
    print('✗ OpenCV not found')

try:
    import numpy as np
    print(f'✓ NumPy {np.__version__}')
except ImportError:
    print('✗ NumPy not found')

try:
    import tensorrt as trt
    print(f'✓ TensorRT {trt.__version__}')
except ImportError:
    print('! TensorRT not found (check JetPack installation)')
"

echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run 'python3 jetson_performance.py' to optimize performance (requires sudo)"
echo "2. Test the installation with: python3 demo_jetson.py"
echo "3. Use FP16 models for better performance on Jetson"
echo ""
echo "Performance tips:"
echo "- Use YOLOv5n or YOLOv5s models for real-time performance"
echo "- Monitor system with: sudo tegrastats"
echo "- Check temperature with: cat /sys/devices/virtual/thermal/thermal_zone*/temp"