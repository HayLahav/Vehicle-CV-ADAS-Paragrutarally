# Vehicle-CV-ADAS for NVIDIA Jetson Xavier

This guide provides instructions for setting up and running the Vehicle-CV-ADAS system on NVIDIA Jetson Xavier AGX/NX platforms.

## 🎯 **Jetson Compatibility Overview**

### ✅ **Fully Compatible**
- **Object Detection**: All YOLO variants (v5-v10), EfficientDet
- **Lane Detection**: UFLD v1/v2 models  
- **Road Segmentation**: TwinLiteNet models
- **Object Tracking**: ByteTrack implementation
- **ONNX Runtime**: GPU-accelerated inference
- **TensorRT**: Native Jetson optimization

### 🔧 **Optimizations Applied**
- **Memory Management**: Unified memory architecture support
- **FP16 Precision**: Automatic quantization for performance
- **Thermal Monitoring**: Real-time temperature tracking
- **Performance Modes**: Maximum performance configuration

## 🚀 **Quick Start**

### **Prerequisites**
1. **NVIDIA Jetson Xavier AGX/NX** with JetPack 4.6+ or 5.0+
2. **Minimum 32GB Storage** (64GB recommended)
3. **External Cooling** (fan or heatsink recommended)
4. **USB Camera or Video File** for testing

### **Automatic Installation**
```bash
# Clone the repository
git clone https://github.com/your-repo/Vehicle-CV-ADAS.git
cd Vehicle-CV-ADAS

# Make installation script executable
chmod +x install_jetson.sh

# Run automated installation
./install_jetson.sh
```

### **Manual Installation Steps**

#### **1. System Prerequisites**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-dev cmake \
    libopenblas-dev liblapack-dev libjpeg-dev libpng-dev \
    libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev \
    libatlas-base-dev gfortran pkg-config
```

#### **2. PyTorch Installation**
```bash
# For JetPack 5.0+
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

# For JetPack 4.6
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

#### **3. TorchVision Installation**
```bash
# Install build dependencies
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev

# Build from source
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd .. && rm -rf torchvision
```

#### **4. ONNX Runtime GPU**
```bash
# For JetPack 5.0+
wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.12.0-cp38-cp38-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.12.0-cp38-cp38-linux_aarch64.whl

# For JetPack 4.6
wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
```

#### **5. Project Dependencies**
```bash
pip3 install -r requirements_jetson.txt
```

## ⚙️ **Performance Optimization**

### **Automatic Optimization**
```bash
# Run performance optimization script
python3 jetson_performance.py
```

### **Manual Optimization**
```bash
# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Set GPU maximum frequency (Xavier AGX)
sudo bash -c 'echo 1377000000 > /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq'
```

### **Environment Variables**
```bash
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/tmp/.nv/ComputeCache
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDNN_BENCHMARK=1
```

## 📊 **Model Recommendations**

### **For Real-Time Performance (>20 FPS)**
```python
# Optimized configuration
object_config = {
    "model_path": './ObjectDetector/models/yolov5n-coco_fp16.onnx',
    "model_type": ObjectModelType.YOLOV5,
    "classes_path": './ObjectDetector/models/coco_label.txt',
    "box_score": 0.5,
    "box_nms_iou": 0.5
}

lane_config = {
    "model_path": './TrafficLaneDetector/models/culane_res18_fp16.onnx',
    "model_type": LaneModelType.UFLDV2_CULANE,
}
```

### **Model Size vs Performance Guide**

| Model | FPS (Xavier AGX) | FPS (Xavier NX) | Memory Usage | Accuracy |
|-------|------------------|-----------------|--------------|----------|
| YOLOv5n-FP16 | ~35 FPS | ~25 FPS | ~2GB | Good |
| YOLOv5s-FP16 | ~28 FPS | ~18 FPS | ~3GB | Better |
| YOLOv5m-FP16 | ~18 FPS | ~12 FPS | ~4GB | Best |

## 🏃 **Running the Application**

### **Basic Usage**
```bash
# Run Jetson-optimized demo
python3 demo_jetson.py
```

### **Custom Video Processing**
```python
# Edit demo_jetson.py
video_path = "/path/to/your/video.mp4"

# Or use camera
video_path = 0  # For USB camera
video_path = 1  # For CSI camera
```

### **Live Camera Feed**
```bash
# For CSI camera (Jetson native)
python3 demo_jetson.py --input /dev/video0

# For USB camera
python3 demo_jetson.py --input /dev/video1
```

## 🔧 **Monitoring and Troubleshooting**

### **System Monitoring**
```bash
# Monitor system stats in real-time
sudo tegrastats

# Check temperatures
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Monitor GPU usage
sudo nvtop  # If installed

# Check memory usage
free -h
```

### **Performance Monitoring with Code**
```python
from jetson_utils import setup_jetson_environment, get_jetson_status

# Setup monitoring
optimizer = setup_jetson_environment()

# Get status
status = get_jetson_status()
print(f"Temperature: {status['temperature']}°C")
print(f"Memory Usage: {status['memory_usage']}%")
```

### **Common Issues and Solutions**

#### **1. Out of Memory Errors**
```bash
# Increase swap space
sudo fallocate -l 8G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile

# Make permanent
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

#### **2. Thermal Throttling**
```python
# Check throttling status
from jetson_utils import JetsonOptimizer
optimizer = JetsonOptimizer()
if optimizer.check_thermal_throttling():
    print("⚠️ System is thermal throttling!")
    print("Consider adding cooling or reducing workload")
```

#### **3. Low FPS Performance**
```bash
# Ensure performance mode is active
sudo nvpmodel -q  # Should show mode 0
sudo jetson_clocks --show

# Check if GPU is being used
nvidia-smi  # Should show CUDA processes
```

#### **4. TensorRT Engine Errors**
```python
# Fallback to ONNX if TensorRT fails
# The system automatically handles this in coreEngine.py
# Check logs for "Falling back to ONNX" messages
```

## 📈 **Performance Benchmarking**

### **Benchmark Models**
```python
from jetson_utils import JetsonModelOptimizer

optimizer = JetsonModelOptimizer()
stats = optimizer.benchmark_model(
    model_path="./models/yolov5n-coco_fp16.onnx",
    input_shape=(1, 3, 640, 640),
    iterations=100
)
print(f"Average FPS: {stats['fps']:.1f}")
```

### **Expected Performance Targets**

#### **Xavier AGX (32GB)**
- **Complete ADAS Pipeline**: 15-20 FPS
- **Object Detection Only**: 25-35 FPS  
- **Lane Detection Only**: 40-60 FPS
- **Road Segmentation Only**: 20-30 FPS

#### **Xavier NX (8GB)**
- **Complete ADAS Pipeline**: 10-15 FPS
- **Object Detection Only**: 18-25 FPS
- **Lane Detection Only**: 30-45 FPS
- **Road Segmentation Only**: 15-25 FPS

## 🛠️ **Development Tips**

### **Model Optimization**
```python
# Convert ONNX to TensorRT for better performance
from jetson_utils import JetsonModelOptimizer

optimizer = JetsonModelOptimizer()
success = optimizer.convert_to_tensorrt(
    onnx_path="./models/yolov5n-coco.onnx",
    output_path="./models/yolov5n-coco_jetson.trt",
    fp16=True
)
```

### **Memory Optimization**
```python
# Use smaller input resolutions
object_config["input_size"] = (416, 416)  # Instead of (640, 640)

# Reduce batch size (should be 1 for real-time)
batch_size = 1

# Enable CUDA memory pooling
import os
os.environ['CUDA_CACHE_DISABLE'] = '0'
```

### **Debugging**
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor resource usage
from jetson_utils import setup_jetson_environment
optimizer = setup_jetson_environment()
# Automatic monitoring will start
```

## 🎯 **Production Deployment**

### **Systemd Service Setup**
```bash
# Create service file
sudo nano /etc/systemd/system/vehicle-adas.service
```

```ini
[Unit]
Description=Vehicle ADAS System
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/Vehicle-CV-ADAS
ExecStart=/usr/bin/python3 demo_jetson.py
Restart=always
Environment=PYTHONPATH=/home/nvidia/Vehicle-CV-ADAS

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable vehicle-adas.service
sudo systemctl start vehicle-adas.service
```

### **Docker Deployment** (Advanced)
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app
COPY . .

RUN pip3 install -r requirements_jetson.txt

CMD ["python3", "demo_jetson.py"]
```

## 📚 **Additional Resources**

- **NVIDIA Jetson Documentation**: https://docs.nvidia.com/jetson/
- **JetPack SDK**: https://developer.nvidia.com/embedded/jetpack
- **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
- **Jetson Community Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

## 🐛 **Reporting Issues**

When reporting issues, please include:
1. **Jetson Model**: AGX Xavier, NX, etc.
2. **JetPack Version**: Check with `jetson_release`
3. **Error Logs**: Full error messages
4. **System Stats**: Output of `tegrastats`
5. **Model Configuration**: Which models you're using

## 📄 **License**

This project is licensed under GPLv3 - see the [License](License) file for details.

---

**⚡ Happy ADAS Development on Jetson Xavier! ⚡**
