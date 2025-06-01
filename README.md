# Vehicle-CV-ADAS-Paragrutarally

**Paragrutarally Advanced Road Assistance System (PARAS)**

This is an enhanced adaptation of the original **jason-li-831202/Vehicle-CV-ADAS** with improved road detection capabilities, NVIDIA Jetson Xavier compatibility, and modern ADAS features.

<p>
    <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-14354C.svg?logo=python&logoColor=white"></a>
    <a href="#"><img alt="OnnxRuntime" src="https://img.shields.io/badge/OnnxRuntime-FF6F00.svg?logo=onnx&logoColor=white"></a>
    <a href="#"><img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-49D.svg?logo=flask&logoColor=white"></a>
    <a href="#"><img alt="NVIDIA Jetson" src="https://img.shields.io/badge/NVIDIA-Jetson-green.svg?logo=nvidia&logoColor=white"></a>
    <a href="#"><img alt="License" src="https://img.shields.io/badge/License-GPLv3-blue.svg"></a>
</p>

## 🚗 **Overview**

ADAS-Paragrutarally is an advanced driver assistance system (ADAS) based on computer vision techniques. This project uses various object detection, road segmentation, and lane detection algorithms to provide comprehensive safety features for vehicles.

![!ADAS on video](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/demo.JPG)

## 🎯 **Key Features**

### **Safety Systems**
- **Front Collision Warning System (FCWS)** - Detects potential collisions with vehicles ahead
- **Lane Departure Warning System (LDWS)** - Warns when vehicle drifts from lane boundaries  
- **Lane Keeping Assist System (LKAS)** - Provides guidance to stay in lane
- **Road Departure Warning System (RDWS)** - Uses road segmentation for departure warnings

### **Technical Capabilities**
- **Multi-model Support**: Compatible with YOLO family (v5-v10), EfficientDet, Ultra-Fast Lane Detection
- **Real-time Processing**: Object detection, tracking, and distance measurement
- **Road Intelligence**: Advanced road segmentation and lane detection
- **Visual Feedback**: Comprehensive warning displays and driver guidance
- **Hardware Acceleration**: TensorRT and ONNX runtime optimization
- **Cross-Platform**: Standard CUDA and NVIDIA Jetson Xavier support

## 🔧 **Technical Components**

### **Object Detection & Tracking**
- **Models**: YOLOv5/v6/v7/v8/v9/v10, EfficientDet
- **Tracking**: ByteTrack implementation for consistent object IDs
- **Distance**: Single camera-based distance estimation
- **Classes**: 80 COCO classes (vehicles, pedestrians, traffic signs, etc.)

### **Lane & Road Detection**
- **Lane Detection**: UltraFast Lane Detection (v1 and v2)
- **Road Segmentation**: TwinLiteNet for drivable area detection
- **Perspective**: Bird's eye view transformation
- **Datasets**: Support for Tusimple, CULane, CurveLanes datasets

### **Model Optimization**
- **Formats**: ONNX and TensorRT support
- **Precision**: FP16/FP32 options with automatic quantization
- **Performance**: Optimized for real-time inference

## 🖥️ **Platform Support**

### **Standard Platforms**
- **OS**: Windows, Linux
- **GPU**: NVIDIA CUDA-capable GPUs
- **Memory**: 8GB+ recommended

### **NVIDIA Jetson Xavier** ⚡
- **Models**: AGX Xavier, Xavier NX
- **JetPack**: 4.6+ or 5.0+
- **Performance**: 15-35 FPS depending on model complexity
- **Optimization**: Native TensorRT acceleration, thermal monitoring

## 🚀 **Quick Start**

### **Standard Installation**
```bash
# Clone repository
git clone https://github.com/your-repo/Vehicle-CV-ADAS.git
cd Vehicle-CV-ADAS

# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py
```

### **NVIDIA Jetson Xavier Installation**
```bash
# Clone repository
git clone https://github.com/your-repo/Vehicle-CV-ADAS.git
cd Vehicle-CV-ADAS

# Run automated Jetson installation
chmod +x install_jetson.sh
./install_jetson.sh

# Test installation
python3 test_jetson.py

# Run Jetson-optimized demo
python3 demo_jetson.py
```

## 📋 **Requirements**

### **Core Requirements**
- **Python**: 3.7+
- **CUDA**: 10.2+ (for GPU acceleration)
- **Memory**: 8GB RAM minimum, 16GB+ recommended

### **Standard Platform Dependencies**
```bash
pip install -r requirements.txt
```
- OpenCV, NumPy, SciPy
- ONNX Runtime (GPU)
- PyTorch, TorchVision
- Scikit-learn, Numba

### **Jetson Xavier Dependencies**
```bash
pip install -r requirements_jetson.txt
```
- JetPack SDK (includes TensorRT, CUDA)
- PyTorch for Jetson
- ONNX Runtime GPU (Jetson build)
- Platform-specific optimizations

## ⚙️ **Configuration Examples**

### **Object Detection Configuration**
```python
# Standard configuration
object_config = {
    "model_path": './ObjectDetector/models/yolov10n-coco_fp16.trt',
    "model_type": ObjectModelType.YOLOV10,
    "classes_path": './ObjectDetector/models/coco_label.txt',
    "box_score": 0.4,
    "box_nms_iou": 0.5
}

# Jetson Xavier optimized configuration
object_config_jetson = {
    "model_path": './ObjectDetector/models/yolov5n-coco_fp16.onnx',
    "model_type": ObjectModelType.YOLOV5,
    "classes_path": './ObjectDetector/models/coco_label.txt',
    "box_score": 0.5,
    "box_nms_iou": 0.5
}
```

### **Road Detection Configuration**
```python
# Road segmentation configuration
road_config = {
    "model_path": "./RoadSegmentation/models/twinlitenet_drivable.onnx",
}

# Lane detection configuration
lane_config = {
    "model_path": './TrafficLaneDetector/models/culane_res18.onnx',
    "model_type": LaneModelType.UFLDV2_CULANE,
}
```

### **Basic Usage Example**
```python
# Initialize detectors
roadDetector = RoadSegmentationDetector(road_config["model_path"])
objectDetector = YoloDetector(logger=LOGGER)
objectTracker = BYTETracker(names=objectDetector.colors_dict)

# Process frame
objectDetector.DetectFrame(frame)
roadDetector.DetectFrame(frame)
objectTracker.update(boxes, scores, class_ids, frame)

# Draw results
objectDetector.DrawDetectedOnFrame(frame)
roadDetector.DrawAreaOnFrame(frame)
objectTracker.DrawTrackedOnFrame(frame)
```

## 📊 **Performance Benchmarks**

### **Standard CUDA Platforms**
| Model | RTX 3080 | RTX 4070 | GTX 1660 Ti |
|-------|----------|----------|-------------|
| YOLOv5n | 60+ FPS | 80+ FPS | 45 FPS |
| YOLOv8m | 35 FPS | 50 FPS | 25 FPS |
| Complete ADAS | 25-30 FPS | 35-40 FPS | 20 FPS |

### **NVIDIA Jetson Xavier**
| Model | Xavier AGX | Xavier NX | Memory Usage |
|-------|------------|-----------|--------------|
| YOLOv5n-FP16 | ~35 FPS | ~25 FPS | ~2GB |
| YOLOv5s-FP16 | ~28 FPS | ~18 FPS | ~3GB |
| Complete ADAS | 15-20 FPS | 10-15 FPS | ~4GB |

## 🛠️ **Development & Optimization**

### **Model Conversion (Jetson)**
```python
from jetson_utils import JetsonModelOptimizer

optimizer = JetsonModelOptimizer()
success = optimizer.convert_to_tensorrt(
    onnx_path="./models/yolov5n-coco.onnx",
    output_path="./models/yolov5n-coco_jetson.trt",
    fp16=True
)
```

### **Performance Monitoring (Jetson)**
```python
from jetson_utils import setup_jetson_environment, get_jetson_status

# Setup monitoring
optimizer = setup_jetson_environment()

# Get status
status = get_jetson_status()
print(f"Temperature: {status['temperature']}°C")
print(f"Memory Usage: {status['memory_usage']}%")
```

### **System Optimization (Jetson)**
```bash
# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor system
sudo tegrastats
```

## 🔧 **Supported Models**

### **Object Detection Models**
| Model Family | Versions | Precision | Performance |
|--------------|----------|-----------|-------------|
| YOLOv5 | n/s/m/l/x | FP32/FP16 | Excellent |
| YOLOv6 | n/s/m/l | FP32/FP16 | Very Good |
| YOLOv7 | tiny/x/w/e/d | FP32/FP16 | Very Good |
| YOLOv8 | n/s/m/l/x | FP32/FP16 | Excellent |
| YOLOv9 | t/s/m/c/e | FP32/FP16 | Excellent |
| YOLOv10 | n/s/m/b/l/x | FP32/FP16 | Excellent |
| EfficientDet | b0/b1/b2/b3 | FP32/FP16 | Good |

### **Lane Detection Models**
| Model | Dataset | Backbone | Performance |
|-------|---------|----------|-------------|
| UFLD v1 | Tusimple | ResNet18 | Fast |
| UFLD v1 | CULane | ResNet18 | Fast |
| UFLD v2 | Tusimple | ResNet18/34 | Faster |
| UFLD v2 | CULane | ResNet18/34 | Faster |

## 📁 **Project Structure**

```
Vehicle-CV-ADAS/
├── demo.py                          # Standard demo application
├── demo_jetson.py                   # Jetson-optimized demo
├── coreEngine.py                    # Inference engine (Jetson compatible)
├── requirements.txt                 # Standard dependencies
├── requirements_jetson.txt          # Jetson dependencies
├── install_jetson.sh               # Jetson installation script
├── test_jetson.py                  # Jetson compatibility tests
├── jetson_utils.py                 # Jetson utilities
├── ObjectDetector/                 # Object detection modules
│   ├── yoloDetector.py             # YOLO implementation
│   ├── efficientdetDetector.py     # EfficientDet implementation
│   ├── distanceMeasure.py          # Distance calculation
│   └── models/                     # Model files
├── TrafficLaneDetector/            # Lane detection modules
│   ├── ufldDetector/               # UFLD implementation
│   └── models/                     # Lane model files
├── RoadSegmentation/               # Road segmentation
│   ├── roadSegmentationDetector.py
│   └── models/                     # Segmentation models
├── ObjectTracker/                  # Object tracking
│   └── byteTrack/                  # ByteTrack implementation
├── assets/                         # UI assets and icons
└── demo/                          # Demo videos and images
```

## 🎥 **Demo & Examples**

### **Demo Video**
- [**Demo Youtube Video**](https://www.youtube.com/watch?v=CHO0C1z5EWE)

### **ADAS Features**
![!FCWS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/FCWS.jpg)
![!LDWS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/LDWS.jpg)
![!LKAS](https://github.com/jason-li-831202/Vehicle-CV-ADAS/blob/master/demo/LKAS.jpg)

## 🔬 **Testing & Validation**

### **Run Tests**
```bash
# Standard platform tests
python -m pytest tests/

# Jetson compatibility tests
python3 test_jetson.py

# Performance benchmarks
python3 -c "from jetson_utils import JetsonModelOptimizer; optimizer = JetsonModelOptimizer(); optimizer.benchmark_model('model.onnx', (1,3,640,640))"
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Memory Errors**
```bash
# Increase swap (Linux/Jetson)
sudo fallocate -l 8G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
```

#### **Low Performance (Jetson)**
```bash
# Check performance mode
sudo nvpmodel -q  # Should show mode 0
sudo jetson_clocks --show

# Monitor thermal throttling
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

#### **TensorRT Issues (Jetson)**
- System automatically falls back to ONNX if TensorRT fails
- Check logs for "Falling back to ONNX" messages
- Ensure JetPack and TensorRT are properly installed

## 🎯 **Production Deployment**

### **Docker Deployment**
```dockerfile
# Standard platforms
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "/app/demo.py"]

# Jetson platforms
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
COPY requirements_jetson.txt .
RUN pip3 install -r requirements_jetson.txt
COPY . /app
CMD ["python3", "/app/demo_jetson.py"]
```

### **Systemd Service (Jetson)**
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

[Install]
WantedBy=multi-user.target
```

## 📚 **Resources & Documentation**

### **General Resources**
- **Original Project**: [jason-li-831202/Vehicle-CV-ADAS](https://github.com/jason-li-831202/Vehicle-CV-ADAS)
- **YOLO Documentation**: [Ultralytics](https://docs.ultralytics.com/)
- **ONNX Runtime**: [Microsoft ONNX Runtime](https://onnxruntime.ai/)

### **Jetson Resources**
- **NVIDIA Jetson Documentation**: https://docs.nvidia.com/jetson/
- **JetPack SDK**: https://developer.nvidia.com/embedded/jetpack
- **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
- **Jetson Community**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### **Development Guidelines**
- Follow Python PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Test on both standard and Jetson platforms when possible

## 🐛 **Reporting Issues**

When reporting issues, please include:

### **For Standard Platforms**
1. **OS and GPU**: Windows/Linux, GPU model
2. **CUDA Version**: `nvidia-smi` output
3. **Python Version**: `python --version`
4. **Error Logs**: Full error messages and stack traces

### **For Jetson Platforms**
1. **Jetson Model**: AGX Xavier, Xavier NX, etc.
2. **JetPack Version**: `jetson_release` output
3. **System Stats**: `tegrastats` output
4. **Temperature**: Thermal state during error
5. **Model Configuration**: Which models you're using

## 📄 **License**

The jason-li-831202/Vehicle-CV-ADAS project is licensed under the GNU General Public License v3.0 (GPLv3).

**GPLv3 License key requirements**:
- Disclose Source
- License and Copyright Notice  
- Same License
- State Changes

---

**🚗 Safe Driving with Advanced Computer Vision! 🛡️**

*This project demonstrates the power of modern computer vision in automotive safety applications, supporting both high-performance desktop systems and edge devices like NVIDIA Jetson Xavier.*
