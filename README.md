# Vehicle-CV-ADAS-Paragrutarally

Paragrutarally Advanced Road Assistance System (PARAS)
##Overview
Paragrutarally is an advanced driver assistance system (ADAS) based on computer vision techniques. The project uses various object detection, road segmentation, and lane detection algorithms to provide safety features including:

Front Collision Warning System (FCWS)
Lane Departure Warning System (LDWS)
Lane Keeping Assist System (LKAS)
Real-time object detection and tracking

This project is an adaptation of the original  **jason-li-831202/Vehicle-CV-ADAS**  with enhanced road detection capabilities and Jeston Xavier adaptations.
Key Features

Multi-model support: Compatible with YOLO family models (v5-v10), EfficientDet, and Ultra-Fast Lane Detection models
Real-time object detection and tracking: Identifies and tracks vehicles, pedestrians, and other road obstacles
Distance measurement: Estimates distance to detected objects
Road segmentation: Identifies drivable areas using semantic segmentation
Lane detection: Detects lane markings and road boundaries
Bird's eye view transformation: Provides overhead perspective of the road scene
Visual feedback: Includes warning displays and guidance for the driver
Hardware acceleration: Supports TensorRT and ONNX runtimes for optimized inference

Technical Components

Object Detection: Using modern architectures (YOLOv5/v6/v7/v8/v9/v10, EfficientDet)
Object Tracking: ByteTrack implementation for consistent object IDs across frames
Lane Detection: UltraFast Lane Detection (v1 and v2)
Road Segmentation: Semantic segmentation for drivable area detection
Distance Measurement: Single camera-based distance estimation

Model Optimization

Support for ONNX and TensorRT formats
Model quantization for improved inference speed
FP16/FP32 precision options

Requirements

Python 3.7+
CUDA-capable GPU
Libraries: OpenCV, scikit-learn, ONNX Runtime, PyCUDA, PyTorch

## Usage
The system can process videos from dashcams or other sources, and provides both visual feedback and warning indications for potential hazards on the road.
python# Set up configurations
road_config = {
    "model_path": "./RoadSegmentation/models/twinlitenet_drivable.onnx",
}

object_config = {
    "model_path": './ObjectDetector/models/yolov10n-coco_fp16.trt',
    "model_type": ObjectModelType.YOLOV10,
    "classes_path": './ObjectDetector/models/coco_label.txt',
    "box_score": 0.4,
    "box_nms_iou": 0.5
}

# Initialize detectors
roadDetector = RoadSegmentationDetector(road_config["model_path"])
objectDetector = YoloDetector(logger=LOGGER)
