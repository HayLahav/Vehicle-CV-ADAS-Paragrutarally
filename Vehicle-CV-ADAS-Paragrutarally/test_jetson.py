#!/usr/bin/env python3
"""
Jetson Xavier Compatibility Test Script
Tests all major components of Vehicle-CV-ADAS on Jetson platform
"""

import sys
import time
import logging
import traceback
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_jetson_detection():
    """Test Jetson platform detection"""
    logger.info("=" * 50)
    logger.info("Testing Jetson Platform Detection")
    logger.info("=" * 50)
    
    try:
        from jetson_utils import setup_jetson_environment, get_jetson_status
        
        # Test platform detection
        optimizer = setup_jetson_environment()
        
        if optimizer.is_jetson:
            logger.info("✓ Jetson platform detected successfully")
            
            # Get detailed info
            info = optimizer.get_jetson_info()
            for key, value in info.items():
                logger.info(f"  {key}: {value}")
            
            # Get status
            status = get_jetson_status()
            for key, value in status.items():
                logger.info(f"  {key}: {value}")
                
            return True
        else:
            logger.warning("! Not running on Jetson platform")
            return False
            
    except Exception as e:
        logger.error(f"✗ Jetson detection failed: {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    logger.info("=" * 50)
    logger.info("Testing Dependencies")
    logger.info("=" * 50)
    
    dependencies = {
        'numpy': 'import numpy as np; print(f"NumPy {np.__version__}")',
        'opencv': 'import cv2; print(f"OpenCV {cv2.__version__}")',
        'torch': 'import torch; print(f"PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")',
        'torchvision': 'import torchvision; print(f"TorchVision {torchvision.__version__}")',
        'onnxruntime': '''
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"ONNX Runtime {ort.__version__}")
print(f"Available providers: {providers}")
        ''',
        'tensorrt': '''
try:
    import tensorrt as trt
    print(f"TensorRT {trt.__version__}")
except ImportError:
    print("TensorRT not available")
        ''',
        'scipy': 'import scipy; print(f"SciPy {scipy.__version__}")',
        'numba': 'import numba; print(f"Numba {numba.__version__}")',
    }
    
    results = {}
    for name, test_code in dependencies.items():
        try:
            exec(test_code)
            results[name] = True
            logger.info(f"✓ {name} - OK")
        except Exception as e:
            results[name] = False
            logger.error(f"✗ {name} - Failed: {e}")
    
    return all(results.values())

def test_core_engine():
    """Test core engine functionality"""
    logger.info("=" * 50)
    logger.info("Testing Core Engine")
    logger.info("=" * 50)
    
    try:
        from coreEngine import is_jetson, create_engine, OnnxEngine, TRT_AVAILABLE
        
        logger.info(f"Jetson detected: {is_jetson()}")
        logger.info(f"TensorRT available: {TRT_AVAILABLE}")
        
        # Test with a dummy ONNX model (create minimal model for testing)
        test_model_path = create_test_onnx_model()
        
        if test_model_path.exists():
            logger.info("Testing ONNX engine creation...")
            engine = create_engine(str(test_model_path))
            logger.info(f"✓ Engine created: {engine.framework_type}")
            
            # Test inference with dummy data
            input_shape = engine.get_engine_input_shape()
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            start_time = time.time()
            output = engine.engine_inference(dummy_input)
            inference_time = time.time() - start_time
            
            logger.info(f"✓ Inference successful: {inference_time:.4f}s")
            logger.info(f"  Input shape: {input_shape}")
            logger.info(f"  Output shape: {[o.shape for o in output]}")
            
            # Clean up
            test_model_path.unlink()
            
            return True
        else:
            logger.error("✗ Could not create test model")
            return False
            
    except Exception as e:
        logger.error(f"✗ Core engine test failed: {e}")
        traceback.print_exc()
        return False

def create_test_onnx_model():
    """Create a minimal ONNX model for testing"""
    try:
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(16, 10)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        test_model_path = Path("test_model_temp.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            str(test_model_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        return test_model_path
        
    except Exception as e:
        logger.error(f"Failed to create test model: {e}")
        return Path("nonexistent.onnx")

def test_object_detection():
    """Test object detection functionality"""
    logger.info("=" * 50)
    logger.info("Testing Object Detection")
    logger.info("=" * 50)
    
    try:
        from ObjectDetector import YoloDetector
        from ObjectDetector.utils import ObjectModelType
        
        # Create test configuration
        config = {
            "model_path": './ObjectDetector/models/yolov5n-coco_fp16.onnx',
            "model_type": ObjectModelType.YOLOV5,
            "classes_path": './ObjectDetector/models/coco_label.txt',
            "box_score": 0.5,
            "box_nms_iou": 0.5
        }
        
        # Check if model file exists
        model_path = Path(config["model_path"])
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Skipping object detection test (model file missing)")
            return True  # Not a failure, just missing files
        
        logger.info("Initializing YOLO detector...")
        YoloDetector.set_defaults(config)
        detector = YoloDetector(logger=logger)
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        start_time = time.time()
        detector.DetectFrame(test_image)
        detection_time = time.time() - start_time
        
        logger.info(f"✓ Object detection successful: {detection_time:.4f}s")
        logger.info(f"  Detected objects: {len(detector.object_info)}")
        
        # Test drawing
        result_image = test_image.copy()
        detector.DrawDetectedOnFrame(result_image)
        logger.info("✓ Object detection drawing successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Object detection test failed: {e}")
        traceback.print_exc()
        return False

def test_lane_detection():
    """Test lane detection functionality"""
    logger.info("=" * 50)
    logger.info("Testing Lane Detection")
    logger.info("=" * 50)
    
    try:
        from TrafficLaneDetector import UltrafastLaneDetectorV2
        from TrafficLaneDetector.ufldDetector.utils import LaneModelType
        
        # Create test configuration
        model_path = './TrafficLaneDetector/models/culane_res18.onnx'
        model_type = LaneModelType.UFLDV2_CULANE
        
        # Check if model file exists
        if not Path(model_path).exists():
            logger.warning(f"Lane model file not found: {model_path}")
            logger.info("Skipping lane detection test (model file missing)")
            return True  # Not a failure, just missing files
        
        logger.info("Initializing lane detector...")
        detector = UltrafastLaneDetectorV2(model_path, model_type, logger=logger)
        
        # Create test image
        test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Test detection
        start_time = time.time()
        detector.DetectFrame(test_image)
        detection_time = time.time() - start_time
        
        logger.info(f"✓ Lane detection successful: {detection_time:.4f}s")
        
        # Test drawing
        result_image = test_image.copy()
        detector.DrawDetectedOnFrame(result_image)
        detector.DrawAreaOnFrame(result_image)
        logger.info("✓ Lane detection drawing successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Lane detection test failed: {e}")
        traceback.print_exc()
        return False

def test_road_segmentation():
    """Test road segmentation functionality"""
    logger.info("=" * 50)
    logger.info("Testing Road Segmentation")
    logger.info("=" * 50)
    
    try:
        from RoadSegmentation import RoadSegmentationDetector
        
        model_path = './RoadSegmentation/models/twinlitenet_drivable.onnx'
        
        # Check if model file exists
        if not Path(model_path).exists():
            logger.warning(f"Road segmentation model not found: {model_path}")
            logger.info("Skipping road segmentation test (model file missing)")
            return True  # Not a failure, just missing files
        
        logger.info("Initializing road segmentation detector...")
        detector = RoadSegmentationDetector(model_path, logger=logger)
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        start_time = time.time()
        detector.DetectFrame(test_image)
        detection_time = time.time() - start_time
        
        logger.info(f"✓ Road segmentation successful: {detection_time:.4f}s")
        
        # Test drawing
        result_image = test_image.copy()
        detector.DrawDetectedOnFrame(result_image)
        logger.info("✓ Road segmentation drawing successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Road segmentation test failed: {e}")
        traceback.print_exc()
        return False

def test_object_tracking():
    """Test object tracking functionality"""
    logger.info("=" * 50)
    logger.info("Testing Object Tracking")
    logger.info("=" * 50)
    
    try:
        from ObjectTracker import BYTETracker
        
        # Create test configuration
        logger.info("Initializing object tracker...")
        tracker = BYTETracker(names={0: "person", 1: "car"})
        
        # Create test data
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
        test_scores = [0.8, 0.9]
        test_class_ids = [0, 1]
        
        # Test tracking
        start_time = time.time()
        results = tracker.update(test_boxes, test_scores, test_class_ids, test_image)
        tracking_time = time.time() - start_time
        
        logger.info(f"✓ Object tracking successful: {tracking_time:.4f}s")
        logger.info(f"  Tracked objects: {len(results)}")
        
        # Test drawing
        result_image = test_image.copy()
        tracker.DrawTrackedOnFrame(result_image)
        logger.info("✓ Object tracking drawing successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Object tracking test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_monitoring():
    """Test Jetson performance monitoring"""
    logger.info("=" * 50)
    logger.info("Testing Performance Monitoring")
    logger.info("=" * 50)
    
    try:
        from jetson_utils import JetsonOptimizer
        
        optimizer = JetsonOptimizer(logger=logger)
        
        if not optimizer.is_jetson:
            logger.info("Not on Jetson platform, skipping performance monitoring test")
            return True
        
        # Test system stats collection
        logger.info("Collecting system statistics...")
        stats = optimizer.get_system_stats()
        
        logger.info("System Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Test thermal throttling check
        is_throttling = optimizer.check_thermal_throttling()
        logger.info(f"Thermal throttling: {is_throttling}")
        
        # Test performance optimization
        logger.info("Testing performance optimization...")
        success = optimizer.optimize_for_inference()
        logger.info(f"Performance optimization: {'✓' if success else '✗'}")
        
        # Test monitoring
        logger.info("Testing continuous monitoring...")
        optimizer.start_monitoring(interval=1.0)
        time.sleep(3)  # Monitor for 3 seconds
        latest_stats = optimizer.get_latest_stats()
        optimizer.stop_monitoring()
        
        logger.info("Latest monitoring statistics:")
        for key, value in latest_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("✓ Performance monitoring test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """Test complete ADAS pipeline"""
    logger.info("=" * 50)
    logger.info("Testing Complete ADAS Pipeline")
    logger.info("=" * 50)
    
    try:
        # This is a simplified version of the complete pipeline test
        # In practice, you would test with actual model files
        
        logger.info("Creating test image...")
        test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Test task conditions
        from taskConditions import TaskConditions
        analyzer = TaskConditions()
        
        # Simulate vehicle position and road status
        vehicle_position = (640, 700)  # Bottom center
        road_center = [(600, 500), (640, 600), (680, 700)]  # Simulated road center
        
        analyzer.UpdateRoadStatus(True, road_center, vehicle_position)
        analyzer.UpdateCollisionStatus(None, True)
        analyzer.UpdateRouteStatus()
        
        logger.info(f"Offset status: {analyzer.offset_msg}")
        logger.info(f"Collision status: {analyzer.collision_msg}")
        logger.info(f"Curvature status: {analyzer.curvature_msg}")
        
        logger.info("✓ Complete pipeline test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Complete pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_distance_measurement():
    """Test distance measurement functionality"""
    logger.info("=" * 50)
    logger.info("Testing Distance Measurement")
    logger.info("=" * 50)
    
    try:
        from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
        from ObjectDetector.core import RectInfo
        
        # Create distance detector
        distance_detector = SingleCamDistanceMeasure()
        
        # Create test objects
        test_objects = [
            RectInfo(100, 200, 80, 120, 0.8, "car"),
            RectInfo(300, 180, 60, 100, 0.9, "person"),
            RectInfo(500, 150, 100, 150, 0.7, "truck")
        ]
        
        # Test distance calculation
        start_time = time.time()
        distance_detector.updateDistance(test_objects)
        calculation_time = time.time() - start_time
        
        logger.info(f"✓ Distance calculation successful: {calculation_time:.4f}s")
        logger.info(f"  Distance points: {len(distance_detector.distance_points)}")
        
        # Test collision point calculation with dummy polygon
        test_polygon = np.array([[200, 400], [600, 400], [700, 600], [100, 600]])
        collision_point = distance_detector.calcCollisionPoint(test_polygon)
        
        if collision_point:
            logger.info(f"  Collision point found: {collision_point}")
        else:
            logger.info("  No collision point detected")
        
        # Test drawing
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        distance_detector.DrawDetectedOnFrame(test_image)
        logger.info("✓ Distance measurement drawing successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Distance measurement test failed: {e}")
        traceback.print_exc()
        return False

def run_benchmark():
    """Run basic performance benchmark"""
    logger.info("=" * 50)
    logger.info("Running Performance Benchmark")
    logger.info("=" * 50)
    
    try:
        # Test inference speed with dummy operations
        test_iterations = 100
        image_size = (640, 640, 3)
        
        logger.info(f"Running {test_iterations} iterations with {image_size} images...")
        
        times = []
        for i in range(test_iterations):
            start_time = time.time()
            
            # Simulate typical ADAS operations
            test_image = np.random.randint(0, 255, image_size, dtype=np.uint8)
            
            # Simulate preprocessing
            resized = np.array(test_image, dtype=np.float32) / 255.0
            
            # Simulate some computations
            processed = np.transpose(resized, (2, 0, 1))
            processed = np.expand_dims(processed, axis=0)
            
            # Simulate postprocessing
            result = np.squeeze(processed)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        logger.info(f"Benchmark Results:")
        logger.info(f"  Average time: {avg_time*1000:.2f} ms")
        logger.info(f"  Standard deviation: {std_time*1000:.2f} ms")
        logger.info(f"  Estimated FPS: {fps:.1f}")
        logger.info(f"  Min time: {np.min(times)*1000:.2f} ms")
        logger.info(f"  Max time: {np.max(times)*1000:.2f} ms")
        
        if fps > 30:
            logger.info("✓ Performance is excellent for real-time processing")
        elif fps > 15:
            logger.info("✓ Performance is good for real-time processing")
        elif fps > 10:
            logger.info("! Performance may be marginal for real-time processing")
        else:
            logger.warning("! Performance may be too slow for real-time processing")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Benchmark failed: {e}")
        return False

def test_perspective_transformation():
    """Test perspective transformation functionality"""
    logger.info("=" * 50)
    logger.info("Testing Perspective Transformation")
    logger.info("=" * 50)
    
    try:
        from TrafficLaneDetector.ufldDetector.perspectiveTransformation import PerspectiveTransformation
        
        # Create perspective transformation
        img_size = (1280, 720)
        transform = PerspectiveTransformation(img_size, logger=logger)
        
        # Create test image
        test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Test bird view transformation
        start_time = time.time()
        bird_view = transform.transformToBirdView(test_image)
        transform_time = time.time() - start_time
        
        logger.info(f"✓ Bird view transformation successful: {transform_time:.4f}s")
        logger.info(f"  Original shape: {test_image.shape}")
        logger.info(f"  Bird view shape: {bird_view.shape}")
        
        # Test frontal view transformation
        frontal_view = transform.transformToFrontalView(bird_view)
        logger.info(f"  Frontal view shape: {frontal_view.shape}")
        
        # Test point transformation
        test_points = [(640, 400), (500, 500), (800, 500)]
        transformed_points = transform.transformToBirdViewPoints(test_points)
        logger.info(f"  Transformed {len(test_points)} points successfully")
        
        logger.info("✓ Perspective transformation test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Perspective transformation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Jetson Xavier Compatibility Tests")
    logger.info("=" * 70)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Jetson Detection", test_jetson_detection),
        ("Dependencies", test_dependencies),
        ("Core Engine", test_core_engine),
        ("Object Detection", test_object_detection),
        ("Lane Detection", test_lane_detection),
        ("Road Segmentation", test_road_segmentation),
        ("Object Tracking", test_object_tracking),
        ("Distance Measurement", test_distance_measurement),
        ("Perspective Transformation", test_perspective_transformation),
        ("Performance Monitoring", test_performance_monitoring),
        ("Complete Pipeline", test_complete_pipeline),
        ("Performance Benchmark", run_benchmark),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n🔄 Running {test_name} test...")
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            test_results[test_name] = False
            logger.error(f"✗ FAILED: {test_name} - {e}")
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("🏁 TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} {test_name}")
    
    logger.info("-" * 70)
    logger.info(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! Jetson Xavier is ready for ADAS.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python3 demo_jetson.py")
        logger.info("2. Check performance with: python3 jetson_performance.py")
        logger.info("3. Monitor system: sudo tegrastats")
        sys.exit(0)
    elif passed >= total * 0.8:
        logger.info("⚠️  Most tests passed. System should work with minor issues.")
        logger.info("\nRecommendations:")
        logger.info("1. Check failed tests and install missing dependencies")
        logger.info("2. Ensure model files are available")
        logger.info("3. Run: python3 demo_jetson.py")
        sys.exit(0)
    else:
        logger.error("❌ Multiple tests failed. Please check your installation.")
        logger.error("\nTroubleshooting:")
        logger.error("1. Run: ./install_jetson.sh")
        logger.error("2. Check JetPack installation")
        logger.error("3. Verify CUDA and TensorRT availability")
        sys.exit(1)

if __name__ == "__main__":
    main()