#!/usr/bin/env python3
"""
Test script for Moondream2 VLM integration on Jetson Orin Nano
Verifies VLM functionality and integration with ADAS components
"""

import cv2
import numpy as np
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_jetson():
    """Check if running on Jetson platform"""
    try:
        with open('/etc/nv_tegra_release') as f:
            return True
    except:
        return False

def test_vlm_imports():
    """Test VLM module imports"""
    logger.info("="*60)
    logger.info("TEST 1: Testing VLM module imports")
    logger.info("="*60)

    try:
        from VLMDetector import MoondreamVLMDetector, VLMSceneAnalyzer
        logger.info("✓ VLM module imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ VLM module import failed: {e}")
        return False

def test_vlm_initialization():
    """Test VLM detector initialization"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Testing VLM detector initialization")
    logger.info("="*60)

    try:
        from VLMDetector import MoondreamVLMDetector

        logger.info("Initializing Moondream2 VLM detector...")
        vlm = MoondreamVLMDetector(logger=logger)

        if vlm.is_available():
            logger.info("✓ VLM detector initialized successfully")
            logger.info(f"  Model ID: {vlm.model_id}")
            logger.info(f"  Revision: {vlm.revision}")
            logger.info(f"  Device: {vlm.device}")
            return vlm
        else:
            logger.warning("✗ VLM detector initialized but not available")
            logger.warning("  This may be due to missing dependencies")
            return None
    except Exception as e:
        logger.error(f"✗ VLM initialization failed: {e}")
        return None

def test_vlm_image_caption(vlm):
    """Test VLM image captioning"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Testing VLM image captioning")
    logger.info("="*60)

    if vlm is None or not vlm.is_available():
        logger.warning("⊘ Skipping - VLM not available")
        return False

    try:
        # Create a test image (synthetic road scene)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw a simple road scene
        cv2.rectangle(test_image, (100, 300), (540, 480), (50, 50, 50), -1)  # Road
        cv2.rectangle(test_image, (200, 200), (280, 300), (0, 0, 200), -1)  # Car
        cv2.circle(test_image, (100, 100), 30, (0, 200, 200), -1)  # Sun

        logger.info("Testing image captioning...")
        result = vlm.caption_image(test_image, length="short")

        logger.info(f"✓ Caption generated successfully")
        logger.info(f"  Caption: {result['caption']}")
        logger.info(f"  Inference time: {result['inference_time']:.3f}s")

        return True
    except Exception as e:
        logger.error(f"✗ Image captioning failed: {e}")
        return False

def test_vlm_visual_qa(vlm):
    """Test VLM visual question answering"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Testing VLM visual question answering")
    logger.info("="*60)

    if vlm is None or not vlm.is_available():
        logger.warning("⊘ Skipping - VLM not available")
        return False

    try:
        # Create a test image with text
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "STOP", (250, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        logger.info("Testing visual question answering...")
        question = "What text is visible in the image?"
        result = vlm.query_image(test_image, question)

        logger.info(f"✓ Question answered successfully")
        logger.info(f"  Question: {result['question']}")
        logger.info(f"  Answer: {result['answer']}")
        logger.info(f"  Inference time: {result['inference_time']:.3f}s")

        return True
    except Exception as e:
        logger.error(f"✗ Visual QA failed: {e}")
        return False

def test_vlm_scene_analyzer(vlm):
    """Test VLM scene analyzer"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Testing VLM scene analyzer")
    logger.info("="*60)

    if vlm is None or not vlm.is_available():
        logger.warning("⊘ Skipping - VLM not available")
        return False

    try:
        from VLMDetector import VLMSceneAnalyzer

        logger.info("Initializing scene analyzer...")
        analyzer = VLMSceneAnalyzer(vlm, logger=logger)
        analyzer.set_analysis_interval(1)  # Analyze every frame for testing

        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (0, 300), (640, 480), (50, 50, 50), -1)
        cv2.rectangle(test_frame, (250, 200), (330, 300), (0, 0, 200), -1)

        logger.info("Testing frame analysis...")
        result = analyzer.analyze_frame(test_frame, force=True)

        if result:
            logger.info(f"✓ Scene analysis successful")
            logger.info(f"  Frame number: {result['frame_number']}")
            logger.info(f"  Description: {result['scene_description']}")
            logger.info(f"  Inference time: {result['inference_time']:.3f}s")
            return True
        else:
            logger.warning("✗ Scene analysis returned None")
            return False
    except Exception as e:
        logger.error(f"✗ Scene analyzer failed: {e}")
        return False

def test_vlm_with_real_image(vlm):
    """Test VLM with a real image if available"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Testing VLM with real driving scene (if available)")
    logger.info("="*60)

    if vlm is None or not vlm.is_available():
        logger.warning("⊘ Skipping - VLM not available")
        return False

    # Try to find a real image
    test_image_paths = [
        "./TrafficLaneDetector/temp/demo-7.mp4",  # Video file
        "./demo/demo.JPG",
        "./assets/test_frame.jpg"
    ]

    real_image = None
    for path in test_image_paths:
        if Path(path).exists():
            if path.endswith('.mp4'):
                # Extract first frame from video
                cap = cv2.VideoCapture(path)
                ret, real_image = cap.read()
                cap.release()
                if ret:
                    logger.info(f"Using frame from video: {path}")
                    break
            else:
                real_image = cv2.imread(path)
                if real_image is not None:
                    logger.info(f"Using image: {path}")
                    break

    if real_image is None:
        logger.warning("⊘ No real image found - skipping test")
        return False

    try:
        logger.info("Analyzing real driving scene...")

        # Test comprehensive scene analysis
        analysis = vlm.analyze_driving_scene(real_image)

        logger.info(f"✓ Driving scene analysis successful")
        logger.info(f"  Scene description: {analysis['scene_description']}")
        logger.info(f"  Weather: {analysis['weather']}")
        logger.info(f"  Time of day: {analysis['time_of_day']}")
        logger.info(f"  Total inference time: {analysis['total_inference_time']:.2f}s")

        return True
    except Exception as e:
        logger.error(f"✗ Real image analysis failed: {e}")
        return False

def test_performance_benchmark(vlm):
    """Benchmark VLM performance"""
    logger.info("\n" + "="*60)
    logger.info("TEST 7: VLM performance benchmark")
    logger.info("="*60)

    if vlm is None or not vlm.is_available():
        logger.warning("⊘ Skipping - VLM not available")
        return False

    try:
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Reset stats
        vlm.reset_stats()

        # Run multiple inferences
        num_runs = 5
        logger.info(f"Running {num_runs} captioning inferences...")

        for i in range(num_runs):
            result = vlm.caption_image(test_image, length="short")
            logger.info(f"  Run {i+1}: {result['inference_time']:.3f}s")

        avg_time = vlm.get_avg_inference_time()
        logger.info(f"✓ Performance benchmark completed")
        logger.info(f"  Average inference time: {avg_time:.3f}s")
        logger.info(f"  Estimated FPS: {1/avg_time:.2f}")

        # Jetson-specific info
        if is_jetson():
            logger.info(f"  Platform: Jetson Orin Nano")
            logger.info(f"  Note: VLM runs at ~{args_vlm_interval} frame intervals in production")

        return True
    except Exception as e:
        logger.error(f"✗ Performance benchmark failed: {e}")
        return False

def check_dependencies():
    """Check required dependencies"""
    logger.info("\n" + "="*60)
    logger.info("Checking dependencies")
    logger.info("="*60)

    dependencies = {
        'transformers': False,
        'torch': False,
        'PIL': False,
        'timm': False,
        'einops': False
    }

    for dep in dependencies:
        try:
            if dep == 'PIL':
                __import__('PIL')
            else:
                __import__(dep)
            dependencies[dep] = True
            logger.info(f"✓ {dep} is installed")
        except ImportError:
            logger.warning(f"✗ {dep} is NOT installed")

    all_installed = all(dependencies.values())

    if not all_installed:
        logger.warning("\nMissing dependencies detected!")
        logger.warning("Install with: pip install transformers torch pillow timm einops")

    return all_installed

def main():
    """Run all VLM tests"""
    logger.info("\n" + "="*60)
    logger.info("MOONDREAM2 VLM INTEGRATION TEST SUITE")
    logger.info("="*60)

    # Platform info
    if is_jetson():
        logger.info("Platform: Jetson Orin Nano detected")
        try:
            with open('/etc/nv_tegra_release') as f:
                logger.info(f"Tegra Info: {f.read().strip()}")
        except:
            pass
    else:
        logger.info("Platform: Standard system (non-Jetson)")

    # Check dependencies first
    deps_ok = check_dependencies()
    if not deps_ok:
        logger.error("\n" + "="*60)
        logger.error("CRITICAL: Missing dependencies - VLM tests will fail")
        logger.error("Please install required packages:")
        logger.error("  pip install transformers torch pillow timm einops")
        logger.error("="*60)

    # Run tests
    results = {}

    # Test 1: Imports
    results['imports'] = test_vlm_imports()

    if not results['imports']:
        logger.error("\n" + "="*60)
        logger.error("CRITICAL: Module import failed - stopping tests")
        logger.error("="*60)
        sys.exit(1)

    # Test 2: Initialization
    vlm = test_vlm_initialization()
    results['initialization'] = vlm is not None

    # Only run remaining tests if VLM is available
    if vlm and vlm.is_available():
        results['caption'] = test_vlm_image_caption(vlm)
        results['visual_qa'] = test_vlm_visual_qa(vlm)
        results['scene_analyzer'] = test_vlm_scene_analyzer(vlm)
        results['real_image'] = test_vlm_with_real_image(vlm)
        results['performance'] = test_performance_benchmark(vlm)
    else:
        logger.warning("\nVLM not available - skipping functional tests")
        results['caption'] = False
        results['visual_qa'] = False
        results['scene_analyzer'] = False
        results['real_image'] = False
        results['performance'] = False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name.upper()}: {status}")

    logger.info("-"*60)
    logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
    logger.info("="*60)

    if passed_tests == total_tests:
        logger.info("\n🎉 All tests passed! VLM integration is working correctly.")
        return 0
    elif results['imports'] and results['initialization']:
        logger.warning("\n⚠️  Some tests failed, but basic VLM functionality is available.")
        logger.warning("The VLM may work with limitations.")
        return 1
    else:
        logger.error("\n❌ Critical tests failed. VLM integration is not working.")
        logger.error("Please check dependencies and error messages above.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
