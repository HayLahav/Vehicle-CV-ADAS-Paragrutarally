# Moondream2 VLM Integration for ADAS

This document describes the Moondream2 Vision-Language Model (VLM) integration for the Vehicle-CV-ADAS-Paragrutarally project on Jetson Orin Nano.

## Overview

Moondream2 is a tiny 1.8B parameter vision-language model that enhances the ADAS system with natural language scene understanding capabilities. It can:

- Generate scene descriptions
- Answer questions about driving scenes
- Detect specific objects and hazards
- Provide weather and road condition analysis

## Features

### Core VLM Capabilities

1. **Image Captioning**: Generate short, normal, or detailed descriptions of driving scenes
2. **Visual Question Answering (VQA)**: Ask natural language questions about the scene
3. **Object Detection**: Detect specific objects using natural language queries
4. **Scene Analysis**: Comprehensive driving scene understanding including:
   - Weather conditions
   - Time of day
   - Road conditions
   - Hazard detection

### Integration with ADAS

The VLM system integrates seamlessly with existing ADAS components:

- **Object Detection**: Enhanced with natural language understanding
- **Lane Detection**: Complemented with scene-level context
- **Road Segmentation**: Augmented with weather and visibility analysis
- **Collision Warning**: Enriched with hazard detection

## Installation

### Prerequisites

Ensure your Jetson Orin Nano has:
- JetPack 5.0+ or 6.0+
- At least 8GB RAM (16GB recommended)
- 10GB free storage for model weights
- CUDA 11.4+ (included in JetPack)

### Install VLM Dependencies

```bash
cd Vehicle-CV-ADAS-Paragrutarally

# Install VLM-specific dependencies
pip install -r requirements_vlm.txt

# For Jetson, you may need to install PyTorch from NVIDIA's repository
# Follow: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

### Verify Installation

Run the test script to verify VLM integration:

```bash
cd Vehicle-CV-ADAS-Paragrutarally
python3 test_vlm_jetson.py
```

Expected output:
```
MOONDREAM2 VLM INTEGRATION TEST SUITE
Platform: Jetson Orin Nano detected
✓ VLM module imports successful
✓ VLM detector initialized successfully
✓ All tests passed!
```

## Usage

### Basic VLM Usage

```python
from VLMDetector import MoondreamVLMDetector
import cv2

# Initialize VLM
vlm = MoondreamVLMDetector()

# Load image
image = cv2.imread('road_scene.jpg')

# Generate caption
result = vlm.caption_image(image, length="normal")
print(f"Scene: {result['caption']}")

# Ask questions
answer = vlm.query_image(image, "Are there any pedestrians?")
print(f"Answer: {answer['answer']}")

# Analyze driving scene
analysis = vlm.analyze_driving_scene(image)
print(f"Weather: {analysis['weather']}")
print(f"Time: {analysis['time_of_day']}")
```

### Running ADAS with VLM

#### Basic Usage

```bash
# Run ADAS with VLM enabled (default: analyze every 60 frames)
python3 demo_jetson_vlm.py --video ./path/to/video.mp4 --enable-vlm

# Run without VLM
python3 demo_jetson_vlm.py --video ./path/to/video.mp4
```

#### Advanced Options

```bash
# Analyze every 30 frames (more frequent but slower)
python3 demo_jetson_vlm.py --video input.mp4 --enable-vlm --vlm-interval 30

# Specify output path
python3 demo_jetson_vlm.py --video input.mp4 --enable-vlm --output output_vlm.mp4

# Full example
python3 demo_jetson_vlm.py \
    --video ./TrafficLaneDetector/temp/demo-7.mp4 \
    --enable-vlm \
    --vlm-interval 60 \
    --output demo_with_vlm.mp4
```

### Integration in Custom Code

```python
from VLMDetector import MoondreamVLMDetector, VLMSceneAnalyzer

# Initialize VLM
vlm = MoondreamVLMDetector(logger=your_logger)

# Create scene analyzer
analyzer = VLMSceneAnalyzer(vlm, logger=your_logger)
analyzer.set_analysis_interval(60)  # Analyze every 60 frames

# In your processing loop
for frame in video_frames:
    # Your existing ADAS processing...

    # VLM analysis (runs only at intervals)
    vlm_result = analyzer.analyze_frame(frame)

    if vlm_result:
        print(f"Scene: {vlm_result['scene_description']}")

    # Draw VLM info on frame
    analyzer.draw_vlm_info(frame)
```

## Performance

### Jetson Orin Nano Performance

| Configuration | FPS (ADAS only) | FPS (ADAS + VLM) | VLM Inference Time |
|--------------|-----------------|------------------|--------------------|
| YOLOv5n + UFLD + VLM (60 frame interval) | 25-30 | 24-28 | ~2-3s |
| YOLOv5s + UFLD + VLM (60 frame interval) | 18-22 | 17-21 | ~2-3s |

**Notes:**
- VLM runs periodically (e.g., every 60 frames), so average FPS impact is minimal
- First inference is slower due to model loading (~5-10s)
- FP16 precision is automatically enabled on Jetson for optimal performance
- Memory usage: ~3-4GB additional for VLM model

### Optimization Tips

1. **Adjust Analysis Interval**: Increase `--vlm-interval` to reduce VLM overhead
   ```bash
   --vlm-interval 120  # Analyze every 120 frames (4 seconds at 30 FPS)
   ```

2. **Monitor Temperature**: VLM increases GPU load; monitor thermals
   ```bash
   watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'
   ```

3. **Performance Mode**: Ensure Jetson is in max performance mode
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

## API Reference

### MoondreamVLMDetector

Main VLM detector class for Moondream2 integration.

#### Methods

- `caption_image(image, length='normal')`: Generate image caption
  - `length`: 'short', 'normal', or 'long'
  - Returns: `{'caption': str, 'inference_time': float}`

- `query_image(image, question)`: Visual question answering
  - `question`: Natural language question
  - Returns: `{'answer': str, 'inference_time': float, 'question': str}`

- `detect_objects(image, object_type)`: Detect specific objects
  - `object_type`: Object to detect (e.g., 'car', 'person')
  - Returns: `{'objects': list, 'count': int, 'inference_time': float}`

- `analyze_driving_scene(image)`: Comprehensive scene analysis
  - Returns: Dict with scene description, weather, time of day, road condition, hazards

### VLMSceneAnalyzer

Higher-level scene analyzer for ADAS integration.

#### Methods

- `analyze_frame(frame, force=False)`: Analyze frame with VLM
  - Respects analysis interval unless `force=True`
  - Returns: Analysis results or None if skipped

- `check_specific_hazard(frame, hazard_type)`: Check for specific hazard
  - `hazard_type`: Type of hazard to detect
  - Returns: Hazard detection result

- `set_analysis_interval(interval)`: Set analysis interval
  - `interval`: Number of frames between analyses

- `draw_vlm_info(frame, x=10, y=500)`: Draw VLM info on frame

## Architecture

```
VLMDetector/
├── __init__.py          # Module exports
└── core.py             # MoondreamVLMDetector and VLMSceneAnalyzer

Integration Flow:
Frame Input
    ↓
Object Detection (YOLO)
    ↓
Lane Detection (UFLD)
    ↓
Road Segmentation
    ↓
VLM Analysis (periodic) ←→ Moondream2 Model
    ↓
Combined ADAS Output
```

## Troubleshooting

### VLM Not Loading

**Problem**: `VLM not available` message

**Solutions**:
```bash
# Check dependencies
pip list | grep -E "transformers|torch|timm|einops"

# Reinstall VLM dependencies
pip install --upgrade -r requirements_vlm.txt

# Test VLM separately
python3 test_vlm_jetson.py
```

### Out of Memory

**Problem**: CUDA out of memory error

**Solutions**:
1. Increase analysis interval: `--vlm-interval 120`
2. Use smaller YOLO model: YOLOv5n instead of YOLOv5s
3. Reduce video resolution
4. Add swap space:
   ```bash
   sudo fallocate -l 8G /var/swapfile
   sudo chmod 600 /var/swapfile
   sudo mkswap /var/swapfile
   sudo swapon /var/swapfile
   ```

### Slow Performance

**Problem**: Low FPS with VLM enabled

**Solutions**:
1. Increase VLM interval: `--vlm-interval 90`
2. Ensure performance mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
3. Monitor temperature throttling
4. Close background applications

### Model Download Fails

**Problem**: HuggingFace model download timeout

**Solutions**:
```bash
# Pre-download model manually
python3 -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'vikhyatk/moondream2',
    revision='2024-08-26',
    trust_remote_code=True
)
"
```

## Examples

### Example 1: Scene Description

```python
from VLMDetector import MoondreamVLMDetector
import cv2

vlm = MoondreamVLMDetector()
image = cv2.imread('highway.jpg')

result = vlm.caption_image(image)
print(result['caption'])
# Output: "A highway with multiple lanes, cars driving in both directions,
#          clear sky, daytime conditions"
```

### Example 2: Hazard Detection

```python
from VLMDetector import VLMSceneAnalyzer, MoondreamVLMDetector

vlm = MoondreamVLMDetector()
analyzer = VLMSceneAnalyzer(vlm)

image = cv2.imread('scene.jpg')

# Check for pedestrian crossing
hazard = analyzer.check_specific_hazard(image, "pedestrian crossing the road")

if hazard['detected']:
    print(f"WARNING: {hazard['description']}")
```

### Example 3: Weather Analysis

```python
vlm = MoondreamVLMDetector()
image = cv2.imread('rainy_road.jpg')

weather = vlm.query_image(image, "What are the weather conditions?")
print(f"Weather: {weather['answer']}")
# Output: "Weather: Rainy conditions with wet roads visible"
```

## Model Information

- **Model Name**: Moondream2
- **Parameters**: 1.8 billion
- **HuggingFace**: vikhyatk/moondream2
- **Revision**: 2024-08-26 (default)
- **License**: Apache 2.0
- **Size**: ~3.7GB (FP32), ~1.9GB (FP16)

## Citation

If you use Moondream2 in your research, please cite:

```bibtex
@misc{moondream2,
  author = {Vikhyat Korrapati},
  title = {Moondream},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/vikhyat/moondream}
}
```

## License

The VLM integration follows the same GPLv3 license as the main ADAS project. Moondream2 itself is licensed under Apache 2.0.

## Support

For issues specific to VLM integration:
1. Run the test script: `python3 test_vlm_jetson.py`
2. Check logs for error messages
3. Open an issue on GitHub with test results

For general ADAS issues, refer to the main README.md.
