#!/usr/bin/env python3
"""
Moondream2 VLM Detector for ADAS
Provides vision-language understanding for enhanced scene analysis
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple
import time

# Check if we're on Jetson
def is_jetson():
    """Check if running on Jetson platform"""
    try:
        with open('/etc/nv_tegra_release') as f:
            return True
    except:
        return False


class MoondreamVLMDetector:
    """
    Moondream2 Vision-Language Model Detector for ADAS applications
    Provides scene understanding, visual question answering, and detailed descriptions
    """

    def __init__(self, model_id: str = "vikhyatk/moondream2",
                 revision: str = "2024-08-26",
                 device: str = "auto",
                 logger=None):
        """
        Initialize Moondream2 VLM Detector

        Args:
            model_id: HuggingFace model identifier
            revision: Model revision/version
            device: Device to use ('cuda', 'cpu', or 'auto')
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_id = model_id
        self.revision = revision
        self.model = None
        self.tokenizer = None
        self.device = device
        self.is_initialized = False
        self.last_scene_description = ""
        self.last_qa_result = ""

        # Performance tracking
        self.inference_times = []

        # Try to initialize the model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Moondream2 model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.logger.info(f"Loading Moondream2 VLM model: {self.model_id}")
            self.logger.info(f"Revision: {self.revision}")

            # Determine device
            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    device_map = {"": "cuda"}
                    self.logger.info("Using CUDA device for VLM")
                else:
                    device_map = {"": "cpu"}
                    self.logger.info("Using CPU device for VLM")
            else:
                device_map = {"": self.device}

            # Load model with optimizations for Jetson
            if is_jetson():
                self.logger.info("Detected Jetson platform - applying optimizations")
                # Use FP16 for faster inference on Jetson
                import torch
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    trust_remote_code=True,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    trust_remote_code=True,
                    device_map=device_map
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision
            )

            self.is_initialized = True
            self.logger.info("Moondream2 VLM initialized successfully")

        except ImportError as e:
            self.logger.error(f"Missing dependencies for Moondream2: {e}")
            self.logger.error("Please install: pip install transformers timm einops torch pillow")
            self.is_initialized = False
        except Exception as e:
            self.logger.error(f"Failed to initialize Moondream2 VLM: {e}")
            self.is_initialized = False

    def is_available(self) -> bool:
        """Check if VLM is available and initialized"""
        return self.is_initialized and self.model is not None

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL Image"""
        # OpenCV uses BGR, PIL uses RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    def caption_image(self, image: np.ndarray, length: str = "normal") -> Dict[str, str]:
        """
        Generate a caption for the image

        Args:
            image: OpenCV image (BGR format)
            length: Caption length ('short', 'normal', 'long')

        Returns:
            Dict with 'caption' and 'inference_time'
        """
        if not self.is_available():
            return {"caption": "VLM not available", "inference_time": 0.0}

        try:
            start_time = time.time()
            pil_image = self._cv2_to_pil(image)

            # Generate caption
            result = self.model.caption(pil_image, length=length)

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            caption = result.get("caption", "")
            self.last_scene_description = caption

            self.logger.debug(f"Caption generated in {inference_time:.3f}s: {caption}")

            return {
                "caption": caption,
                "inference_time": inference_time
            }

        except Exception as e:
            self.logger.error(f"Caption generation failed: {e}")
            return {"caption": "", "inference_time": 0.0}

    def query_image(self, image: np.ndarray, question: str) -> Dict[str, str]:
        """
        Ask a question about the image (Visual Question Answering)

        Args:
            image: OpenCV image (BGR format)
            question: Question to ask about the image

        Returns:
            Dict with 'answer' and 'inference_time'
        """
        if not self.is_available():
            return {"answer": "VLM not available", "inference_time": 0.0}

        try:
            start_time = time.time()
            pil_image = self._cv2_to_pil(image)

            # Query the image
            result = self.model.query(pil_image, question)

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            answer = result.get("answer", "")
            self.last_qa_result = f"Q: {question}\nA: {answer}"

            self.logger.debug(f"Query answered in {inference_time:.3f}s: {answer}")

            return {
                "answer": answer,
                "inference_time": inference_time,
                "question": question
            }

        except Exception as e:
            self.logger.error(f"Image query failed: {e}")
            return {"answer": "", "inference_time": 0.0, "question": question}

    def detect_objects(self, image: np.ndarray, object_type: str) -> Dict:
        """
        Detect specific objects in the image

        Args:
            image: OpenCV image (BGR format)
            object_type: Type of object to detect (e.g., 'car', 'person', 'traffic sign')

        Returns:
            Dict with 'objects', 'count', and 'inference_time'
        """
        if not self.is_available():
            return {"objects": [], "count": 0, "inference_time": 0.0}

        try:
            start_time = time.time()
            pil_image = self._cv2_to_pil(image)

            # Detect objects
            result = self.model.detect(pil_image, object_type)

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            objects = result.get("objects", [])

            self.logger.debug(f"Detected {len(objects)} {object_type}(s) in {inference_time:.3f}s")

            return {
                "objects": objects,
                "count": len(objects),
                "inference_time": inference_time,
                "object_type": object_type
            }

        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return {"objects": [], "count": 0, "inference_time": 0.0}

    def analyze_driving_scene(self, image: np.ndarray) -> Dict:
        """
        Perform comprehensive driving scene analysis

        Args:
            image: OpenCV image (BGR format)

        Returns:
            Dict with scene analysis results
        """
        if not self.is_available():
            return {
                "available": False,
                "scene_description": "VLM not available",
                "weather": "unknown",
                "time_of_day": "unknown",
                "road_condition": "unknown",
                "hazards": []
            }

        analysis = {
            "available": True,
            "scene_description": "",
            "weather": "unknown",
            "time_of_day": "unknown",
            "road_condition": "unknown",
            "hazards": [],
            "total_inference_time": 0.0
        }

        try:
            # Get scene description
            caption_result = self.caption_image(image, length="normal")
            analysis["scene_description"] = caption_result["caption"]
            analysis["total_inference_time"] += caption_result["inference_time"]

            # Analyze weather conditions
            weather_result = self.query_image(image, "What are the weather conditions? (sunny, cloudy, rainy, foggy)")
            analysis["weather"] = weather_result["answer"]
            analysis["total_inference_time"] += weather_result["inference_time"]

            # Analyze time of day
            time_result = self.query_image(image, "What time of day is it? (daytime, dusk, night)")
            analysis["time_of_day"] = time_result["answer"]
            analysis["total_inference_time"] += time_result["inference_time"]

            # Check for hazards
            hazard_result = self.query_image(image, "Are there any road hazards or obstacles visible?")
            if hazard_result["answer"].lower() not in ["no", "none", "no hazards"]:
                analysis["hazards"].append(hazard_result["answer"])

            self.logger.info(f"Scene analysis completed in {analysis['total_inference_time']:.2f}s")

        except Exception as e:
            self.logger.error(f"Driving scene analysis failed: {e}")

        return analysis

    def get_avg_inference_time(self) -> float:
        """Get average inference time"""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times = []


class VLMSceneAnalyzer:
    """
    Higher-level scene analyzer that combines VLM with ADAS detectors
    for enhanced situational awareness
    """

    def __init__(self, vlm_detector: MoondreamVLMDetector, logger=None):
        """
        Initialize VLM Scene Analyzer

        Args:
            vlm_detector: MoondreamVLMDetector instance
            logger: Logger instance
        """
        self.vlm = vlm_detector
        self.logger = logger or logging.getLogger(__name__)
        self.frame_counter = 0
        self.analysis_interval = 30  # Analyze every N frames to save computation
        self.last_analysis = None

    def should_analyze(self) -> bool:
        """Determine if we should run VLM analysis on this frame"""
        self.frame_counter += 1
        return self.frame_counter % self.analysis_interval == 0

    def analyze_frame(self, frame: np.ndarray, force: bool = False) -> Optional[Dict]:
        """
        Analyze frame with VLM

        Args:
            frame: OpenCV image
            force: Force analysis regardless of interval

        Returns:
            Analysis results or None if skipped
        """
        if not force and not self.should_analyze():
            return self.last_analysis

        if not self.vlm.is_available():
            return None

        # Get quick scene description
        caption_result = self.vlm.caption_image(frame, length="short")

        analysis = {
            "frame_number": self.frame_counter,
            "scene_description": caption_result["caption"],
            "inference_time": caption_result["inference_time"],
            "timestamp": time.time()
        }

        self.last_analysis = analysis
        return analysis

    def check_specific_hazard(self, frame: np.ndarray, hazard_type: str) -> Dict:
        """
        Check for specific hazard in the scene

        Args:
            frame: OpenCV image
            hazard_type: Type of hazard to check (e.g., 'pedestrian crossing', 'stopped vehicle')

        Returns:
            Hazard detection result
        """
        if not self.vlm.is_available():
            return {"detected": False, "description": "VLM not available"}

        question = f"Is there a {hazard_type} in the image? Answer yes or no and explain briefly."
        result = self.vlm.query_image(frame, question)

        detected = "yes" in result["answer"].lower()

        return {
            "detected": detected,
            "hazard_type": hazard_type,
            "description": result["answer"],
            "inference_time": result["inference_time"]
        }

    def set_analysis_interval(self, interval: int):
        """Set how often to run VLM analysis (in frames)"""
        self.analysis_interval = max(1, interval)
        self.logger.info(f"VLM analysis interval set to every {interval} frames")

    def draw_vlm_info(self, frame: np.ndarray, x: int = 10, y: int = 500):
        """
        Draw VLM analysis information on frame

        Args:
            frame: OpenCV image to draw on
            x: X position
            y: Y position
        """
        if self.last_analysis is None:
            return

        # Create semi-transparent background
        overlay = frame.copy()
        text_height = 100
        cv2.rectangle(overlay, (x-5, y-25), (x+600, y+text_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw VLM info
        cv2.putText(frame, "VLM Scene Analysis:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Truncate long descriptions
        description = self.last_analysis.get("scene_description", "")
        if len(description) > 80:
            description = description[:77] + "..."

        cv2.putText(frame, description, (x, y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show inference time
        inf_time = self.last_analysis.get("inference_time", 0)
        cv2.putText(frame, f"VLM Time: {inf_time:.2f}s", (x, y+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
