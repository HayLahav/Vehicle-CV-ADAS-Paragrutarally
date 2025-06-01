import cv2
import time
import numpy as np
import logging
import platform
import subprocess
import os

from ObjectTracker import BYTETracker
from taskConditions import TaskConditions, Logger
from ObjectDetector import YoloDetector, EfficientdetDetector
from ObjectDetector.utils import ObjectModelType, CollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# Import both lane and road detection classes
from TrafficLaneDetector import UltrafastLaneDetector, UltrafastLaneDetectorV2
from TrafficLaneDetector.ufldDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ufldDetector.utils import LaneModelType, OffsetType, CurvatureType
from RoadSegmentation import RoadSegmentationDetector

LOGGER = Logger(None, logging.INFO, logging.INFO)

# Jetson Detection Functions
def is_jetson():
    """Check if running on Jetson platform"""
    try:
        with open('/etc/nv_tegra_release') as f:
            tegra_info = f.read()
            return 'tegra' in tegra_info.lower()
    except:
        return False

def get_jetson_info():
    """Get Jetson platform information"""
    try:
        with open('/etc/nv_tegra_release') as f:
            return f.read().strip()
    except:
        return "Unknown Jetson platform"

def optimize_jetson_performance():
    """Optimize Jetson Xavier performance"""
    if not is_jetson():
        return
    
    try:
        # Set max performance mode
        subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=False, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'jetson_clocks'], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Set GPU max frequency for Xavier AGX
        subprocess.run(['sudo', 'bash', '-c', 
                       'echo 1377000000 > /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq'], 
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        LOGGER.info("Jetson performance optimized")
    except Exception as e:
        LOGGER.war(f"Could not optimize Jetson performance: {e}")

def monitor_jetson_stats():
    """Get Jetson system statistics"""
    if not is_jetson():
        return {}
    
    stats = {}
    try:
        # Get temperature
        temp_zones = ['/sys/devices/virtual/thermal/thermal_zone0/temp',
                     '/sys/devices/virtual/thermal/thermal_zone1/temp']
        temps = []
        for zone in temp_zones:
            try:
                with open(zone, 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    temps.append(temp)
            except:
                pass
        if temps:
            stats['temperature'] = max(temps)
        
        # Get memory usage
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemAvailable:' in line:
                    available = int(line.split()[1]) // 1024  # MB
                    stats['memory_available'] = available
                elif 'MemTotal:' in line:
                    total = int(line.split()[1]) // 1024  # MB
                    stats['memory_total'] = total
        
        if 'memory_total' in stats and 'memory_available' in stats:
            stats['memory_used'] = stats['memory_total'] - stats['memory_available']
            stats['memory_usage_percent'] = (stats['memory_used'] / stats['memory_total']) * 100
        
    except Exception as e:
        LOGGER.war(f"Could not get Jetson stats: {e}")
    
    return stats

# Video path - update this to your test video
video_path = "./TrafficLaneDetector/temp/demo-7.mp4"

# Jetson-optimized configurations
if is_jetson():
    # Road segmentation config for Jetson
    road_config = {
        "model_path": "./RoadSegmentation/models/twinlitenet_drivable.onnx",
    }
    
    # Optimized object detection config for Jetson
    object_config = {
        "model_path": './ObjectDetector/models/yolov5n-coco_fp16.onnx',  # Smaller, faster model
        "model_type": ObjectModelType.YOLOV5,
        "classes_path": './ObjectDetector/models/coco_label.txt',
        "box_score": 0.5,  # Higher threshold for performance
        "box_nms_iou": 0.5
    }
    
    # Lane detection config for Jetson
    lane_config = {
        "model_path": './TrafficLaneDetector/models/culane_res18_fp16.onnx',
        "model_type": LaneModelType.UFLDV2_CULANE,
    }
else:
    # Standard configurations for non-Jetson platforms
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
    
    lane_config = {
        "model_path": './TrafficLaneDetector/models/culane_res18.onnx',
        "model_type": LaneModelType.UFLDV2_CULANE,
    }

# Priority: FCWS > RDWS > RKAS
class ControlPanel(object):
    CollisionDict = {
        CollisionType.UNKNOWN: (0, 255, 255),
        CollisionType.NORMAL: (0, 255, 0),
        CollisionType.PROMPT: (0, 102, 255),
        CollisionType.WARNING: (0, 0, 255)
    }

    OffsetDict = {
        OffsetType.UNKNOWN: (0, 255, 255),
        OffsetType.RIGHT: (0, 0, 255),
        OffsetType.LEFT: (0, 0, 255),
        OffsetType.CENTER: (0, 255, 0)
    }

    CurvatureDict = {
        CurvatureType.UNKNOWN: (0, 255, 255),
        CurvatureType.STRAIGHT: (0, 255, 0),
        CurvatureType.EASY_LEFT: (0, 102, 255),
        CurvatureType.EASY_RIGHT: (0, 102, 255),
        CurvatureType.HARD_LEFT: (0, 0, 255),
        CurvatureType.HARD_RIGHT: (0, 0, 255)
    }

    def __init__(self):
        # Load warning and guidance images with error handling
        self.images_loaded = False
        try:
            collision_warning_img = cv2.imread('./assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
            self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100)) if collision_warning_img is not None else None
            
            collision_prompt_img = cv2.imread('./assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
            self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100)) if collision_prompt_img is not None else None
            
            collision_normal_img = cv2.imread('./assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
            self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100)) if collision_normal_img is not None else None
            
            left_curve_img = cv2.imread('./assets/left_turn.png', cv2.IMREAD_UNCHANGED)
            self.left_curve_img = cv2.resize(left_curve_img, (200, 200)) if left_curve_img is not None else None
            
            right_curve_img = cv2.imread('./assets/right_turn.png', cv2.IMREAD_UNCHANGED)
            self.right_curve_img = cv2.resize(right_curve_img, (200, 200)) if right_curve_img is not None else None
            
            keep_straight_img = cv2.imread('./assets/straight.png', cv2.IMREAD_UNCHANGED)
            self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200)) if keep_straight_img is not None else None
            
            determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
            self.determined_img = cv2.resize(determined_img, (200, 200)) if determined_img is not None else None
            
            left_lanes_img = cv2.imread('./assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
            self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200)) if left_lanes_img is not None else None
            
            right_lanes_img = cv2.imread('./assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
            self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200)) if right_lanes_img is not None else None
            
            self.images_loaded = True
        except Exception as e:
            LOGGER.war(f"Could not load UI images: {e}. Using text-only interface.")

        # FPS and performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start = time.time()
        self.curve_status = None
        
        # Jetson-specific monitoring
        self.jetson_stats = {}
        self.stats_update_counter = 0

    def _updateFPS(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count >= 30:
            self.end = time.time()
            self.fps = self.frame_count / (self.end - self.start)
            self.frame_count = 0
            self.start = time.time()
            
            # Update Jetson stats every 30 frames
            if is_jetson():
                self.jetson_stats = monitor_jetson_stats()

    def _draw_image_with_alpha(self, frame, img, x, y):
        """Draw image with alpha channel support"""
        if img is None or img.shape[2] < 4:
            return
        
        try:
            h, w = img.shape[:2]
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, frame.shape[1] - w))
            y = max(0, min(y, frame.shape[0] - h))
            
            alpha_channel = img[:, :, 3] / 255.0
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha_channel * img[:, :, c] + 
                    (1 - alpha_channel) * frame[y:y+h, x:x+w, c]
                )
        except Exception as e:
            # Fallback to simple text if image drawing fails
            cv2.putText(frame, "IMG", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25):
        """Display BirdView Panel on image with Jetson optimizations"""
        try:
            W = int(main_show.shape[1] * show_ratio)
            H = int(main_show.shape[0] * show_ratio)

            min_birdview_show = cv2.resize(min_show, (W, H))
            min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, 
                                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            # Ensure we don't exceed frame boundaries
            end_y = min(min_birdview_show.shape[0], main_show.shape[0])
            end_x = min(min_birdview_show.shape[1], main_show.shape[1])
            start_x = max(0, main_show.shape[1] - min_birdview_show.shape[1])
            
            main_show[0:end_y, start_x:main_show.shape[1]] = min_birdview_show[0:end_y, 0:(main_show.shape[1]-start_x)]
        except Exception as e:
            LOGGER.war(f"Error displaying bird view panel: {e}")

    def DisplayRoadSignsPanel(self, main_show, offset_type, curvature_type):
        """Display road guidance signs with Jetson optimizations"""
        try:
            W = 400
            H = 365
            
            # Ensure panel fits within frame
            W = min(W, main_show.shape[1])
            H = min(H, main_show.shape[0])
            
            widget = np.copy(main_show[:H, :W])
            widget //= 2
            widget[0:3, :] = [0, 0, 255]  # top
            widget[-3:-1, :] = [0, 0, 255]  # bottom
            widget[:, 0:3] = [0, 0, 255]  # left
            widget[:, -3:-1] = [0, 0, 255]  # right
            main_show[:H, :W] = widget

            # Display appropriate warning images or text
            if self.images_loaded:
                if curvature_type == CurvatureType.UNKNOWN and offset_type in {OffsetType.UNKNOWN, OffsetType.CENTER}:
                    self._draw_image_with_alpha(main_show, self.determined_img, W//2-100, 10)
                    self.curve_status = None
                elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status == "Left") and \
                     (curvature_type not in {CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT}):
                    self._draw_image_with_alpha(main_show, self.left_curve_img, W//2-100, 10)
                    self.curve_status = "Left"
                elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status == "Right") and \
                     (curvature_type not in {CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT}):
                    self._draw_image_with_alpha(main_show, self.right_curve_img, W//2-100, 10)
                    self.curve_status = "Right"

                # Display road position guidance
                if offset_type == OffsetType.RIGHT:
                    self._draw_image_with_alpha(main_show, self.left_lanes_img, W//2-150, 10)
                elif offset_type == OffsetType.LEFT:
                    self._draw_image_with_alpha(main_show, self.right_lanes_img, W//2-150, 10)
                elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight":
                    self._draw_image_with_alpha(main_show, self.keep_straight_img, W//2-100, 10)
                    self.curve_status = "Straight"
            else:
                # Text-only fallback
                status_text = f"{curvature_type.name}"
                cv2.putText(main_show, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Update and display text information
            self._updateFPS()
            
            # System information
            cv2.putText(main_show, "RDWS: " + offset_type.value, (10, 240), 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
                       color=self.OffsetDict[offset_type], thickness=2)
            cv2.putText(main_show, "RKAS: " + curvature_type.value, org=(10, 280), 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
                       color=self.CurvatureDict[curvature_type], thickness=2)
            cv2.putText(main_show, "FPS: %.1f" % self.fps, (10, widget.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Jetson-specific information
            if is_jetson() and self.jetson_stats:
                if 'temperature' in self.jetson_stats:
                    temp_color = (0, 255, 0) if self.jetson_stats['temperature'] < 70 else (0, 255, 255)
                    if self.jetson_stats['temperature'] > 80:
                        temp_color = (0, 0, 255)
                    cv2.putText(main_show, f"Temp: {self.jetson_stats['temperature']:.1f}C", 
                               (10, widget.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 1)
                
                if 'memory_usage_percent' in self.jetson_stats:
                    mem_color = (0, 255, 0) if self.jetson_stats['memory_usage_percent'] < 80 else (0, 255, 255)
                    if self.jetson_stats['memory_usage_percent'] > 90:
                        mem_color = (0, 0, 255)
                    cv2.putText(main_show, f"Mem: {self.jetson_stats['memory_usage_percent']:.1f}%", 
                               (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mem_color, 1)
                
        except Exception as e:
            LOGGER.war(f"Error displaying road signs panel: {e}")

    def DisplayCollisionPanel(self, main_show, collision_type, object_infer_time, road_infer_time, show_ratio=0.25):
        """Display Collision Panel with Jetson optimizations"""
        try:
            W = int(main_show.shape[1] * show_ratio)
            H = int(main_show.shape[0] * show_ratio)

            # Ensure panel positions are within bounds
            start_y = min(H + 20, main_show.shape[0] - H)
            end_y = min(start_y + H, main_show.shape[0])
            start_x = max(main_show.shape[1] - W - 20, 0)
            end_x = min(start_x + W, main_show.shape[1])
            
            if start_y >= end_y or start_x >= end_x:
                return  # Panel doesn't fit, skip

            widget = np.copy(main_show[start_y:end_y, start_x:end_x])
            widget //= 2
            widget[0:3, :] = [0, 0, 255]  # top
            widget[-3:-1, :] = [0, 0, 255]  # bottom
            widget[:, -3:-1] = [0, 0, 255]  # left
            widget[:, 0:3] = [0, 0, 255]  # right
            main_show[start_y:end_y, start_x:end_x] = widget

            # Display appropriate collision warning image or text
            if self.images_loaded:
                img_y = start_y + 50
                img_x = start_x + 10
                
                if collision_type == CollisionType.WARNING and self.collision_warning_img is not None:
                    self._draw_image_with_alpha(main_show, self.collision_warning_img, img_x, img_y)
                elif collision_type == CollisionType.PROMPT and self.collision_prompt_img is not None:
                    self._draw_image_with_alpha(main_show, self.collision_prompt_img, img_x, img_y)
                elif collision_type == CollisionType.NORMAL and self.collision_normal_img is not None:
                    self._draw_image_with_alpha(main_show, self.collision_normal_img, img_x, img_y)
            else:
                # Text-only fallback
                cv2.putText(main_show, collision_type.name, (start_x + 10, start_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CollisionDict[collision_type], 2)

            # Display warning text and performance metrics
            text_y_base = 240
            text_x = max(20, main_show.shape[1] - W + 10)
            
            cv2.putText(main_show, "FCWS: " + collision_type.value, (text_x, text_y_base), 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                       color=self.CollisionDict[collision_type], thickness=2)
            cv2.putText(main_show, "obj: %.2f s" % object_infer_time, (text_x, text_y_base + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(main_show, "road: %.2f s" % road_infer_time, (text_x, text_y_base + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            
        except Exception as e:
            LOGGER.war(f"Error displaying collision panel: {e}")


if __name__ == "__main__":
    # Platform detection and optimization
    if is_jetson():
        LOGGER.info(f"[Platform] Running on NVIDIA Jetson")
        LOGGER.info(f"[Platform] {get_jetson_info()}")
        optimize_jetson_performance()
        
        # Set Jetson-specific environment variables
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        os.environ['CUDA_CACHE_PATH'] = '/tmp/.nv/ComputeCache'
    else:
        LOGGER.info("[Platform] Running on standard CUDA platform")

    # Initialize video capture and output
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Video path is error. Please check it.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter(video_path[:-4]+'_jetson_out.mp4', fourcc, 30.0, (width, height))
    cv2.namedWindow("ADAS Simulation - Jetson", cv2.WINDOW_NORMAL)
    
    #==========================================================
    #                   Initialize Classes
    #==========================================================
    LOGGER.info("-"*40)

    # Initialize road segmentation detector
    LOGGER.info("Initializing Road Segmentation Detector")
    try:
        roadDetector = RoadSegmentationDetector(road_config["model_path"], logger=LOGGER)
    except Exception as e:
        LOGGER.error(f"Failed to initialize road detector: {e}")
        exit(1)
    
    # Initialize perspective transformation for bird's eye view
    transformView = PerspectiveTransformation((width, height), logger=LOGGER)

    # Initialize object detection model
    LOGGER.info("ObjectDetector Model Type: {}".format(object_config["model_type"].name))
    try:
        if ObjectModelType.EfficientDet == object_config["model_type"]:
            EfficientdetDetector.set_defaults(object_config)
            objectDetector = EfficientdetDetector(logger=LOGGER)
        else:
            YoloDetector.set_defaults(object_config)
            objectDetector = YoloDetector(logger=LOGGER)
    except Exception as e:
        LOGGER.error(f"Failed to initialize object detector: {e}")
        exit(1)
    
    # Initialize distance measurement and object tracking
    distanceDetector = SingleCamDistanceMeasure()
    objectTracker = BYTETracker(names=objectDetector.colors_dict)

    # Initialize display panel and task condition analyzer
    displayPanel = ControlPanel()
    analyzeMsg = TaskConditions()
    
    LOGGER.info("All systems initialized successfully")
    LOGGER.info(f"Processing video: {video_path}")
    LOGGER.info("-"*40)
    
    frame_counter = 0
    total_start_time = time.time()
    
    # Main processing loop
    try:
        while cap.isOpened():
            ret, frame = cap.read()  # Read frame from the video
            if ret:
                frame_show = frame.copy()
                frame_counter += 1

                #========================== Detect Objects =========================
                object_time = time.time()
                try:
                    objectDetector.DetectFrame(frame)
                    object_infer_time = round(time.time() - object_time, 3)
                except Exception as e:
                    LOGGER.war(f"Object detection failed: {e}")
                    object_infer_time = 0

                # Track objects if tracking is enabled
                if objectTracker and len(objectDetector.object_info) > 0:
                    try:
                        box = [obj.tolist(format_type="xyxy") for obj in objectDetector.object_info]
                        score = [obj.conf for obj in objectDetector.object_info]
                        id = [obj.label for obj in objectDetector.object_info]
                        objectTracker.update(box, score, id, frame)
                    except Exception as e:
                        LOGGER.war(f"Object tracking failed: {e}")

                #========================== Detect Road ===========================
                road_time = time.time()
                try:
                    roadDetector.DetectFrame(frame)
                    road_infer_time = round(time.time() - road_time, 4)
                except Exception as e:
                    LOGGER.war(f"Road detection failed: {e}")
                    road_infer_time = 0

                #========================= Analyze Status ========================
                try:
                    # Update distance measurements
                    distanceDetector.updateDistance(objectDetector.object_info)
                    
                    # Use road boundary for collision detection
                    vehicle_distance = distanceDetector.calcCollisionPoint(roadDetector.road_info.road_boundary)
                    
                    # Calculate vehicle position (bottom center of frame)
                    vehicle_position = (frame.shape[1] // 2, frame.shape[0] - 10)
                    
                    # Update road status in the analyzer
                    analyzeMsg.UpdateRoadStatus(
                        roadDetector.road_info.is_on_road,
                        roadDetector.road_info.road_center,
                        vehicle_position
                    )
                    
                    # Update collision and route status
                    analyzeMsg.UpdateCollisionStatus(vehicle_distance, roadDetector.road_info.is_on_road)
                    analyzeMsg.UpdateRouteStatus()
                except Exception as e:
                    LOGGER.war(f"Status analysis failed: {e}")
                
                # Create bird's eye view
                birdview_show = frame.copy()  # Placeholder for bird's eye view
                
                try:
                    if roadDetector.road_info.road_mask is not None:
                        # Create a bird's eye view of the road mask
                        road_mask_colored = np.zeros_like(frame)
                        if roadDetector.road_info.road_mask is not None:
                            road_mask_rgb = cv2.cvtColor(roadDetector.road_info.road_mask, cv2.COLOR_GRAY2BGR)
                            road_mask_colored[roadDetector.road_info.road_mask > 0] = [0, 255, 0]
                        birdview_show = transformView.transformToBirdView(road_mask_colored)
                except Exception as e:
                    LOGGER.war(f"Bird view creation failed: {e}")

                #========================== Draw Results =========================
                try:
                    # Draw road detection
                    roadDetector.DrawDetectedOnFrame(frame_show)
                    roadDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
                    
                    # Draw object detection and tracking
                    objectDetector.DrawDetectedOnFrame(frame_show)
                    objectTracker.DrawTrackedOnFrame(frame_show, False)
                    distanceDetector.DrawDetectedOnFrame(frame_show)

                    # Display panels
                    displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
                    displayPanel.DisplayRoadSignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)
                    displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, object_infer_time, road_infer_time)
                except Exception as e:
                    LOGGER.war(f"Drawing failed: {e}")
                
                # Show result
                cv2.imshow("ADAS Simulation - Jetson", frame_show)

                # Write frame to output video
                try:
                    vout.write(frame_show)
                except Exception as e:
                    LOGGER.war(f"Video writing failed: {e}")
                
                # Performance monitoring for Jetson
                if is_jetson() and frame_counter % 100 == 0:
                    avg_fps = frame_counter / (time.time() - total_start_time)
                    stats = monitor_jetson_stats()
                    LOGGER.info(f"Frame {frame_counter}: Avg FPS: {avg_fps:.1f}")
                    if 'temperature' in stats:
                        LOGGER.info(f"Temperature: {stats['temperature']:.1f}°C")
                    if 'memory_usage_percent' in stats:
                        LOGGER.info(f"Memory Usage: {stats['memory_usage_percent']:.1f}%")

            else:
                break
                
            # Check for exit key
            if cv2.waitKey(1) == ord('q'):  # Press key q to stop
                break

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    except Exception as e:
        LOGGER.error(f"Unexpected error: {e}")
    finally:
        # Clean up
        total_time = time.time() - total_start_time
        avg_fps = frame_counter / total_time if total_time > 0 else 0
        
        LOGGER.info("-"*40)
        LOGGER.info(f"Processing completed:")
        LOGGER.info(f"Total frames: {frame_counter}")
        LOGGER.info(f"Total time: {total_time:.2f}s")
        LOGGER.info(f"Average FPS: {avg_fps:.2f}")
        
        if is_jetson():
            final_stats = monitor_jetson_stats()
            if 'temperature' in final_stats:
                LOGGER.info(f"Final temperature: {final_stats['temperature']:.1f}°C")
        
        vout.release()
        cap.release()
        cv2.destroyAllWindows()