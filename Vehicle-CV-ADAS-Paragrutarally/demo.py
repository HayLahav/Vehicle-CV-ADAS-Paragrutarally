import cv2, time
import numpy as np
import logging
import pycuda.driver as drv

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

# Video path - update this to your test video
video_path = "./TrafficLaneDetector/temp/demo-7.mp4"

# Configuration for road segmentation instead of lane detection
road_config = {
    "model_path": "./RoadSegmentation/models/twinlitenet_drivable.onnx",
}

# Object detection configuration
object_config = {
    "model_path": './ObjectDetector/models/yolov10n-coco_fp16.trt',
    "model_type": ObjectModelType.YOLOV10,
    "classes_path": './ObjectDetector/models/coco_label.txt',
    "box_score": 0.4,
    "box_nms_iou": 0.5
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
        # Load warning and guidance images
        collision_warning_img = cv2.imread('./assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
        self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100))
        collision_prompt_img = cv2.imread('./assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
        self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100))
        collision_normal_img = cv2.imread('./assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
        self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100))
        left_curve_img = cv2.imread('./assets/left_turn.png', cv2.IMREAD_UNCHANGED)
        self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
        right_curve_img = cv2.imread('./assets/right_turn.png', cv2.IMREAD_UNCHANGED)
        self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
        keep_straight_img = cv2.imread('./assets/straight.png', cv2.IMREAD_UNCHANGED)
        self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
        determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
        self.determined_img = cv2.resize(determined_img, (200, 200))
        left_lanes_img = cv2.imread('./assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
        self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
        right_lanes_img = cv2.imread('./assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
        self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))

        # FPS
        self.fps = 0
        self.frame_count = 0
        self.start = time.time()

        self.curve_status = None

    def _updateFPS(self):
        """
        Update FPS.

        Args:
            None

        Returns:
            None
        """
        self.frame_count += 1
        if self.frame_count >= 30:
            self.end = time.time()
            self.fps = self.frame_count / (self.end - self.start)
            self.frame_count = 0
            self.start = time.time()

    def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25):
        """
        Display BirdView Panel on image.

        Args:
            main_show: video image.
            min_show: bird view image.
            show_ratio: display scale of bird view image.

        Returns:
            main_show: Draw bird view on frame.
        """
        W = int(main_show.shape[1] * show_ratio)
        H = int(main_show.shape[0] * show_ratio)

        min_birdview_show = cv2.resize(min_show, (W, H))
        min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]:] = min_birdview_show

    def DisplayRoadSignsPanel(self, main_show, offset_type, curvature_type):
        """
        Display road guidance signs for road segmentation.

        Args:
            main_show: image.
            offset_type: offset status (UNKNOWN/CENTER/RIGHT/LEFT)
            curvature_type: curvature status

        Returns:
            main_show: Draw signs info on frame.
        """
        W = 400
        H = 365
        widget = np.copy(main_show[:H, :W])
        widget //= 2
        widget[0:3, :] = [0, 0, 255]  # top
        widget[-3:-1, :] = [0, 0, 255]  # bottom
        widget[:, 0:3] = [0, 0, 255]  # left
        widget[:, -3:-1] = [0, 0, 255]  # right
        main_show[:H, :W] = widget

        # Display appropriate warning images
        if curvature_type == CurvatureType.UNKNOWN and offset_type in {OffsetType.UNKNOWN, OffsetType.CENTER}:
            y, x = self.determined_img[:, :, 3].nonzero()
            main_show[y+10, x-100+W//2] = self.determined_img[y, x, :3]
            self.curve_status = None
        elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status == "Left") and \
             (curvature_type not in {CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT}):
            y, x = self.left_curve_img[:, :, 3].nonzero()
            main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
            self.curve_status = "Left"
        elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status == "Right") and \
             (curvature_type not in {CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT}):
            y, x = self.right_curve_img[:, :, 3].nonzero()
            main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
            self.curve_status = "Right"

        # Display road position guidance
        if offset_type == OffsetType.RIGHT:
            y, x = self.left_lanes_img[:, :, 2].nonzero()
            main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
        elif offset_type == OffsetType.LEFT:
            y, x = self.right_lanes_img[:, :, 2].nonzero()
            main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
        elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight":
            y, x = self.keep_straight_img[:, :, 3].nonzero()
            main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
            self.curve_status = "Straight"

        # Update and display text information
        self._updateFPS()
        cv2.putText(main_show, "RDWS: " + offset_type.value, (10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
        cv2.putText(main_show, "RKAS: " + curvature_type.value, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
        cv2.putText(main_show, "FPS: %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def DisplayCollisionPanel(self, main_show, collision_type, object_infer_time, road_infer_time, show_ratio=0.25):
        """
        Display Collision Panel on image.

        Args:
            main_show: image.
            collision_type: collision status by CollisionType.
            object_infer_time: object detection time (seconds)
            road_infer_time: road detection time (seconds)

        Returns:
            main_show: Draw collision info on frame.
        """
        W = int(main_show.shape[1] * show_ratio)
        H = int(main_show.shape[0] * show_ratio)

        widget = np.copy(main_show[H+20:2*H, -W-20:])
        widget //= 2
        widget[0:3, :] = [0, 0, 255]  # top
        widget[-3:-1, :] = [0, 0, 255]  # bottom
        widget[:, -3:-1] = [0, 0, 255]  # left
        widget[:, 0:3] = [0, 0, 255]  # right
        main_show[H+20:2*H, -W-20:] = widget

        # Display appropriate collision warning image
        if collision_type == CollisionType.WARNING:
            y, x = self.collision_warning_img[:, :, 3].nonzero()
            main_show[H+y+50, (x-W-5)] = self.collision_warning_img[y, x, :3]
        elif collision_type == CollisionType.PROMPT:
            y, x = self.collision_prompt_img[:, :, 3].nonzero()
            main_show[H+y+50, (x-W-5)] = self.collision_prompt_img[y, x, :3]
        elif collision_type == CollisionType.NORMAL:
            y, x = self.collision_normal_img[:, :, 3].nonzero()
            main_show[H+y+50, (x-W-5)] = self.collision_normal_img[y, x, :3]

        # Display warning text
        cv2.putText(main_show, "RFCWS: " + collision_type.value, (main_show.shape[1] - int(W) + 100, 240), 
                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=self.CollisionDict[collision_type], thickness=2)
        cv2.putText(main_show, "object-infer: %.2f s" % object_infer_time, (main_show.shape[1] - int(W) + 100, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(main_show, "road-infer: %.2f s" % road_infer_time, (main_show.shape[1] - int(W) + 100, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


if __name__ == "__main__":
    # Initialize video capture and output
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Video path is error. Please check it.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc, 30.0, (width, height))
    cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)
    
    #==========================================================
    #                   Initialize Classes
    #==========================================================
    LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
    LOGGER.info("[Driver] Cuda Version: {}".format(drv.get_driver_version()))
    LOGGER.info("-"*40)

    # Initialize road segmentation detector instead of lane detector
    LOGGER.info("Initializing Road Segmentation Detector")
    roadDetector = RoadSegmentationDetector(road_config["model_path"], logger=LOGGER)
    
    # Initialize perspective transformation for bird's eye view
    transformView = PerspectiveTransformation((width, height), logger=LOGGER)

    # Initialize object detection model
    LOGGER.info("ObjectDetector Model Type: {}".format(object_config["model_type"].name))
    if ObjectModelType.EfficientDet == object_config["model_type"]:
        EfficientdetDetector.set_defaults(object_config)
        objectDetector = EfficientdetDetector(logger=LOGGER)
    else:
        YoloDetector.set_defaults(object_config)
        objectDetector = YoloDetector(logger=LOGGER)
    
    # Initialize distance measurement and object tracking
    distanceDetector = SingleCamDistanceMeasure()
    objectTracker = BYTETracker(names=objectDetector.colors_dict)

    # Initialize display panel and task condition analyzer
    displayPanel = ControlPanel()
    analyzeMsg = TaskConditions()
    
    # Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()  # Read frame from the video
        if ret:
            frame_show = frame.copy()

            #========================== Detect Objects =========================
            object_time = time.time()
            objectDetector.DetectFrame(frame)
            object_infer_time = round(time.time() - object_time, 2)

            # Track objects if tracking is enabled
            if objectTracker:
                box = [obj.tolist(format_type="xyxy") for obj in objectDetector.object_info]
                score = [obj.conf for obj in objectDetector.object_info]
                id = [obj.label for obj in objectDetector.object_info]
                objectTracker.update(box, score, id, frame)

            #========================== Detect Road ===========================
            road_time = time.time()
            roadDetector.DetectFrame(frame)
            road_infer_time = round(time.time() - road_time, 4)

            #========================= Analyze Status ========================
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
            
            # Create bird's eye view
            birdview_show = frame.copy()  # Placeholder for bird's eye view
            
            if roadDetector.road_info.road_mask is not None:
                # Create a bird's eye view of the road mask
                road_mask_colored = np.zeros_like(frame)
                if roadDetector.road_info.road_mask is not None:
                    road_mask_rgb = cv2.cvtColor(roadDetector.road_info.road_mask, cv2.COLOR_GRAY2BGR)
                    road_mask_colored[roadDetector.road_info.road_mask > 0] = [0, 255, 0]
                birdview_show = transformView.transformToBirdView(road_mask_colored)

            #========================== Draw Results =========================
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
            
            # Show result
            cv2.imshow("ADAS Simulation", frame_show)

        else:
            break
            
        # Write frame to output video
        vout.write(frame_show)
        
        # Check for exit key
        if cv2.waitKey(1) == ord('q'):  # Press key q to stop
            break

    # Clean up
    vout.release()
    cap.release()
    cv2.destroyAllWindows()