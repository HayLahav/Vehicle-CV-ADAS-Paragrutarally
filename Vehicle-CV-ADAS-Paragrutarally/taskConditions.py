import numpy as np
import logging
import ctypes
from ObjectDetector.utils import CollisionType
from TrafficLaneDetector.ufldDetector.utils import OffsetType, CurvatureType

STD_OUTPUT_HANDLE= -11
def set_color(color, handle=None):
    return True
class LimitedList(list):
    def __init__(self, maxlen):
        super().__init__()
        self._maxlen = maxlen
        self._is_full = False

    def full(self):
        return self._is_full
    
    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(LimitedList, self).append(element)
        if len(self) < self._maxlen:
            self._is_full = False
        else:
            self._is_full = True
            
    def extend(self, elements):
        for element in elements:
            self.append(element)
            
    def clear(self):
        super(LimitedList, self).__init__()
        self._is_full = False

class Logger:
    FOREGROUND_WHITE = 0x0007
    FOREGROUND_BLUE = 0x01 # text color contains blue.
    FOREGROUND_GREEN= 0x02 # text color contains green.
    FOREGROUND_RED  = 0x04 # text color contains red.
    FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN

    def __init__(self, path, clevel = logging.DEBUG, Flevel = logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        self.clevel = clevel
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # CMD
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        self.logger.addHandler(sh)
        if (path != None):
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            fh.setLevel(Flevel)
            self.logger.addHandler(fh)

    def changelevel(self, clevel):
        self.clevel = clevel
        self.logger.setLevel(clevel)

    def debug(self, message):
        self.logger.debug(message)
 
    def info(self, message, color=FOREGROUND_BLUE):
        set_color(color)
        self.logger.info(message)
        set_color(self.FOREGROUND_WHITE)
 
    def war(self, message, color=FOREGROUND_YELLOW):
        set_color(color)
        self.logger.warn(message)
        set_color(self.FOREGROUND_WHITE)
 
    def error(self, message, color=FOREGROUND_RED):
        set_color(color)
        self.logger.error(message)
        set_color(self.FOREGROUND_WHITE)
 
    def cri(self, message):
        self.logger.critical(message)

class TaskConditions(object):
    
    def __init__(self):
        # Status messages for different systems
        self.collision_msg = CollisionType.UNKNOWN
        self.offset_msg = OffsetType.UNKNOWN
        self.curvature_msg = CurvatureType.UNKNOWN
        
        # Record lists for averaging measurements
        self.vehicle_collision_record = LimitedList(5)
        self.vehicle_offset_record = LimitedList(5)
        self.vehicle_curvature_record = LimitedList(10)
        
        # Bird view transformation status
        self.transform_status = None
        self.toggle_status = "Default"
        self.toggle_oscillator_status = [False, False]
        self.toggle_status_counter = {"Offset": 0, "Curvae": 0, "BirdViewAngle": 0}
        
        # Road status tracking for road segmentation
        self.is_on_road = False
        self.road_departure_threshold = 0.2  # Distance to road edge that triggers warning
        self.road_center = None
        self.vehicle_position = None
        

    def _calibration_curve(self, vehicle_curvature, frequency=3, curvae_thres=15000):
        """
        Calibration road when curvae smooth.

        Args:
            vehicle_curvature: calc curvature values from birdView.
            frequency: when togger count > frequency, toggle_status will Default means BirdView will revise.
            curvae_thres: The larger the values, it means curvae smooth on BirdView.

        Returns:
            None
        """
        if (self.toggle_status_counter["BirdViewAngle"] <= frequency):
            if (vehicle_curvature >= curvae_thres):
                self.toggle_status_counter["BirdViewAngle"] += 1
            else:
                self.toggle_status_counter["BirdViewAngle"] = 0
        else:
            self.toggle_status_counter["BirdViewAngle"] = 0
            self.toggle_status = "Default"

    def _calc_deviation(self, offset, offset_thres):
        """
        Get offset status.

        Args:
            offset: Get avg offset values.
            offset_thres: Determine whether the lane line is offset from the center.

        Returns:
            OffsetType
        """
        if (abs(offset) > offset_thres):
            if (offset > 0 and self.curvature_msg not in {CurvatureType.HARD_LEFT, CurvatureType.EASY_LEFT}):
                msg = OffsetType.RIGHT
            elif (offset < 0 and self.curvature_msg not in {CurvatureType.HARD_RIGHT, CurvatureType.EASY_RIGHT}):
                msg = OffsetType.LEFT
            else:
                msg = OffsetType.UNKNOWN
        else:
            msg = OffsetType.CENTER

        return msg

    def _calc_direction(self, curvature, curvae_dir, curvae_thres):
        """
        Get curvature status.

        Args:
            curvature: Get avg curvature values.
            curvae_dir: Get avg curvae direction.
            curvae_thres: Determine whether the lane line is hard or easy curvae.

        Returns:
            CurvatureType
        """
        if (curvature <= curvae_thres):
            if (curvae_dir == "L" and self.curvature_msg != CurvatureType.EASY_RIGHT):
                msg = CurvatureType.HARD_LEFT
            elif (curvae_dir == "R" and self.curvature_msg != CurvatureType.EASY_LEFT):
                msg = CurvatureType.HARD_RIGHT
            else:
                msg = CurvatureType.UNKNOWN
        else:
            if (curvae_dir == "L"):
                msg = CurvatureType.EASY_LEFT
            elif (curvae_dir == "R"):
                msg = CurvatureType.EASY_RIGHT
            else:
                msg = CurvatureType.STRAIGHT
        return msg

    def CheckStatus(self):
        """
        Determine whether to update Bird View perspective transform.

        Args:
            None

        Returns:
            Bool
        """
        if (self.curvature_msg == CurvatureType.UNKNOWN and self.offset_msg == OffsetType.UNKNOWN):
            self.toggle_oscillator_status = [False, False]

        if self.toggle_status != self.transform_status:
            self.transform_status = self.toggle_status
            self.toggle_status = None
            return True
        else:
            return False

    def UpdateRoadStatus(self, is_on_road, road_center=None, vehicle_position=None):
        """
        Update road status based on road segmentation results.
        
        Args:
            is_on_road: Boolean indicating if vehicle is on the road
            road_center: Road center point (x, y) or array of points
            vehicle_position: Position of the vehicle (x, y)
            
        Returns:
            None
        """
        self.is_on_road = is_on_road
        self.road_center = road_center
        self.vehicle_position = vehicle_position
        
        if not is_on_road:
            self.offset_msg = OffsetType.UNKNOWN
            return
            
        # Calculate offset from road center
        if road_center is not None and vehicle_position is not None:
            try:
                # Handle different road_center formats
                if isinstance(road_center, (tuple, list)) and len(road_center) == 2:
                    # Single center point (x, y)
                    center_x, center_y = road_center
                    vehicle_x, vehicle_y = vehicle_position
                    
                    # Calculate normalized offset (-1 to 1)
                    img_width = 1280  # Assuming standard width
                    normalized_offset = (vehicle_x - center_x) / (img_width / 2)
                    self.vehicle_offset_record.append(normalized_offset)
                    
                    if self.vehicle_offset_record.full():
                        avg_offset = np.median(self.vehicle_offset_record)
                        
                        # Determine offset type
                        if abs(avg_offset) < self.road_departure_threshold:
                            self.offset_msg = OffsetType.CENTER
                        elif avg_offset > 0:
                            self.offset_msg = OffsetType.RIGHT
                        else:
                            self.offset_msg = OffsetType.LEFT
                    else:
                        self.offset_msg = OffsetType.UNKNOWN
                        
                elif isinstance(road_center, np.ndarray) and road_center.ndim == 2 and road_center.shape[1] == 2:
                    # Array of road center points
                    vehicle_x, vehicle_y = vehicle_position
                    
                    # Calculate distance to each center point
                    vehicle_point = np.array([vehicle_x, vehicle_y])
                    distances = np.sum((road_center - vehicle_point) ** 2, axis=1)
                    
                    if len(distances) > 0:
                        closest_idx = np.argmin(distances)
                        center_x = road_center[closest_idx][0]
                        
                        # Calculate normalized offset (-1 to 1)
                        img_width = 1280  # Assuming standard width
                        normalized_offset = (vehicle_x - center_x) / (img_width / 2)
                        self.vehicle_offset_record.append(normalized_offset)
                        
                        if self.vehicle_offset_record.full():
                            avg_offset = np.median(self.vehicle_offset_record)
                            
                            # Determine offset type
                            if abs(avg_offset) < self.road_departure_threshold:
                                self.offset_msg = OffsetType.CENTER
                            elif avg_offset > 0:
                                self.offset_msg = OffsetType.RIGHT
                            else:
                                self.offset_msg = OffsetType.LEFT
                        else:
                            self.offset_msg = OffsetType.UNKNOWN
                    else:
                        self.offset_msg = OffsetType.UNKNOWN
                else:
                    self.offset_msg = OffsetType.UNKNOWN
            except Exception as e:
                # If there's any error in processing road center, set to UNKNOWN
                self.offset_msg = OffsetType.UNKNOWN
        else:
            self.offset_msg = OffsetType.UNKNOWN

    def UpdateOffsetStatus(self, vehicle_offset, offset_thres=0.65):
        """
        Judging the state of the avg offset.
        For lane-based systems, not road segmentation.

        Args:
            vehicle_offset: Calc offset values from birdView.
            offset_thres: Determine whether the lane line is offset from the center.

        Returns:
            None
        """
        if (vehicle_offset != None):
            self.vehicle_offset_record.append(vehicle_offset)
            if self.vehicle_offset_record.full():
                avg_vehicle_offset = np.median(self.vehicle_offset_record)
                self.offset_msg = self._calc_deviation(avg_vehicle_offset, offset_thres)

                plus = [i for i in self.vehicle_offset_record if i > 0.2]
                mius = [i for i in self.vehicle_offset_record if i < -0.2]
                if (self.toggle_status_counter["Offset"] >= 10):
                    if (len(plus) == len(self.vehicle_offset_record)):
                        self.toggle_oscillator_status[0] = True
                        self.toggle_status_counter["Offset"] = 0
                    if (len(mius) == len(self.vehicle_offset_record)):
                        self.toggle_oscillator_status[1] = True
                        self.toggle_status_counter["Offset"] = 0
                    if (np.array(self.toggle_oscillator_status).all()):
                        self.toggle_status = "Top"
                        self.toggle_oscillator_status = [False, False]
                    else:
                        self.toggle_status_counter["Offset"] = 0
                else:
                    self.toggle_status_counter["Offset"] += 1
            else:
                self.offset_msg = OffsetType.UNKNOWN
        else:
            self.offset_msg = OffsetType.UNKNOWN
            self.vehicle_offset_record.clear()

    def UpdateRouteStatus(self, vehicle_direction=None, vehicle_curvature=None, curvae_thres=500):
        """
        Judging the state of the avg curvature.
        Works with both lane detection and road segmentation.

        Args:
            vehicle_direction: Calc preliminary curvae direction from birdView.
            vehicle_curvature: Calc curvature values from birdView.
            curvae_thres: Determine whether the lane line is hard or easy curvae.

        Returns:
            None
        """
        # For road segmentation, use simplified approach
        if hasattr(self, 'is_on_road') and self.is_on_road:
            if self.offset_msg == OffsetType.CENTER:
                # When on the road and centered, assume straight path
                self.curvature_msg = CurvatureType.STRAIGHT
                return
        
        # Original lane-based logic
        if (vehicle_curvature != None):
            if (vehicle_direction != None and self.offset_msg == OffsetType.CENTER):
                self.vehicle_curvature_record.append([vehicle_direction, vehicle_curvature])

                if self.vehicle_curvature_record.full():
                    try:
                        # Safely extract direction and curvature data
                        directions = []
                        curvatures = []
                        for record in self.vehicle_curvature_record:
                            if isinstance(record, (list, tuple)) and len(record) >= 2:
                                directions.append(record[0])
                                curvatures.append(float(record[1]))
                        
                        if directions and curvatures:
                            avg_direction = max(set(directions), key=directions.count)
                            avg_curvature = np.median(curvatures)
                            self.curvature_msg = self._calc_direction(avg_curvature, avg_direction, curvae_thres)
                            
                            if (self.toggle_status_counter["Curvae"] >= 10):
                                if (self.curvature_msg != CurvatureType.STRAIGHT and 
                                    len(self.vehicle_offset_record) > 0 and
                                    abs(self.vehicle_offset_record[-1]) < 0.2 and 
                                    not np.array(self.toggle_oscillator_status).any()):
                                    self.toggle_status = "Bottom"
                                else:
                                    self.toggle_status_counter["Curvae"] = 0
                            else:
                                self.toggle_status_counter["Curvae"] += 1
                        else:
                            self.curvature_msg = CurvatureType.UNKNOWN
                    except (IndexError, ValueError):
                        self.curvature_msg = CurvatureType.UNKNOWN
                else:
                    self.curvature_msg = CurvatureType.UNKNOWN
            else:
                self.vehicle_curvature_record.clear()
                self.curvature_msg = CurvatureType.UNKNOWN

            self._calibration_curve(vehicle_curvature)

        else:
            self.vehicle_curvature_record.clear()
            self.curvature_msg = CurvatureType.UNKNOWN

    def UpdateCollisionStatus(self, vehicle_distance, road_area=True, distance_thres=1.5): 
        """
        Judging the state of the avg distance.
        Works with both lane detection and road segmentation.

        Args:
            vehicle_distance: Calc preliminary distance from SingleCamDistanceMeasure Class.
            road_area: Boolean indicating if a valid road area is detected.
            distance_thres: Distance when deciding to warn.

        Returns:
            None
        """
        if (vehicle_distance != None):
            x, y, d = vehicle_distance
            self.vehicle_collision_record.append(d)
            if self.vehicle_collision_record.full():
                avg_vehicle_collision = np.median(self.vehicle_collision_record)
                if (avg_vehicle_collision <= distance_thres):
                    self.collision_msg = CollisionType.WARNING
                elif (distance_thres < avg_vehicle_collision <= 2*distance_thres):
                    self.collision_msg = CollisionType.PROMPT
                else:
                    self.collision_msg = CollisionType.NORMAL
        else:
            if (road_area):
                self.collision_msg = CollisionType.NORMAL
            else:
                self.collision_msg = CollisionType.UNKNOWN
            self.vehicle_collision_record.clear()
