import cv2
import numpy as np
import typing
from typing import Union, List, Tuple
from ObjectDetector.core import RectInfo
from ObjectTracker.core import putText_shadow

class SingleCamDistanceMeasure(object):
    # 1 cm = 0.39 inch, original size h x w 
    INCH = 0.39
    RefSizeDict = { 
                    "person" : (160*INCH, 50*INCH), 
                    "bicycle" : (98*INCH, 65*INCH),
                    "motorbike" : (100*INCH, 100*INCH),
                    "car" : (150*INCH, 180*INCH ),
                    "bus" : (319*INCH, 250*INCH), 
                    "truck" : (346*INCH, 250*INCH), 
                 }

    def __init__(self, object_list: list = ["person", "bicycle", "car", "motorbike", "bus", "truck"]):
        self.object_list = object_list
        self.f = 100  # focal length
        self.distance_points = []

    def __isInsidePolygon(self, pt: tuple, poly: np.ndarray) -> bool:
        """
        Judgment point is within the polygon range.

        Args:
            pt: the object points.
            poly: is a polygonal points. [[x1, y1], [x2, y2], [x3, y3] ... [xn, yn]]

        Returns:
            total number of all feature vector.
        """
        c = False
        i = -1
        l = len(poly)
        j = l - 1
        while i < l - 1:
            i += 1
            if((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or (
                poly[j][0] <= pt[0] and pt[0] < poly[i][0])):
                if(pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (
                    poly[j][0] - poly[i][0]) + poly[i][1]):
                    c = not c
            j = i
        return c

    def updateDistance(self, boxes: typing.List[RectInfo]) -> None:
        """
        Update the distance of the target object through the size of pixels.

        Args:
            boxes: coordinate information and labels of the target object.

        Returns:
            None
        """
        self.distance_points = []
        if (len(boxes) != 0):
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                label = box.label
                
                if label in self.object_list and ymax <= 650:
                    point_x = (xmax + xmin) // 2
                    point_y = ymax

                    try:
                        distance = (self.RefSizeDict[label][0] * self.f) / (ymax - ymin)
                        distance = distance / 12 * 0.3048  # 1ft = 0.3048 m
                        self.distance_points.append([point_x, point_y, distance])
                    except:
                        pass

    def calcCollisionPoint(self, poly: np.ndarray) -> Union[list, None]:
        """
        Determine whether the target object is within the road boundaries.
        
        This method can work with both lane polygons and road boundary polygons,
        making it compatible with both lane detection and road segmentation approaches.

        Args:
            poly: is a polygonal points. [[x1, y1], [x2, y2], [x3, y3] ... [xn, yn]]
                 Can be either lane area polygon or road boundary contour.

        Returns:
            [Xcenter, Ybottom, distance] or None if no collision point
        """
        if len(self.distance_points) == 0 or poly is None or len(poly) == 0:
            return None
            
        sorted_distance_points = sorted(self.distance_points, key=lambda arr: arr[2])
        for x, y, d in sorted_distance_points:
            # Check if point is inside the polygon
            # This works with both lane area polygons and road boundary contours
            status = cv2.pointPolygonTest(poly, (x, y), False) >= 0
            
            # Alternative method using numpy array if poly is in the right format
            # status = self.__isInsidePolygon((x, y), np.squeeze(poly))
            
            if status:
                return [x, y, d]
        return None

    def DrawDetectedOnFrame(self, frame_show: cv2) -> None: 
        """
        Draw distance information on the frame for detected objects.
        
        Args:
            frame_show: The image to draw on
            
        Returns:
            None
        """
        if len(self.distance_points) != 0:
            for x, y, d in self.distance_points:
                # Draw a circle at the base of the object
                cv2.circle(frame_show, (x, y), 4, (255, 255, 255), thickness=-1)

                # Format distance text
                unit = 'm'
                if d < 0:
                    text = ' {} {}'.format("unknown", unit)
                else:
                    text = ' {:.2f} {}'.format(d, unit)
                
                # Scale font based on distance (closer objects get larger text)
                fontScale = max(0.4, min(1, 1/d))
                
                # Calculate text position
                textsize = cv2.getTextSize(text, 0, fontScale=fontScale, thickness=3)[0]
                textX = int((x - textsize[0]/2))
                textY = int((y + textsize[1]))

                # Draw text with shadow for better visibility
                putText_shadow(frame_show, text, (textX + 1, textY + 5), 
                              fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontScale,  
                              color=(255, 255, 255), thickness=1, 
                              shadow_color=(150, 150, 150))