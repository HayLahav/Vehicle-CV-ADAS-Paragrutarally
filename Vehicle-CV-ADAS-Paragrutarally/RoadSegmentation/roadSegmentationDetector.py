import cv2
import numpy as np
import logging
from typing import List, Tuple, Union

class RoadInfo:
    """Class to store road detection information"""
    def __init__(self):
        self.road_mask = None          # Binary mask of the drivable area
        self.road_boundary = None      # Contour points of the road boundary
        self.road_center = None        # Center line points of the road
        self.is_on_road = False        # Flag indicating if vehicle is on the road

class RoadSegmentationDetector:
    """Detector class for road segmentation using TwinLiteNet or similar models"""
    
    def __init__(self, model_path: str, logger=None):
        """Initialize the RoadSegmentationDetector
        
        Args:
            model_path: Path to the ONNX or TensorRT model
            logger: Logger object for debugging
        """
        self.logger = logger
        self.model_path = model_path
        self.road_info = RoadInfo()
        
        # Import here to avoid circular imports
        import onnxruntime
        
        try:
            # Load the ONNX model
            self.session = onnxruntime.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            
            if self.logger:
                self.logger.info(f"Road segmentation model loaded from {model_path}")
                self.logger.info(f"Input shape: {self.input_shape}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load model: {e}")
            else:
                print(f"Failed to load model: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image for the model
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed image ready for inference
        """
        # Get input dimensions from the model
        _, _, height, width = self.input_shape
        
        # Resize to model's expected input size
        img = cv2.resize(image, (width, height))
        
        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Transpose to get CHW format and add batch dimension
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def calculate_road_center(self, mask: np.ndarray) -> np.ndarray:
        """Calculate the center line of the road
        
        Args:
            mask: Binary mask of the road
            
        Returns:
            Array of center line points
        """
        # Skeletonize the mask to get center line
        # Simple approach: for each row, find the middle point between road edges
        center_points = []
        h, w = mask.shape
        
        for y in range(0, h, 10):  # Sample every 10 rows
            row = mask[y, :]
            road_pixels = np.where(row > 0)[0]
            if len(road_pixels) > 0:
                left_edge = road_pixels[0]
                right_edge = road_pixels[-1]
                center_x = (left_edge + right_edge) // 2
                center_points.append((center_x, y))
        
        return np.array(center_points)
    
    def DetectFrame(self, image: np.ndarray) -> None:
        """Detect road in the input frame
        
        Args:
            image: Input BGR image
        """
        # Get original dimensions
        original_h, original_w = image.shape[:2]
        
        # Preprocess the image
        input_data = self.preprocess_image(image)
        
        # Run inference
        results = self.session.run([self.output_name], {self.input_name: input_data})
        
        # Get the drivable area mask (first channel is usually drivable area)
        mask = results[0][0, 0]  # Shape: [batch, channel, height, width]
        
        # Convert to binary mask
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Resize mask to original image size
        resized_mask = cv2.resize(binary_mask, (original_w, original_h))
        
        # Apply morphological operations to clean the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours to get the road boundary
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Update road info
        self.road_info.road_mask = cleaned_mask
        
        # Get the largest contour (assuming it's the road)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            self.road_info.road_boundary = largest_contour
            
            # Calculate road center line
            self.road_info.road_center = self.calculate_road_center(cleaned_mask)
            
            # Check if vehicle is on the road (assuming vehicle is at the bottom center of the image)
            vehicle_position = (original_w // 2, original_h - 10)
            is_on_road = cv2.pointPolygonTest(largest_contour, vehicle_position, False) >= 0
            self.road_info.is_on_road = is_on_road
        else:
            self.road_info.road_boundary = None
            self.road_info.road_center = None
            self.road_info.is_on_road = False
    
    def DrawDetectedOnFrame(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Draw road detection on the frame
        
        Args:
            frame: Input BGR image
            alpha: Transparency of the overlay
            
        Returns:
            Frame with road detection visualization
        """
        if self.road_info.road_mask is not None:
            overlay = frame.copy()
            
            # Create a colored overlay for the road mask
            road_overlay = np.zeros_like(frame)
            road_overlay[self.road_info.road_mask > 0] = [0, 255, 0]  # Green for drivable area
            
            # Apply overlay
            cv2.addWeighted(road_overlay, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Draw road boundary
            if self.road_info.road_boundary is not None:
                cv2.drawContours(overlay, [self.road_info.road_boundary], -1, (0, 0, 255), 2)
            
            # Draw road center line
            if self.road_info.road_center is not None and len(self.road_info.road_center) > 1:
                for i in range(len(self.road_info.road_center) - 1):
                    pt1 = tuple(self.road_info.road_center[i])
                    pt2 = tuple(self.road_info.road_center[i + 1])
                    cv2.line(overlay, pt1, pt2, (255, 0, 0), 2)
            
            frame[:] = overlay
        
        return frame
    
    def DrawAreaOnFrame(self, frame: np.ndarray, color: tuple = (255, 191, 0), alpha: float = 0.85) -> np.ndarray:
        """Draw road area on the frame with specified color
        
        Args:
            frame: Input BGR image
            color: Color for the road area
            alpha: Transparency of the overlay
            
        Returns:
            Frame with road area visualization
        """
        if self.road_info.road_mask is not None:
            H, W, _ = frame.shape
            lane_segment_img = frame.copy()
            
            # Create a colored overlay for the road mask
            road_overlay = np.zeros_like(frame)
            road_overlay[self.road_info.road_mask > 0] = color
            
            # Apply overlay
            cv2.addWeighted(lane_segment_img, alpha, road_overlay, 1 - alpha, 0, lane_segment_img)
            frame[:H, :W, :] = lane_segment_img
        
        return frame