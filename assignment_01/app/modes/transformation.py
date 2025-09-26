import cv2
import numpy as np
from typing import Tuple
from ..constants import Constants
from ..utils import UIUtils, TrackbarConfig
from .base_modes import TrackbarMode


class TransformationMode(TrackbarMode):
    """Image transformation mode with translation, rotation, and scaling."""
    
    def __init__(self):
        super().__init__()
        self.translate_x = Constants.DEFAULT_TRANSLATE_X
        self.translate_y = Constants.DEFAULT_TRANSLATE_Y
        self.rotation = Constants.DEFAULT_ROTATION
        self.scale = Constants.DEFAULT_SCALE
        self._setup_trackbars()
    
    def _setup_trackbars(self) -> None:
        """Setup trackbar configurations for transformations."""
        self.trackbar_configs = [
            TrackbarConfig('Translate X', self.translate_x + Constants.MAX_TRANSLATE, 
                          Constants.MAX_TRANSLATE * 2, self._on_translate_x_change),
            TrackbarConfig('Translate Y', self.translate_y + Constants.MAX_TRANSLATE, 
                          Constants.MAX_TRANSLATE * 2, self._on_translate_y_change),
            TrackbarConfig('Rotation', self.rotation, Constants.MAX_ROTATION, self._on_rotation_change),
            TrackbarConfig('Scale %', self.scale, Constants.MAX_SCALE, self._on_scale_change)
        ]
    
    def _on_translate_x_change(self, val: int) -> None:
        """Handle translation X trackbar change."""
        self.translate_x = val - Constants.MAX_TRANSLATE  # Convert from 0-400 to -200 to +200
    
    def _on_translate_y_change(self, val: int) -> None:
        """Handle translation Y trackbar change."""
        self.translate_y = val - Constants.MAX_TRANSLATE  # Convert from 0-400 to -200 to +200
    
    def _on_rotation_change(self, val: int) -> None:
        """Handle rotation trackbar change."""
        self.rotation = val
    
    def _on_scale_change(self, val: int) -> None:
        """Handle scale trackbar change."""
        self.scale = max(Constants.MIN_SCALE, val)  # Ensure minimum scale
    
    def _get_transformation_matrix(self, frame_center: Tuple[float, float]) -> np.ndarray:
        """Calculate the combined transformation matrix."""
        try:
            cx, cy = frame_center
            
            # Create individual transformation matrices
            # 1. Translation matrix
            T = np.array([
                [1, 0, self.translate_x],
                [0, 1, self.translate_y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # 2. Rotation matrix (around center)
            angle_rad = np.radians(self.rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Translate to origin, rotate, translate back
            R = np.array([
                [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a],
                [sin_a, cos_a, cy * (1 - cos_a) - cx * sin_a],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # 3. Scale matrix (around center)
            scale_factor = self.scale / 100.0
            S = np.array([
                [scale_factor, 0, cx * (1 - scale_factor)],
                [0, scale_factor, cy * (1 - scale_factor)],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Combine transformations: T * R * S
            combined_matrix = T @ R @ S
            
            return combined_matrix
            
        except Exception as e:
            print(f"Error calculating transformation matrix: {e}")
            return np.eye(3, dtype=np.float32)
    
    def _draw_transformation_info(self, frame: np.ndarray) -> None:
        """Draw current transformation parameters on the frame."""
        try:
            info_lines = [
                f"Translation: ({self.translate_x:+d}, {self.translate_y:+d}) px",
                f"Rotation: {self.rotation}°",
                f"Scale: {self.scale}%"
            ]
            
            y_start = frame.shape[0] - 120
            for i, line in enumerate(info_lines):
                UIUtils.draw_text(frame, line, (10, y_start + i * 25), 
                                 Constants.FONT_SCALE_SMALL, Constants.COLOR_YELLOW, 
                                 Constants.FONT_THICKNESS_NORMAL)
                
        except Exception as e:
            print(f"Error drawing transformation info: {e}")
    
    def _draw_center_crosshair(self, frame: np.ndarray, center: Tuple[int, int]) -> None:
        """Draw a crosshair at the transformation center."""
        try:
            cx, cy = center
            crosshair_size = 20
            
            # Draw crosshair lines
            cv2.line(frame, (cx - crosshair_size, cy), (cx + crosshair_size, cy), 
                    Constants.COLOR_RED, 2)
            cv2.line(frame, (cx, cy - crosshair_size), (cx, cy + crosshair_size), 
                    Constants.COLOR_RED, 2)
            
            # Draw center circle
            cv2.circle(frame, (cx, cy), 3, Constants.COLOR_RED, -1)
            
        except Exception as e:
            print(f"Error drawing crosshair: {e}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply transformations to the frame."""
        try:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            
            # Get transformation matrix
            transform_matrix = self._get_transformation_matrix(center)
            
            # Apply transformation
            transformed = cv2.warpAffine(frame, transform_matrix[:2], (w, h), 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(0, 0, 0))
            
            # Draw transformation center crosshair on original position
            self._draw_center_crosshair(transformed, center)
            
            # Draw transformation info
            self._draw_transformation_info(transformed)
            
            return transformed
            
        except Exception as e:
            print(f"Error applying transformations: {e}")
            return frame