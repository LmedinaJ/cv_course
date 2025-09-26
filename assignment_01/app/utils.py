import cv2
import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
from .constants import Constants

@dataclass
class TrackbarConfig:
    name: str
    initial_value: int
    max_value: int
    callback: Callable[[int], None]

class UIUtils:
    @staticmethod
    def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int], 
                  font_scale: float = Constants.FONT_SCALE_MEDIUM, 
                  color: Tuple[int, int, int] = Constants.UI_MENU_TEXT,
                  thickness: int = Constants.FONT_THICKNESS_NORMAL) -> None:
        """Draw text on frame with consistent styling."""
        try:
            cv2.putText(frame, text, position, Constants.FONT, font_scale, color, thickness)
        except Exception as e:
            print(f"Error drawing text '{text}': {e}")
    
    @staticmethod
    def draw_text_list(frame: np.ndarray, text_list: List[str], 
                      start_y: int = Constants.UI_TEXT_Y_OFFSET,
                      font_scale: float = Constants.FONT_SCALE_LARGE,
                      color: Tuple[int, int, int] = Constants.UI_MENU_TEXT) -> None:
        """Draw a list of text lines with consistent spacing."""
        y_offset = start_y
        for text in text_list:
            UIUtils.draw_text(frame, text, (10, y_offset), font_scale, color, Constants.FONT_THICKNESS_BOLD)
            y_offset += Constants.UI_TEXT_Y_SPACING

class ImageUtils:
    @staticmethod
    def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error converting to grayscale: {e}")
            return frame
    
    @staticmethod
    def grayscale_to_bgr(gray_frame: np.ndarray) -> np.ndarray:
        """Convert grayscale frame back to BGR for display."""
        try:
            return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Error converting grayscale to BGR: {e}")
            return gray_frame
    
    @staticmethod
    def ensure_odd_kernel_size(size: int, minimum: int = 1) -> int:
        """Ensure kernel size is odd and >= minimum."""
        size = max(minimum, size)
        return size if size % 2 == 1 else size + 1
    
    @staticmethod
    def validate_range(value: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
        """Validate value is within specified range."""
        value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value