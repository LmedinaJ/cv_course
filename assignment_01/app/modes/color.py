import cv2
import numpy as np
from typing import List
from .base_modes import BaseFilterMode
from ..utils import ImageUtils

class ColorConversionMode(BaseFilterMode):
    """Color conversion mode with RGB, Gray, and HSV options."""
    
    def __init__(self):
        super().__init__({
            'r': ('rgb', self._rgb_mode),
            'g': ('gray', self._gray_mode),
            'h': ('hsv', self._hsv_mode)
        })
    
    def _rgb_mode(self, frame: np.ndarray) -> np.ndarray:
        return frame
    
    def _gray_mode(self, frame: np.ndarray) -> np.ndarray:
        gray = ImageUtils.convert_to_grayscale(frame)
        return ImageUtils.grayscale_to_bgr(gray)
    
    def _hsv_mode(self, frame: np.ndarray) -> np.ndarray:
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(f"Error converting to HSV: {e}")
            return frame
    
    def switch_conversion(self, key: str) -> bool:
        return self.switch_option(key)
    
    def get_conversion_list(self) -> List[str]:
        return self.get_option_list()