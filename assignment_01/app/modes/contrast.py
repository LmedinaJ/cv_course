import cv2
import numpy as np
from .base_modes import TrackbarMode
from ..constants import Constants
from ..utils import TrackbarConfig, ImageUtils

class ContrastBrightnessMode(TrackbarMode):
    """Contrast and brightness adjustment mode."""
    
    def __init__(self):
        super().__init__()
        self.contrast = Constants.DEFAULT_CONTRAST
        self.brightness = Constants.DEFAULT_BRIGHTNESS
        self._setup_trackbars()
    
    def _setup_trackbars(self) -> None:
        """Setup trackbar configurations."""
        self.trackbar_configs = [
            TrackbarConfig('Contrast', self.contrast, Constants.MAX_CONTRAST, self._on_contrast_change),
            TrackbarConfig('Brightness', self.brightness + 100, Constants.MAX_BRIGHTNESS, self._on_brightness_change)
        ]
    
    def _on_contrast_change(self, val: int) -> None:
        self.contrast = ImageUtils.validate_range(val, 1)
    
    def _on_brightness_change(self, val: int) -> None:
        self.brightness = val - 100
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            alpha = self.contrast / 10.0
            beta = self.brightness
            return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        except Exception as e:
            print(f"Error applying contrast/brightness: {e}")
            return frame