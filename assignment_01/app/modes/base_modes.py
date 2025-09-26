import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable
from ..constants import Constants
from ..utils import TrackbarConfig

class BaseMode(ABC):
    """Abstract base class for all video processing modes."""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a video frame and return the result."""
        pass

class BaseFilterMode(BaseMode):
    """Base class for modes that have multiple sub-options."""
    
    def __init__(self, options: Dict[str, Tuple[str, Callable]]):
        self.options = options
        self.current_option = list(options.values())[0][0] if options else None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using current option."""
        for key, (option_name, option_func) in self.options.items():
            if option_name == self.current_option:
                return option_func(frame)
        return frame
    
    def switch_option(self, key: str) -> bool:
        """Switch to a different option."""
        if key in self.options:
            self.current_option = self.options[key][0]
            return True
        return False
    
    def get_option_list(self) -> List[str]:
        """Get list of available options."""
        return [f"Press '{key}' for {option_name}" for key, (option_name, _) in self.options.items()]

class TrackbarMode(BaseMode):
    """Base class for modes that use trackbars."""
    
    def __init__(self):
        self.trackbars_created = False
        self.trackbar_configs: List[TrackbarConfig] = []
    
    def create_trackbars(self) -> None:
        """Create trackbars based on configuration."""
        if not self.trackbars_created:
            try:
                cv2.namedWindow(Constants.CAMERA_WINDOW)
                for config in self.trackbar_configs:
                    cv2.createTrackbar(config.name, Constants.CAMERA_WINDOW, 
                                     config.initial_value, config.max_value, config.callback)
                self.trackbars_created = True
            except Exception as e:
                print(f"Error creating trackbars: {e}")
    
    def destroy_trackbars(self) -> None:
        """Destroy all trackbars."""
        if self.trackbars_created:
            try:
                cv2.destroyAllWindows()
                cv2.namedWindow(Constants.CAMERA_WINDOW)
                self.trackbars_created = False
            except Exception as e:
                print(f"Error destroying trackbars: {e}")