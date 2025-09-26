import numpy as np
from .base_modes import BaseMode

class NormalMode(BaseMode):
    """Normal mode - no processing."""
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame