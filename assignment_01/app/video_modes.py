import numpy as np
from typing import List

from .constants import Constants
from .modes.normal import NormalMode
from .modes.color import ColorConversionMode
from .modes.contrast import ContrastBrightnessMode
from .modes.histogram import HistogramMode
from .modes.transformation import TransformationMode
from .modes.calibration import CameraCalibrationMode
from .modes.filters import FiltersMode
from .modes.panorama import PanoramaMode
from .modes.ar_mode import AugmentedRealityMode

class VideoModes:
    """Main video modes manager."""
    
    def __init__(self):
        self.main_modes = {
            '1': ('normal', NormalMode()),
            '2': ('color_conversion', ColorConversionMode()),
            '3': ('contrast_brightness', ContrastBrightnessMode()),
            '4': ('histogram', HistogramMode()),
            '5': ('transformation', TransformationMode()),
            '6': ('calibration', CameraCalibrationMode()),
            '7': ('filters', FiltersMode()),
            '8': ('panorama', PanoramaMode()),
            '9': ('AR', AugmentedRealityMode())
        }
        self.current_main_mode = 'normal'
        self.current_mode_obj = self.main_modes['1'][1]
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with current mode."""
        try:
            return self.current_mode_obj.process_frame(frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def switch_main_mode(self, key: str) -> bool:
        """Switch to a different main mode."""
        if key in self.main_modes:
            # Destroy trackbars from current mode before switching
            if hasattr(self.current_mode_obj, 'destroy_trackbars'):
                self.current_mode_obj.destroy_trackbars()
            
            self.current_main_mode = self.main_modes[key][0]
            self.current_mode_obj = self.main_modes[key][1]
            
            # Create trackbars for modes that need them
            if hasattr(self.current_mode_obj, 'create_trackbars'):
                self.current_mode_obj.create_trackbars()
            
            return True
        return False
    
    def handle_sub_mode_key(self, key: str) -> bool:
        """Handle sub-mode key presses."""
        try:
            if self.current_main_mode == 'filters':
                return self.current_mode_obj.switch_filter(key)
            elif self.current_main_mode == 'color_conversion':
                return self.current_mode_obj.switch_conversion(key)
            elif self.current_main_mode == 'calibration':
                return self.current_mode_obj.handle_key(key)
            elif self.current_main_mode == 'panorama':
                if key == 'c':
                    return True
                elif key == 'r':
                    self.current_mode_obj.reset()
                    return True
                elif key == 'p':
                    self.current_mode_obj.create_panorama()
                    return True
            return False
        except Exception as e:
            print(f"Error handling sub-mode key: {e}")
            return False
    
    def get_mode_info(self) -> List[str]:
        """Get current mode information for display."""
        info = []
        
        # Main mode options
        info.append("Main Modes:")
        info.extend([f"  {key}: {mode_name}" for key, (mode_name, _) in self.main_modes.items()])
        info.append(f"Current: {self.current_main_mode}")
        info.append("")
        
        # Sub-mode options
        try:
            if self.current_main_mode == 'filters':
                info.append("Filter Options:")
                info.append(f"  Current filter: {self.current_mode_obj.current_option}")
                info.extend([f"  {item}" for item in self.current_mode_obj.get_filter_list()])
                if self.current_mode_obj.current_option in ['blur', 'canny', 'bilateral', 'hough_lines']:
                    info.append("  Use trackbar sliders above for settings")
            elif self.current_main_mode == 'color_conversion':
                info.append("Color Conversion Options:")
                info.append(f"  Current: {self.current_mode_obj.current_option}")
                info.extend([f"  {item}" for item in self.current_mode_obj.get_conversion_list()])
            elif self.current_main_mode == 'contrast_brightness':
                info.append("Contrast & Brightness:")
                info.append(f"  Contrast: {self.current_mode_obj.contrast / 10.0:.1f}")
                info.append(f"  Brightness: {self.current_mode_obj.brightness}")
                info.append("  Use trackbar sliders above to adjust")
            elif self.current_main_mode == 'histogram':
                info.append("Histogram Display:")
                info.append("  Shows RGB channel histograms")
                info.append("  Blue (B), Green (G), Red (R) channels")
                info.append("  Overlayed in top-right corner")
            elif self.current_main_mode == 'transformation':
                info.append("Image Transformation:")
                info.append(f"  Translation: ({self.current_mode_obj.translate_x:+d}, {self.current_mode_obj.translate_y:+d}) px")
                info.append(f"  Rotation: {self.current_mode_obj.rotation}°")
                info.append(f"  Scale: {self.current_mode_obj.scale}%")
                info.append("  Use trackbar sliders above to adjust")
            elif self.current_main_mode == 'calibration':
                info.append("Camera Calibration:")
                if self.current_mode_obj.calibration_complete:
                    info.append(f"  Status: Complete! Error: {self.current_mode_obj.calibration_error:.4f}")
                    info.append("  Press 'r' to calibrate again")
                elif self.current_mode_obj.is_calibrating:
                    info.append(f"  Status: In Progress ({self.current_mode_obj.images_captured}/{Constants.TARGET_CALIBRATION_IMAGES})")
                    info.append("  Press 's' to stop")
                else:
                    info.append("  Status: Ready")
                    info.append("  Press 'c' to start calibration")
                    info.append("  Need 9x6 chessboard pattern")
            elif self.current_main_mode == 'panorama':
                info.append("Panorama Options:")
                info.extend([
                    "  Press 'c' to capture image",
                    "  Press 'r' to reset panorama",
                    "  Press 'p' to create panorama (needs 2+ images)"
                ])
            elif self.current_main_mode == 'T-Rex AR':
                info.append("T-Rex Augmented Reality Mode:")
                info.extend([
                    "  Detects ArUco markers from sources folder",
                    "  Renders 3D T-Rex model on markers",
                    f"  Model: {len(self.current_mode_obj.trex_model.vertices)} vertices",
                    "  Requires camera calibration for accuracy",
                    "  Use A4_ArUco_Marker.png from sources"
                ])
            elif self.current_main_mode == 'normal':
                info.append("Normal Mode - No additional options")
        except Exception as e:
            info.append(f"Error getting mode info: {e}")
        
        return info
    
    def get_main_mode_list(self) -> List[str]:
        """Get list of main modes."""
        return [f"Press '{key}' for {mode_name} mode" for key, (mode_name, _) in self.main_modes.items()]