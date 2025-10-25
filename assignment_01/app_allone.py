import cv2
import numpy as np
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

# Constants
class Constants:
    # Window settings
    CAMERA_WINDOW = 'Camera'
    
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.7
    FONT_SCALE_MEDIUM = 0.6
    FONT_SCALE_SMALL = 0.5
    FONT_SCALE_TINY = 0.4
    FONT_SCALE_MICRO = 0.3
    FONT_THICKNESS_BOLD = 2
    FONT_THICKNESS_NORMAL = 1
    
    # Colors (BGR format)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_GRAY = (200, 200, 200)
    COLOR_BLACK = (0, 0, 0)
    
    # Histogram settings
    HIST_HEIGHT = 200
    HIST_WIDTH = 512
    HIST_BINS = 256
    HIST_RANGE = [0, 256]
    HIST_POSITION_X_OFFSET = 20
    HIST_POSITION_Y = 80
    HIST_BACKGROUND_ALPHA = 0.3
    HIST_LINE_THICKNESS = 2
    
    # Default filter parameters
    DEFAULT_BLUR_KERNEL = 15
    DEFAULT_CANNY_LOW = 100
    DEFAULT_CANNY_HIGH = 200
    DEFAULT_BILATERAL_D = 9
    DEFAULT_BILATERAL_SIGMA_COLOR = 75
    DEFAULT_BILATERAL_SIGMA_SPACE = 75
    DEFAULT_HOUGH_THRESHOLD = 100
    DEFAULT_HOUGH_MIN_LINE_LENGTH = 50
    DEFAULT_HOUGH_MAX_LINE_GAP = 10
    DEFAULT_CONTRAST = 10
    DEFAULT_BRIGHTNESS = 0
    
    # Default transformation parameters
    DEFAULT_TRANSLATE_X = 0
    DEFAULT_TRANSLATE_Y = 0
    DEFAULT_ROTATION = 0
    DEFAULT_SCALE = 100
    
    # Trackbar limits
    MAX_BLUR_KERNEL = 50
    MAX_CANNY_THRESHOLD = 300
    MAX_BILATERAL_D = 25
    MAX_BILATERAL_SIGMA = 200
    MAX_HOUGH_THRESHOLD = 200
    MAX_HOUGH_LINE_LENGTH = 200
    MAX_HOUGH_LINE_GAP = 50
    MAX_CONTRAST = 30
    MAX_BRIGHTNESS = 200
    
    # Transformation limits
    MAX_TRANSLATE = 200  # -200 to +200 pixels
    MAX_ROTATION = 360   # 0 to 360 degrees
    MIN_SCALE = 10       # 10% minimum scale
    MAX_SCALE = 300      # 300% maximum scale
    
    # Camera calibration settings
    CHESSBOARD_SIZE = (9, 6)  # Number of internal corners (width, height)
    SQUARE_SIZE_MM = 25       # Real-world size of chessboard square in mm
    TARGET_CALIBRATION_IMAGES = 20  # Number of images needed for calibration
    CALIBRATION_CAPTURE_DELAY = 2   # Seconds between automatic captures
    CALIBRATION_FILE = 'assigment_01/sources/calibration.npz'  # Output file for calibration data
    
    # UI layout
    UI_TEXT_Y_OFFSET = 30
    UI_TEXT_Y_SPACING = 35
    UI_QUIT_TEXT_Y_OFFSET = 20

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
                  color: Tuple[int, int, int] = Constants.COLOR_GREEN,
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
                      color: Tuple[int, int, int] = Constants.COLOR_GREEN) -> None:
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

class HistogramRenderer:
    """Handles histogram visualization separately from histogram calculation."""
    
    def __init__(self):
        self.hist_h = Constants.HIST_HEIGHT
        self.hist_w = Constants.HIST_WIDTH
        self.bin_w = int(round(self.hist_w / Constants.HIST_BINS))
    
    def calculate_histograms(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate histograms for BGR channels."""
        try:
            hist_b = cv2.calcHist([frame], [0], None, [Constants.HIST_BINS], Constants.HIST_RANGE)
            hist_g = cv2.calcHist([frame], [1], None, [Constants.HIST_BINS], Constants.HIST_RANGE)
            hist_r = cv2.calcHist([frame], [2], None, [Constants.HIST_BINS], Constants.HIST_RANGE)
            return hist_b, hist_g, hist_r
        except Exception as e:
            print(f"Error calculating histograms: {e}")
            return np.zeros((Constants.HIST_BINS, 1)), np.zeros((Constants.HIST_BINS, 1)), np.zeros((Constants.HIST_BINS, 1))
    
    def create_histogram_image(self, hist_b: np.ndarray, hist_g: np.ndarray, hist_r: np.ndarray) -> np.ndarray:
        """Create histogram visualization image."""
        hist_img = np.zeros((self.hist_h, self.hist_w, 3), dtype=np.uint8)
        
        # Normalize histograms
        cv2.normalize(hist_b, hist_b, 0, self.hist_h, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, self.hist_h, cv2.NORM_MINMAX)
        cv2.normalize(hist_r, hist_r, 0, self.hist_h, cv2.NORM_MINMAX)
        
        # Draw histogram lines
        for i in range(1, Constants.HIST_BINS):
            # Blue channel
            cv2.line(hist_img,
                    (self.bin_w * (i-1), self.hist_h - int(hist_b[i-1])),
                    (self.bin_w * i, self.hist_h - int(hist_b[i])),
                    Constants.COLOR_BLUE, Constants.HIST_LINE_THICKNESS)
            # Green channel
            cv2.line(hist_img,
                    (self.bin_w * (i-1), self.hist_h - int(hist_g[i-1])),
                    (self.bin_w * i, self.hist_h - int(hist_g[i])),
                    Constants.COLOR_GREEN, Constants.HIST_LINE_THICKNESS)
            # Red channel
            cv2.line(hist_img,
                    (self.bin_w * (i-1), self.hist_h - int(hist_r[i-1])),
                    (self.bin_w * i, self.hist_h - int(hist_r[i])),
                    Constants.COLOR_RED, Constants.HIST_LINE_THICKNESS)
        
        return hist_img
    
    def overlay_histogram(self, frame: np.ndarray, hist_img: np.ndarray) -> np.ndarray:
        """Overlay histogram on frame with labels."""
        frame_h, frame_w = frame.shape[:2]
        
        # Position histogram
        start_x = frame_w - self.hist_w - Constants.HIST_POSITION_X_OFFSET
        start_y = Constants.HIST_POSITION_Y
        end_x = start_x + self.hist_w
        end_y = start_y + self.hist_h
        
        # Check bounds
        if end_x <= frame_w and end_y + 40 <= frame_h:
            # Add semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (start_x - 10, start_y - 25), (end_x + 10, end_y + 35), Constants.COLOR_BLACK, -1)
            cv2.addWeighted(overlay, Constants.HIST_BACKGROUND_ALPHA, frame, 0.7, 0, frame)
            
            # Add histogram
            frame[start_y:end_y, start_x:end_x] = hist_img
            
            # Add labels
            self._draw_histogram_labels(frame, start_x, start_y, end_x, end_y)
        
        return frame
    
    def _draw_histogram_labels(self, frame: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """Draw histogram labels and axis information."""
        # Title
        UIUtils.draw_text(frame, "Histogram RGB", (start_x, start_y - 10), 
                         Constants.FONT_SCALE_MEDIUM, Constants.COLOR_WHITE, Constants.FONT_THICKNESS_BOLD)
        
        # Y-axis label
        UIUtils.draw_text(frame, "Freq", (start_x - 35, start_y + self.hist_h//2), 
                         Constants.FONT_SCALE_TINY, Constants.COLOR_WHITE)
        
        # X-axis label
        UIUtils.draw_text(frame, "Intensity", (start_x + self.hist_w//2 - 30, end_y + 30), 
                         Constants.FONT_SCALE_TINY, Constants.COLOR_WHITE)
        
        # X-axis ticks
        UIUtils.draw_text(frame, "0", (start_x - 5, end_y + 15), 
                         Constants.FONT_SCALE_MICRO, Constants.COLOR_GRAY)
        UIUtils.draw_text(frame, "128", (start_x + self.hist_w//2 - 10, end_y + 15), 
                         Constants.FONT_SCALE_MICRO, Constants.COLOR_GRAY)
        UIUtils.draw_text(frame, "255", (start_x + self.hist_w - 15, end_y + 15), 
                         Constants.FONT_SCALE_MICRO, Constants.COLOR_GRAY)
        
        # Color channel legend
        UIUtils.draw_text(frame, "B", (start_x + 10, end_y + 15), 
                         Constants.FONT_SCALE_SMALL, Constants.COLOR_BLUE, Constants.FONT_THICKNESS_BOLD)
        UIUtils.draw_text(frame, "G", (start_x + 30, end_y + 15), 
                         Constants.FONT_SCALE_SMALL, Constants.COLOR_GREEN, Constants.FONT_THICKNESS_BOLD)
        UIUtils.draw_text(frame, "R", (start_x + 50, end_y + 15), 
                         Constants.FONT_SCALE_SMALL, Constants.COLOR_RED, Constants.FONT_THICKNESS_BOLD)

class NormalMode(BaseMode):
    """Normal mode - no processing."""
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

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

class HistogramMode(BaseMode):
    """Histogram display mode."""
    
    def __init__(self):
        self.renderer = HistogramRenderer()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        hist_b, hist_g, hist_r = self.renderer.calculate_histograms(frame)
        hist_img = self.renderer.create_histogram_image(hist_b, hist_g, hist_r)
        return self.renderer.overlay_histogram(frame, hist_img)

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

class CameraCalibrationMode(BaseMode):
    """Camera calibration mode using chessboard pattern."""
    
    def __init__(self):
        self.is_calibrating = False
        self.calibration_complete = False
        self.images_captured = 0
        self.last_capture_time = 0.0
        
        # Calibration data storage
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_error = 0.0
        
        # Setup calibration parameters
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._setup_object_points()
    
    def _setup_object_points(self) -> None:
        """Setup 3D object points for chessboard pattern."""
        try:
            # Create 3D points for chessboard corners
            self.objp = np.zeros((Constants.CHESSBOARD_SIZE[0] * Constants.CHESSBOARD_SIZE[1], 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:Constants.CHESSBOARD_SIZE[0], 
                                       0:Constants.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
            # Scale to actual square size in mm
            self.objp = self.objp * Constants.SQUARE_SIZE_MM
        except Exception as e:
            print(f"Error setting up object points: {e}")
    
    def start_calibration(self) -> None:
        """Start the calibration process."""
        self.is_calibrating = True
        self.calibration_complete = False
        self.images_captured = 0
        self.objpoints.clear()
        self.imgpoints.clear()
        self.last_capture_time = time.time()
        print("Starting camera calibration...")
        print(f"Show the {Constants.CHESSBOARD_SIZE} chessboard to the camera from various angles.")
    
    def stop_calibration(self) -> None:
        """Stop the calibration process."""
        self.is_calibrating = False
        print("Calibration stopped.")
    
    def reset_calibration(self) -> None:
        """Reset calibration data."""
        self.is_calibrating = False
        self.calibration_complete = False
        self.images_captured = 0
        self.objpoints.clear()
        self.imgpoints.clear()
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_error = 0.0
        print("Calibration data reset.")
    
    def _detect_and_capture_chessboard(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect chessboard pattern and capture if found."""
        try:
            gray = ImageUtils.convert_to_grayscale(frame)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, Constants.CHESSBOARD_SIZE, None)
            
            if ret and corners is not None:
                # Draw corners for visual feedback
                display_frame = frame.copy()
                cv2.drawChessboardCorners(display_frame, Constants.CHESSBOARD_SIZE, corners, ret)
                
                # Capture image if enough time has passed
                current_time = time.time()
                if (current_time - self.last_capture_time > Constants.CALIBRATION_CAPTURE_DELAY and 
                    self.images_captured < Constants.TARGET_CALIBRATION_IMAGES):
                    
                    # Refine corner locations to sub-pixel accuracy
                    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    
                    # Store object and image points
                    self.objpoints.append(self.objp)
                    self.imgpoints.append(refined_corners)
                    
                    self.images_captured += 1
                    self.last_capture_time = current_time
                    
                    print(f"Captured calibration image {self.images_captured}/{Constants.TARGET_CALIBRATION_IMAGES}")
                
                return True, display_frame
            else:
                return False, frame
                
        except Exception as e:
            print(f"Error detecting chessboard: {e}")
            return False, frame
    
    def _perform_calibration(self, frame_shape: Tuple[int, int]) -> bool:
        """Perform camera calibration calculation."""
        try:
            print("Performing camera calibration...")
            
            # Perform calibration
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, frame_shape[::-1], None, None)
            
            if ret:
                self.camera_matrix = mtx
                self.distortion_coeffs = dist
                
                # Calculate re-projection error
                total_error = 0
                for i in range(len(self.objpoints)):
                    imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    total_error += error
                
                self.calibration_error = total_error / len(self.objpoints)
                
                # Save calibration data
                self._save_calibration_data(mtx, dist, rvecs, tvecs)
                
                self.calibration_complete = True
                self.is_calibrating = False
                
                print(f"Calibration successful! Re-projection error: {self.calibration_error:.4f}")
                return True
            else:
                print("Calibration failed!")
                return False
                
        except Exception as e:
            print(f"Error during calibration: {e}")
            return False
    
    def _save_calibration_data(self, mtx: np.ndarray, dist: np.ndarray, 
                              rvecs: List[np.ndarray], tvecs: List[np.ndarray]) -> None:
        """Save calibration data to file."""
        try:
            np.savez(Constants.CALIBRATION_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print(f"Calibration data saved to '{Constants.CALIBRATION_FILE}'")
        except Exception as e:
            print(f"Error saving calibration data: {e}")
    
    def _draw_calibration_info(self, frame: np.ndarray, pattern_detected: bool) -> None:
        """Draw calibration information on frame (positioned on the right side)."""
        try:
            frame_h, frame_w = frame.shape[:2]
            
            # Position text on the right side to avoid menu overlap
            x_position = frame_w - 400  # 400 pixels from right edge
            y_offset = 30
            
            if self.calibration_complete:
                # Show calibration results
                UIUtils.draw_text(frame, "Calibration Complete!", (x_position, y_offset), 
                                 Constants.FONT_SCALE_LARGE, Constants.COLOR_GREEN, 
                                 Constants.FONT_THICKNESS_BOLD)
                y_offset += 40
                
                UIUtils.draw_text(frame, f"Re-projection error: {self.calibration_error:.4f}", 
                                 (x_position, y_offset), Constants.FONT_SCALE_MEDIUM, Constants.COLOR_GREEN)
                y_offset += 30
                
                UIUtils.draw_text(frame, f"Data saved to: {Constants.CALIBRATION_FILE}", 
                                 (x_position, y_offset), Constants.FONT_SCALE_SMALL, Constants.COLOR_GREEN)
                y_offset += 30
                
                UIUtils.draw_text(frame, "Press 'r' to reset and calibrate again", 
                                 (x_position, y_offset), Constants.FONT_SCALE_SMALL, Constants.COLOR_YELLOW)
                
            elif self.is_calibrating:
                # Show calibration progress
                UIUtils.draw_text(frame, "Camera Calibration in Progress", (x_position, y_offset), 
                                 Constants.FONT_SCALE_LARGE, Constants.COLOR_BLUE, 
                                 Constants.FONT_THICKNESS_BOLD)
                y_offset += 40
                
                UIUtils.draw_text(frame, f"Images: {self.images_captured}/{Constants.TARGET_CALIBRATION_IMAGES}", 
                                 (x_position, y_offset), Constants.FONT_SCALE_MEDIUM, 
                                 Constants.COLOR_GREEN if pattern_detected else Constants.COLOR_RED,
                                 Constants.FONT_THICKNESS_BOLD)
                y_offset += 30
                
                status_text = "Pattern detected - capturing..." if pattern_detected else "Show chessboard pattern"
                status_color = Constants.COLOR_GREEN if pattern_detected else Constants.COLOR_YELLOW
                UIUtils.draw_text(frame, status_text, (x_position, y_offset), 
                                 Constants.FONT_SCALE_MEDIUM, status_color)
                y_offset += 30
                
                UIUtils.draw_text(frame, "Press 's' to stop calibration", 
                                 (x_position, y_offset), Constants.FONT_SCALE_SMALL, Constants.COLOR_YELLOW)
                
            else:
                # Show calibration instructions
                UIUtils.draw_text(frame, "Camera Calibration Mode", (x_position, y_offset), 
                                 Constants.FONT_SCALE_LARGE, Constants.COLOR_BLUE, 
                                 Constants.FONT_THICKNESS_BOLD)
                y_offset += 40
                
                instructions = [
                    "Instructions:",
                    "1. Print a 9x6 chessboard pattern",
                    "2. Press 'c' to start calibration",
                    "3. Show chessboard from various angles",
                    f"4. Need {Constants.TARGET_CALIBRATION_IMAGES} good captures",
                    "5. Calibration runs automatically"
                ]
                
                for instruction in instructions:
                    UIUtils.draw_text(frame, instruction, (x_position, y_offset), 
                                     Constants.FONT_SCALE_SMALL, Constants.COLOR_WHITE)
                    y_offset += 25
            
        except Exception as e:
            print(f"Error drawing calibration info: {e}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for camera calibration."""
        try:
            pattern_detected = False
            result_frame = frame.copy()
            
            if self.is_calibrating and not self.calibration_complete:
                # Try to detect and capture chessboard
                pattern_detected, processed_frame = self._detect_and_capture_chessboard(frame)
                if processed_frame is not None:
                    result_frame = processed_frame
                
                # Check if we have enough images to calibrate
                if self.images_captured >= Constants.TARGET_CALIBRATION_IMAGES:
                    self._perform_calibration(frame.shape[:2])
            
            # Draw calibration information
            self._draw_calibration_info(result_frame, pattern_detected)
            
            return result_frame
            
        except Exception as e:
            print(f"Error processing calibration frame: {e}")
            return frame
    
    def handle_key(self, key: str) -> bool:
        """Handle key presses for calibration control."""
        try:
            if key == 'c' and not self.is_calibrating and not self.calibration_complete:
                self.start_calibration()
                return True
            elif key == 's' and self.is_calibrating:
                self.stop_calibration()
                return True
            elif key == 'r':
                self.reset_calibration()
                return True
            return False
        except Exception as e:
            print(f"Error handling calibration key: {e}")
            return False

class FiltersMode(BaseFilterMode, TrackbarMode):
    """Image filters mode with multiple filter options."""
    
    def __init__(self):
        BaseFilterMode.__init__(self, {
            'g': ('gray', self._gray_filter),
            'b': ('blur', self._blur_filter),
            'c': ('canny', self._canny_filter),
            'd': ('bilateral', self._bilateral_filter),
            'h': ('hough_lines', self._hough_lines_filter)
        })
        TrackbarMode.__init__(self)
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        """Initialize filter parameters."""
        self.blur_kernel = Constants.DEFAULT_BLUR_KERNEL
        self.canny_low = Constants.DEFAULT_CANNY_LOW
        self.canny_high = Constants.DEFAULT_CANNY_HIGH
        self.bilateral_d = Constants.DEFAULT_BILATERAL_D
        self.bilateral_sigma_color = Constants.DEFAULT_BILATERAL_SIGMA_COLOR
        self.bilateral_sigma_space = Constants.DEFAULT_BILATERAL_SIGMA_SPACE
        self.hough_threshold = Constants.DEFAULT_HOUGH_THRESHOLD
        self.hough_min_line_length = Constants.DEFAULT_HOUGH_MIN_LINE_LENGTH
        self.hough_max_line_gap = Constants.DEFAULT_HOUGH_MAX_LINE_GAP
    
    def _setup_trackbars_for_filter(self, filter_name: str) -> None:
        """Setup trackbars for specific filter."""
        self.trackbar_configs.clear()
        
        if filter_name == 'blur':
            self.trackbar_configs = [
                TrackbarConfig('Blur Kernel', self.blur_kernel, Constants.MAX_BLUR_KERNEL, self._on_blur_change)
            ]
        elif filter_name == 'canny':
            self.trackbar_configs = [
                TrackbarConfig('Canny Low', self.canny_low, Constants.MAX_CANNY_THRESHOLD, self._on_canny_low_change),
                TrackbarConfig('Canny High', self.canny_high, Constants.MAX_CANNY_THRESHOLD, self._on_canny_high_change)
            ]
        elif filter_name == 'bilateral':
            self.trackbar_configs = [
                TrackbarConfig('Diameter', self.bilateral_d, Constants.MAX_BILATERAL_D, self._on_bilateral_d_change),
                TrackbarConfig('Sigma Color', self.bilateral_sigma_color, Constants.MAX_BILATERAL_SIGMA, self._on_bilateral_sigma_color_change),
                TrackbarConfig('Sigma Space', self.bilateral_sigma_space, Constants.MAX_BILATERAL_SIGMA, self._on_bilateral_sigma_space_change)
            ]
        elif filter_name == 'hough_lines':
            self.trackbar_configs = [
                TrackbarConfig('Threshold', self.hough_threshold, Constants.MAX_HOUGH_THRESHOLD, self._on_hough_threshold_change),
                TrackbarConfig('Min Line Length', self.hough_min_line_length, Constants.MAX_HOUGH_LINE_LENGTH, self._on_hough_min_line_length_change),
                TrackbarConfig('Max Line Gap', self.hough_max_line_gap, Constants.MAX_HOUGH_LINE_GAP, self._on_hough_max_line_gap_change)
            ]
    
    def switch_filter(self, key: str) -> bool:
        """Switch filter and update trackbars."""
        if self.switch_option(key):
            self._setup_trackbars_for_filter(self.current_option)
            if self.trackbars_created:
                self.destroy_trackbars()
                self.create_trackbars()
            return True
        return False
    
    def get_filter_list(self) -> List[str]:
        return self.get_option_list()
    
    # Trackbar callbacks
    def _on_blur_change(self, val: int) -> None:
        self.blur_kernel = ImageUtils.ensure_odd_kernel_size(val, 1)
    
    def _on_canny_low_change(self, val: int) -> None:
        self.canny_low = val
    
    def _on_canny_high_change(self, val: int) -> None:
        self.canny_high = val
    
    def _on_bilateral_d_change(self, val: int) -> None:
        self.bilateral_d = ImageUtils.validate_range(val, 1)
    
    def _on_bilateral_sigma_color_change(self, val: int) -> None:
        self.bilateral_sigma_color = ImageUtils.validate_range(val, 1)
    
    def _on_bilateral_sigma_space_change(self, val: int) -> None:
        self.bilateral_sigma_space = ImageUtils.validate_range(val, 1)
    
    def _on_hough_threshold_change(self, val: int) -> None:
        self.hough_threshold = ImageUtils.validate_range(val, 1)
    
    def _on_hough_min_line_length_change(self, val: int) -> None:
        self.hough_min_line_length = ImageUtils.validate_range(val, 1)
    
    def _on_hough_max_line_gap_change(self, val: int) -> None:
        self.hough_max_line_gap = ImageUtils.validate_range(val, 1)
    
    # Filter implementations
    def _gray_filter(self, frame: np.ndarray) -> np.ndarray:
        gray = ImageUtils.convert_to_grayscale(frame)
        return ImageUtils.grayscale_to_bgr(gray)
    
    def _blur_filter(self, frame: np.ndarray) -> np.ndarray:
        try:
            kernel_size = ImageUtils.ensure_odd_kernel_size(self.blur_kernel)
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        except Exception as e:
            print(f"Error applying blur filter: {e}")
            return frame
    
    def _canny_filter(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray = ImageUtils.convert_to_grayscale(frame)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            return ImageUtils.grayscale_to_bgr(edges)
        except Exception as e:
            print(f"Error applying Canny filter: {e}")
            return frame
    
    def _bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        try:
            return cv2.bilateralFilter(frame, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
        except Exception as e:
            print(f"Error applying bilateral filter: {e}")
            return frame
    
    def _hough_lines_filter(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray = ImageUtils.convert_to_grayscale(frame)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi/180,
                threshold=self.hough_threshold,
                minLineLength=self.hough_min_line_length,
                maxLineGap=self.hough_max_line_gap
            )
            
            result = frame.copy()
            line_count = 0
            
            if lines is not None:
                line_count = len(lines)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(result, (x1, y1), (x2, y2), Constants.COLOR_GREEN, 2)
            
            # Draw line count
            color = Constants.COLOR_GREEN if line_count > 0 else Constants.COLOR_RED
            UIUtils.draw_text(result, f"Lines: {line_count}", (10, 30), 
                             Constants.FONT_SCALE_LARGE, color, Constants.FONT_THICKNESS_BOLD)
            
            return result
        except Exception as e:
            print(f"Error applying Hough lines filter: {e}")
            return frame

class CustomPanoramaStitcher:
    """Custom panorama stitching implementation without OpenCV's built-in stitcher."""
    
    def __init__(self):
        self.detector = None
        self.matcher = None
        try:
            # Use SIFT detector for feature detection
            self.detector = cv2.SIFT_create()
            # Use FLANN matcher for feature matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        except Exception as e:
            print(f"Error initializing feature detector/matcher: {e}")
    
    def detect_and_compute_features(self, image: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors for an image."""
        try:
            if self.detector is None:
                return None, None
            
            gray = ImageUtils.convert_to_grayscale(image)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            return keypoints, descriptors
        except Exception as e:
            print(f"Error detecting features: {e}")
            return None, None
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between two images."""
        try:
            if self.matcher is None or desc1 is None or desc2 is None:
                return []
            
            # Find matches using FLANN matcher
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            return good_matches
        except Exception as e:
            print(f"Error matching features: {e}")
            return []
    
    def find_homography_ransac(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                              max_iterations: int = 1000, threshold: float = 4.0) -> Optional[np.ndarray]:
        """Find homography matrix using RANSAC algorithm."""
        try:
            if len(src_pts) < 4 or len(dst_pts) < 4:
                return None
            
            best_homography = None
            best_inliers = 0
            
            for _ in range(max_iterations):
                # Randomly select 4 points
                indices = np.random.choice(len(src_pts), 4, replace=False)
                src_sample = src_pts[indices]
                dst_sample = dst_pts[indices]
                
                # Compute homography from 4 points
                H = self.compute_homography_4points(src_sample, dst_sample)
                if H is None:
                    continue
                
                # Count inliers
                inliers = self.count_inliers(src_pts, dst_pts, H, threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_homography = H
            
            return best_homography
        except Exception as e:
            print(f"Error computing homography: {e}")
            return None
    
    def compute_homography_4points(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """Compute homography matrix from 4 corresponding points."""
        try:
            # Set up the linear system Ah = 0
            A = []
            for i in range(4):
                x, y = src_pts[i]
                u, v = dst_pts[i]
                A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
                A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
            
            A = np.array(A)
            
            # Solve using SVD
            U, S, Vt = np.linalg.svd(A)
            h = Vt[-1, :]
            
            # Reshape to 3x3 matrix
            H = h.reshape((3, 3))
            
            # Normalize
            if H[2, 2] != 0:
                H = H / H[2, 2]
            
            return H
        except Exception as e:
            print(f"Error computing 4-point homography: {e}")
            return None
    
    def count_inliers(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                     H: np.ndarray, threshold: float) -> int:
        """Count number of inlier points for given homography."""
        try:
            # Transform source points using homography
            ones = np.ones((len(src_pts), 1))
            src_homogeneous = np.hstack([src_pts, ones])
            transformed = H @ src_homogeneous.T
            
            # Convert from homogeneous coordinates
            transformed_pts = transformed[:2, :] / transformed[2, :]
            transformed_pts = transformed_pts.T
            
            # Calculate distances
            distances = np.linalg.norm(dst_pts - transformed_pts, axis=1)
            
            # Count inliers
            return np.sum(distances < threshold)
        except Exception as e:
            print(f"Error counting inliers: {e}")
            return 0
    
    def warp_image(self, image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """Warp image using homography matrix."""
        try:
            return cv2.warpPerspective(image, H, output_shape)
        except Exception as e:
            print(f"Error warping image: {e}")
            return image
    
    def blend_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Blend two images using simple averaging where they overlap."""
        try:
            # Create masks for valid regions
            mask1 = np.any(img1 > 0, axis=2).astype(np.float32)
            mask2 = np.any(img2 > 0, axis=2).astype(np.float32)
            
            # Find overlap region
            overlap = mask1 * mask2
            
            # Create blended image
            result = np.zeros_like(img1, dtype=np.float32)
            
            # Areas with only img1
            only_img1 = (mask1 > 0) & (mask2 == 0)
            result[only_img1] = img1[only_img1]
            
            # Areas with only img2
            only_img2 = (mask1 == 0) & (mask2 > 0)
            result[only_img2] = img2[only_img2]
            
            # Overlap areas - blend with weights
            overlap_mask = overlap > 0
            if np.any(overlap_mask):
                # Simple averaging in overlap region
                result[overlap_mask] = (img1[overlap_mask].astype(np.float32) + 
                                      img2[overlap_mask].astype(np.float32)) / 2
            
            return result.astype(np.uint8)
        except Exception as e:
            print(f"Error blending images: {e}")
            return img1
    
    def stitch_images(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """Stitch multiple images into a panorama."""
        try:
            if len(images) < 2:
                print("Need at least 2 images for stitching")
                return None
            
            print(f"Starting panorama stitching with {len(images)} images...")
            
            # Start with the first image
            result = images[0].copy()
            
            for i in range(1, len(images)):
                print(f"Stitching image {i+1}/{len(images)}...")
                
                # Detect features in current result and next image
                kp1, desc1 = self.detect_and_compute_features(result)
                kp2, desc2 = self.detect_and_compute_features(images[i])
                
                if desc1 is None or desc2 is None:
                    print(f"Failed to detect features in image {i+1}")
                    continue
                
                # Match features
                matches = self.match_features(desc1, desc2)
                
                if len(matches) < 10:
                    print(f"Not enough matches found for image {i+1} ({len(matches)} matches)")
                    continue
                
                print(f"Found {len(matches)} feature matches")
                
                # Extract matched points
                src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                
                # Find homography
                H = self.find_homography_ransac(src_pts, dst_pts)
                
                if H is None:
                    print(f"Failed to compute homography for image {i+1}")
                    continue
                
                # Calculate output canvas size
                h1, w1 = result.shape[:2]
                h2, w2 = images[i].shape[:2]
                
                # Transform corners of second image to find bounding box
                corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, H)
                
                # Find bounding box
                all_corners = np.concatenate([
                    np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
                    transformed_corners
                ])
                
                x_min = int(np.min(all_corners[:, 0, 0]))
                x_max = int(np.max(all_corners[:, 0, 0]))
                y_min = int(np.min(all_corners[:, 0, 1]))
                y_max = int(np.max(all_corners[:, 0, 1]))
                
                # Adjust for negative coordinates
                translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
                H_adjusted = translation @ H
                
                # Calculate output size
                output_width = x_max - x_min
                output_height = y_max - y_min
                
                # Warp second image
                warped = self.warp_image(images[i], H_adjusted, (output_width, output_height))
                
                # Warp first image (result) with translation only
                result_warped = self.warp_image(result, translation, (output_width, output_height))
                
                # Blend images
                result = self.blend_images(result_warped, warped)
                
                print(f"Successfully stitched image {i+1}")
            
            print("Panorama stitching completed!")
            return result
            
        except Exception as e:
            print(f"Error during stitching: {e}")
            return None

class PanoramaMode(BaseMode):
    """Panorama creation mode with custom stitching."""
    
    def __init__(self):
        self.images: List[np.ndarray] = []
        self.stitcher = CustomPanoramaStitcher()
        self.current_panorama: Optional[np.ndarray] = None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        
        # Draw image count and instructions
        UIUtils.draw_text(result, f"Images captured: {len(self.images)}", 
                         (10, result.shape[0] - 80), Constants.FONT_SCALE_MEDIUM, 
                         Constants.COLOR_YELLOW, Constants.FONT_THICKNESS_BOLD)
        
        if len(self.images) >= 2:
            UIUtils.draw_text(result, "Press 'p' to create panorama", 
                             (10, result.shape[0] - 50), Constants.FONT_SCALE_SMALL, 
                             Constants.COLOR_GREEN, Constants.FONT_THICKNESS_NORMAL)
        
        return result
    
    def capture_image(self, frame: np.ndarray) -> None:
        """Capture current frame for panorama."""
        self.images.append(frame.copy())
        print(f"Captured image {len(self.images)}")
    
    def reset(self) -> None:
        """Reset captured images and panorama."""
        self.images.clear()
        self.current_panorama = None
        # Close panorama window if open
        try:
            cv2.destroyWindow('Panorama')
        except:
            pass
        print("Panorama reset - all images cleared")
    
    def create_panorama(self) -> bool:
        """Create panorama from captured images using custom stitching."""
        if len(self.images) < 2:
            print("Need at least 2 images to create panorama")
            return False
        
        print("Creating panorama with custom stitching algorithm...")
        self.current_panorama = self.stitcher.stitch_images(self.images)
        
        if self.current_panorama is not None:
            # Display panorama in separate window
            self.display_panorama()
            return True
        else:
            print("Failed to create panorama")
            return False
    
    def display_panorama(self) -> None:
        """Display the created panorama in a separate window."""
        if self.current_panorama is not None:
            try:
                # Resize panorama if too large for display
                h, w = self.current_panorama.shape[:2]
                max_display_width = 1200
                max_display_height = 800
                
                if w > max_display_width or h > max_display_height:
                    scale = min(max_display_width / w, max_display_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    display_panorama = cv2.resize(self.current_panorama, (new_w, new_h))
                else:
                    display_panorama = self.current_panorama
                
                # Create window and display
                cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)
                cv2.imshow('Panorama', display_panorama)
                print(f"Panorama displayed! Size: {w}x{h} pixels")
                print("Press any key in the Panorama window to close it")
                
            except Exception as e:
                print(f"Error displaying panorama: {e}")

class OBJModel:
    """Simple OBJ model loader for 3D rendering."""
    
    def __init__(self, obj_path: str):
        self.vertices = []
        self.faces = []
        self.load_model(obj_path)
    
    def load_model(self, obj_path: str) -> None:
        """Load OBJ model from file."""
        try:
            if not os.path.exists(obj_path):
                print(f"OBJ file not found: {obj_path}")
                return
            
            with open(obj_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):  # Vertex
                        parts = line.split()
                        if len(parts) >= 4:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.vertices.append(vertex)
                    elif line.startswith('f '):  # Face
                        parts = line.split()
                        if len(parts) >= 4:
                            # Handle faces with texture/normal indices (v/vt/vn)
                            face_vertices = []
                            for part in parts[1:]:
                                vertex_idx = int(part.split('/')[0]) - 1  # OBJ indices start at 1
                                face_vertices.append(vertex_idx)
                            self.faces.append(face_vertices)
            
            # Convert to numpy arrays and normalize scale
            self.vertices = np.array(self.vertices, dtype=np.float32)
            self.normalize_model()
            
            print(f"Loaded OBJ model: {len(self.vertices)} vertices, {len(self.faces)} faces")
            
        except Exception as e:
            print(f"Error loading OBJ model: {e}")
            self.vertices = []
            self.faces = []
    
    def normalize_model(self) -> None:
        """Normalize model to fit within a reasonable size."""
        if len(self.vertices) == 0:
            return
        
        # Center the model
        center = np.mean(self.vertices, axis=0)
        self.vertices -= center
        
        # Scale to fit properly within camera view without calibration
        max_extent = np.max(np.abs(self.vertices))
        if max_extent > 0:
            scale_factor = 0.06 / max_extent  # Larger size - 5cm total size for better visibility
            self.vertices *= scale_factor
        
        # Move model slightly above the marker plane
        self.vertices[:, 2] -= 0.005  # Move down in Z to sit on marker
        
        # Rotate model to stand upright (assuming Y is up in the model)
        # Rotate -90 degrees around X to make Y point up in camera coordinates
        rotation_x = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        self.vertices = np.dot(self.vertices, rotation_x.T)
    
    def get_triangulated_faces(self) -> List[List[int]]:
        """Convert faces to triangles."""
        triangulated = []
        for face in self.faces:
            if len(face) == 3:
                triangulated.append(face)
            elif len(face) == 4:  # Quad -> 2 triangles
                triangulated.append([face[0], face[1], face[2]])
                triangulated.append([face[0], face[2], face[3]])
            elif len(face) > 4:  # Polygon -> fan triangulation
                for i in range(1, len(face) - 1):
                    triangulated.append([face[0], face[i], face[i + 1]])
        return triangulated

class AugmentedRealityMode(BaseMode):
    """Augmented Reality mode using ArUco markers."""
    
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Load camera calibration data
        self.mtx = np.eye(3)
        self.dist = np.zeros((1, 5))
        self._load_calibration()
        
        # Load T-Rex 3D model
        self.sources_path = "assigment_01/sources"
        trex_model_path = os.path.join(self.sources_path, "trex_model.obj")
        self.trex_model = OBJModel(trex_model_path)
        
        # Check if ArUco marker image exists and get its specific ID
        self.marker_image_path = os.path.join(self.sources_path, "A4_ArUco_Marker.png")
        self.marker_available = os.path.exists(self.marker_image_path)
        self.target_marker_id = 42  # Specific ID for T-Rex marker from sources folder
        
        print(f"AR Mode: T-Rex will render only on marker ID {self.target_marker_id}")
    
    def _load_calibration(self) -> None:
        """Load camera calibration data from file."""
        try:
            import os
            calibration_path = 'assigment_01/sources/calibration.npz'
            if os.path.exists(calibration_path):
                with np.load(calibration_path) as X:
                    self.mtx, self.dist = [X[i] for i in ('mtx', 'dist')]
                print("AR Mode: Calibration data loaded.")
            else:
                print("AR Mode: WARNING - 'calibration.npz' not found in sources folder. AR mode will not be accurate.")
                print("AR Mode: Please run camera calibration mode first.")
        except Exception as e:
            print(f"AR Mode: Error loading calibration: {e}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with AR cube rendering on ArUco markers."""
        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None:
            # Use solvePnP for OpenCV 4.12.0 compatibility (systematically tested)
            marker_length = 0.05  # 5cm marker size
            marker_points = np.array([
                [-marker_length/2, marker_length/2, 0],
                [marker_length/2, marker_length/2, 0],
                [marker_length/2, -marker_length/2, 0],
                [-marker_length/2, -marker_length/2, 0]
            ], dtype=np.float32)
            
            # Check which markers are detected and process them
            trex_markers_found = []
            other_markers_found = []
            
            for i, marker_id in enumerate(ids.flatten()):
                # Use solvePnP for each marker (proven to work)
                # Reshape corners from (1, 4, 2) to (4, 2)
                corner_2d = corners[i].reshape(4, 2).astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(marker_points, corner_2d, self.mtx, self.dist)
                
                if success:
                    if marker_id == self.target_marker_id:
                        # This is our T-Rex marker!
                        trex_markers_found.append((rvec, tvec, corners[i], marker_id))
                    else:
                        # Other markers get basic visualization
                        other_markers_found.append((corners[i], marker_id))
            
            # Draw T-Rex 3D model ONLY on target marker (ID 42)
            for rvec, tvec, corner, marker_id in trex_markers_found:
                frame = self._draw_3d_model(frame, rvec, tvec)
                # Draw special border for T-Rex marker
                corner_reshaped = corner.reshape(4, 2).astype(np.int32)
                cv2.polylines(frame, [corner_reshaped], True, (0, 255, 0), 3)
                # Label it as T-Rex marker (fix position format)
                text_pos = (int(corner_reshaped[0][0]), int(corner_reshaped[0][1]) - 10)
                cv2.putText(frame, f"T-Rex Marker {marker_id}", 
                          text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0, 255, 0), 2)
            
            # Draw other markers with basic visualization
            for corner, marker_id in other_markers_found:
                corner_reshaped = corner.reshape(4, 2).astype(np.int32)
                cv2.polylines(frame, [corner_reshaped], True, (0, 0, 255), 2)
                text_pos = (int(corner_reshaped[0][0]), int(corner_reshaped[0][1]) - 10)
                cv2.putText(frame, f"ID {marker_id}", 
                          text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 0, 255), 2)
            
            # Add detection info
            self._add_ar_info(frame, len(ids), len(trex_markers_found))
        else:
            # No markers detected
            self._add_no_markers_info(frame)
        
        return frame
    
    def _draw_3d_model(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Draw the T-Rex 3D model on the detected marker."""
        if len(self.trex_model.vertices) == 0:
            return self._draw_fallback_cube(frame, rvec, tvec)
            
        # Project 3D model vertices to 2D image plane
        model_vertices = self.trex_model.vertices.reshape(-1, 1, 3)
        img_points, _ = cv2.projectPoints(model_vertices, rvec, tvec, self.mtx, self.dist)
        img_points = img_points.reshape(-1, 2).astype(np.int32)
        
        # Get triangulated faces for rendering
        triangulated_faces = self.trex_model.get_triangulated_faces()
        
        # Simplified rendering - just draw triangles without complex depth sorting
        # Define colors for different parts of the T-Rex
        colors = [
            (50, 150, 50),    # Dark green
            (70, 180, 70),    # Medium green  
            (90, 200, 90),    # Light green
            (40, 120, 40),    # Very dark green
            (100, 220, 100),  # Bright green
            (60, 160, 60),    # Another green tone
        ]
        
        # Draw all faces for complete rendering
        max_faces_to_draw = len(triangulated_faces)  # Draw all faces
        
        # Draw faces
        for i in range(max_faces_to_draw):  # Draw every face
            if i < len(triangulated_faces):
                face = triangulated_faces[i]
                
                if len(face) >= 3 and all(0 <= idx < len(img_points) for idx in face):
                    # Get triangle vertices
                    triangle_points = img_points[face[:3]]
                    
                    # Ensure triangle_points is the right shape and type
                    if triangle_points.shape[0] == 3 and triangle_points.shape[1] == 2:
                        # Check if triangle is in reasonable bounds
                        x_coords = triangle_points[:, 0]
                        y_coords = triangle_points[:, 1]
                        
                        # Use numpy functions that work with arrays
                        if (np.min(x_coords) >= -100 and np.max(x_coords) < frame.shape[1] + 100 and
                            np.min(y_coords) >= -100 and np.max(y_coords) < frame.shape[0] + 100):
                            
                            # Use color based on face index for variety
                            color = colors[i % len(colors)]
                            
                            # Fill triangle - ensure int32 type
                            triangle_int = triangle_points.astype(np.int32)
                            cv2.fillConvexPoly(frame, triangle_int, color, lineType=cv2.LINE_AA)
        
        return frame
    
    def _draw_fallback_cube(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Draw a simple cube as fallback when model fails."""
        try:
            # Define simple cube
            axis_length = 0.025
            cube_points = np.float32([
                [0, 0, 0], [axis_length, 0, 0], [axis_length, axis_length, 0], [0, axis_length, 0],
                [0, 0, -axis_length], [axis_length, 0, -axis_length], [axis_length, axis_length, -axis_length], [0, axis_length, -axis_length]
            ])
            
            img_points, _ = cv2.projectPoints(cube_points, rvec, tvec, self.mtx, self.dist)
            img_points = np.int32(img_points).reshape(-1, 2)
            
            # Draw simple cube faces
            faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
            colors = [(100,100,255), (100,255,100), (255,100,100), (255,255,100), (255,100,255), (100,255,255)]
            
            for face, color in zip(faces, colors):
                cv2.fillConvexPoly(frame, img_points[face], color, lineType=cv2.LINE_AA)
                cv2.polylines(frame, [img_points[face]], True, (0,0,0), 2, cv2.LINE_AA)
            
            return frame
        except Exception as e:
            print(f"Error drawing fallback cube: {e}")
            return frame
    
    def _add_ar_info(self, frame: np.ndarray, total_markers: int, trex_markers: int) -> None:
        """Add AR status information to frame."""
        info_lines = [
            "AR Mode - T-Rex Augmented Reality",
            f"Total markers detected: {total_markers}",
            f"T-Rex markers (ID 42): {trex_markers}",
            f"T-Rex model: {len(self.trex_model.vertices)} vertices, {len(self.trex_model.faces)} faces",
            "3D T-Rex renders ONLY on marker ID 42",
            "",
            "Instructions:",
            "- Use the A4_ArUco_Marker.png from sources folder",
            "- This specific marker has ID 42 and triggers T-Rex",
            "- Other markers will be outlined but not render T-Rex"
        ]
        
        # Position text on the right side to avoid menu overlap
        frame_w = frame.shape[1]
        x_position = frame_w - 400  # 400 pixels from right edge
        y_offset = 30
        
        for i, line in enumerate(info_lines):
            if line:  # Skip empty lines
                cv2.putText(frame, line, (x_position, y_offset + i * 25), 
                          Constants.FONT, Constants.FONT_SCALE_SMALL, 
                          Constants.COLOR_WHITE, Constants.FONT_THICKNESS_NORMAL)
    
    def _add_no_markers_info(self, frame: np.ndarray) -> None:
        """Add no markers detected information."""
        info_lines = [
            "AR Mode - T-Rex Augmented Reality",
            "Status: No ArUco markers detected",
            f"T-Rex model loaded: {len(self.trex_model.vertices)} vertices",
            f"Looking for marker ID {self.target_marker_id} specifically",
            "",
            "Instructions:",
            "1. Print A4_ArUco_Marker.png from sources folder",
            "2. This marker has ID 42 and will show T-Rex",
            "3. Show marker to camera with good lighting",
            "4. Hold marker flat and steady"
        ]
        
        # Position text on the right side to avoid menu overlap
        frame_w = frame.shape[1]
        x_position = frame_w - 400  # 400 pixels from right edge
        y_offset = 30
        
        for i, line in enumerate(info_lines):
            color = Constants.COLOR_YELLOW if "No ArUco" in line else Constants.COLOR_WHITE
            cv2.putText(frame, line, (x_position, y_offset + i * 25), 
                      Constants.FONT, Constants.FONT_SCALE_SMALL, 
                      color, Constants.FONT_THICKNESS_NORMAL)
    
    def _add_error_info(self, frame: np.ndarray, error_msg: str) -> None:
        """Add error information to frame."""
        info_lines = [
            "AR Mode - Error",
            f"Error: {error_msg}",
            "Check camera calibration and ArUco setup"
        ]
        
        # Position text on the right side to avoid menu overlap
        frame_w = frame.shape[1]
        x_position = frame_w - 400  # 400 pixels from right edge
        y_offset = 30
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (x_position, y_offset + i * 25), 
                      Constants.FONT, Constants.FONT_SCALE_SMALL, 
                      Constants.COLOR_RED, Constants.FONT_THICKNESS_NORMAL)

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
            '9': ('T-Rex AR', AugmentedRealityMode())
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

class CameraApp:
    """Main camera application."""
    
    def __init__(self):
        self.cap = None
        self.video_modes = VideoModes()
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Cannot open camera")
                return False
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def run(self) -> None:
        """Run the main application loop."""
        if not self.initialize_camera():
            return
        
        # Print initial instructions
        print("Controls:")
        print("- Press 'q' to quit")
        for mode_info in self.video_modes.get_main_mode_list():
            print(f"- {mode_info}")
        print()
        print("Sub-mode controls will be shown on screen")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Cannot read frame")
                    break
                
                # Process frame
                processed_frame = self.video_modes.process_frame(frame)
                
                # Draw UI
                self._draw_ui(processed_frame)
                
                # Display frame
                cv2.imshow(Constants.CAMERA_WINDOW, processed_frame)
                
                # Handle input
                if not self._handle_input(frame):
                    break
                    
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def _draw_ui(self, frame: np.ndarray) -> None:
        """Draw user interface on frame."""
        try:
            # Draw mode information
            UIUtils.draw_text_list(frame, self.video_modes.get_mode_info())
            
            # Draw quit instruction
            UIUtils.draw_text(frame, "Press 'q' to quit", 
                             (10, frame.shape[0] - Constants.UI_QUIT_TEXT_Y_OFFSET),
                             Constants.FONT_SCALE_MEDIUM, Constants.COLOR_GREEN, 
                             Constants.FONT_THICKNESS_BOLD)
        except Exception as e:
            print(f"Error drawing UI: {e}")
    
    def _handle_input(self, frame: np.ndarray) -> bool:
        """Handle keyboard input. Returns False to quit."""
        try:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            
            if key < 128:  # Valid ASCII
                key_str = chr(key)
                if not self.video_modes.switch_main_mode(key_str):
                    if self.video_modes.current_main_mode == 'panorama' and key_str == 'c':
                        self.video_modes.current_mode_obj.capture_image(frame)
                    else:
                        self.video_modes.handle_sub_mode_key(key_str)
            
            return True
        except Exception as e:
            print(f"Error handling input: {e}")
            return True

def main():
    """Main entry point."""
    app = CameraApp()
    app.run()

if __name__ == "__main__":
    main()