import cv2
import numpy as np
from typing import List
from ..constants import Constants
from ..utils import UIUtils, ImageUtils, TrackbarConfig
from .base_modes import BaseFilterMode, TrackbarMode


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
                    cv2.line(result, (x1, y1), (x2, y2), Constants.UI_HIGHLIGHT, 2)
            
            # Draw line count
            color = Constants.CALIBRATION_SUCCESS if line_count > 0 else Constants.UI_ERROR_TEXT
            UIUtils.draw_text(result, f"Lines: {line_count}", (10, 30), 
                             Constants.FONT_SCALE_LARGE, color, Constants.FONT_THICKNESS_BOLD)
            
            return result
        except Exception as e:
            print(f"Error applying Hough lines filter: {e}")
            return frame