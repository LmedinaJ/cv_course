import cv2
import numpy as np
from typing import Tuple
from ..constants import Constants
from ..utils import UIUtils
from .base_modes import BaseMode


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
                    Constants.UI_HIGHLIGHT, Constants.HIST_LINE_THICKNESS)
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
                         Constants.FONT_SCALE_SMALL, Constants.UI_INFO_TEXT, Constants.FONT_THICKNESS_BOLD)
        UIUtils.draw_text(frame, "R", (start_x + 50, end_y + 15), 
                         Constants.FONT_SCALE_SMALL, Constants.COLOR_RED, Constants.FONT_THICKNESS_BOLD)


class HistogramMode(BaseMode):
    """Histogram display mode."""
    
    def __init__(self):
        self.renderer = HistogramRenderer()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        hist_b, hist_g, hist_r = self.renderer.calculate_histograms(frame)
        hist_img = self.renderer.create_histogram_image(hist_b, hist_g, hist_r)
        return self.renderer.overlay_histogram(frame, hist_img)