import cv2
import numpy as np
import time
from typing import Tuple, Optional, List
from ..constants import Constants
from ..utils import UIUtils, ImageUtils
from .base_modes import BaseMode


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
                                 Constants.FONT_SCALE_LARGE, Constants.CALIBRATION_SUCCESS, 
                                 Constants.FONT_THICKNESS_BOLD)
                y_offset += 40
                
                UIUtils.draw_text(frame, f"Re-projection error: {self.calibration_error:.4f}", 
                                 (x_position, y_offset), Constants.FONT_SCALE_MEDIUM, Constants.UI_INFO_TEXT)
                y_offset += 30
                
                UIUtils.draw_text(frame, f"Data saved to: {Constants.CALIBRATION_FILE}", 
                                 (x_position, y_offset), Constants.FONT_SCALE_SMALL, Constants.UI_INFO_TEXT)
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
                                 Constants.CALIBRATION_SUCCESS if pattern_detected else Constants.UI_ERROR_TEXT,
                                 Constants.FONT_THICKNESS_BOLD)
                y_offset += 30
                
                status_text = "Pattern detected - capturing..." if pattern_detected else "Show chessboard pattern"
                status_color = Constants.CALIBRATION_SUCCESS if pattern_detected else Constants.UI_HIGHLIGHT
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