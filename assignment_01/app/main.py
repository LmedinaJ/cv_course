import cv2
import numpy as np

from .constants import Constants
from .utils import UIUtils
from .video_modes import VideoModes

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
                             Constants.FONT_SCALE_MEDIUM, Constants.UI_MENU_TEXT, 
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