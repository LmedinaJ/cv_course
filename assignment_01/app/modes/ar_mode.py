import cv2
import numpy as np
import os
from typing import List, Tuple
from ..constants import Constants
from .base_modes import BaseMode


class OBJModel:
    """OBJ model loader for 3D rendering."""
    
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
        self.sources_path = "sources"
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
            calibration_path = 'sources/calibration.npz'
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