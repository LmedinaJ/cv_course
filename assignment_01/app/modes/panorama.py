import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..constants import Constants
from ..utils import UIUtils, ImageUtils
from .base_modes import BaseMode


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
                             Constants.UI_INFO_TEXT, Constants.FONT_THICKNESS_NORMAL)
        
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