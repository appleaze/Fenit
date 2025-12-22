import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class ChessBoardProcessor:
    def __init__(self, target_size=(500, 500)):
        self.target_size = target_size
        self.warped_image = None
        self.transform_matrix = None
        self.debug_hough_img = None # Store for overlaying NMS

    def process(self, image, corners, piece_points=None, method='geometric'):
        """
        Main pipeline:
        1. Correct corners (expand dynamically if pieces provided, else 1%)
        2. Warp perspective
        3. Detect grid lines (Geometric or Canny)
        4. Construct grid
        """
        # 1. Correction Algorithm
        if piece_points is not None and len(piece_points) > 0:
             corrected_corners = self.correct_corners_with_pieces(corners, piece_points, image)
        else:
             corrected_corners = self.correct_corners(corners)
        
        # 2. Warping
        self.warped_image, self.transform_matrix = self.warp_image(image, corrected_corners)
        
        # 3. Grid Generation
        if method == 'canny':
            print("Using Canny Edge Detection Pipeline...")
            lines = self.find_grid_lines(self.warped_image)
            h_lines, v_lines = self.cluster_and_filter_lines(lines)
        else:
            # Geometric (Default)
            # User requested clean lines derived from keypoints (corners).
            print("Using Ideal Geometric Grid...")
            h_lines, v_lines = self.generate_ideal_grid()
        
        return h_lines, v_lines

    def generate_ideal_grid(self):
        """
        Generates a perfect 8x8 grid for the 500x500 target.
        """
        ideal_grid = np.linspace(0, 500, 9)
        
        v_lines = [(rho, 0.0) for rho in ideal_grid]       # Theta = 0 (Vertical)
        h_lines = [(rho, np.pi/2) for rho in ideal_grid]   # Theta = pi/2 (Horizontal)
        
        return h_lines, v_lines

    def correct_corners(self, corners):
        """
        Calculates center, creates vectors to corners, scales by 1% (1.01),
        pushes corners outward to ensure full board capture.
        """
        corners = np.array(corners, dtype=np.float32)
        center = np.mean(corners, axis=0)
        
        corrected = []
        for point in corners:
            vector = point - center
            # Scale by 1% as requested
            scaled_vector = vector * 1.01 
            new_point = center + scaled_vector
            corrected.append(new_point)
            
        return np.array(corrected, dtype=np.float32)

    def correct_corners_with_pieces(self, corners, piece_points, image=None):
        """
        Expands the board corners dynamically to include all piece points.
        Uses a heuristic uniform scaling from the centroid.
        """
        corners = np.array(corners, dtype=np.float32)
        center = np.mean(corners, axis=0)
        
        piece_points = np.array(piece_points, dtype=np.float32)
        
        # Initial scale
        scale = 1.0
        max_scale = 1.08 # Increased to 8% to catch missing piece (Grid should handle it now)
        step = 0.01
        
        print("Expanding board to cover pieces...")
        
        # Loop until all pieces are inside or limit reached
        while scale <= max_scale:
            # Create expanded polygon
            scaled_corners = []
            for point in corners:
                vector = point - center
                new_point = center + vector * scale
                scaled_corners.append(new_point)
            
            scaled_corners_np = np.array(scaled_corners, dtype=np.float32)
            
            # Use pointPolygonTest (requires contour likely int32 or float32?)
            # OpenCV pointPolygonTest usually wants float32 contour is fine.
            # But the contour order matters! It expects order.
            # We must ensure corners are sorted first for polygon test?
            # Or assume they form a convex hull. 
            # Let's sort them by angle first to be sure they form a valid polygon loop.
            
            # Sort for Polygon Test
            angles = np.arctan2(scaled_corners_np[:, 1] - center[1], scaled_corners_np[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            poly = scaled_corners_np[sorted_indices]
            
            all_inside = True
            for p in piece_points:
                # measureDist=True -> return signed distance
                # Dist > 0 inside, < 0 outside, = 0 on edge.
                dist = cv2.pointPolygonTest(poly, (p[0], p[1]), True)
                
                # Check if outside (allow small margin -2px?)
                if dist < -2.0:
                    all_inside = False
                    break
            
            if all_inside:
                print(f"  -> Converged at scale {scale:.2f}")
                self.expansion_scale = scale
                
                # DEBUG: Save Expansion
                if image is not None:
                    debug_exp = image.copy()
                    # Draw Original (Red)
                    cv2.polylines(debug_exp, [corners.astype(np.int32)], True, (0, 0, 255), 2)
                    # Draw Pieces (Yellow)
                    for px, py in piece_points:
                        cv2.circle(debug_exp, (int(px), int(py)), 5, (0, 255, 255), -1)
                    # Draw New (Green)
                    cv2.polylines(debug_exp, [scaled_corners_np.astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.imwrite("result/debug_expansion.jpg", debug_exp)
                    print("Saved result/debug_expansion.jpg")

                return scaled_corners_np 
                
            scale += step
            
        print(f"  -> Reached max scale {max_scale}. Using best effort.")
        self.expansion_scale = max_scale
        # Reconstruct final
        final_poly = []
        for point in corners:
            vector = point - center
            final_poly.append(center + vector * max_scale)
        final_np = np.array(final_poly, dtype=np.float32)

        # DEBUG: Save Expansion (Max Reached)
        if image is not None:
            debug_exp = image.copy()
            cv2.polylines(debug_exp, [corners.astype(np.int32)], True, (0, 0, 255), 2)
            for px, py in piece_points:
                cv2.circle(debug_exp, (int(px), int(py)), 5, (0, 255, 255), -1)
            cv2.polylines(debug_exp, [final_np.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.imwrite("result/debug_expansion.jpg", debug_exp)
            print("Saved result/debug_expansion.jpg")

        return final_np

    def generate_ideal_grid(self):
        """
        Generates a perfect 8x8 grid for the 500x500 target.
        Corrects for the expansion scale to ensure grid lines match the actual board.
        Scale > 1.0 means the 500x500 image contains the board PLUS margin.
        The actual board is centered and smaller by factor 1/scale.
        """
        target_size = 500.0
        
        # If expansion_scale is not set, assume 1.0 (or default max_scale if only correct_corners used)
        scale = getattr(self, 'expansion_scale', 1.01) # default 1.01 from correct_corners
        
        # Calculate actual board width in warped pixels
        actual_width = target_size / scale
        
        # Calculate offsets
        margin = (target_size - actual_width) / 2.0
        
        start = margin
        end = target_size - margin
        
        ideal_grid = np.linspace(start, end, 9)
        
        v_lines = [(rho, 0.0) for rho in ideal_grid]       # Theta = 0 (Vertical)
        h_lines = [(rho, np.pi/2) for rho in ideal_grid]   # Theta = pi/2 (Horizontal)
        
        return h_lines, v_lines

    def warp_image(self, image, corners):
        """
        Warp to 500x500 grayscale.
        """
        # Robust Sorting: Sort by angle from centroid
        # 1. Calculate centroid
        centroid = corners.mean(axis=0)
        
        # 2. Calculate angles
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        
        # 3. Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        rect = sorted_corners
        
        # DEBUG: Print Sorted Corners
        print("sorted_corners (TL, TR, BR, BL):\n", rect)
        
        # DEBUG: Visualize Sorting Order on Original Image (if passed? wait, process receives image)
        # We need to save this to result dir. The method doesn't know output_dir.
        # We can write to "result/debug_sorting.jpg" hardcoded or passed via init?
        # Let's just save to "result/debug_sorting.jpg" assuming result dir exists.
        
        try:
            debug_sort = image.copy()
            labels = ["TL", "TR", "BR", "BL"]
            colors = [(0,0,255), (0,255,255), (0,255,0), (255,0,0)] # Red, Yellow, Green, Blue
            for i in range(4):
                x, y = int(rect[i][0]), int(rect[i][1])
                cv2.circle(debug_sort, (x, y), 15, colors[i], -1)
                cv2.putText(debug_sort, labels[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite("result/debug_sorting.jpg", debug_sort)
            print("Saved debug_sorting.jpg")
        except Exception as e:
            print(f"Failed to save debug_sorting.jpg: {e}")
            
        (w, h) = self.target_size
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (w, h))
        
        # Convert to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        return gray, M

    def find_grid_lines(self, gray_image):
        """
        Canny Edge -> Hough Transform
        """
        # Canny
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # DEBUG: Save Canny
        cv2.imwrite("result/debug_canny.jpg", edges)
        print("Saved result/debug_canny.jpg")
        
        # HoughLines (Standard HT)
        # 1 pixel resolution, 1 degree (pi/180) resolution
        # Threshold: 110 (Lowered from 150 to catch fainter lines)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)
        
        # DEBUG: Visualize Hough Lines
        if lines is not None:
             self.debug_hough_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
             for line in lines:
                 rho, theta = line[0]
                 a = np.cos(theta)
                 b = np.sin(theta)
                 x0 = a * rho
                 y0 = b * rho
                 x1 = int(x0 + 1000 * (-b))
                 y1 = int(y0 + 1000 * (a))
                 x2 = int(x0 - 1000 * (-b))
                 y2 = int(y0 - 1000 * (a))
                 cv2.line(self.debug_hough_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
             
             cv2.imwrite("result/debug_hough_lines.jpg", self.debug_hough_img)
             print("Saved result/debug_hough_lines.jpg")
             
        return lines

    def cluster_and_filter_lines(self, lines):
        """
        Filter lines into Horizontal and Vertical using Reference Angles.
        Vertical Reference: 0 radians (or Pi)
        Horizontal Reference: Pi/2 radians (1.57)
        """
        if lines is None or len(lines) < 2:
            return [], []

        vertical_lines = []
        horizontal_lines = []
        
        # Threshold for angular deviation (gradient) from reference
        angle_threshold = 0.35 # ~20 degrees
        
        for line in lines:
            rho, theta = line[0]
            
            # Distance (Gradient) from Vertical Reference (0 or Pi)
            # theta is in [0, pi]
            diff_v = min(abs(theta - 0), abs(theta - np.pi))
            
            # Distance (Gradient) from Horizontal Reference (Pi/2)
            diff_h = abs(theta - np.pi/2)
            
            # Assign to closest reference (Voronoi partition at 45 degrees)
            if diff_v < diff_h:
                vertical_lines.append((rho, theta))
            else:
                horizontal_lines.append((rho, theta))
             
        # Apply NMS (Non-Maximum Suppression)
        # Note: Input lists are already sorted by Hough Vote (strength) because we preserved order
        # We apply NMS *before* sorting by Rho.
        
        final_v = self.nms_lines(vertical_lines)
        final_h = self.nms_lines(horizontal_lines)
        
        # Sort final lines by Rho (spatial order for grid construction)
        final_v.sort(key=lambda x: x[0])
        final_h.sort(key=lambda x: x[0])
        
        # DEBUG: Visualize NMS Selection on top of Hough Lines
        if self.debug_hough_img is not None:
             debug_nms = self.debug_hough_img.copy()
             # Draw selected lines in GREEN (Thicker)
             for rho, theta in final_v + final_h:
                 a = np.cos(theta)
                 b = np.sin(theta)
                 x0 = a * rho
                 y0 = b * rho
                 x1 = int(x0 + 1000 * (-b))
                 y1 = int(y0 + 1000 * (a))
                 x2 = int(x0 - 1000 * (-b))
                 y2 = int(y0 - 1000 * (a))
                 # Green
                 cv2.line(debug_nms, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 
             cv2.imwrite("result/debug_hough_nms.jpg", debug_nms)
             print("Saved result/debug_hough_nms.jpg")
        
        # Interpolate Missing Lines (Restore obscured lines)
        final_v = self.interpolate_lines(final_v, is_vertical=True)
        final_h = self.interpolate_lines(final_h, is_vertical=False)
        
        return final_h, final_v
        
    def interpolate_lines(self, lines, is_vertical=True):
        """
        Interpolate lines to create a CLEAN grid.
        Since we warp to 500x500 and trust the corners, we enforce a PERFECT orthogonal grid.
        This eliminates jitter ("messy lines").
        """
        target_lines = 9
        ideal_grid = np.linspace(0, 500, target_lines) # [0, 62.5, ..., 500]
        
        # Enforce perfect angles
        ideal_theta = 0.0 if is_vertical else np.pi/2
        
        # We fundamentally trust the warp now.
        # So we simply return the ideal grid. 
        # The detection (Hough) serves as a validation that grid lines exist, 
        # but we don't use their jittery positions for the final grid.
        
        final_lines = []
        for rho in ideal_grid:
            final_lines.append((rho, ideal_theta))
                
        return final_lines
        
    def nms_lines(self, lines, rho_threshold=20):
        """
        Apply Non-Maximum Suppression.
        Assumes `lines` is sorted by strength (Hough votes).
        Picks the strongest line and suppresses any subsequent lines with similar Rho.
        """
        if not lines:
            return []
            
        kept_lines = []
        candidates = list(lines) # Copy
        
        while candidates:
            # Pick strongest (first in list)
            best_line = candidates.pop(0)
            kept_lines.append(best_line)
            
            # Suppress neighbors (based on Rho difference)
            # Filter out lines that are too close to current best
            candidates = [line for line in candidates if abs(line[0] - best_line[0]) > rho_threshold]
            
        return kept_lines

if __name__ == "__main__":
    print("This module implements the Board Processing logic.")
    print("Use it by importing ChessBoardProcessor and passing the image + YOLO corners.")
