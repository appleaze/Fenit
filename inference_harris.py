import cv2
import numpy as np
from ultralytics import YOLO
from chess_board import ChessBoardProcessor
import os
from pathlib import Path

def get_piece_mapping():
    # Matches 'names' in data.yaml
    return {
        1: 'p', 2: 'P',
        3: 'r', 4: 'R',
        5: 'b', 6: 'B',
        7: 'n', 8: 'N',
        9: 'q', 10: 'Q',
        11: 'k', 12: 'K'
    }

def board_to_fen(board):
    fen_rows = []
    for row in board:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    return "/".join(fen_rows) + " w - - 0 1"

def interpolate_grid(lines, max_dim=500):
    if not lines or len(lines) < 2:
        return np.linspace(0, max_dim, 9)
    
    rhos = [l[0] for l in lines]
    ideal_grid = np.linspace(0, max_dim, 9)
    final_grid = ideal_grid.copy()
    
    for rho in rhos:
        idx = (np.abs(ideal_grid - rho)).argmin()
        diff = rho - ideal_grid[idx]
        if abs(diff) < 30:
             final_grid[idx] = rho
             
    return final_grid

def get_square_from_grid(x, y, h_grid, v_grid):
    col = -1
    row = -1
    margin_tolerance = 25.0 
    
    if x >= v_grid[0] and x < v_grid[-1]:
        for i in range(8):
            if v_grid[i] <= x < v_grid[i+1]:
                col = i
                break
    else:
        if v_grid[0] - margin_tolerance <= x < v_grid[0]:
            col = 0
        elif v_grid[-1] <= x < v_grid[-1] + margin_tolerance:
            col = 7
            
    if y >= h_grid[0] and y < h_grid[-1]:
        for i in range(8):
            if h_grid[i] <= y < h_grid[i+1]:
                row = i
                break
    else:
         if h_grid[0] - margin_tolerance <= y < h_grid[0]:
             row = 0
         elif h_grid[-1] <= y < h_grid[-1] + margin_tolerance:
             row = 7
            
    return col, row

def detect_board_harris(image, output_dir="result/"):
    """
    Detects chessboard corners using Harris Corner Detection.
    Returns: numpy array of 4 corners (TL, TR, BR, BL) or None
    """
    print("Running Harris Corner Detection for Board...")
    
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Harris Corner Detection
    # blockSize - size of neighbourhood considered for corner detection
    # ksize - aperture parameter of Sobel derivative used
    # k - Harris detector free parameter in the equation
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    # Dilate result to merge nearby corners
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value
    threshold = 0.01 * dst.max()
    
    # Create a mask of corners
    corner_mask = np.zeros_like(gray, dtype=np.uint8)
    corner_mask[dst > threshold] = 255
    
    # Find centroids of these corner clusters
    # centroids is (x, y)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_mask)
    
    # We expect many corners (approx 7x7=49 inner corners + outer ones)
    print(f"Harris found {len(centroids)} potential corner points.")
    
    if len(centroids) < 4:
        print("Error: Too few corners found to form a board.")
        return None

    # Determine Board Boundary
    # Strategy: Find the Largest Convex Hull of these points, 
    # then approximate it to a Quadrilateral (4 points).
    
    # Convert centroids to correct format for convexHull
    points = np.float32(centroids[1:]) # Skip the first centroid (background)
    
    if len(points) < 4:
         return None

    # Find Hull
    hull = cv2.convexHull(points)
    
    # Approximate Polygon to 4 points
    # Epsilon: perimeter * factor
    perimeter = cv2.arcLength(hull, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    # If approx has > 4 points, increase epsilon. If < 4, decrease or fallback.
    # Simple loop to enforce 4 points
    max_iter = 10
    iter_count = 0
    while len(approx) != 4 and iter_count < max_iter:
        if len(approx) > 4:
            epsilon *= 1.5
        else:
            epsilon *= 0.5
        approx = cv2.approxPolyDP(hull, epsilon, True)
        iter_count += 1
        
    print(f"Approximated Polygon has {len(approx)} points.")
    
    # Debug Visualization
    if output_dir:
        debug_harris = image.copy()
        # Draw all centroids
        for i, pt in enumerate(points):
            cv2.circle(debug_harris, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
            
        # Draw Hull
        cv2.drawContours(debug_harris, [hull.astype(int)], 0, (0, 255, 0), 2)
        
        # Draw Approx
        cv2.drawContours(debug_harris, [approx.astype(int)], 0, (255, 0, 0), 3)
        
        cv2.imwrite(os.path.join(output_dir, "debug_harris_detection.jpg"), debug_harris)

    if len(approx) != 4:
        # Fallback: Just return the minAreaRect of the hull
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)
        print("Using MinAreaRect as fallback.")
        return box
    
    # Reshape approx to (4, 2)
    corners = approx.reshape(4, 2).astype(np.float32)
    return corners

def run_inference(image_path, model_path="runs/chess_pose_train/weights/best.pt", output_dir="result/", grid_method="geometric"):
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception:
        print(f"Error: Could not load model at {model_path}")
        return

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Run YOLO inference for PIECES only
    print("Running YOLO detection (Pieces)...")
    results = model(img)[0]
    
    # -------------------------------------------------------------
    # REPLACE MODEL BOARD DETECTION WITH HARRIS CORNER DETECTION
    # -------------------------------------------------------------
    board_corners = detect_board_harris(img, output_dir)
    
    if board_corners is None:
        print("Harris Detection failed to find a valid board.")
        return
        
    print("Using Harris-Detected Board Corners.")
    
    # Collect Piece Points for Expansion (if needed, though Harris should capture the grid)
    piece_points_for_expansion = []
    
    # Process Board (Warp & Grid)
    processor = ChessBoardProcessor()
    
    # We pass piece_points as None or empty because Harris ideally found the grid itself?
    # Actually, Harris finds the outer boundary of the SQUARES.
    # The actual board might be slightly larger, but for warping the 8x8 grid, 
    # the outer corners of the pattern are exactly what we want.
    # So we shouldn't need massive expansion unless Harris found the *inner* 6x6.
    
    # Let's pass piece points anyway just in case the processor wants to sanity check.
    result = processor.process(img, board_corners, piece_points=piece_points_for_expansion, method=grid_method)
    
    if result is not None:
        h_lines, v_lines = result
        v_grid = interpolate_grid(v_lines, 500)
        h_grid = interpolate_grid(h_lines, 500)
    else:
        print("Warning: Grid line detection failed. Using uniform grid.")
        v_grid = np.linspace(0, 500, 9)
        h_grid = np.linspace(0, 500, 9)
        
    M = processor.transform_matrix
    
    if processor.warped_image is not None and output_dir:
         warped_path = os.path.join(output_dir, "debug_warped_harris.jpg")
         cv2.imwrite(warped_path, processor.warped_image)

    if M is None:
        print("Error: Perspective warp failed.")
        return

    # Initialize 8x8 board
    board_state = [['.' for _ in range(8)] for _ in range(8)]
    board_conf = [[-1.0 for _ in range(8)] for _ in range(8)]
    piece_map = get_piece_mapping()
    candidate_pieces = [] 
    
    # Process detected pieces
    for kpts, cls, conf in zip(results.keypoints.data, results.boxes.cls, results.boxes.conf):
        class_id = int(cls)
        if class_id == 0: continue # Skip board (we used Harris)
        
        confidence = float(conf)
        points = kpts.cpu().numpy()
        
        if len(points) >= 3:
             base_points = points[:3, :2]
        else:
             base_points = points[:, :2] # Fallback
             
        piece_x = np.mean(base_points[:, 0])
        piece_y = np.mean(base_points[:, 1])
        
        pt = np.array([[[piece_x, piece_y]]], dtype=np.float32)
        if M is not None:
             warped_pt = cv2.perspectiveTransform(pt, M)[0][0]
             col, row = get_square_from_grid(warped_pt[0], warped_pt[1], h_grid, v_grid)
             
             piece_char = piece_map.get(class_id, '?')
             candidate_pieces.append({
                 'char': piece_char,
                 'conf': confidence,
                 'col': col,
                 'row': row,
                 'x': warped_pt[0],
                 'y': warped_pt[1]
             })

    candidate_pieces.sort(key=lambda x: x['conf'], reverse=True)
    
    for cand in candidate_pieces:
        c, r = cand['col'], cand['row']
        if c != -1 and r != -1:
            if board_state[r][c] == '.':
                 board_state[r][c] = cand['char']
                 board_conf[r][c] = cand['conf']

    # Orientation Logic (Simplified for brevity, same as original)
    white_count = 0
    black_count = 0
    # ... (Reusing simpler orientation assumption or skipping for this test)
    # Let's verify correctness first.
    
    print("\nDetected Board State:")
    print("  a b c d e f g h")
    for i, row in enumerate(board_state):
        print(f"{8-i} " + " ".join(row))

    fen = board_to_fen(board_state)
    print(f"\nPredicted FEN: {fen}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem + "_harris"
        
        fen_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(fen_path, "w") as f:
            f.write(fen)
        print(f"Saved FEN to {fen_path}")
        
        save_path = os.path.join(output_dir, f"{base_name}.jpg")
        res_plotted = results.plot()
        
        # Visualize Harris corners on result
        for pt in board_corners:
            cv2.circle(res_plotted, (int(pt[0]), int(pt[1])), 15, (0, 255, 0), 4)

        cv2.imwrite(save_path, res_plotted)
        print(f"Saved inference result to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="runs/chess_pose_train/weights/best.pt", help="Path to trained model")
    parser.add_argument("--output-dir", default="result/", help="Directory to save results")
    parser.add_argument("--grid-method", default="geometric", choices=["geometric", "canny"], help="Grid detection method")
    args = parser.parse_args()
    
    run_inference(args.image, args.model, args.output_dir, args.grid_method)
