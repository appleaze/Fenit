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
    
    # Default FEN suffix: White to move, no castling, no en passant, 0 halfmoves, 1 fullmove
    # You can adjust this if your model detects whose turn it is
    # You can adjust this if your model detects whose turn it is
    return "/".join(fen_rows) + " w - - 0 1"

def estimate_board_from_pieces(results):
    """
    Estimates the board corners based on the locations of detected pieces.
    Returns: numpy array of 4 corners (TL, TR, BR, BL) or None
    """
    piece_points = []
    
    # Collect all piece positions
    for kpts, cls in zip(results.keypoints.data, results.boxes.cls):
        class_id = int(cls)
        if class_id == 0: continue # Skip board
        
        points = kpts.cpu().numpy()
        if len(points) >= 3:
            # keypoints: [base1, base2, base3, top]
            # Use average of base points for board contact
            base_points = points[:3, :2]
            piece_x = np.mean(base_points[:, 0])
            piece_y = np.mean(base_points[:, 1])
            piece_points.append([piece_x, piece_y])
            
    if len(piece_points) < 4:
        print(f"Not enough pieces ({len(piece_points)}) to estimate board.")
        return None
        
    points = np.array(piece_points, dtype=np.float32)
    
    # Find Min Area Rotated Rectangle
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int8(box) # Keeping it float is better for precision, but boxPoints returns float32 usually? 
    # cv2.boxPoints returns float32, let's keep it
    box = np.array(cv2.boxPoints(rect), dtype=np.float32)
    
    # Expand the box slightly (e.g. 10%) to include the squares and margins
    # The rect result is (center(x, y), (width, height), angle)
    center, size, angle = rect
    w, h = size
    
    # Expansion factor: Pieces are on squares, we want the board edge.
    # Assuming pieces are filling the board reasonably well, maybe 10-15% padding?
    padding = 1.15 
    new_size = (w * padding, h * padding)
    
    new_rect = (center, new_size, angle)
    expanded_box = cv2.boxPoints(new_rect)
    expanded_box = np.array(expanded_box, dtype=np.float32)

    return expanded_box

def interpolate_grid(lines, max_dim=500):
    """
    Interpolate 9 grid lines (boundaries 0..8) from detected lines.
    If detected lines are good, use them. Else, fallback to uniform spacing.
    Returns: list of 9 coordinates (offsets).
    """
    if not lines or len(lines) < 2:
        # Fallback to uniform
        return np.linspace(0, max_dim, 9)
    
    # Extract Rhos (assumed sorted)
    rhos = [l[0] for l in lines]
    
    # Simple strategy:
    # 1. Find the first and last valid lines (closest to 0 and 500)
    # 2. Or if we have 9 lines, just use them?
    # 3. Often we catch only inner lines.
    
    # Robust approach: Linear Fit
    # Assume lines correspond to indices 0..8 (or 1..7).
    # We don't know WHICH index they map to.
    # But usually board covers 0-500.
    
    # Let's assume the first line is near index I and last near index J.
    # Uniform spacing d ~ 500/8 = 62.5
    
    # Actually, for this task, the warp is FORCED to be 500x500.
    # So we *expect* lines at 0, 62.5, 125, ... 500.
    # We can snap detected lines to these slots and refine the slots.
    
    ideal_grid = np.linspace(0, max_dim, 9)
    final_grid = ideal_grid.copy()
    
    # Snap detected lines
    for rho in rhos:
        # Find closest ideal line
        idx = (np.abs(ideal_grid - rho)).argmin()
        diff = rho - ideal_grid[idx]
        
        # If match is reasonably close (e.g. within 20px), update the grid point
        # But we want to preserve uniform spacing? 
        # Actually, if the board is warped non-linearly, non-uniform is better.
        # But perspective transform should handle linearity.
        # Let's just trust valid lines.
        
        if abs(diff) < 30:
             final_grid[idx] = rho
             
    # Fill gaps (interpolate between locked points)
    # This is slightly complex. Simpler: just return fixed grid if lines are sparse.
    # Or: Just return the lines we found and fill the rest?
    # Let's stick to FIXED grid if lines are messy, or use the lines if we have 9.
    
    # Given the complexity and "loose" estimation, relying on the lines might be unstable
    # unless we force them to be 9 lines.
    
    # Let's try: JUST return fixed grid for now, but use lines if they are perfect.
    # Actually, the user wants Phase 3.
    # Let's use `chess_board.py` lines to refine the grid.
    
    return final_grid

def get_square_from_grid(x, y, h_grid, v_grid):
    """
    Find which row/col (0-7) the point (x,y) falls into.
    h_grid: 9 y-coords (boundaries)
    v_grid: 9 x-coords (boundaries)
    Includes a tolerance margin to snap pieces slightly outside the grid.
    """
    col = -1
    row = -1
    
    # Tolerance for snapping to edge (e.g. 25px to match 1.08 scale margin ~18px)
    margin_tolerance = 25.0 
    
    # Check Columns (X)
    # Check if inside grid range 0..7
    if x >= v_grid[0] and x < v_grid[-1]:
        for i in range(8):
            if v_grid[i] <= x < v_grid[i+1]:
                col = i
                break
    else:
        # Check margin
        if v_grid[0] - margin_tolerance <= x < v_grid[0]:
            col = 0 # Snap to Left Edge
        elif v_grid[-1] <= x < v_grid[-1] + margin_tolerance:
            col = 7 # Snap to Right Edge
            
    # Check Rows (Y)
    if y >= h_grid[0] and y < h_grid[-1]:
        for i in range(8):
            if h_grid[i] <= y < h_grid[i+1]:
                row = i
                break
    else:
         # Check margin
         if h_grid[0] - margin_tolerance <= y < h_grid[0]:
             row = 0 # Snap to Top Edge
         elif h_grid[-1] <= y < h_grid[-1] + margin_tolerance:
             row = 7 # Snap to Bottom Edge
            
    return col, row

def run_inference(image_path, model_path="runs/chess_pose_train/weights/best.pt", output_dir="result/", grid_method="geometric"):
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception:
        print(f"Error: Could not load model at {model_path}")
        print("Please train the model first using train.py!")
        return

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Run YOLO inference
    print("Running YOLO detection...")
    results = model(img)[0]
    
    # Extract Board Corners (Class 0)
    best_board_idx = -1
    max_conf = -1.0
    
    print(f"Detected {len(results.boxes)} objects.")
    
    # Extract Board Corners (Class 0)
    best_board_idx = -1
    max_conf = -1.0
    
    print(f"Detected {len(results.boxes)} objects.")
    
    for i, (kpts, get_cls, conf, box) in enumerate(zip(results.keypoints.data, results.boxes.cls, results.boxes.conf, results.boxes.xyxy)):
        class_id = int(get_cls)
        confidence = float(conf)
        
        if class_id == 0: # Board
            print(f"Board candidate {i}: Confidence={confidence:.4f}")
            points = kpts.cpu().numpy()
            
            # DEBUG: Save visualization for THIS candidate
            debug_img = img.copy()
            
            # Draw Box
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Keypoints
            for k_idx, point in enumerate(points):
                 x, y = int(point[0]), int(point[1])
                 cv2.circle(debug_img, (x, y), 8, (0, 0, 255), -1)
                 cv2.putText(debug_img, f"{k_idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if output_dir:
                 candidate_path = os.path.join(output_dir, f"debug_candidate_{i}_conf_{confidence:.2f}.jpg")
                 cv2.imwrite(candidate_path, debug_img)
                 print(f"  -> Saved debug image to {candidate_path}")

            if len(points) >= 4:
                # Check if this is the best board
                if confidence > max_conf:
                    max_conf = confidence
                    best_board_idx = i
            else:
                 print(f"  -> Skipped selection (not enough keypoints: {len(points)})")

    board_corners = None
    if best_board_idx != -1:
        print(f"Selected Best Board (Index {best_board_idx}) with Confidence {max_conf:.4f}")
        kpts = results.keypoints.data[best_board_idx]
        points = kpts.cpu().numpy()
        board_corners = points[:4, :2]
    else:
        print("No board detected by model with sufficient confidence.")

    # Fallback / Override: Estimate from pieces
    # If the user says model board is wrong, we should prioritize estimation or provide it as option.
    # Let's try estimation and compare or just use it if available for this specific debugging case.
    print("Attempting to estimate board from pieces...")
    estimated_corners = estimate_board_from_pieces(results)
    
    if estimated_corners is not None:
        if board_corners is None:
             print("Using ESTIMATED board from piece locations (Fallback).")
             board_corners = estimated_corners
        else:
             print("Using Model-Detected Board Corners for Warping (Preferred).")
        
        # DEBUG: Visualize estimated corners
        debug_est = img.copy()
        for i, point in enumerate(estimated_corners):
             x, y = int(point[0]), int(point[1])
             cv2.circle(debug_est, (x, y), 10, (0, 255, 0), -1)
             cv2.putText(debug_est, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Draw all pieces to see what defined the hull
        for kpts, cls in zip(results.keypoints.data, results.boxes.cls):
            if int(cls) == 0: continue
            pts = kpts.cpu().numpy()
            if len(pts) >= 3:
                bx = np.mean(pts[:3, 0])
                by = np.mean(pts[:3, 1])
                cv2.circle(debug_est, (int(bx), int(by)), 5, (255, 255, 0), -1)
                
        if output_dir:
             cv2.imwrite(os.path.join(output_dir, "debug_estimated_board.jpg"), debug_est)
    
    # Collect Piece Points for Dynamic Expansion
    piece_points_for_expansion = []
    for kpts, cls in zip(results.keypoints.data, results.boxes.cls):
        class_id = int(cls)
        if class_id == 0: continue # Skip board
        
        pts = kpts.cpu().numpy()
        if len(pts) >= 3:
            # Use base points as the footprint
            base_points = pts[:3, :2]
            # Add all base points individually or just the mean?
            # Dynamic expansion should cover the whole piece footprint on the board.
            # Adding all 3 base points is safer.
            for bp in base_points:
                piece_points_for_expansion.append(bp)
    
    # Process Board (Warp & Grid)
    processor = ChessBoardProcessor()
    # Phase 3: Detect Grid Lines
    # Pass piece_points for dynamic expansion
    result = processor.process(img, board_corners, piece_points=piece_points_for_expansion, method=grid_method)
    
    if result is not None:
        h_lines, v_lines = result
        # Determine Grid Boundaries
        v_grid = interpolate_grid(v_lines, 500)
        h_grid = interpolate_grid(h_lines, 500)
    else:
        print("Warning: Grid line detection failed. Using uniform grid.")
        h_lines, v_lines = [], []
        v_grid = np.linspace(0, 500, 9)
        h_grid = np.linspace(0, 500, 9)
        
    M = processor.transform_matrix
    
    if processor.warped_image is not None and output_dir:
         warped_path = os.path.join(output_dir, "debug_warped.jpg")
         debug_warped = processor.warped_image.copy()
         
         # Draw Grid Lines
         for x in v_grid:
             cv2.line(debug_warped, (int(x), 0), (int(x), 500), (0, 0, 255), 1) # Red Vertical
         for y in h_grid:
             cv2.line(debug_warped, (0, int(y)), (500, int(y)), (255, 0, 0), 1) # Blue Horizontal
             
         cv2.imwrite(warped_path, debug_warped)
         print(f"Saved debug warped image to {warped_path}")

    if M is None:
        print("Error: Perspective warp failed.")
        return

    # Initialize 8x8 board (list of lists)
    board_state = [['.' for _ in range(8)] for _ in range(8)]
    # Keep track of confidence to resolve duplicates
    board_conf = [[-1.0 for _ in range(8)] for _ in range(8)]
    
    piece_map = get_piece_mapping()

    candidate_pieces = [] # list of dictionaries
    
    # Process detected pieces
    for kpts, cls, conf in zip(results.keypoints.data, results.boxes.cls, results.boxes.conf):
        class_id = int(cls)
        if class_id == 0: continue # Skip board
        
        confidence = float(conf)
        points = kpts.cpu().numpy()
        
        # Check dimensions
        if len(points) >= 3:
             base_points = points[:3, :2]
        else:
             base_points = points[:, :2] # Fallback
             
        # Mean of base points
        piece_x = np.mean(base_points[:, 0])
        piece_y = np.mean(base_points[:, 1])
        
        # Transform to grid
        pt = np.array([[[piece_x, piece_y]]], dtype=np.float32)
        if M is not None:
             warped_pt = cv2.perspectiveTransform(pt, M)[0][0] # (x, y)
             
             # Get Grid indices
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
             
             # Draw on debug image
             if debug_warped is not None:
                  cv2.circle(debug_warped, (int(warped_pt[0]), int(warped_pt[1])), 4, (0, 255, 255), -1)

    # Sort candidates by confidence (Descending)
    candidate_pieces.sort(key=lambda x: x['conf'], reverse=True)
    
    # ---------------------------------------------------------
    # NO CONFLICT RESOLUTION (Overwrite Standard)
    # User requested removal of "missing knight" logic.
    # ---------------------------------------------------------
    for cand in candidate_pieces:
        c, r = cand['col'], cand['row']
        if c != -1 and r != -1:
            # Only place if empty or overwrite with higher confidence?
            # List is sorted by conf DESC. So first one wins.
            if board_state[r][c] == '.':
                 board_state[r][c] = cand['char']
                 board_conf[r][c] = cand['conf']
            else:
                 # Already occupied by higher confidence piece
                 pass

    # ---------------------------------------------------------
    # ORIENTATION LOGIC
    # ---------------------------------------------------------
    white_y_sum, white_x_sum, white_count = 0, 0, 0
    black_y_sum, black_x_sum, black_count = 0, 0, 0
    
    for r in range(8):
        for c in range(8):
            char = board_state[r][c]
            if char.isupper():
                white_y_sum += r
                white_x_sum += c
                white_count += 1
            elif char.islower():
                black_y_sum += r
                black_x_sum += c
                black_count += 1
                
    if white_count > 0 and black_count > 0:
        avg_white_y = white_y_sum / white_count
        avg_black_y = black_y_sum / black_count
        avg_white_x = white_x_sum / white_count
        avg_black_x = black_x_sum / black_count
        
        print(f"Orientation Check: White Y={avg_white_y:.2f} X={avg_white_x:.2f}, Black Y={avg_black_y:.2f} X={avg_black_x:.2f}")
        
        # Determine Rotation
        rotation = 0
        
        # Check Y-axis spread
        if abs(avg_white_y - avg_black_y) > 2.0:
            if avg_white_y < avg_black_y: # White is Up (Top)
                rotation = 180
                print("Orientation: White Top -> Rotate 180")
            else:
                print("Orientation: White Bottom (Standard)")
        else:
            # Check X-axis spread (Sideways)
            if abs(avg_white_x - avg_black_x) > 2.0:
                if avg_white_x > avg_black_x: # White is Right
                    rotation = 90 # CW
                    print("Orientation: White Right -> Rotate 90 CW")
                else: # White is Left
                    rotation = 270 # CW (or -90)
                    print("Orientation: White Left -> Rotate 270 CW")
        
        # Apply Rotation
        if rotation == 90:
             # (r, c) -> (c, 7-r)
             new_state = [['.' for _ in range(8)] for _ in range(8)]
             for r in range(8):
                 for c in range(8):
                     new_state[c][7-r] = board_state[r][c]
             board_state = new_state
        elif rotation == 180:
             # (r, c) -> (7-r, 7-c)
             new_state = [['.' for _ in range(8)] for _ in range(8)]
             for r in range(8):
                 for c in range(8):
                     new_state[7-r][7-c] = board_state[r][c]
             board_state = new_state
        elif rotation == 270:
             # (r, c) -> (7-c, r)
             new_state = [['.' for _ in range(8)] for _ in range(8)]
             for r in range(8):
                 for c in range(8):
                     new_state[7-c][r] = board_state[r][c]
             board_state = new_state
    
    # Debug Print Board
    print("\nDetected Board State:")
    print("  a b c d e f g h")
    for i, row in enumerate(board_state):
        print(f"{8-i} " + " ".join(row))

    # Generate FEN
    fen = board_to_fen(board_state)
    print(f"\nPredicted FEN: {fen}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Save FEN
        fen_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(fen_path, "w") as f:
            f.write(fen)
        print(f"Saved FEN to {fen_path}")
        
        # Save Annotated Image
        save_path = os.path.join(output_dir, f"{base_name}.jpg")
        
        # Get the plotted results (BGR)
        res_plotted = results.plot()
        
        # 1. Overlay the detected grid coordinates
        # 2. Draw the estimated grid lines for debugging
        
        # Draw 8x8 Grid on original image (remapped from normalized warp coords covers entire image?)
        # No, simpler to just overlay text at piece positions like before, but let's try to draw the grid lines.
        # To draw grid lines on the ORIGINAL image, we need to inverse transform the 500x500 grid points.
        
        # Calculate Inverse Matrix
        if M is not None:
             try:
                 M_inv = np.linalg.inv(M)
                 
                 # Draw Vertical Lines (from v_grid)
                 for x in v_grid:
                     pt_top = np.array([[[x, 0]]], dtype=np.float32)
                     pt_bot = np.array([[[x, 500]]], dtype=np.float32)
                     
                     orig_top = cv2.perspectiveTransform(pt_top, M_inv)[0][0]
                     orig_bot = cv2.perspectiveTransform(pt_bot, M_inv)[0][0]
                     
                     cv2.line(res_plotted, (int(orig_top[0]), int(orig_top[1])), (int(orig_bot[0]), int(orig_bot[1])), (0, 255, 255), 2)

                 # Draw Horizontal Lines (from h_grid)
                 for y in h_grid:
                     pt_left = np.array([[[0, y]]], dtype=np.float32)
                     pt_right = np.array([[[500, y]]], dtype=np.float32)
                     
                     orig_left = cv2.perspectiveTransform(pt_left, M_inv)[0][0]
                     orig_right = cv2.perspectiveTransform(pt_right, M_inv)[0][0]
                     
                     cv2.line(res_plotted, (int(orig_left[0]), int(orig_left[1])), (int(orig_right[0]), int(orig_right[1])), (0, 255, 255), 2)
                     
                 # Key Request: Visualize the 4 Keypoints (Corners) used for the grid
                 corners_warped = [
                     [0, 0],     # TL
                     [500, 0],   # TR
                     [500, 500], # BR
                     [0, 500]    # BL
                 ]
                 for cx, cy in corners_warped:
                     pt = np.array([[[cx, cy]]], dtype=np.float32)
                     orig_pt = cv2.perspectiveTransform(pt, M_inv)[0][0]
                     # Draw Large Red Dot
                     cv2.circle(res_plotted, (int(orig_pt[0]), int(orig_pt[1])), 10, (0, 0, 255), -1)
                     # Draw Circle Outline for visibility
                     cv2.circle(res_plotted, (int(orig_pt[0]), int(orig_pt[1])), 12, (255, 255, 255), 2)
                     
             except Exception as e:
                 print(f"Could not draw grid lines: {e}")

        # Label Pieces
        for row in range(8):
            for col in range(8):
                char = board_state[row][col]
                if char != '.':
                     # Find which piece corresponds to this (reverse mapping is hard specifically)
                     # Instead, we should have stored the piece coordinates in the loop above.
                     pass

        # Re-iterating results to draw text (inefficient but simple for debug)
        for kpts, cls in zip(results.keypoints.data, results.boxes.cls):
             if int(cls) == 0: continue
             points = kpts.cpu().numpy()
             if len(points) >= 3:
                 base_points = points[:3, :2]
                 piece_x = np.mean(base_points[:, 0])
                 piece_y = np.mean(base_points[:, 1])
                 
                 # Transform
                 src_point = np.array([[[piece_x, piece_y]]], dtype=np.float32)
                 dst_point = cv2.perspectiveTransform(src_point, M)[0][0]
                 wx, wy = dst_point
                 c = int(wx // 62.5)
                 r = int(wy // 62.5)
                 
                 if 0 <= c < 8 and 0 <= r < 8:
                     coord = f"{chr(97+c)}{8-r}"
                     cv2.putText(res_plotted, coord, (int(piece_x), int(piece_y)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imwrite(save_path, res_plotted)
        print(f"Saved inference result to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="runs/chess_pose_train/weights/best.pt", help="Path to trained model")
    parser.add_argument("--output-dir", default="result/", help="Directory to save results")
    parser.add_argument("--grid-method", default="geometric", choices=["geometric", "canny"], help="Grid detection method: geometric (default) or canny")
    args = parser.parse_args()
    
    run_inference(args.image, args.model, args.output_dir, args.grid_method)
