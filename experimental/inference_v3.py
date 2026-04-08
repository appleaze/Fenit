import cv2
import numpy as np
from ultralytics import YOLO
from chess_board import ChessBoardProcessor
import os
from pathlib import Path
import random

def get_piece_mapping():
    # Matches 'names' in manual_annotate_dataset/data.yaml
    return {
        1: 'K',  # White-King
        2: 'P',  # White-Pawn
        3: 'p',  # Black-Pawn
        4: 'k',  # Black-King
        5: 'Q',  # White-Queen
        6: 'B',  # White-Bishop
        7: 'N',  # White-Knight
        8: 'R',  # White-Rook
        9: 'b',  # Black-Bishop
        10: 'r', # Black-Rook
        11: 'n', # Black-Knight
        12: 'q'  # Black-Queen
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
    if not lines or len(lines) < 2:
        return np.linspace(0, max_dim, 9)
    ideal_grid = np.linspace(0, max_dim, 9)
    final_grid = ideal_grid.copy()
    rhos = [l[0] for l in lines]
    for rho in rhos:
        idx = (np.abs(ideal_grid - rho)).argmin()
        if abs(rho - ideal_grid[idx]) < 30:
            final_grid[idx] = rho
    return final_grid


def refine_grid_with_chessboard_corners(warped_img, warp_size=500):
    """
    Option A — OpenCV findChessboardCorners grid refinement.

    Detects the 7×7 inner corner lattice of the warped chessboard image
    and fits a precise 9-point grid per axis using linear regression on
    the detected corner positions.  This is completely independent of YOLO
    board keypoint quality.

    Returns (h_grid, v_grid) as shape-(9,) float arrays, or None on failure.
    """
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) if warped_img.ndim == 3 else warped_img.copy()

    # Enhance contrast — helps with dark/uneven lighting
    gray = cv2.equalizeHist(gray)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, (7, 7), flags)

    if not found or corners is None:
        return None

    # Sub-pixel refinement on the detected inner corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # corners shape: (49, 1, 2) → reshape to (7, 7, 2)
    # Row-major order: corners[row*7 + col] = (x, y)
    c = corners.reshape(7, 7, 2)

    # Average column x-positions over all 7 rows → 7 inner column positions
    col_x = c[:, :, 0].mean(axis=0)   # shape (7,)
    # Average row y-positions over all 7 columns → 7 inner row positions
    row_y = c[:, :, 1].mean(axis=1)   # shape (7,)

    # Fit a line through the 7 detected inner positions (index 1..7 in the 9-point grid)
    idx = np.arange(7, dtype=np.float64)
    col_a, col_b = np.polyfit(idx, col_x, 1)   # col_x[i] ≈ col_a*i + col_b
    row_a, row_b = np.polyfit(idx, row_y, 1)   # row_y[i] ≈ row_a*i + row_b

    # Extrapolate to the two boundary positions (index -1 and 7 in fit space)
    # Inner corners are at grid positions 1..7 → board boundaries are at 0 and 8
    v_grid = np.array([col_b - col_a] +
                      [col_a * i + col_b for i in range(7)] +
                      [col_a * 7 + col_b], dtype=np.float64)
    h_grid = np.array([row_b - row_a] +
                      [row_a * i + row_b for i in range(7)] +
                      [row_a * 7 + row_b], dtype=np.float64)

    # Sanity check: grid must stay roughly within [0, warp_size]
    if v_grid[0] < -warp_size * 0.2 or v_grid[-1] > warp_size * 1.2:
        return None
    if h_grid[0] < -warp_size * 0.2 or h_grid[-1] > warp_size * 1.2:
        return None

    return h_grid, v_grid

def get_square_from_grid(x, y, h_grid, v_grid):
    """
    [FIXED] Nearest-cell-centre assignment.
    Assigns (x, y) to the grid cell whose centre is closest, rather than
    using hard boundary comparisons.  This tolerates pieces sitting exactly
    on a grid line and corrects off-by-one errors from small warp inaccuracies.
    Rejects points that are more than one cell-width outside the board.
    """
    cell_w = (v_grid[-1] - v_grid[0]) / 8   # nominal cell width  (~62.5 px)
    cell_h = (h_grid[-1] - h_grid[0]) / 8   # nominal cell height (~62.5 px)
    margin = 0.75  # allow up to 75% of a cell outside the board edge

    # Column centres
    v_centres = np.array([(v_grid[i] + v_grid[i+1]) / 2.0 for i in range(8)])
    # Row centres
    h_centres = np.array([(h_grid[i] + h_grid[i+1]) / 2.0 for i in range(8)])

    # Reject if way outside the board
    if x < v_grid[0] - cell_w * margin or x > v_grid[-1] + cell_w * margin:
        col = -1
    else:
        col = int(np.argmin(np.abs(v_centres - x)))

    if y < h_grid[0] - cell_h * margin or y > h_grid[-1] + cell_h * margin:
        row = -1
    else:
        row = int(np.argmin(np.abs(h_centres - y)))

    return col, row


def compute_iou(boxA, boxB):
    """Compute Intersection-over-Union of two [xmin, ymin, xmax, ymax] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)


def get_square_iou(base_points_warped, h_grid, v_grid, occupied_cells=None, use_hybrid=False):
    """
    Assigns a piece (described by its warped base-keypoint footprint) to the
    best-matching unoccupied grid cell using IoU, with an optional hybrid
    centre-distance penalty.
    Returns (col, row) — both -1 if no cell scores above the minimum IoU.
    """
    if occupied_cells is None:
        occupied_cells = set()

    xs = [p[0] for p in base_points_warped]
    ys = [p[1] for p in base_points_warped]
    padding = 5.0
    piece_box = [
        min(xs) - padding, min(ys) - padding,
        max(xs) + padding, max(ys) + padding,
    ]
    piece_center_x = (piece_box[0] + piece_box[2]) / 2.0
    piece_center_y = (piece_box[1] + piece_box[3]) / 2.0

    cell_w = (v_grid[-1] - v_grid[0]) / 8.0
    cell_h = (h_grid[-1] - h_grid[0]) / 8.0

    best_score = -1.0
    best_iou   = 0.0
    best_c = best_r = -1

    for r in range(8):
        for c in range(8):
            if (r, c) in occupied_cells:
                continue
            cell_box = [v_grid[c], h_grid[r], v_grid[c + 1], h_grid[r + 1]]
            iou = compute_iou(piece_box, cell_box)
            if use_hybrid:
                cell_cx = (cell_box[0] + cell_box[2]) / 2.0
                cell_cy = (cell_box[1] + cell_box[3]) / 2.0
                dist_x  = abs(piece_center_x - cell_cx) / cell_w
                dist_y  = abs(piece_center_y - cell_cy) / cell_h
                score   = iou - 0.1 * (dist_x ** 2 + dist_y ** 2) ** 0.5
            else:
                score = iou
            if score > best_score:
                best_score = score
                best_iou   = iou
                best_r = r
                best_c = c

    if best_iou < 0.15:
        return -1, -1
    return best_c, best_r


def process_single_image(image_path, model, output_dir="result/", grid_method="geometric",
                         force_estimate=False, use_hybrid=False):
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
    if best_board_idx != -1 and not force_estimate:
        print(f"Selected Best Board (Index {best_board_idx}) with Confidence {max_conf:.4f}")
        kpts = results.keypoints.data[best_board_idx]
        points = kpts.cpu().numpy()
        board_corners = points[:4, :2].astype(np.float32)

        # [FIX] Sub-pixel corner refinement — reduces off-by-one grid errors
        try:
            gray_refine = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # cornerSubPix requires shape (N, 1, 2) and contiguous float32
            corners_in  = np.ascontiguousarray(board_corners.reshape(-1, 1, 2), dtype=np.float32)
            refined     = cv2.cornerSubPix(gray_refine, corners_in, (11, 11), (-1, -1), criteria)
            board_corners = refined.reshape(-1, 2)
            print("Applied sub-pixel corner refinement.")
        except Exception as e:
            print(f"Sub-pixel refinement skipped: {e}")

    elif force_estimate:
        print("Ignoring Model-Detected Board (Force Estimate Enabled).")
    else:
        print("No board detected by model with sufficient confidence.")

    # Fallback / Override: Estimate from pieces
    # If the user says model board is wrong, we should prioritize estimation or provide it as option.
    # Let's try estimation and compare or just use it if available for this specific debugging case.
    print("Attempting to estimate board from pieces...")
    estimated_corners = estimate_board_from_pieces(results)
    
    if estimated_corners is not None:
        if board_corners is None:
             print("Using ESTIMATED board from piece locations (Fallback/Forced).")
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
        if output_dir:
             cv2.imwrite(os.path.join(output_dir, "debug_estimated_board.jpg"), debug_est)
             
    if board_corners is None:
        print("Error: Could not determine board corners from model or estimation. Skipping image.")
        return None
    
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
    debug_warped = None

    # ── Option A: findChessboardCorners grid refinement ──────────────────
    # Attempt to detect the 7×7 inner lattice on the warped board image.
    # If successful, this replaces the uniform/hough grid with a precise
    # self-calibrating grid that is independent of YOLO board keypoints.
    grid_source = "uniform"  # track which method was used
    if processor.warped_image is not None:
        refined = refine_grid_with_chessboard_corners(processor.warped_image)
        if refined is not None:
            h_grid, v_grid = refined
            grid_source = "findChessboardCorners"
            print("[GRID] findChessboardCorners succeeded — using precise 7×7 lattice grid.")
        else:
            print("[GRID] findChessboardCorners failed — falling back to uniform/hough grid.")
    print(f"[GRID] Source: {grid_source}")

    if processor.warped_image is not None and output_dir:
        warped_path = os.path.join(output_dir, "debug_warped.jpg")
        debug_warped = processor.warped_image.copy()

        # Draw Grid Lines
        for x in v_grid:
            cv2.line(debug_warped, (int(x), 0), (int(x), 500), (0, 0, 255), 1)  # Red Vertical
        for y in h_grid:
            cv2.line(debug_warped, (0, int(y)), (500, int(y)), (255, 0, 0), 1)  # Blue Horizontal

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

    candidate_pieces = []  # list of dictionaries

    # Process detected pieces — collect warped base-keypoint footprints
    for kpts, cls, conf in zip(results.keypoints.data, results.boxes.cls, results.boxes.conf):
        class_id = int(cls)
        if class_id == 0:
            continue  # skip board

        confidence = float(conf)
        points     = kpts.cpu().numpy()

        if len(points) >= 3:
            base_points = points[:3, :2]
        else:
            base_points = points[:, :2]  # fallback

        # Mean centre for debug drawing
        piece_x = np.mean(base_points[:, 0])
        piece_y = np.mean(base_points[:, 1])

        if M is not None:
            # Transform each base keypoint individually → IoU footprint
            pts = np.array([[[bp[0], bp[1]]] for bp in base_points], dtype=np.float32)
            warped_pts = cv2.perspectiveTransform(pts, M)
            base_points_warped = [(wpt[0][0], wpt[0][1]) for wpt in warped_pts]

            # Also transform the mean centre for debug drawing
            pt_mean   = np.array([[[piece_x, piece_y]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(pt_mean, M)[0][0]

            piece_char = piece_map.get(class_id, '?')
            candidate_pieces.append({
                'char':               piece_char,
                'conf':               confidence,
                'base_points_warped': base_points_warped,
                'x':                  warped_pt[0],
                'y':                  warped_pt[1],
            })

            if debug_warped is not None:
                cv2.circle(debug_warped, (int(warped_pt[0]), int(warped_pt[1])), 4, (0, 255, 255), -1)

    # Sort candidates by confidence descending so higher-confidence pieces
    # claim their cell first (no-conflict IoU assignment below).
    candidate_pieces.sort(key=lambda x: x['conf'], reverse=True)

    # --- IoU-based piece-to-cell assignment (key fix for accuracy) ---
    occupied_cells = set()
    for cand in candidate_pieces:
        col, row = get_square_iou(
            cand['base_points_warped'], h_grid, v_grid,
            occupied_cells=occupied_cells, use_hybrid=use_hybrid
        )
        cand['col'] = col
        cand['row'] = row
        if col != -1 and row != -1:
            occupied_cells.add((row, col))
    
    # ---------------------------------------------------------
    # PHASE 1: Populate Temp Board for Orientation Inference
    # ---------------------------------------------------------
    temp_board_state = [['.' for _ in range(8)] for _ in range(8)]
    for cand in candidate_pieces:
        c, r = cand.get('col', -1), cand.get('row', -1)
        if c != -1 and r != -1:
            if temp_board_state[r][c] == '.':
                temp_board_state[r][c] = cand['char']

    # ----------------------------------------------------------------
    # ORIENTATION LOGIC [FIXED v2] — 3-tier system
    #
    #  Tier 1 (PRIMARY)  : Corner-colour detection
    #    In standard orientation (white at bottom), the top-left warped
    #    cell corresponds to square a8, which is a LIGHT square.
    #    If the sampled cell is DARK, the board is upside-down → rotate 180°.
    #    This is physics-level reliable and works without any pieces.
    #
    #  Tier 2 (SECONDARY): King-anchor
    #    If corner-colour is ambiguous (uniform board, bad lighting), fall back
    #    to comparing white-king vs black-king row/col positions.
    #
    #  Tier 3 (FALLBACK) : Piece-average heuristic (original method)
    # ----------------------------------------------------------------
    rotation = 0
    orientation_method = "none"

    # ── Tier 1: Corner colour ──────────────────────────────────────────
    if processor.warped_image is not None:
        warped_for_orient = processor.warped_image
        cell_size = warped_for_orient.shape[0] // 8   # ~62 px

        # Sample the 4 corner cells to get a contrast estimate
        tl_cell = warped_for_orient[2:cell_size-2, 2:cell_size-2]  # a8 → should be LIGHT
        tr_cell = warped_for_orient[2:cell_size-2, -cell_size+2:-2] # h8 → should be DARK
        tl_bright = float(np.mean(tl_cell if tl_cell.ndim == 2 else cv2.cvtColor(tl_cell, cv2.COLOR_BGR2GRAY)))
        tr_bright = float(np.mean(tr_cell if tr_cell.ndim == 2 else cv2.cvtColor(tr_cell, cv2.COLOR_BGR2GRAY)))
        contrast  = abs(tl_bright - tr_bright)

        print(f"Corner colour — TL(a8) brightness={tl_bright:.1f}, "
              f"TR(h8) brightness={tr_bright:.1f}, contrast={contrast:.1f}")

        if contrast > 20:  # sufficient contrast to trust the reading
            # a8 should be LIGHT in standard orientation
            if tl_bright < tr_bright:   # TL is darker → a8 is dark → board is flipped
                rotation = 180
                print("Corner colour: a8 is DARK → board flipped → Rotate 180°")
            else:
                print("Corner colour: a8 is LIGHT → standard orientation (no rotation)")
            orientation_method = "corner_colour"
        else:
            print(f"Corner colour: contrast too low ({contrast:.1f}) — falling back to king-anchor.")

    # ── Tier 2: King-anchor (if corner colour was inconclusive) ───────
    if orientation_method == "none":
        white_king_pos = None
        black_king_pos = None
        for r in range(8):
            for c in range(8):
                ch = temp_board_state[r][c]
                if ch == 'K': white_king_pos = (r, c)
                elif ch == 'k': black_king_pos = (r, c)

        if white_king_pos and black_king_pos:
            wr, wc = white_king_pos
            br, bc = black_king_pos
            print(f"King-anchor: White K at (row={wr},col={wc}), Black k at (row={br},col={bc})")
            dy, dx = abs(wr - br), abs(wc - bc)
            if dy >= dx:
                if wr < br:
                    rotation = 180
                    print("King-anchor: White King at top → Rotate 180°")
                else:
                    print("King-anchor: White King at bottom (no rotation)")
            else:
                if wc > bc:
                    rotation = 90
                    print("King-anchor: White King at right → Rotate 90° CW")
                else:
                    rotation = 270
                    print("King-anchor: White King at left → Rotate 270° CW")
            orientation_method = "king_anchor"

    # ── Tier 3: Piece-average fallback ────────────────────────────────
    if orientation_method == "none":
        print("Orientation fallback: using piece-average heuristic.")
        wy = wx = wc_cnt = by = bx = bc_cnt = 0
        for r in range(8):
            for c in range(8):
                ch = temp_board_state[r][c]
                if ch.isupper():   wy += r; wx += c; wc_cnt += 1
                elif ch.islower(): by += r; bx += c; bc_cnt += 1
        if wc_cnt > 0 and bc_cnt > 0:
            ayw, axw = wy / wc_cnt, wx / wc_cnt
            ayb, axb = by / bc_cnt, bx / bc_cnt
            print(f"  Piece avg — White (row={ayw:.1f},col={axw:.1f}), Black (row={ayb:.1f},col={axb:.1f})")
            if abs(ayw - ayb) > 2.0:
                if ayw < ayb: rotation = 180
            elif abs(axw - axb) > 2.0:
                rotation = 90 if axw > axb else 270
        print(f"  Fallback rotation: {rotation}°")
        orientation_method = "piece_average"

    print(f"Orientation method used: {orientation_method} → rotation={rotation}°")

    # ── Apply Rotation & Phase 2 Placement with Constraints ─────────────
    
    # Initialize 8x8 final rotated board
    board_state = [['.' for _ in range(8)] for _ in range(8)]
    board_conf = [[-1.0 for _ in range(8)] for _ in range(8)]
    
    white_kings_placed = 0
    black_kings_placed = 0

    def rotate_coords(r, c, rot):
        if rot == 90: return c, 7-r
        elif rot == 180: return 7-r, 7-c
        elif rot == 270: return 7-c, r
        return r, c
        
    for cand in candidate_pieces:
        c, r = cand['col'], cand['row']
        if c == -1 or r == -1: continue
        
        final_r, final_c = rotate_coords(r, c, rotation)
        char = cand['char']
        
        # --- Logical Chess Engine Constraints ---
        
        # 1. Pawns cannot be on the 1st or 8th rank
        if char.lower() == 'p' and (final_r == 0 or final_r == 7):
            print(f"  -> Constraint: Skipping {char} at {chr(97+final_c)}{8-final_r} (Pawns cannot be on edge ranks)")
            continue
            
        # 2. Maximum of 1 White King and 1 Black King
        if char == 'K':
            if white_kings_placed >= 1:
                print(f"  -> Constraint: Skipping extra White King at {chr(97+final_c)}{8-final_r}")
                continue
        elif char == 'k':
            if black_kings_placed >= 1:
                print(f"  -> Constraint: Skipping extra Black King at {chr(97+final_c)}{8-final_r}")
                continue
        
        # Place piece if empty
        if board_state[final_r][final_c] == '.':
            board_state[final_r][final_c] = char
            board_conf[final_r][final_c] = cand['conf']
            if char == 'K': white_kings_placed += 1
            if char == 'k': black_kings_placed += 1
    
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

    return fen

def run_inference(image_source,
                  model_path="runs/manual_annotate_tuning/trial_11/weights/last.pt",
                  output_dir="result/",
                  grid_method="geometric",
                  force_estimate=False,
                  use_hybrid=False,
                  use_all=False):
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error: Could not load model at {model_path} - {e}")
        return

    if os.path.isdir(image_source):
        all_image_paths = []
        for root, _, files in os.walk(image_source):
            for f in files:
                if "_aug_" in f.lower():
                    continue
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(root, f))

        if not all_image_paths:
            print(f"No images found in {image_source} or its subdirectories.")
            return

        if use_all:
            image_paths = all_image_paths
            print(f"Found {len(all_image_paths)} images. Evaluating ALL.")
        else:
            sample_size = min(len(all_image_paths), 20)
            image_paths = random.sample(all_image_paths, sample_size)
            print(f"Found {len(all_image_paths)} images. Sampled {sample_size}. Use --all to evaluate everything.")
    else:
        image_paths = [image_source]

    correct_count = 0
    total_count = 0  # Only count images that represent FENs

    for img_path in image_paths:
        print(f"\n--- Processing {img_path} ---")
        pred_fen = process_single_image(img_path, model, output_dir, grid_method, force_estimate, use_hybrid)
        
        if pred_fen is None:
            continue
            
        base_name = Path(img_path).stem
        
        # Check if it has 7 underscores or dashes (a valid FEN has 8 parts => 7 separators)
        underscore_count = base_name.count('_')
        dash_count = base_name.count('-')
        
        if underscore_count == 7 or dash_count == 7:
            if underscore_count == 7:
                 raw_fen = base_name.replace("_", "/")
            else:
                 raw_fen = base_name.replace("-", "/")
                 
            # Clean up suffixes like " w", "(board)", or "_aug_X"
            clean_fen = raw_fen.split(" ")[0].split("(")[0].split("_aug_")[0].split("/aug/")[0]
            
            # Our prediction usually contains the turn info: " w - - 0 1"
            # So extract just the board part:
            pred_board_fen = pred_fen.split(" ")[0]
            
            if pred_board_fen == clean_fen:
                print(f"MATCH! Predicted FEN matches filename.")
                correct_count += 1
            else:
                print(f"MISMATCH! \nPred: {pred_board_fen}\nTrue: {clean_fen}")
                
            total_count += 1
        else:
            print(f"Filename '{base_name}' doesn't look like a FEN. Skipping eval comparison.")

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100 
        print(f"\n==========================================")
        print(f"Testing Complete!")
        print(f"Total FEN Images Evaluated: {total_count}")
        print(f"Correct FENs: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"==========================================")
    else:
        print("\nNo FEN images were evaluated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="YOLO chess inference pipeline v3 — fixed pipeline with Trial 11 weights."
    )
    parser.add_argument("image", help="Path to image file or directory")
    parser.add_argument(
        "--model",
        default="runs/manual_annotate_tuning/trial_11/weights/last.pt",
        help="Path to trained model (default: Trial 11)"
    )
    parser.add_argument("--output-dir",    default="result/",  help="Directory to save results")
    parser.add_argument("--grid-method",   default="geometric", choices=["geometric", "canny"],
                        help="Grid detection method (default: geometric)")
    parser.add_argument("--force-estimate", action="store_true",
                        help="Force board estimation from piece locations")
    parser.add_argument("--use-hybrid",    action="store_true",
                        help="Use hybrid IoU + centre-distance scoring for cell assignment")
    parser.add_argument("--all",           action="store_true", dest="use_all",
                        help="Evaluate ALL images in a directory (default: random sample of 20)")
    args = parser.parse_args()

    run_inference(
        args.image, args.model, args.output_dir, args.grid_method,
        force_estimate=args.force_estimate,
        use_hybrid=args.use_hybrid,
        use_all=args.use_all,
    )
