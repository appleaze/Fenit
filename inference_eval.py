"""
inference_eval.py
=================
Original YOLO chess-piece inference pipeline.

Pipeline stages
---------------
1.  YOLO pose inference       → detections (box, class, keypoints)
2.  Board-corner selection    → highest-confidence class-0 detection
3.  Piece-hull fallback       → estimate corners from piece positions
4.  Perspective warp          → normalise board to 500 × 500 px
5.  Grid line detection       → geometric or Canny-based 9-line grids
6.  Piece-to-cell assignment  → simple boundary lookup
7.  Orientation correction    → piece-average heuristic
8.  FEN generation            → board_to_fen()

Usage
-----
  # single image
  python inference_eval.py path/to/image.jpg

  # run on a directory (random sample of 20)
  python inference_eval.py path/to/folder/

  # specify model / output dir
  python inference_eval.py image.jpg \\
      --model runs/manual_annotate_tuning/trial_11/weights/last.pt \\
      --output-dir result/

  # force board estimation from piece convex hull
  python inference_eval.py image.jpg --force-estimate
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from chess_board import ChessBoardProcessor

# ── Default paths ──────────────────────────────────────────────────────────
DEFAULT_MODEL     = "runs/manual_annotate_tuning/trial_11/weights/last.pt"
DEFAULT_OUTPUT    = "result/"
DEFAULT_GRID      = "geometric"
WARP_SIZE         = 500   # square side length after perspective warp (px)
SNAP_MARGIN       = 25.0  # px tolerance to snap a piece just outside the grid


# ══════════════════════════════════════════════════════════════════════════
# PIECE MAPPING
# ══════════════════════════════════════════════════════════════════════════

def get_piece_mapping() -> dict[int, str]:
    """Return the YOLO class-ID → FEN character mapping.

    Class IDs match the 'names' list in manual_annotate_dataset/data.yaml.
    Upper-case = White pieces, lower-case = Black pieces.
    """
    return {
        1:  'K',   # White King
        2:  'P',   # White Pawn
        3:  'p',   # Black Pawn
        4:  'k',   # Black King
        5:  'Q',   # White Queen
        6:  'B',   # White Bishop
        7:  'N',   # White Knight
        8:  'R',   # White Rook
        9:  'b',   # Black Bishop
        10: 'r',   # Black Rook
        11: 'n',   # Black Knight
        12: 'q',   # Black Queen
    }


# ══════════════════════════════════════════════════════════════════════════
# FEN UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def board_to_fen(board: list[list[str]]) -> str:
    """Convert an 8×8 board (list-of-lists, '.' = empty) to a FEN string.

    Appends the standard suffix for "White to move, no castling, no en-
    passant, 0 half-moves, move 1".
    """
    fen_rows = []
    for row in board:
        empty = 0
        row_str = ""
        for cell in row:
            if cell == '.':
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += cell
        if empty:
            row_str += str(empty)
        fen_rows.append(row_str)

    return "/".join(fen_rows) + " w - - 0 1"


def parse_fen_from_filename(stem: str) -> str | None:
    """Extract a board-FEN from a filename that encodes FEN with _ or - separators.

    A valid 8-rank FEN has exactly 7 separators → 8 parts.
    Returns the board-FEN portion (before any trailing space / suffix), or None.
    """
    n_under = stem.count('_')
    n_dash  = stem.count('-')

    if n_under == 7:
        raw = stem.replace('_', '/')
    elif n_dash == 7:
        raw = stem.replace('-', '/')
    else:
        return None  # not a FEN filename

    # Strip trailing metadata such as " w", "(board)", "_aug_N"
    return raw.split(' ')[0].split('(')[0].strip()


# ══════════════════════════════════════════════════════════════════════════
# BOARD-CORNER ESTIMATION
# ══════════════════════════════════════════════════════════════════════════

def estimate_board_from_pieces(results) -> np.ndarray | None:
    """Estimate board corners from the convex hull of all detected piece bases.

    Returns a (4, 2) float32 array [TL, TR, BR, BL] expanded by 15 % outward
    to reach the board edge, or None if fewer than 4 piece points are found.
    """
    piece_points = []

    for kpts, cls in zip(results.keypoints.data, results.boxes.cls):
        if int(cls) == 0:
            continue  # skip the board class

        pts = kpts.cpu().numpy()
        if len(pts) >= 3:
            # Use mean of the three base keypoints as the piece's board contact
            base = pts[:3, :2]
            piece_points.append([np.mean(base[:, 0]), np.mean(base[:, 1])])

    if len(piece_points) < 4:
        print(f"  [estimate] Only {len(piece_points)} pieces — too few to estimate board.")
        return None

    pts_arr = np.array(piece_points, dtype=np.float32)
    rect    = cv2.minAreaRect(pts_arr)
    center, (w_r, h_r), angle = rect

    # Expand the minimal bounding rectangle to cover the full board
    expanded_rect = (center, (w_r * 1.15, h_r * 1.15), angle)
    expanded_box  = np.array(cv2.boxPoints(expanded_rect), dtype=np.float32)
    return expanded_box


# ══════════════════════════════════════════════════════════════════════════
# GRID UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def interpolate_grid(detected_lines: list, max_dim: int = WARP_SIZE) -> np.ndarray:
    """Snap detected Hough lines onto a 9-point ideal uniform grid.

    Detected lines (rho values) that are within 30 px of an ideal grid
    boundary are used to refine that boundary.  All others stay at the
    uniform position.

    Returns a shape-(9,) float64 array of grid boundaries.
    """
    ideal = np.linspace(0, max_dim, 9)

    if not detected_lines or len(detected_lines) < 2:
        return ideal  # fallback: pure uniform grid

    final = ideal.copy()
    for rho, in detected_lines if isinstance(detected_lines[0], (int, float)) \
            else [(l[0],) for l in detected_lines]:
        idx = int(np.argmin(np.abs(ideal - rho)))
        if abs(rho - ideal[idx]) < 30:
            final[idx] = rho

    return final


def get_square_from_grid(
        x: float, y: float,
        h_grid: np.ndarray, v_grid: np.ndarray
) -> tuple[int, int]:
    """Return (col, row) for a warped point (x, y), with ±25 px snap margin.

    Returns (-1, -1) if the point is outside the grid + margin.
    """
    col = -1
    row = -1

    # ── Column (X direction) ──────────────────────────────────────────────
    if v_grid[0] <= x < v_grid[-1]:
        for i in range(8):
            if v_grid[i] <= x < v_grid[i + 1]:
                col = i
                break
    elif v_grid[0] - SNAP_MARGIN <= x < v_grid[0]:
        col = 0   # snap to left edge
    elif v_grid[-1] <= x < v_grid[-1] + SNAP_MARGIN:
        col = 7   # snap to right edge

    # ── Row (Y direction) ─────────────────────────────────────────────────
    if h_grid[0] <= y < h_grid[-1]:
        for i in range(8):
            if h_grid[i] <= y < h_grid[i + 1]:
                row = i
                break
    elif h_grid[0] - SNAP_MARGIN <= y < h_grid[0]:
        row = 0   # snap to top edge
    elif h_grid[-1] <= y < h_grid[-1] + SNAP_MARGIN:
        row = 7   # snap to bottom edge

    return col, row


# ══════════════════════════════════════════════════════════════════════════
# ORIENTATION
# ══════════════════════════════════════════════════════════════════════════

def _rotate_board_180(board: list[list[str]]) -> list[list[str]]:
    return [[board[7 - r][7 - c] for c in range(8)] for r in range(8)]

def _rotate_board_90(board: list[list[str]]) -> list[list[str]]:
    """90° clockwise: (r, c) → (c, 7-r)."""
    new = [['.' for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            new[c][7 - r] = board[r][c]
    return new

def _rotate_board_270(board: list[list[str]]) -> list[list[str]]:
    """270° clockwise: (r, c) → (7-c, r)."""
    new = [['.' for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            new[7 - c][r] = board[r][c]
    return new


def correct_orientation(board: list[list[str]]) -> list[list[str]]:
    """Rotate the board so White pieces are at the bottom (rows 6-7).

    Uses the average row of all White pieces vs all Black pieces.
    If the separation is predominantly horizontal (Y spread < X spread),
    uses the average column instead.

    Returns the (possibly rotated) board.
    """
    white_rows, white_cols = [], []
    black_rows, black_cols = [], []

    for r in range(8):
        for c in range(8):
            ch = board[r][c]
            if ch.isupper():
                white_rows.append(r); white_cols.append(c)
            elif ch.islower():
                black_rows.append(r); black_cols.append(c)

    if not white_rows or not black_rows:
        print("  [orient] Cannot determine orientation — missing one colour.")
        return board

    avg_wy = sum(white_rows) / len(white_rows)
    avg_by = sum(black_rows) / len(black_rows)
    avg_wx = sum(white_cols) / len(white_cols)
    avg_bx = sum(black_cols) / len(black_cols)

    print(f"  [orient] White avg (row={avg_wy:.2f}, col={avg_wx:.2f}), "
          f"Black avg (row={avg_by:.2f}, col={avg_bx:.2f})")

    y_spread = abs(avg_wy - avg_by)
    x_spread = abs(avg_wx - avg_bx)

    if y_spread > 2.0:
        if avg_wy < avg_by:             # White at top → rotate 180
            print("  [orient] White at top → rotate 180°")
            return _rotate_board_180(board)
        else:
            print("  [orient] Standard (White at bottom) — no rotation")
            return board
    elif x_spread > 2.0:
        if avg_wx > avg_bx:             # White at right → rotate 90 CW
            print("  [orient] White at right → rotate 90° CW")
            return _rotate_board_90(board)
        else:                           # White at left → rotate 270 CW
            print("  [orient] White at left → rotate 270° CW")
            return _rotate_board_270(board)

    print("  [orient] Spread too small — returning board unchanged.")
    return board


# ══════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _save_candidate_debug(img, kpts, box, index: int, conf: float, output_dir: str):
    """Save a debug image highlighting one board-detection candidate."""
    debug = img.copy()
    x1, y1, x2, y2 = map(int, box.cpu().numpy())
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for ki, pt in enumerate(kpts.cpu().numpy()):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(debug, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(debug, str(ki), (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    path = os.path.join(output_dir, f"debug_candidate_{index}_conf_{conf:.2f}.jpg")
    cv2.imwrite(path, debug)
    print(f"  [debug] Candidate saved → {path}")


def _draw_grid_on_warped(warped_img, h_grid, v_grid) -> np.ndarray:
    """Return a copy of the warped image with grid lines drawn on it."""
    dbg = warped_img.copy()
    for x in v_grid:
        cv2.line(dbg, (int(x), 0), (int(x), WARP_SIZE), (0, 0, 255), 1)
    for y in h_grid:
        cv2.line(dbg, (0, int(y)), (WARP_SIZE, int(y)), (255, 0, 0), 1)
    return dbg


def _draw_grid_on_original(annotated, M, h_grid, v_grid):
    """Project the 500 × 500 grid back onto the original image via M_inv."""
    try:
        M_inv = np.linalg.inv(M)
    except Exception as e:
        print(f"  [debug] Could not invert warp matrix: {e}")
        return annotated

    def _project(pt_warp):
        pt = np.array([[[float(pt_warp[0]), float(pt_warp[1])]]], dtype=np.float32)
        return cv2.perspectiveTransform(pt, M_inv)[0][0]

    # Vertical grid lines
    for x in v_grid:
        p1 = tuple(map(int, _project((x, 0))))
        p2 = tuple(map(int, _project((x, WARP_SIZE))))
        cv2.line(annotated, p1, p2, (0, 255, 255), 2)

    # Horizontal grid lines
    for y in h_grid:
        p1 = tuple(map(int, _project((0, y))))
        p2 = tuple(map(int, _project((WARP_SIZE, y))))
        cv2.line(annotated, p1, p2, (0, 255, 255), 2)

    # Board-corner dots
    for cx, cy in [(0, 0), (WARP_SIZE, 0), (WARP_SIZE, WARP_SIZE), (0, WARP_SIZE)]:
        orig = tuple(map(int, _project((cx, cy))))
        cv2.circle(annotated, orig, 10, (0, 0, 255), -1)
        cv2.circle(annotated, orig, 12, (255, 255, 255), 2)

    return annotated


# ══════════════════════════════════════════════════════════════════════════
# CORE INFERENCE
# ══════════════════════════════════════════════════════════════════════════

def process_single_image(
        image_path: str,
        model,
        output_dir: str  = DEFAULT_OUTPUT,
        grid_method: str = DEFAULT_GRID,
        force_estimate: bool = False,
) -> str | None:
    """Run the full detection → FEN pipeline on a single image.

    Parameters
    ----------
    image_path    : Path to the input image.
    model         : Loaded YOLO model instance.
    output_dir    : Directory for debug/result files (None = skip saving).
    grid_method   : 'geometric' or 'canny' (passed to ChessBoardProcessor).
    force_estimate: If True, ignore the YOLO board class and always estimate
                    corners from piece positions.

    Returns
    -------
    The predicted FEN string (e.g. "rnbqkbnr/... w - - 0 1"), or None on failure.
    """
    # ── 1. Load image ─────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Cannot read image: {image_path}")
        return None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── 2. YOLO inference ─────────────────────────────────────────────────
    print("Running YOLO detection...")
    results = model(img)[0]
    print(f"  Detected {len(results.boxes)} objects.")

    # ── 3. Board-corner selection ─────────────────────────────────────────
    board_corners = None

    if not force_estimate:
        best_idx  = -1
        best_conf = -1.0

        for i, (kpts, cls, conf, box) in enumerate(zip(
                results.keypoints.data, results.boxes.cls,
                results.boxes.conf,   results.boxes.xyxy)):

            if int(cls) != 0:
                continue  # only interested in the board class

            c = float(conf)
            print(f"  Board candidate {i}: conf={c:.4f}")

            if output_dir:
                _save_candidate_debug(img, kpts, box, i, c, output_dir)

            pts = kpts.cpu().numpy()
            if len(pts) >= 4 and c > best_conf:
                best_conf = c
                best_idx  = i

        if best_idx != -1:
            print(f"  Selected board candidate {best_idx} (conf={best_conf:.4f})")
            pts = results.keypoints.data[best_idx].cpu().numpy()
            board_corners = pts[:4, :2].astype(np.float32)
        else:
            print("  No board detected with ≥4 keypoints.")
    else:
        print("  Force-estimate mode — ignoring YOLO board class.")

    # ── 4. Piece-hull fallback ────────────────────────────────────────────
    print("Estimating board from piece positions (for comparison / fallback)...")
    estimated = estimate_board_from_pieces(results)

    if estimated is not None:
        if board_corners is None:
            print("  Using estimated corners (no YOLO board / force mode).")
            board_corners = estimated
        else:
            print("  YOLO board corners preferred over estimate.")

        if output_dir:
            dbg_est = img.copy()
            for j, pt in enumerate(estimated):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(dbg_est, (px, py), 10, (0, 255, 0), -1)
                cv2.putText(dbg_est, str(j), (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(output_dir, "debug_estimated_board.jpg"), dbg_est)

    if board_corners is None:
        print("[Error] Could not determine board corners. Skipping.")
        return None

    # ── 5. Piece base-points for dynamic board expansion ──────────────────
    piece_bases = []
    for kpts, cls in zip(results.keypoints.data, results.boxes.cls):
        if int(cls) == 0:
            continue
        pts = kpts.cpu().numpy()
        if len(pts) >= 3:
            for bp in pts[:3, :2]:
                piece_bases.append(bp)

    # ── 6. Perspective warp + grid detection ─────────────────────────────
    processor = ChessBoardProcessor()
    grid_result = processor.process(
        img, board_corners,
        piece_points=piece_bases,
        method=grid_method,
    )

    if grid_result is not None:
        h_lines, v_lines = grid_result
        h_grid = interpolate_grid(h_lines, WARP_SIZE)
        v_grid = interpolate_grid(v_lines, WARP_SIZE)
    else:
        print("  [warn] Grid detection failed — using uniform grid.")
        h_grid = np.linspace(0, WARP_SIZE, 9)
        v_grid = np.linspace(0, WARP_SIZE, 9)

    M = processor.transform_matrix

    # Save warped + grid debug image
    debug_warped = None
    if processor.warped_image is not None and output_dir:
        debug_warped = _draw_grid_on_warped(processor.warped_image, h_grid, v_grid)
        cv2.imwrite(os.path.join(output_dir, "debug_warped.jpg"), debug_warped)
        print(f"  [debug] Warped board saved.")

    if M is None:
        print("[Error] Perspective warp matrix is None.")
        return None

    # ── 7. Piece-to-cell assignment ───────────────────────────────────────
    piece_map   = get_piece_mapping()
    candidates  = []   # [{char, conf, col, row, wx, wy}]

    for kpts, cls, conf in zip(
            results.keypoints.data, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0:
            continue

        pts = kpts.cpu().numpy()
        base_pts = pts[:3, :2] if len(pts) >= 3 else pts[:, :2]

        mean_x = float(np.mean(base_pts[:, 0]))
        mean_y = float(np.mean(base_pts[:, 1]))

        # Warp piece base-point centre to the 500×500 grid
        src = np.array([[[mean_x, mean_y]]], dtype=np.float32)
        wx, wy = map(float, cv2.perspectiveTransform(src, M)[0][0])

        col, row = get_square_from_grid(wx, wy, h_grid, v_grid)

        candidates.append({
            'char': piece_map.get(int(cls), '?'),
            'conf': float(conf),
            'col':  col,
            'row':  row,
            'wx':   wx,
            'wy':   wy,
        })

        if debug_warped is not None:
            cv2.circle(debug_warped, (int(wx), int(wy)), 4, (0, 255, 255), -1)

    # Higher-confidence pieces fill cells first (they win ties)
    candidates.sort(key=lambda c: c['conf'], reverse=True)

    # ── 8. Populate 8×8 board ─────────────────────────────────────────────
    board = [['.' for _ in range(8)] for _ in range(8)]

    for cand in candidates:
        c, r = cand['col'], cand['row']
        if c == -1 or r == -1:
            continue
        if board[r][c] == '.':
            board[r][c] = cand['char']

    # ── 9. Orientation correction ─────────────────────────────────────────
    board = correct_orientation(board)

    # Print board for debugging
    print("\n  Board state:")
    print("    a b c d e f g h")
    for i, row in enumerate(board):
        print(f"  {8-i} {' '.join(row)}")

    # ── 10. FEN generation ────────────────────────────────────────────────
    fen = board_to_fen(board)
    print(f"\n  Predicted FEN: {fen}")

    # ── 11. Save outputs ──────────────────────────────────────────────────
    if output_dir:
        stem = Path(image_path).stem

        # FEN text file
        fen_path = os.path.join(output_dir, f"{stem}.txt")
        with open(fen_path, 'w') as f:
            f.write(fen)
        print(f"  [save] FEN  → {fen_path}")

        # Annotated image with grid overlay
        annotated = results.plot()
        annotated = _draw_grid_on_original(annotated, M, h_grid, v_grid)
        img_path  = os.path.join(output_dir, f"{stem}.jpg")
        cv2.imwrite(img_path, annotated)
        print(f"  [save] IMG  → {img_path}")

    return fen


# ══════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_inference(
        image_source: str,
        model_path: str      = DEFAULT_MODEL,
        output_dir: str      = DEFAULT_OUTPUT,
        grid_method: str     = DEFAULT_GRID,
        force_estimate: bool = False,
        sample_size: int     = 20,
) -> None:
    """Load the model and run inference on one image OR a directory.

    When given a directory, a random sample of up to `sample_size`
    non-augmented images is chosen.  If the filenames encode a FEN
    (7 underscores), accuracy is measured automatically.

    Parameters
    ----------
    image_source   : Path to an image file or directory.
    model_path     : Path to YOLO weights.
    output_dir     : Where to write debug/result files.
    grid_method    : 'geometric' or 'canny'.
    force_estimate : Override board-class detection with piece-hull estimate.
    sample_size    : Max images to sample when `image_source` is a directory.
    """
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[Error] Could not load model — {e}")
        return

    # ── Collect image paths ───────────────────────────────────────────────
    if os.path.isdir(image_source):
        all_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(image_source)
            for f in files
            if not '_aug_' in f.lower()
            and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not all_paths:
            print(f"[Error] No images found in {image_source}")
            return

        n = min(sample_size, len(all_paths))
        image_paths = random.sample(all_paths, n)
        print(f"  {len(all_paths)} eligible images found — evaluating {n}.")
    else:
        image_paths = [image_source]

    # ── Run and evaluate ──────────────────────────────────────────────────
    correct = 0
    total   = 0

    for img_path in image_paths:
        print(f"\n{'─'*60}")
        print(f"Image: {img_path}")

        pred_fen = process_single_image(
            img_path, model, output_dir, grid_method, force_estimate
        )
        if pred_fen is None:
            continue

        true_fen = parse_fen_from_filename(Path(img_path).stem)
        if true_fen is None:
            print("  (No ground-truth FEN in filename — skipping accuracy check.)")
            continue

        pred_board = pred_fen.split(' ')[0]
        total += 1

        if pred_board == true_fen:
            print("  ✓ MATCH")
            correct += 1
        else:
            print(f"  ✗ MISMATCH")
            print(f"    Pred: {pred_board}")
            print(f"    True: {true_fen}")

    # ── Summary ───────────────────────────────────────────────────────────
    if total:
        acc = correct / total * 100
        print(f"\n{'='*60}")
        print(f"  FEN Images Evaluated : {total}")
        print(f"  Correct              : {correct}")
        print(f"  Accuracy             : {acc:.2f} %")
        print(f"{'='*60}")
    else:
        print("\n  No FEN-named images to evaluate.")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLO chess inference pipeline — original version."
    )
    parser.add_argument(
        "image",
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO weights path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help=f"Directory for result/debug files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--grid-method",
        default=DEFAULT_GRID,
        choices=["geometric", "canny"],
        help="Grid detection method (default: geometric)",
    )
    parser.add_argument(
        "--force-estimate",
        action="store_true",
        help="Estimate board corners from piece convex hull, ignoring the YOLO board class.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        metavar="N",
        help="Max images to randomly sample when input is a directory (default: 20).",
    )
    args = parser.parse_args()

    run_inference(
        args.image,
        model_path     = args.model,
        output_dir     = args.output_dir,
        grid_method    = args.grid_method,
        force_estimate = args.force_estimate,
        sample_size    = args.sample,
    )
