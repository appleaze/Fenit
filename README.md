# YOLOv8 Chess Pose Training Project

This project implements a YOLOv8-Pose model training pipeline for detecting chess board keypoints and pieces.

## Project Structure
- `dataset_final/`: consolidated dataset (Train/Val split)
- `runs/`: Training outputs (logs, weights, examples)
- `train.py`: Main training script
- `prepare_dataset.py`: Script used to organize the raw data
- `inference.py`: Inference script for FEN generation
- `cvat_workflow.py`: Tools for CVAT auto-annotation and format conversion
- `chess_board.py`: Board processing and grid detection logic
- `data.yaml`: YOLO configuration file

## 1. Dataset Preparation
The dataset was organized using `prepare_dataset.py`, which:
*   Scanned source folders (`original`, `augmented`, `More Augmented`).
*   Found **2,187 images**.
*   Consolidated them into `dataset_final/`.
*   Split: 1749 Train / 438 Val.

## 2. Training
The training uses the **YOLOv8l-pose** model.

### Prerequisites
- Python 3.11+
- Ultralytics (`pip install ultralytics`)
- PyTorch with CUDA support (for GPU training)

### Running Training
To run the full training with the original paper's hyperparameters:

1.  Open `train.py`
2.  Ensure these settings are uncommented/set:
    ```python
    epochs=150,
    device=0,  # Use GPU
    pose=24.0  # Increased pose loss
    ```
3.  Run the script:
    ```bash
    python train.py
    ```

> [!NOTE]
> The script is currently configured for a 1-epoch CPU test run (`device='cpu'`) to verify the pipeline.

## 3. Inference
Run detection and FEN generation on a single image:

```bash
python inference.py path/to/image.jpg
```

**Arguments:**
- `--model`: Path to model weights (default: `runs/chess_pose_train/weights/best.pt`)
- `--output-dir`: Output directory for FEN string and debug images (default: `result/`)

## 4. CVAT Workflow
Utilities for integrating with CVAT for dataset management.

### Auto-Annotation
Connect to a local CVAT instance to auto-annotate a task:
```bash
python cvat_workflow.py annotate --task-id <ID> --model runs/chess_pose_train/weights/best.pt
```

### Conversion
Convert CVAT-exported COCO JSON annotations to YOLO format:
```bash
python cvat_workflow.py convert --annotations-dir <dir_with_jsons>
```

## 5. Results
Output logs and weights will be saved to `runs/chess_pose_train`. Check `runs/chess_pose_train/train_batch0.jpg` to verify data loading.
