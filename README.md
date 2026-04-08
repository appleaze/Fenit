# YOLOv8 Chess Pose Estimation Project

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Pose-blue" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Python-3.11+-green" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

## Abstract

This repository contains the official inference and training pipeline for the thesis project on **YOLO-based Chess Piece Detection & Forsyth-Edwards Notation (FEN) Generation**. By leveraging the YOLOv8 Pose model, this system effectively identifies individual chess pieces, estimates their bases as keypoints, and predicts a robust chessboard grid layout to overcome aggressive perspective distortion.

This allows for highly accurate chessboard digitisation from monocular RGB images without relying on standard 2D bounding boxes. 

## Repository Structure

The directory has been streamlined to highlight the core operations. Experimental scripts and obsolete generations are housed in `experimental/`.

*   `api.py` — FastAPI deployment script for FEN inferences.
*   `inference_eval.py` — The core inference pipeline (Pose extraction, perspective warping, piece placement, and FEN generation).
*   `chess_board.py` — Grid line detection and perspective-warping geometry logic. 
*   `train.py` & `train_custom.py` — YOLOv8 training scripts utilizing the default structural hyperparameters.
*   `prepare_dataset.py` & `augment_dataset.py` — Data aggregation and automated augmentation.
*   `cvat_workflow.py` — Automation tools for importing/exporting datasets to CVAT.
*   `experimental/` — Deprecated inference loops and archived hyperparameter search databases (Optuna).

## Getting Started

### Prerequisites
* Python 3.11+
* PyTorch with CUDA support (for accelerated rendering and training)

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/chess-yolov8-pose.git
cd chess-yolov8-pose
pip install -r requirements_api.txt
```

## The Pipeline Workflow

### 1. Data Preparation and CVAT Annotation
Our dataset relies on a combination of purely annotated real-world boards and artificially augmented samples. Use the utilities in `cvat_workflow.py` to auto-annotate and push data to a local CVAT instance for QA:
```bash
python cvat_workflow.py annotate --task-id <ID> --model path/to/weights.pt
```
Once approved, data is structured into `dataset_final/` using `prepare_dataset.py` and subsequently expanded with `augment_dataset.py`.

### 2. Training the Model
To reproduce the experimental training sequence, run:
```bash
python train.py
```
This loads our `YOLOv8l-pose` configuration, scales the pose loss factor (hyperparameter: `pose=24.0`), and writes logs, weights, and validation charts to `runs/`.

### 3. Running Core Inference
To predict the FEN of a single image and output debug visualizations to `result/`:
```bash
python inference_eval.py path/to/image.jpg --model path/to/best_weights.pt
```
_Add the `--force-estimate` flag to calculate grid limits based solely on piece convex-hulls when YOLO board recognition struggles on blurred samples._

### 4. API Deployment
A Fast API layer is provided for mobile app interaction or integration into automated judging pipelines:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
POST an image payload to `/predict/` to receive a real-time JSON response carrying the predicted FEN state.

## Citation

If you utilized this codebase or our methodology in your research, please cite:
```bibtex
@mastersthesis{YourThesis2026,
  author       = {Your Name},
  title        = {YOLOv8 Pose Estimation for Chessboard Recognition},
  school       = {Your University},
  year         = {2026},
}
```

## Acknowledgments
* Built utilizing the [Ultralytics YOLOv8 architecture](https://github.com/ultralytics/ultralytics). 
* Special thanks to CVAT community developers.
