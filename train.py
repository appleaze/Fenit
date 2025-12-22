import argparse
import optuna
from ultralytics import YOLO

# Constants
# Using yolov8l-pose.pt as the base model
MODEL_PATH = "yolov8l-pose.pt" 
DATA_YAML = "g:/Dean/Thesis/yolo/dataset_final/data.yaml"

def objective(trial):
    # Trial 0: Default parameters suggestion to anchor the search
    if trial.number == 0:
        args = {"epochs": 100, "batch": 16}
    else:
        args = {
            "optimizer": "AdamW",
            "lr0": trial.suggest_float("lr0", 0.001, 0.5),
            "momentum": trial.suggest_float("momentum", 0.9, 0.999),
            "weight_decay": trial.suggest_float("weight_decay", 0.0001, 0.001),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 3, 10),
            "warmup_momentum": trial.suggest_float("warmup_momentum", 0.01, 0.99),
            "warmup_bias_lr": trial.suggest_float("warmup_bias_lr", 0.1, 0.9),
            "box": trial.suggest_float("box", 0.5, 2.0),
            "cls": trial.suggest_float("cls", 6.0, 12.0),
            "pose": trial.suggest_float("pose", 12.0, 24.0),
            "kobj": trial.suggest_float("kobj", 2.0, 12.0),
            "batch": trial.suggest_int("batch", 8, 16),
            "epochs": trial.suggest_int("epochs", 25, 100),
            "dropout": trial.suggest_float("dropout", 0.05, 0.2),
        }

    # Initialize model
    model = YOLO(MODEL_PATH)

    # Train with suggested parameters
    print(f"\n--- Starting Trial {trial.number} ---")
    model.train(
        data=DATA_YAML,
        imgsz=640,
        val=False, # Skip validation during training loop to save time if desired, or set True
        fliplr=0.5, # Default augmentation usually 0.5, user code had 0? Keeping user prefs.
        flipud=0,
        scale=0, # User code had 0?
        translate=0, # User code had 0?
        hsv_v=0,
        hsv_s=0,
        hsv_h=0,
        erasing=0,
        augment=False, # User code had False
        project="g:/Dean/Thesis/yolo/runs",
        name=f"trial_{trial.number}",
        exist_ok=True,
        **args,
    )
    
    # Validate to get fitness metric
    # output of val() is a DetMetrics or PoseMetrics object
    metrics = model.val(data=DATA_YAML)
    
    # metrics.fitness is the weighted combination of precision, recall, mAP, etc.
    return metrics.fitness

def tune():
    print("Starting Optuna Hyperparameter Tuning...")
    
    # JournalFileStorage uses symlinks which fail on Windows without Admin privileges.
    # We switch to SQLite which is robust and standard.
    storage = "sqlite:///optuna_chess.db"

    study = optuna.create_study(
        direction="maximize",
        study_name="chess-optune",
        storage=storage,
        load_if_exists=True,
    )

    # N_TRIALS: How many experiments to run? 
    # User original code had n_trials=1, which is just for testing.
    # I will set it to 50 for a real session, or 1 if they just want to verify.
    # Let's start with 10 to be safe, or let user configure. 
    # Changing to 100 as per typical tuning sessions, but user can Ctrl+C.
    study.optimize(objective, n_trials=100)
    
    print("Best params:", study.best_params)

def train_single_run():
    """
    Standard training function (Fallback or single run mode).
    """
    model = YOLO(MODEL_PATH)
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        pose=24.0,
        project="g:/Dean/Thesis/yolo/runs",
        name="chess_pose_train",
        exist_ok=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", action="store_true", help="Run a single training session instead of tuning")
    args = parser.parse_args()

    if args.single_run:
        train_single_run()
    else:
        tune()