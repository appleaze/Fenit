import sys
import logging
import optuna
from ultralytics import YOLO

# Display Optuna logs clearly
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.INFO)

# Constants
MODEL_PATH = "yolov8l-pose.pt" 
DATA_YAML = "g:/Dean/Thesis/yolo/manual_annotate_dataset/data.yaml"
PROJECT_DIR_TUNE = "g:/Dean/Thesis/yolo/runs/manual_annotate_tuning"

def objective(trial):
    # Trial 0: Default parameters suggestion to anchor the search
    if trial.number == 0:
        args = {"epochs": 50, "batch": 8}
    else:
        args = {
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD", "auto"]),
            "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
            "momentum": trial.suggest_float("momentum", 0.85, 0.99),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
            "warmup_momentum": trial.suggest_float("warmup_momentum", 0.01, 0.99),
            "warmup_bias_lr": trial.suggest_float("warmup_bias_lr", 0.1, 0.9),
            "box": trial.suggest_float("box", 1.0, 10.0),
            "cls": trial.suggest_float("cls", 0.1, 2.0),
            "pose": trial.suggest_float("pose", 12.0, 30.0),
            "kobj": trial.suggest_float("kobj", 1.0, 5.0),
            "batch": trial.suggest_categorical("batch", [8, 16]),
            "epochs": trial.suggest_int("epochs", 30, 80),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "fliplr": trial.suggest_categorical("fliplr", [0.0, 0.5]),
            "scale": trial.suggest_float("scale", 0.0, 0.5),
            "translate": trial.suggest_float("translate", 0.0, 0.2),
            "erasing": trial.suggest_float("erasing", 0.0, 0.4),
        }

    # Initialize model
    model = YOLO(MODEL_PATH)

    # Train with suggested parameters
    print(f"\n--- Starting Trial {trial.number} ---")
    
    results = model.train(
        data=DATA_YAML,
        imgsz=640,
        val=True, # enable validation to capture fitness
        device=0,
        project=PROJECT_DIR_TUNE,
        name=f"trial_{trial.number}",
        exist_ok=True,
        **args,
    )
    
    # ultralytics results object should have fitness attribute
    if hasattr(results, 'fitness'):
        return results.fitness
    
    # Fallback validation check
    metrics = model.val(data=DATA_YAML)
    return metrics.fitness


def tune_custom():
    print("Starting Optuna Hyperparameter Tuning over `manual_annotate_dataset`...")
    
    # Isolated db specific for the new dataset
    storage = "sqlite:///optuna_manual_annotate.db"

    study = optuna.create_study(
        direction="maximize",
        study_name="manual-annotate-optune",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=50) # run up to 50 trials
    
    print("\n==================================")
    print("Best Trial parameters:", study.best_params)
    print("Best Fitness:", study.best_value)
    print("==================================")

if __name__ == "__main__":
    tune_custom()
