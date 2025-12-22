import os
import sys
import json
import shutil
import argparse
from typing import List
from pathlib import Path

import PIL
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco

try:
    from cvat_sdk import models, make_client
    import cvat_sdk.auto_annotation as cvataa
except ImportError:
    print("Warning: cvat-sdk not installed. Auto-annotation features will not work.")

class AnnotShapes:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    @property
    def spec(self):
        # type hint omitted to avoid import error if cvat_sdk missing during convert-only run
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.skeleton_label_spec(
                    name,
                    id,
                    [
                        models.SublabelRequest(str(x), id=x, type="points")
                        for x in range(1, 5)
                    ],
                )
                for id, name in self.model.names.items()
            ],
        )

    def detect(self, context, image: PIL.Image.Image) -> List:
        print(f"=== Annotating image {context.frame_name} ===")
        results = self.model(image, verbose=False)

        return [
            cvataa.skeleton(
                int(label.item()),
                elements=[
                    {
                        "frame": 0,
                        "type": "points",
                        "label_id": idx + 1,
                        "points": sk.tolist(),
                    }
                    for idx, sk in enumerate(skeleton.xy[0])
                ],
            )
            for result in results
            for skeleton, label in zip(result.keypoints, result.boxes.cls)
        ]

def auto_annotate(task_id: int, model_path: str, host: str, port: int, user: str, password: str):
    """
    Connects to CVAT and runs auto-annotation on the specified task.
    """
    print(f"Connecting to CVAT at {host}:{port} as {user}...")
    try:
        with make_client(host=host, port=port, credentials=(user, password)) as client:
            print(f"Running AUTO ANNOTATION on TASK {task_id}")
            cvataa.annotate_task(client, int(task_id), AnnotShapes(model_path), clear_existing=True)
    except Exception as e:
        print(f"\nError: Could not connect to CVAT at {host}:8080")
        print(f"Details: {e}")
        print("\nPlease ensure:")
        print("1. CVAT is running (check Docker or your terminal).")
        print("2. You are using the correct host/port (default is localhost:8080).")

def handle_files(annotations_dir: Path):
    """
    Pre-processes COCO JSON files exported from CVAT.
    Applies user-specific ID mapping and BBox corrections.
    """
    print(f"Processing JSON files in {annotations_dir}...")
    files = list(annotations_dir.glob("*.json"))

    if not files:
        print("No JSON files found.")
        return

    for f in files:
        print(f"  Fixing {f.name}...")
        with open(f, "r") as file:
            data = json.load(file)
        
        # User-provided logic for ID remapping
        # Original: data["categories"][idx]["id"] = int(c["id"] / 5 + 1)
        # Note: This implies specific CVAT schema behavior. Keeping as requested.
        for idx, c in enumerate(data.get("categories", [])):
            if "id" in c:
                data["categories"][idx]["id"] = int(c["id"] / 5 + 1)
        
        for idx, a in enumerate(data.get("annotations", [])):
            if "category_id" in a:
                data["annotations"][idx]["category_id"] = int(a["category_id"] / 5 + 1)
            
            # BBox corrections
            bbox = a.get("bbox", [0,0,0,0])
            if bbox[2] == 0:
                data["annotations"][idx]["bbox"][2] = 50
            if bbox[3] == 0:
                data["annotations"][idx]["bbox"][3] = 50
            
            # Keypoint corrections
            keypoints = a.get("keypoints", [])
            # User logic: v1, v2, v3, v4 = keypoints[2], keypoints[5], keypoints[8], keypoints[11]
            if len(keypoints) >= 12:
                # 4 keypoints * 3 values (x,y,v) = 12 values
                indices = [2, 5, 8, 11] 
                for i, k_idx in enumerate(indices):
                    v = keypoints[k_idx]
                    if v == 0:
                        # If visibility is 0, zero out x and y? 
                        # User code: data["annotations"][idx]["keypoints"][idx - 1] = 0
                        # Wait, 'idx' in user code inner loop refers to enumeration of [v1,v2...]?
                        # Re-implementing strictly as:
                        # for idx, v in enumerate([v1, v2, v3, v4]):
                        #    if v == 0:
                        #        data["annotations"][idx]["keypoints"][idx - 1] = 0
                        # This looks buggy in original code (idx variable shadowing), 
                        # but I will try to interpret intent: set x,y to 0 if v is 0.
                        base_idx = k_idx - 2 
                        data["annotations"][idx]["keypoints"][base_idx] = 0     # x
                        data["annotations"][idx]["keypoints"][base_idx + 1] = 0 # y

        with open(f, "w") as file:
            json.dump(data, file)

def convert(annotations_dir: Path, output_dir: Path, dataset_labels_dir: Path):
    """
    Converts processed COCO JSON to YOLO format using Ultralytics.
    """
    print("Converting COCO to YOLO...")
    # ultralytics.data.converter.convert_coco save to 'yolo_labels' dir by default or similar?
    # signature: convert_coco(labels_dir, save_dir, use_segments=False, use_keypoints=False, cls91to80=False)
    
    convert_coco(
        labels_dir=str(annotations_dir),
        save_dir=str(output_dir),
        use_keypoints=True,
        cls91to80=False,
    )
    
    # Move files
    converted_labels_dir = output_dir / "labels"
    if converted_labels_dir.exists():
        print(f"Moving labels to {dataset_labels_dir}...")
        dataset_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for file in converted_labels_dir.rglob("*.txt"):
            shutil.move(str(file), str(dataset_labels_dir / file.name))
        
        print("Done.")
    else:
        print(f"Warning: No 'labels' directory found in {output_dir}")

def run():
    parser = argparse.ArgumentParser(description="CVAT Workflow: Auto-Annotation & Conversion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for annotation
    annotate_parser = subparsers.add_parser("annotate", help="Run auto-annotation on CVAT task")
    annotate_parser.add_argument("--task-id", type=int, required=True, help="CVAT Task ID")
    annotate_parser.add_argument("--model", type=str, default="yolov8l-pose.pt", help="Path to YOLO model")
    annotate_parser.add_argument("--host", type=str, default="http://localhost", help="CVAT Host URL")
    annotate_parser.add_argument("--port", type=int, default=8080, help="CVAT Port")
    annotate_parser.add_argument("--user", type=str, default="admin", help="CVAT Username")
    annotate_parser.add_argument("--password", type=str, default="admin", help="CVAT Password")

    # Subparser for conversion
    convert_parser = subparsers.add_parser("convert", help="Convert exported COCO JSON to YOLO format")
    convert_parser.add_argument("--annotations-dir", type=str, default="annotations", help="Directory containing exported COCO JSONs")
    convert_parser.add_argument("--temp-output", type=str, default="conversion_temp", help="Temporary output dir for conversion")
    convert_parser.add_argument("--final-labels-dir", type=str, default="dataset_final/labels", help="Final destination for .txt labels")

    args = parser.parse_args()

    if args.command == "annotate":
        auto_annotate(args.task_id, args.model, args.host, args.port, args.user, args.password)
    
    elif args.command == "convert":
        anno_path = Path(args.annotations_dir)
        temp_path = Path(args.temp_output)
        final_path = Path(args.final_labels_dir)
        
        if not anno_path.exists():
            print(f"Error: Annotations directory '{anno_path}' does not exist.")
            return

        handle_files(anno_path)
        convert(anno_path, temp_path, final_path)

if __name__ == "__main__":
    run()
