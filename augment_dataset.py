import os
import shutil
import random
import cv2
import albumentations as A
from tqdm import tqdm

# --- Configuration ---
SRC_IMG_DIR = r"G:\Dean\Thesis\yolo\training\images\Train"
SRC_LBL_DIR = r"G:\Dean\Thesis\yolo\training\labels\Train"
DST_DIR = r"G:\Dean\Thesis\yolo\manual_annotate_dataset"

TRAIN_SPLIT = 0.85
AUGMENT_MULTIPLIER = 3 # 3 augmentations + 1 original for each train image

def main():
    # Ensure directories exist
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DST_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DST_DIR, 'labels', split), exist_ok=True)

    # Find all matching images and labels
    images = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    data = []
    for img_name in images:
        base = os.path.splitext(img_name)[0]
        lbl_name = base + '.txt'
        if os.path.exists(os.path.join(SRC_LBL_DIR, lbl_name)):
            data.append((img_name, lbl_name))

    # Shuffle and split
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * TRAIN_SPLIT)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Total pairs: {len(data)}")
    print(f"Train pairs: {len(train_data)}")
    print(f"Val pairs:   {len(val_data)}")

    # Transformation Pipeline
    try:
        # Tries to use Defocus if available, otherwise falls back to a Blur substitute
        defocus_transform = A.Defocus(p=0.15)
    except AttributeError:
        defocus_transform = A.Blur(blur_limit=7, p=0.15)

    transform = A.Compose([
        A.Affine(scale=(0.8, 1.2), p=0.4),                       # Scaling: 40%
        A.Affine(rotate=(-30, 30), shear=(-30, 30), p=1.0),      # Rotation & Shear: 100%
        A.GaussianBlur(blur_limit=(3, 7), p=0.25),               # Gaussian Blur: 25%
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.20),            # Gaussian Noise: 20%
        defocus_transform                                        # Defocus/Blur: 15%
    ], 
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0.0, min_visibility=0.0),
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def parse_yolo_label(lbl_path, w, h):
        classes = []
        bboxes = []
        keypoints = []
        
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = [float(x) for x in line.strip().split()]
                if len(parts) < 5: continue
                
                classes.append(int(parts[0]))
                bboxes.append([parts[1], parts[2], parts[3], parts[4]]) # x_c, y_c, bw, bh
                
                box_kpts = []
                # Starting from index 5, reading in groups of 3 (x, y, visibility)
                for i in range(5, len(parts), 3):
                    px = parts[i] * w
                    py = parts[i+1] * h
                    pv = parts[i+2]
                    box_kpts.append([px, py, pv])
                keypoints.append(box_kpts)
                
        return classes, bboxes, keypoints

    def save_yolo_label(lbl_path, classes, bboxes, keypoints, w, h):
        lines = []
        for i in range(len(classes)):
            cls_id = classes[i]
            x_c, y_c, bw, bh = bboxes[i]
            
            # constrain bbox centers to image bounds
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            
            line = f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
            
            # format keypoints back to normalized [0, 1]
            for px, py, pv in keypoints[i]:
                nx = px / w
                ny = py / h
                
                # If keypoint is strictly outside bounds, set invisible (visibility=0)
                if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                    pv = 0
                
                # Clamp coordinates just in case it's slightly drifted out of bounds
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                line += f" {nx:.6f} {ny:.6f} {int(pv)}"
            
            lines.append(line)
            
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def process_data(data_split, split_name, do_augment=False):
        for img_name, lbl_name in tqdm(data_split, desc=f"Processing {split_name}"):
            img_src = os.path.join(SRC_IMG_DIR, img_name)
            lbl_src = os.path.join(SRC_LBL_DIR, lbl_name)
            
            img_dst = os.path.join(DST_DIR, 'images', split_name, img_name)
            lbl_dst = os.path.join(DST_DIR, 'labels', split_name, lbl_name)
            
            # Always copy the original unaltered files first
            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)
            
            if do_augment:
                img = cv2.imread(img_src)
                if img is None: 
                    print(f"Could not read {img_src}, skipping augmentations.")
                    continue
                h, w, _ = img.shape
                
                classes, bboxes, keypoints = parse_yolo_label(lbl_src, w, h)
                
                # Flatten keypoints for Albumentations
                flat_kpts = []
                for box_kpts in keypoints:
                    for kp in box_kpts:
                        flat_kpts.append((kp[0], kp[1]))
                
                for aug_i in range(AUGMENT_MULTIPLIER):
                    try:
                        transformed = transform(image=img, bboxes=bboxes, class_labels=classes, keypoints=flat_kpts)
                    except Exception as e:
                        print(f"Skipping augmentation {aug_i} for {img_name} due to error: {e}")
                        continue
                    
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_classes = transformed['class_labels']
                    aug_flat_kpts = transformed['keypoints']
                    
                    # Unflatten keypoints back into their bounding boxes
                    aug_keypoints = []
                    idx = 0
                    for orig_box_kpts in keypoints:
                        box_aug_kpts = []
                        for orig_kp in orig_box_kpts:
                            orig_pv = orig_kp[2]
                            new_px = aug_flat_kpts[idx][0]
                            new_py = aug_flat_kpts[idx][1]
                            box_aug_kpts.append((new_px, new_py, orig_pv))
                            idx += 1
                        aug_keypoints.append(box_aug_kpts)
                    
                    # Save augmented pair
                    base = os.path.splitext(img_name)[0]
                    aug_img_name = f"{base}_aug_{aug_i}.jpg"
                    aug_lbl_name = f"{base}_aug_{aug_i}.txt"
                    
                    cv2.imwrite(os.path.join(DST_DIR, 'images', split_name, aug_img_name), aug_img)
                    save_yolo_label(os.path.join(DST_DIR, 'labels', split_name, aug_lbl_name), 
                                    aug_classes, aug_bboxes, aug_keypoints, w, h)

    print("\n--- Starting validation data copy... ---")
    process_data(val_data, 'val', do_augment=False)

    print("\n--- Starting training data processing & augmentation... ---")
    process_data(train_data, 'train', do_augment=True)

    # Output the YAML logic file by reading the original and modifying paths
    import yaml
    
    orig_yaml_path = os.path.join(r"G:\Dean\Thesis\yolo\training", 'data.yaml')
    out_yaml_path = os.path.join(DST_DIR, 'data.yaml')
    
    try:
        with open(orig_yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}
            
        # Update necessary fields for YOLOv8
        yaml_data['path'] = os.path.abspath(DST_DIR).replace('\\\\', '/')
        yaml_data['train'] = 'images/train'
        yaml_data['val'] = 'images/val'
        yaml_data['test'] = ''
        
        # Remove old keys if they exist
        if 'Train' in yaml_data:
            del yaml_data['Train']
            
        with open(out_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False)
            
    except Exception as e:
        print(f"Warning: Could not parse original data.yaml using PyYAML ({e}). Falling back to simple path replacement.")
        # Fallback if PyYAML isn't available
        with open(orig_yaml_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if line.startswith('path:'):
                clean_path = os.path.abspath(DST_DIR).replace('\\', '/')
                new_lines.append(f"path: {clean_path}\n")
            elif line.startswith('Train:'):
                new_lines.append(f"train: images/train\n")
                new_lines.append(f"val: images/val\n")
            else:
                new_lines.append(line)
                
        with open(out_yaml_path, 'w') as f:
            f.writelines(new_lines)

    print(f"\nDataset successfully created in {DST_DIR}")
    print("New data.yaml has been created in the new directory for YOLOv8 training.")

if __name__ == "__main__":
    main()
