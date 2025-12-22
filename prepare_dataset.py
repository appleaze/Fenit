import os
import shutil
import random
from pathlib import Path

def prepare_dataset():
    # Configuration
    base_dir = Path("g:/Dean/Thesis/yolo/archive (15)")
    dest_dir = Path("g:/Dean/Thesis/yolo/dataset_final")
    
    # Source directories (based on exploration)
    sources = [
        base_dir / "Chess Piece Board Pose Annotation/Chess Piece Board Pose Annotation/original chess",
        base_dir / "Chess Piece Board Pose Annotation/Chess Piece Board Pose Annotation/augmented chess",
        base_dir / "Chess Piece Board Pose Annotation More Augmented Samples/Chess Piece Board Pose Annotation More Augmented Samples/augmented"
    ]
    
    # Create destination structure
    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            (dest_dir / dtype / split).mkdir(parents=True, exist_ok=True)
            
    # Collect all pairs
    all_pairs = []
    
    print("Scanning for files...")
    for source in sources:
        img_dir = source / "images"
        lbl_dir = source / "labels"
        
        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist. Skipping.")
            continue
            
        # Get all images recursively
        images = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpeg"))
        
        for img_path in images:
            # Check for corresponding label
            # Need to find relative path from img_dir to match label structure
            rel_path = img_path.relative_to(img_dir)
            lbl_path = lbl_dir / rel_path.with_suffix(".txt")
            
            # Use 'labels' dir + relative path (e.g. train/img.txt)
            if lbl_path.exists():
                all_pairs.append((img_path, lbl_path))
            else:
                # Try flat structure just in case labels aren't nested the same way
                lbl_path_flat = lbl_dir / (img_path.stem + ".txt")
                if lbl_path_flat.exists():
                    all_pairs.append((img_path, lbl_path_flat))

    print(f"Found {len(all_pairs)} image/label pairs.")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_pairs)
    
    # Split
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    print(f"Training: {len(train_pairs)}")
    print(f"Validation: {len(val_pairs)}")
    
    # Move files
    def move_files(pairs, split_name):
        print(f"Moving {split_name} files...")
        count = 0
        total = len(pairs)
        for img_src, lbl_src in pairs:
            # Handle potential filename collisions by checking usage
            # Simple strategy: If destination exists, rename (append counter)
            
            img_dest = dest_dir / "images" / split_name / img_src.name
            lbl_dest = dest_dir / "labels" / split_name / lbl_src.name
            
            counter = 1
            while img_dest.exists() or lbl_dest.exists():
                stem = img_src.stem
                suffix = img_src.suffix
                new_name = f"{stem}_{counter}{suffix}"
                new_lbl_name = f"{stem}_{counter}.txt"
                
                img_dest = dest_dir / "images" / split_name / new_name
                lbl_dest = dest_dir / "labels" / split_name / new_lbl_name
                counter += 1
            
            shutil.move(str(img_src), str(img_dest))
            shutil.move(str(lbl_src), str(lbl_dest))
            
            count += 1
            if count % 100 == 0:
                print(f"Moved {count}/{total} files...")

    move_files(train_pairs, 'train')
    move_files(val_pairs, 'val')
    
    print("Dataset preparation complete!")
    print(f"Data is now in {dest_dir}")

if __name__ == "__main__":
    prepare_dataset()
