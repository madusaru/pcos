import os
import shutil
import random

# Set random seed for consistency
random.seed(42)

# Source folders (update if path is different)
src_base = "PCOS"
infected_folder = os.path.join(src_base, "infected")
noninfected_folder = os.path.join(src_base, "noninfected")

# Output base folder
output_base = "dataset"
splits = ["train", "val", "test"]

# Create required folders
for split in splits:
    for cls in ["infected", "noninfected"]:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

def split_and_copy(src_dir, class_name):
    files = os.listdir(src_dir)
    random.shuffle(files)

    total = len(files)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    for i, file in enumerate(files):
        src_path = os.path.join(src_dir, file)

        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        # Destination path
        dst_path = os.path.join(output_base, split, class_name, file)
        shutil.copy(src_path, dst_path)

# Perform split for both classes
split_and_copy(infected_folder, "infected")
split_and_copy(noninfected_folder, "noninfected")

print("✅ Dataset split complete! You now have train/val/test folders.")
