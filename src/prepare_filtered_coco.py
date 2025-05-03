"""
LLaVA-Instruct-150K → COCO Image Filter Script

This script processes the LLaVA-Instruct-150K dataset and prepares a filtered COCO dataset.
It does so by:

1. Loading LLaVA-Instruct-150K dataset (JSON format).
2. Extracting the unique image filenames referenced in the dataset.
3. Copying only the required images from the full COCO dataset (train2014/val2014) 
   into a new directory → filtered_coco/.

This filtered_coco/ directory is then later used to create:
    → eval_images (1000 samples)
    → finetune_images (10k samples)
in the next steps of the LLaVA project pipeline.

"""

import os
import json
import shutil
import random
from tqdm import tqdm
import numpy as np
import torch

# ------------------------------
# Configurations
# ------------------------------

SEED = 42

# Path to COCO dataset (original source images)
COCO_TRAIN_DIR = "./coco/train2014"
COCO_VAL_DIR = "./coco/val2014"

# Output directory where filtered images will be copied
OUTPUT_DIR = "./filtered_coco"

# LLaVA-150K JSON dataset path
LLaVA_JSON_PATH = "llava_instruct_150k.json"

# ------------------------------
# Utilities
# ------------------------------

def set_random_seeds(seed):
    """Ensure reproducibility by setting random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_used_images(json_path):
    """
    Extract unique image filenames that are used in the LLaVA-Instruct-150K dataset.

    Args:
        json_path (str): Path to llava_instruct_150k.json file.

    Returns:
        set: A set of unique image filenames.
    """
    used_images = set()

    with open(json_path, "r") as f:
        data = json.load(f)

    for example in tqdm(data, desc="Extracting image names"):
        image_file = example.get("image")
        if image_file:
            used_images.add(image_file.strip())

    print(f"Total unique images needed: {len(used_images)}")
    return used_images

def copy_coco_images(used_images, train_dir, val_dir, output_dir):
    """
    Copy the required COCO images from train2014/val2014 to the filtered output directory.

    Args:
        used_images (set): Set of required image filenames.
        train_dir (str): Path to COCO train2014 directory.
        val_dir (str): Path to COCO val2014 directory.
        output_dir (str): Output directory for filtered images.
    """
    os.makedirs(output_dir, exist_ok=True)

    copied_images = 0
    missing_images = 0

    for img_file in tqdm(sorted(used_images), desc="Copying images"):
        # Check if image exists in train or val directories
        train_path = os.path.join(train_dir, "COCO_train2014_" + img_file)
        val_path = os.path.join(val_dir, "COCO_val2014_" + img_file)
        dest_path = os.path.join(output_dir, img_file)

        if os.path.exists(train_path):
            shutil.copy(train_path, dest_path)
            copied_images += 1
        elif os.path.exists(val_path):
            shutil.copy(val_path, dest_path)
            copied_images += 1
        else:
            missing_images += 1
            print(f"WARNING: Missing image → {img_file}")

    # Final summary
    print("\nImage Copying Summary:")
    print(f"Copied images : {copied_images}")
    print(f"Missing images: {missing_images}")

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    print("==== LLaVA-Instruct-150K → COCO Filter Preparation ====")

    # Step 0: Set seeds
    set_random_seeds(SEED)

    # Step 1: Extract image filenames used in LLaVA-Instruct-150K
    used_images = extract_used_images(LLaVA_JSON_PATH)

    # Step 2: Copy required COCO images into filtered_coco/
    copy_coco_images(used_images, COCO_TRAIN_DIR, COCO_VAL_DIR, OUTPUT_DIR)

    print("\nFiltering Completed. Images stored in →", OUTPUT_DIR)
