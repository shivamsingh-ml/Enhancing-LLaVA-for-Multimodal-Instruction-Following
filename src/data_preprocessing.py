# -*- coding: utf-8 -*-
"""
This script:
1. Loads LLaVA-Instruct-150K dataset
2. Extracts required image filenames
3. Copies related COCO images to a new directory
"""

import os
import json
import shutil
import random
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch

# ------------------------------
# Configurations
# ------------------------------
SEED = 42
COCO_TRAIN_DIR = "./coco/train2014"
COCO_VAL_DIR = "./coco/val2014"
OUTPUT_DIR = "./filtered_coco"
LLaVA_JSON_PATH = "llava_instruct_150k.json"
MAX_TEXT_SAMPLES = 100

# ------------------------------
# Utilities
# ------------------------------

def set_random_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_used_images(json_path):
    """Extract unique image filenames from dataset JSON."""
    used_images = set()

    with open(json_path, "r") as f:
        data = json.load(f)

    for example in tqdm(data, desc="Extracting Image Names"):
        image_file = example.get("image")
        if image_file:
            used_images.add(image_file.strip())

    print(f"Total unique images needed: {len(used_images)}")
    return used_images

def copy_coco_images(used_images, train_dir, val_dir, output_dir):
    """Copy required COCO images to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    missing_images = 0
    copied_images = 0

    for img_file in tqdm(sorted(used_images), desc="Copying Images"):
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
            print(f"Missing image: {img_file}")

    print("\nImage Copying Summary:")
    print(f"Copied images: {copied_images}")
    print(f"Missing images: {missing_images}")



# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    print("==== LLaVA Phase 1 Preprocessing ====")

    # Set seeds
    set_random_seeds(SEED)

    # Step 1: Extract used images from JSON
    used_images = extract_used_images(LLaVA_JSON_PATH)

    # Step 2: Copy required COCO images
    copy_coco_images(used_images, COCO_TRAIN_DIR, COCO_VAL_DIR, OUTPUT_DIR)

    print("\nPhase 1 Preprocessing Completed.")
