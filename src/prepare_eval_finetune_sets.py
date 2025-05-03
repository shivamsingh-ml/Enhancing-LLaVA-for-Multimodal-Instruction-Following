import os
import json
import random
import shutil
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------
# Script: Prepare Evaluation and Finetuning subsets from LLaVA-Instruct-150K dataset
#
# This script does the following:
#   1. Loads full LLaVA-Instruct-150K JSON dataset
#   2. Randomly selects 1000 samples for evaluation
#   3. Randomly selects 10,000 samples for finetuning
#   4. Saves the selected splits into eval_1000_samples.jsonl and finetune_10k_samples.jsonl
#   5. Collects corresponding images from filtered_coco directory and copies them into:
#       → ./eval_images
#       → ./finetune_images
# ----------------------------------------------------------------------------------------------------

# -------------------
# CONFIGURATION
# -------------------

LLAVA_JSON_FILE = "llava_instruct_150k.json"
FILTERED_COCO_PATH = "./filtered_coco/filtered_coco"

EVAL_JSONL_FILE = "eval_1000_samples.jsonl"
FINETUNE_JSONL_FILE = "finetune_10k_samples.jsonl"

EVAL_IMAGE_DIR = "./eval_images"
FINETUNE_IMAGE_DIR = "./finetune_images"

# -------------------
# STEP 1: Load the full dataset
# -------------------

print("Loading LLaVA-Instruct-150K dataset...")
with open(LLAVA_JSON_FILE, "r") as f:
    samples = json.load(f)

print(f"Total loaded samples: {len(samples)}")

# -------------------
# STEP 2: Shuffle and split the dataset
# -------------------

random.seed(42)  # Ensure reproducibility
random.shuffle(samples)

# Select first 1000 for eval
eval_samples = samples[:1000]

# Select next 10,000 for finetuning
finetune_samples = samples[1000:1000 + 10000]

# -------------------
# STEP 3: Save eval and finetune splits to JSONL files
# -------------------

print("Saving eval samples...")
with open(EVAL_JSONL_FILE, "w") as f_eval:
    for sample in eval_samples:
        f_eval.write(json.dumps(sample) + "\n")

print("Saved eval set to:", EVAL_JSONL_FILE)

print("Saving finetune samples...")
with open(FINETUNE_JSONL_FILE, "w") as f_train:
    for sample in finetune_samples:
        f_train.write(json.dumps(sample) + "\n")

print("Saved finetune set to:", FINETUNE_JSONL_FILE)

# -------------------
# STEP 4: Create directories to store filtered images
# -------------------

os.makedirs(EVAL_IMAGE_DIR, exist_ok=True)
os.makedirs(FINETUNE_IMAGE_DIR, exist_ok=True)

# -------------------
# STEP 5: Function to copy images from filtered_coco to target folders
# -------------------

def copy_images(jsonl_file, target_dir):
    with open(jsonl_file, "r") as f:
        for line in tqdm(f, desc=f"Copying images for {jsonl_file}"):
            sample = json.loads(line)
            image_file = sample.get("image", None)
            
            if not image_file:
                continue
            
            # Normalize filename (COCO sometimes has JPG instead of jpg)
            image_file = image_file.strip()
            if image_file.endswith(".JPG"):
                image_file = image_file[:-4] + ".jpg"

            src_path = os.path.join(FILTERED_COCO_PATH, image_file)

            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(target_dir, image_file))
            else:
                print(f"WARNING: Image not found: {src_path}")

# -------------------
# STEP 6: Copy images for eval and finetune splits
# -------------------

copy_images(EVAL_JSONL_FILE, EVAL_IMAGE_DIR)
copy_images(FINETUNE_JSONL_FILE, FINETUNE_IMAGE_DIR)

print("Finished copying images.")
print("Eval images →", EVAL_IMAGE_DIR)
print("Finetune images →", FINETUNE_IMAGE_DIR)
