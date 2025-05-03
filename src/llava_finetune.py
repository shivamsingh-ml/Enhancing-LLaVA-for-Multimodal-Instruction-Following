import os
import json
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, Trainer, TrainingArguments, default_data_collator
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------------
# Configuration
# ---------------------------
SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FINETUNE_JSONL = os.path.join(PROJECT_ROOT, "..", "data", "finetune_10k_samples.jsonl")
FINETUNE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "..", "data", "finetune_images")
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

torch.manual_seed(SEED)

# ---------------------------
# Load Processor and Model (Quantized + LoRA)
# ---------------------------

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False, token=HF_TOKEN)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    token=HF_TOKEN
)

# Apply LoRA adapters
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# ---------------------------
# Dataset Class
# ---------------------------

class LLaVAFinetuneDataset(Dataset):
    def __init__(self, jsonl_path, images_dir, processor):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["conversations"][0]["value"]
        target = sample["conversations"][-1]["value"]

        image_file = sample.get("image")
        if not image_file:
            raise ValueError("No image found.")

        if image_file.endswith(".JPG"):
            image_file = image_file[:-4] + ".jpg"

        img_path = os.path.join(self.images_dir, image_file)
        image = Image.open(img_path).convert("RGB")

        # Limit instruction tokens so that with <image> they fit well
        max_text_tokens = 256  # safe side, LLaVA can easily support 512 total, but let's keep for safety

        instruction_tokens = processor.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=max_text_tokens).input_ids
        instruction_text = processor.tokenizer.decode(instruction_tokens[0], skip_special_tokens=True)

        instruction_with_image = "<image>\n" + instruction_text

        inputs = processor(text=instruction_with_image, images=image, return_tensors="pt", padding="longest", truncation=True)

        input_length = inputs["input_ids"].shape[1]

        # Process target → padding to input length (very important now!)
        labels = processor.tokenizer(target, return_tensors="pt", padding="max_length", truncation=True, max_length=input_length)

        # Replace pad tokens with -100
        labels["input_ids"][labels["input_ids"] == processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels["input_ids"].squeeze(0)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs



# ---------------------------
# Load Dataset
# ---------------------------

dataset = LLaVAFinetuneDataset(FINETUNE_JSONL, FINETUNE_IMAGES_DIR, processor)

# ---------------------------
# Training Arguments
# ---------------------------

training_args = TrainingArguments(
    output_dir="./llava_finetune_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
)

# ---------------------------
# Trainer
# ---------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,  # Automatically pad dynamically
)

# ---------------------------
# Start Training
# ---------------------------

trainer.train()

# ---------------------------
# Save LoRA Adapter (only adapter weights will be saved)
# ---------------------------
model.save_pretrained("./llava_finetune_lora")
print("Finetuning complete. LoRA adapter saved.")