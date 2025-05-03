# 🚀 Robust and Adversarial Instruction Tuning of LLaVA (Multimodal + Quantized Fine-tuning + LLM Supervision)

This repository contains an ongoing research project on **robustifying LLaVA-1.5 7B (Multimodal Model)** through:

- Efficient Quantized Fine-tuning with LoRA (4-bit)
- Multimodal Instruction Tuning (Images + Text)
- Adversarial Dataset Construction & Robust Finetuning
- LLM-based Supervisor Guided Evaluation and Potential Preference Optimization

> **Note:** This project focuses specifically on **multimodal fine-tuning**, which is considerably harder than text-only finetuning. Handling images + text together requires careful prompt, tokenizer, vision encoder coordination and is an active research area.

---

## 📚 Dataset Overview

Primary dataset used:

**[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)**

- ~150K multimodal samples (Images + Instructions + Responses).
- Images mostly from **COCO dataset**, with some synthetic images.
- For efficiency, subset is filtered and used for finetuning and evaluation.

Prepared subsets:

| Set | Samples | Description |
|-----|---------|-------------|
| `eval_1000_samples.jsonl` | 1000 | For baseline and robust evaluation |
| `finetune_10k_samples.jsonl` | 10,000 | For initial LoRA multimodal finetuning |
| `filtered_coco` | N/A | COCO images filtered for above samples |

> ✅ These subsets help in running **multimodal training & evaluation** on limited compute.

---

## ✅ Baseline Multimodal Evaluation Results

| Model | Dataset | BERTScore F1 |
|-------|---------|--------------|
| LLaVA 1.5 Pretrained (Multimodal) | Eval 1000 Samples | **0.8693** |

> Multimodal baseline evaluation is complete using BERTScore.  
> Further evaluation will compare Finetuned and Robust versions.

---

## 📌 Project Roadmap

### PHASE 0 → Baseline Setup and Finetuning ✅

- Filter LLaVA 150K and prepare evaluation + finetune subsets.
- **Baseline Multimodal Evaluation → Pretrained LLaVA**
- Multimodal Fine-tuning → LLaVA (10K samples, LoRA, Quantized)
- Evaluate Finetuned → Compare vs Pretrained

### PHASE 1 → Adversarial Dataset Creation (Upcoming)

- Generate adversarial instructions (LLM-generated + Manual)
- Validate and format (adversarial_5k_samples.jsonl)
- Ready dataset for robust multimodal finetuning.

### PHASE 2 → Robust Fine-tuning (Adversarial + Normal Mix)

- Finetune LLaVA again on adversarial dataset (LoRA + Quantized)
- Save adapter (`llava_finetuned_robust_lora`)

### PHASE 3 → Robustness Evaluation

- Evaluate:
    - Eval Set (Pretrained vs Finetuned vs Robust)
    - Adversarial Eval Set (Pretrained vs Finetuned vs Robust)
- (Optional) Use LLM Supervisor to score robustness.
- Report → Robust LLaVA expected to perform better on adversarial instructions.

### PHASE 4 → Supervisor LLM + Feedback Loop

- Use LLMs (GPT-4, Claude, Gemini) to compare model generations.
- Collect feedback → Rankings, better/worse labels.
- (Optional) Use feedback for preference finetuning → Align Robust LLaVA.

### PHASE 5 → Optional Continual Multimodal Robust Learning

- Merge all datasets → Create Unified Robust Dataset.
- Multimodal fine-tune for universal robustness.
- Zero-shot test on external benchmarks → A-OKVQA, ScienceQA etc.

### FINAL PHASE → Project Reporting & Results

- Quantitative → BERTScore / Human Score Comparisons
- Qualitative → Side-by-side generations
- Report → "Robust and Adversarial Multimodal Instruction Tuning of LLaVA"

---

## 🚧 Project Status

| Task | Status |
|------|--------|
| Baseline Multimodal Evaluation | ✅ Completed |
| Finetune Multimodal LLaVA (10k LoRA) | ✅ In Progress |
| Adversarial Dataset Creation | 🔜 Next |
| Robust Finetuning | 🔜 |
| Supervisor LLM Evaluation | 🔜 |
| Final Reporting | 🔜 |

---

## 📌 Important Notes on Multimodal Finetuning

- Multimodal fine-tuning is **not trivial** like pure text finetuning.
- Aligning Image Features ↔ Text Tokens requires:
    - Prompt formatting
    - Input padding and truncation management
    - Careful dataset creation
- Using Quantization (4-bit) + LoRA → makes this even more challenging.
- This project attempts to make robust multimodal finetuning practical and replicable.

---

## 📎 References

- [LLaVA 1.5 (7B)](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- [LoRA (PEFT)](https://huggingface.co/docs/peft/index)
- [BERTScore](https://github.com/Tiiiger/bert_score)

---

## 💡 Future Ideas

- Explore Supervisor LLM guided training → Reward Modeling / Reinforcement Learning
- Extend robust model to VQA datasets → A-OKVQA, ScienceQA
- Explore Fully Quantized Robust LLaVA deployments

---

