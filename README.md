# 🏥 Medical AI System — Symptom & Skin Disease Diagnosis with SLMs

An end-to-end medical AI pipeline where users submit **text-based symptom queries** or **skin lesion images**, and the system routes inputs through domain-adapted Small Language Models (SLMs) to generate structured **treatment plans and clinical recommendations**.

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🤗 SFT Model | [Yash983/medgemma-2b-sft](https://huggingface.co/Yash983/medgemma-2b-sft) |
| 🤗 DPO Model | [Yash983/medgemma-2b-dpo](https://huggingface.co/Yash983/medgemma-2b-dpo) |
| 🚀 Live Demo | [gemma-2b-sft-streamlit](https://huggingface.co/spaces/Yash983/gemma-2b-sft-streamlit) |

---

## 🧭 System Overview

```
User Input
    ├── Text (symptoms / medical query)
    │       └──▶ MedGemma-2B (SFT → DPO) ──▶ Treatment Plan
    │
    └── Image (skin lesion photo)
            └──▶ ResNet50 DermNet Classifier ──▶ Disease Label
                        └──▶ MedGemma-2B (SFT → DPO) ──▶ Treatment Plan
```

---

## 🧱 Components

### 1. MedGemma-2B-SFT — Supervised Fine-Tuning

**Base Model:** `google/gemma-2b`  
**Method:** QLoRA (Supervised Fine-Tuning)  
**Domain:** Medical QA, Clinical Reasoning

Fine-tuned Gemma-2B on a curated ~190K-sample medical corpus using QLoRA to ground the model in medical language and structured reasoning.

**Dataset (~190K samples after filtering):**
- [PubMedQA](https://huggingface.co/datasets/pubmed_qa) — scientific QA
- [MedMCQA](https://huggingface.co/datasets/medmcqa) — clinical MCQs
- [MedAlpaca](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) — instruction-style medical data

**Training Config:**

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| Target modules | q_proj, v_proj |
| Batch size (effective) | 8 |
| Max sequence length | 512 |
| Optimizer | paged_adamw_8bit |
| Precision | fp16 |
| Epochs | 2 |

**Evaluation Results:**

| Metric | Base Gemma-2B | SFT | Δ |
|---|---|---|---|
| Perplexity ↓ | 46.88 | 17.11 | ↓ 29.77 |
| ROUGE-1 ↑ | 14.95 | 25.16 | +10.21 |
| ROUGE-2 ↑ | 3.81 | 9.30 | +5.49 |
| ROUGE-L ↑ | 12.78 | 21.52 | +8.74 |
| BERTScore-F1 ↑ | 69.63 | 75.40 | +5.77 |
| Win Rate | — | 98.67% | — |

---

### 2. MedGemma-2B-DPO — Preference Alignment

**Base:** `Yash983/medgemma-2b-sft`  
**Method:** Direct Preference Optimization (DPO with IPO loss)

RLHF-aligned version of the SFT model, trained to prefer clinically coherent, well-structured responses over noisy or hallucinated outputs.

**Dataset:**
- [WhiteF4lcon/medical-rlhf-pairs](https://huggingface.co/datasets/WhiteF4lcon/medical-rlhf-pairs) — ~5.9K chosen/rejected preference pairs

**Training Config:**

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Beta (β) | 0.5 |
| Loss type | IPO |
| Learning rate | 1e-5 |
| Epochs | 1 |
| Optimizer | paged_adamw_8bit |

**Evaluation Results:**

| Metric | SFT | DPO | Δ |
|---|---|---|---|
| Perplexity ↓ | 17.11 | 8.23 | ↓ 8.88 ✅ |
| ROUGE-1 ↑ | 25.16 | 23.08 | −2.08 ↔ |
| ROUGE-2 ↑ | 9.30 | 5.33 | −3.97 ❌ |
| ROUGE-L ↑ | 21.52 | 14.46 | −7.06 ❌ |
| Win Rate | — | 99.3% | — |
| Reward Margin | — | 0.95 | — |

> ROUGE regression is expected — DPO optimises for preference alignment, not lexical overlap. Outputs become more verbose and paraphrased.

**Training Stack:**
```
google/gemma-2b
  └── SFT LoRA (merged)   ← medical language grounding
        └── DPO LoRA      ← preference alignment
```

---

### 3. DermNet Skin Disease Classifier

**Architecture:** ResNet50 (ImageNet pretrained)  
**Task:** Multi-class skin disease classification  
**Classes:** 114  
**Validation Accuracy:** 94%

Transfer learning approach with a frozen ResNet50 convolutional base and a custom dense classification head, trained on the DermNet dataset.

**Dataset:** [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) — combined train/test splits  
**Image size:** 224 × 224  
**Augmentation:** zoom, horizontal flip  
**Epochs:** 20  
**Optimizer:** Adam (lr=0.001)

**Model Architecture:**
```
ResNet50 (frozen, ImageNet weights)
  └── GlobalAveragePooling2D
        └── Dense(1024, relu)
              └── Dense(114, softmax)
```

---

## 🚀 Demo

The Streamlit app is deployed on Hugging Face Spaces:  
👉 [gemma-2b-sft-streamlit](https://huggingface.co/spaces/Yash983/gemma-2b-sft-streamlit)

Users can:
- Enter free-text symptom descriptions to receive treatment plan suggestions
- Upload skin lesion images for disease classification followed by clinical recommendations

---

## ⚙️ Inference — Text Query

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE = "google/gemma-2b"
SFT  = "Yash983/medgemma-2b-sft"
DPO  = "Yash983/medgemma-2b-dpo"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, SFT)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, DPO)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token

prompt = "Answer this question truthfully\n\nContext:\nWhat are the symptoms and treatment for diabetes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
```

---

## ⚠️ Disclaimer

This system is a **research prototype** and is **not intended for clinical use**. Always verify outputs against authoritative clinical sources. Do not use for medical diagnosis or treatment decisions.

---

## 📄 Citation

```bibtex
@misc{medgemma2b-sft,
  title  = {MedGemma-2B-SFT: Supervised fine-tuned medical language model},
  author = {Yash},
  year   = {2025},
  url    = {https://huggingface.co/Yash983/medgemma-2b-sft}
}

@misc{medgemma2b-dpo,
  title  = {MedGemma-2B-DPO: DPO-aligned medical language model},
  author = {Yash},
  year   = {2025},
  url    = {https://huggingface.co/Yash983/medgemma-2b-dpo}
}
```