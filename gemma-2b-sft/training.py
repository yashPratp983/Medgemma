"""
train.py — Production-ready QLoRA SFT fine-tuning for MedGemma-2B
"""

import gc
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import (
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Config dataclasses ─────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    save_dir: str = field(default="medical_dataset", metadata={"help": "Directory to save/load processed dataset"})
    load_from_cache: bool = field(default=False, metadata={"help": "Skip dataset download and load from save_dir"})
    train_ratio: float = field(default=0.80)
    val_ratio: float = field(default=0.10)
    test_ratio: float = field(default=0.10)
    min_text_length: int = field(default=80, metadata={"help": "Min chars to keep a sample"})
    dedup_prefix_len: int = field(default=120, metadata={"help": "Chars used for near-dedup key"})
    seed: int = field(default=42)


@dataclass
class ModelConfig:
    model_id: str = field(default="google/gemma-2b")
    output_dir: str = field(default="model")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_seq_length: int = field(default=512)
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True


# ── Dataset helpers ────────────────────────────────────────────────────────────
def _load_pubmedqa() -> list:
    parts = []
    for config_name in ("pqa_labeled", "pqa_unlabeled"):
        try:
            raw = load_dataset("pubmed_qa", config_name, trust_remote_code=True)

            def fmt(x):
                ctx = " ".join(x["context"]["contexts"])[:1000]
                ans = (x.get("long_answer") or "").strip()
                return {"text": f"{x['question'].strip()}\n{ctx}\n{ans}".strip()}

            for split_ds in raw.values():
                parts.append(split_ds.map(fmt, remove_columns=split_ds.column_names))
        except Exception as e:
            logger.warning("PubMedQA config=%s skipped: %s", config_name, e)
    return parts


def _load_medmcqa() -> list:
    raw = load_dataset("medmcqa", trust_remote_code=True)

    def fmt(x):
        opts = (
            f"A) {x.get('opa','')}\n"
            f"B) {x.get('opb','')}\n"
            f"C) {x.get('opc','')}\n"
            f"D) {x.get('opd','')}"
        )
        exp     = (x.get("exp") or "").strip()
        subject = (x.get("subject_name") or "").strip()
        topic   = (x.get("topic_name")   or "").strip()
        header  = f"[{subject} — {topic}]\n" if subject else ""
        return {"text": f"{header}{x['question'].strip()}\n{opts}\n{exp}".strip()}

    return [
        split_ds.map(fmt, remove_columns=split_ds.column_names)
        for split_ds in raw.values()
    ]


def _load_medalpaca() -> list:
    raw = load_dataset(
        "medalpaca/medical_meadow_medical_flashcards",
        trust_remote_code=True,
    )

    def fmt(x):
        fields = [x.get("instruction", ""), x.get("input", ""), x.get("output", "")]
        return {"text": "\n".join(f.strip() for f in fields if f.strip())}

    return [
        split_ds.map(fmt, remove_columns=split_ds.column_names)
        for split_ds in raw.values()
    ]


def build_dataset(cfg: DataConfig) -> DatasetDict:
    """Download, clean, dedup, and split the medical corpus."""
    if cfg.load_from_cache:
        logger.info("Loading dataset from cache: %s", cfg.save_dir)
        return load_from_disk(cfg.save_dir)

    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    loaders = {
        "PubMedQA": _load_pubmedqa,
        "MedMCQA":  _load_medmcqa,
        "MedAlpaca": _load_medalpaca,
    }

    parts = []
    for name, loader_fn in loaders.items():
        try:
            splits = loader_fn()
            n = sum(len(s) for s in splits)
            parts.extend(splits)
            logger.info("%-12s loaded: %s samples", name, f"{n:,}")
        except Exception as e:
            logger.error("%-12s FAILED: %s", name, e)

    if not parts:
        raise RuntimeError("All dataset loaders failed.")

    combined = concatenate_datasets(parts)
    before = len(combined)

    combined = combined.filter(
        lambda x: isinstance(x["text"], str) and len(x["text"].strip()) >= cfg.min_text_length
    )
    logger.info("After length filter: %s → %s", f"{before:,}", f"{len(combined):,}")

    seen: set = set()
    def is_unique(x):
        key = x["text"][: cfg.dedup_prefix_len].strip().lower()
        if key in seen:
            return False
        seen.add(key)
        return True

    combined = combined.filter(is_unique)
    logger.info("After dedup: %s samples", f"{len(combined):,}")

    combined = combined.shuffle(seed=cfg.seed)
    total    = len(combined)
    n_train  = int(total * cfg.train_ratio)
    n_val    = int(total * cfg.val_ratio)

    ds_dict = DatasetDict({
        "train": combined.select(range(n_train)),
        "val":   combined.select(range(n_train, n_train + n_val)),
        "test":  combined.select(range(n_train + n_val, total)),
    })

    logger.info(
        "Split → train: %s | val: %s | test: %s",
        f"{len(ds_dict['train']):,}",
        f"{len(ds_dict['val']):,}",
        f"{len(ds_dict['test']):,}",
    )

    ds_dict.save_to_disk(cfg.save_dir)
    logger.info("Dataset cached to: %s", cfg.save_dir)
    return ds_dict


# ── Model helpers ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Training ───────────────────────────────────────────────────────────────────
def train(model_cfg: ModelConfig, train_ds, val_ds, tokenizer) -> None:
    adapter_path = os.path.join(model_cfg.output_dir, "lora_adapter")

    training_args = TrainingArguments(
        output_dir=model_cfg.output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        hub_model_id=model_cfg.hub_model_id if model_cfg.push_to_hub else None,
        push_to_hub=model_cfg.push_to_hub,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=model_cfg.max_seq_length,
        packing=True,
    )

    logger.info("Starting training...")
    trainer.train()

    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved → %s", adapter_path)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Auth — never hardcode tokens; use env var or .env file
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN env variable not set. Run: export HF_TOKEN=hf_...")
    login(token=hf_token)

    # GPU selection via env var, default to GPU 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logger.info("Using CUDA device: %s", os.environ["CUDA_VISIBLE_DEVICES"])

    parser = HfArgumentParser((DataConfig, ModelConfig))
    data_cfg, model_cfg = parser.parse_args_into_dataclasses()

    set_reproducibility(data_cfg.seed)

    # ── Data ───────────────────────────────────────────────────
    ds       = build_dataset(data_cfg)
    train_ds = ds["train"]
    val_ds   = ds["val"]

    # ── Model ──────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # ── Train ──────────────────────────────────────────────────
    train(model_cfg, train_ds, val_ds, tokenizer)

    logger.info("Done.")