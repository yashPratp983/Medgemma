"""
dpo_train.py — Production-ready DPO alignment for MedGemma-2B
Loads SFT adapter, merges it into base, applies fresh LoRA, trains with DPOTrainer.
"""

import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class DPODataConfig:
    dataset_id: str = field(
        default="WhiteF4lcon/medical-rlhf-pairs",
        metadata={"help": "HuggingFace dataset with instruction/positive/negative columns"},
    )
    val_split: float = field(default=0.10, metadata={"help": "Fraction of data held out for validation"})
    seed: int = field(default=42)


@dataclass
class DPOModelConfig:
    model_id: str = field(default="google/gemma-2b")
    sft_adapter: str = field(default="model/lora_adapter", metadata={"help": "Path to trained SFT LoRA adapter"})
    output_dir: str = field(default="model/dpo_adapter")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)


# ── BnB config (shared between model and ref_model) ───────────────────────────
def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


# ── Model loading ──────────────────────────────────────────────────────────────
def load_and_merge_sft(model_cfg: DPOModelConfig) -> AutoModelForCausalLM:
    """Load base model, attach SFT adapter, merge weights, return plain model."""
    bnb = get_bnb_config()

    logger.info("Loading base model: %s", model_cfg.model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    logger.info("Attaching SFT adapter: %s", model_cfg.sft_adapter)
    sft = PeftModel.from_pretrained(base, model_cfg.sft_adapter)

    logger.info("Merging SFT LoRA into base weights...")
    merged = sft.merge_and_unload()
    merged.config.use_cache = False
    logger.info("Merge complete")
    return merged


def apply_dpo_lora(merged_model: AutoModelForCausalLM, model_cfg: DPOModelConfig):
    """Prepare merged model for k-bit training and attach a fresh DPO LoRA."""
    merged_model = prepare_model_for_kbit_training(merged_model)

    lora_cfg = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(merged_model, lora_cfg)
    model.print_trainable_parameters()
    logger.info("Fresh DPO LoRA applied")
    return model


def load_ref_model(model_cfg: DPOModelConfig) -> AutoModelForCausalLM:
    """Load a frozen reference model (base + SFT adapter, not merged) for KL penalty."""
    bnb = get_bnb_config()
    ref = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    ref.load_adapter(model_cfg.sft_adapter)
    ref.eval()
    logger.info("Reference model loaded")
    return ref


# ── Dataset ────────────────────────────────────────────────────────────────────
def build_dpo_dataset(data_cfg: DPODataConfig):
    logger.info("Loading DPO dataset: %s", data_cfg.dataset_id)
    raw = load_dataset(data_cfg.dataset_id)

    def format_dpo(example):
        return {
            "prompt":   example["instruction"],
            "chosen":   example["positive"],
            "rejected": example["negative"],
        }

    dpo_ds = raw["train"].map(format_dpo, remove_columns=raw["train"].column_names)
    split  = dpo_ds.train_test_split(test_size=data_cfg.val_split, seed=data_cfg.seed)

    logger.info(
        "DPO split → train: %s | val: %s",
        f"{len(split['train']):,}",
        f"{len(split['test']):,}",
    )
    logger.info("Sample prompt: %s", split["train"][0]["prompt"][:120])
    return split["train"], split["test"]


# ── Training ───────────────────────────────────────────────────────────────────
def train_dpo(model, ref_model, tokenizer, train_ds, val_ds, model_cfg: DPOModelConfig) -> None:
    dpo_cfg = DPOConfig(
        output_dir=model_cfg.output_dir,
        beta=0.5,
        loss_type="ipo",
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        weight_decay=0.01,
        fp16=False,
        logging_steps=25,
        save_steps=100,
        save_total_limit=2,
        eval_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.5,
        push_to_hub=model_cfg.push_to_hub,
        hub_model_id=model_cfg.hub_model_id if model_cfg.push_to_hub else None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    trainer.save_model(model_cfg.output_dir)
    tokenizer.save_pretrained(model_cfg.output_dir)
    logger.info("DPO adapter saved → %s", model_cfg.output_dir)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN env variable not set. Run: export HF_TOKEN=hf_...")
    login(token=hf_token)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logger.info("Using CUDA device: %s", os.environ["CUDA_VISIBLE_DEVICES"])

    parser = HfArgumentParser((DPODataConfig, DPOModelConfig))
    data_cfg, model_cfg = parser.parse_args_into_dataclasses()

    set_seed(data_cfg.seed)

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Data ───────────────────────────────────────────────────
    train_ds, val_ds = build_dpo_dataset(data_cfg)

    # ── Models ─────────────────────────────────────────────────
    merged  = load_and_merge_sft(model_cfg)
    model   = apply_dpo_lora(merged, model_cfg)
    ref_model = load_ref_model(model_cfg)

    # ── Train ──────────────────────────────────────────────────
    train_dpo(model, ref_model, tokenizer, train_ds, val_ds, model_cfg)

    del model, ref_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Done.")