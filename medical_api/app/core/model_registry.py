from __future__ import annotations
from typing import Optional, Any


class ModelRegistry:
    """Singleton-style store for all loaded models."""

    def __init__(self):
        # HuggingFace pipeline (replaces OpenAI client)
        self._hf_pipeline: Optional[Any] = None

        # Whisper (lazy)
        self._whisper: Optional[Any] = None

        # Keras skin classifier (lazy)
        self._keras: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # MedGemma HuggingFace pipeline
    # ------------------------------------------------------------------ #
    @property
    def hf_pipeline(self):
        if self._hf_pipeline is None:
            import torch
            from transformers import pipeline, AutoTokenizer
            from peft import AutoPeftModelForCausalLM
            from app.core.config import settings

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(settings.torch_dtype, torch.bfloat16)

            tokenizer = AutoTokenizer.from_pretrained(
                settings.hf_model_id,
                token=settings.hf_token if settings.hf_token else None,
            )

            model = AutoPeftModelForCausalLM.from_pretrained(
                settings.hf_model_id,
                torch_dtype=torch_dtype,
                device_map=settings.device_map,
                token=settings.hf_token if settings.hf_token else None,
            )

            self._hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

        return self._hf_pipeline

    # ------------------------------------------------------------------ #
    # Keras skin classifier (lazy)
    # ------------------------------------------------------------------ #
    @property
    def keras(self):
        if self._keras is None:
            from tensorflow.keras.models import load_model
            from app.core.config import settings
            self._keras = load_model(settings.keras_model_path)
        return self._keras

    def reset(self):
        self._hf_pipeline = None
        self._whisper = None
        self._keras = None


registry = ModelRegistry()
