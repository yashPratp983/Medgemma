from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.model_registry import registry
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the HF pipeline at startup so the first request isn't slow.
    logger.info("Loading MedGemma pipeline …")
    _ = registry.hf_pipeline
    logger.info("MedGemma pipeline ready.")
    yield
    registry.reset()
