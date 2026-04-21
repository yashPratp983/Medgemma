from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.api import api_router

app = FastAPI(
    title="Medical Diagnosis API",
    description=(
        "Two endpoints:\n"
        "- **POST /api/v1/chat** — LLM-based symptom diagnosis from text or audio\n"
        "- **POST /api/v1/classify-image** — Skin-disease image classification + GPT advice"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)
