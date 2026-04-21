from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, DiagnosisResult
from app.core.model_registry import registry
from app.services import llm

router = APIRouter(prefix="/chat", tags=["LLM Chat Diagnosis"])


@router.post(
    "",
    response_model=DiagnosisResult,
    summary="Diagnose disease from symptoms (text or audio) + vitals",
)
async def chat_diagnosis(body: ChatRequest):
    """
    Provide patient vitals plus symptoms either as:
    - **text** – typed symptom description (preferred), or
    - **audio_base64** – base64-encoded audio that will be transcribed via Whisper.
    """
    if not body.pulse_rate:
        raise HTTPException(400, "pulse_rate is required")
    if not body.oxygen_level:
        raise HTTPException(400, "oxygen_level is required")
    if not body.temperature:
        raise HTTPException(400, "temperature is required")

    client = registry.hf_pipeline

    symptoms = llm.extract_symptoms_from_text(client, body.text)


    result = llm.diagnose_from_symptoms(
        client,
        symptoms=symptoms,
        pulse_rate=body.pulse_rate,
        oxygen_level=body.oxygen_level,
        temperature=body.temperature,
    )
    return DiagnosisResult(**result)
