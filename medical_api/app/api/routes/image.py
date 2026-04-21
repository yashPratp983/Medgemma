from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.schemas.image import ImageDiagnosisResponse
from app.core.config import settings
from app.core.model_registry import registry
from app.services import image_classifier, llm

router = APIRouter(prefix="/classify-image", tags=["Image Classification"])


@router.post(
    "",
    response_model=ImageDiagnosisResponse,
    summary="Classify skin-disease image and return GPT treatment advice",
)
async def classify_image(
    image: UploadFile = File(..., description="JPEG skin-lesion photo"),
    pulse_rate: str = Form(...),
    oxygen_level: str = Form(...),
    temperature: str = Form(...),
):
    """
    Upload a JPEG image of the skin lesion together with the patient's vitals.
    Returns the predicted disease class, confidence score, and GPT-generated
    treatment advice.

    If model confidence is below the threshold the diagnosis fields are `null`.
    """
    image_bytes = await image.read()

    try:
        disease, confidence = image_classifier.predict(registry.keras, image_bytes)
    except Exception as e:
        raise HTTPException(422, f"Image processing failed: {e}")

    if confidence < settings.confidence_threshold:
        return ImageDiagnosisResponse(
            predicted_disease="No disease confidently detected",
            treatment_plan="",
        )

    advice = llm.diagnose_from_disease(
        registry.hf_pipeline,
        disease=disease,
        pulse_rate=pulse_rate,
        oxygen_level=oxygen_level,
        temperature=temperature,
    )

    return ImageDiagnosisResponse(
        predicted_disease=disease,
        treatment_plan=advice["treatment_plan"],
    )
