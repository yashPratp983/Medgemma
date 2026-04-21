from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    pulse_rate: str = Field(..., examples=["72"])
    oxygen_level: str = Field(..., examples=["98"])
    temperature: str = Field(..., examples=["98.6"])
    text: Optional[str] = Field(default="", description="Typed symptoms (preferred)")
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded audio clip (used when text is empty)",
    )


class DiagnosisResult(BaseModel):
    treatment_plan: Optional[str]
