from typing import Optional
from pydantic import BaseModel


class ImageDiagnosisResponse(BaseModel):
    predicted_disease: Optional[str]
    treatment_plan: Optional[str]
