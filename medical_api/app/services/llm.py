from __future__ import annotations
from typing import Optional
from app.core.config import settings


_NULL_RESULT: dict = {
    "predicted_disease": None,
    "treatment_plan": None,
    "prescribed_drugs": None,
    "specialization": None,
}

_VITALS_FORMAT = (
    'Pulse Rate: "{pulse_rate}", '
    'Oxygen level: "{oxygen_level}", '
    'Temperature: "{temperature}"'
)

_SYSTEM_SYMPTOMS = (
    "You are a clinical decision-support assistant.\n"
    "Analyze symptoms and vitals and provide a helpful medical response.\n\n"
    "Your response should include:\n"
    "- likely disease\n"
    "- treatment advice\n"
    "- possible medications\n"
    "- which specialist to visit\n\n"
    "Write a concise medical paragraph."
)

_SYSTEM_DISEASE = (
    "You are a clinical decision-support assistant.\n"
    "Given confirmed disease and vitals, provide management guidance.\n\n"
    "Include:\n"
    "- disease explanation\n"
    "- treatment plan\n"
    "- medications\n"
    "- specialist recommendation\n\n"
    "Write a concise medical paragraph."
)

_SYSTEM_EXTRACT = (
    "Extract symptoms from text.\n"
    "Return comma separated symptoms only."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diagnose_from_symptoms(
    pipe,
    symptoms: str,
    pulse_rate: str,
    oxygen_level: str,
    temperature: str,
) -> dict:

    user_msg = (
        f'Symptoms: "{symptoms}", '
        + _VITALS_FORMAT.format(
            pulse_rate=pulse_rate,
            oxygen_level=oxygen_level,
            temperature=temperature,
        )
    )

    text = _generate(pipe, _SYSTEM_SYMPTOMS, user_msg)

    return {
        "treatment_plan": text,
    }


def diagnose_from_disease(
    pipe,
    disease: str,
    pulse_rate: str,
    oxygen_level: str,
    temperature: str,
) -> dict:

    user_msg = (
        f'Disease: "{disease}", '
        + _VITALS_FORMAT.format(
            pulse_rate=pulse_rate,
            oxygen_level=oxygen_level,
            temperature=temperature,
        )
    )

    text = _generate(pipe, _SYSTEM_DISEASE, user_msg)

    return {
        "predicted_disease": disease,
        "treatment_plan": text,
    }


def extract_symptoms_from_text(pipe, transcript: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_EXTRACT},
        {"role": "user", "content": transcript},
    ]

    return _run_pipeline(pipe, messages, max_new_tokens=128).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate(pipe, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return _run_pipeline(pipe, messages, max_new_tokens=settings.max_new_tokens)


def _run_pipeline(pipe, messages: list[dict], max_new_tokens: int) -> str:
    prompt = ""

    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "system":
            prompt += f"<start_of_turn>system\n{content}<end_of_turn>\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"

    prompt += "<start_of_turn>assistant\n"

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    return outputs[0]["generated_text"].strip()