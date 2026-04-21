from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # HuggingFace
    hf_model_id: str = "Yash983/medgemma-2b-sft"
    hf_token: str = ""                  # set if the repo is private/gated
    torch_dtype: str = "bfloat16"       # "bfloat16" | "float16" | "float32"
    device_map: str = "auto"            # "auto" | "cpu" | "cuda"
    max_new_tokens: int = 300

    # Other models
    keras_model_path: str = "/home/yash/Downloads/v1.keras"
    confidence_threshold: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()
