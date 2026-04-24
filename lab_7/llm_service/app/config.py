
from pathlib import Path
from pydantic_settings import BaseSettings


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    llm_provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:0.5b"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    hf_token: str = ""
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    class Config:
        env_file = str(ENV_PATH)
        env_prefix = ""
        extra = "ignore"


settings = Settings()
