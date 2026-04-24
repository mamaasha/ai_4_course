from pathlib import Path
from pydantic_settings import BaseSettings


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "petfinder_profiles"
    llm_service_url: str = "http://llm_service:8000"
    llm_provider: str = "ollama"

    class Config:
        env_file = str(ENV_PATH)
        env_prefix = ""
        extra = "ignore"


settings = Settings()
