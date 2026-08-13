from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # Hugging Face
    HF_API_KEY: str | None = None
    HF_LLM_MODEL: str = "Qwen/Qwen3-4b"
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # GitHub
    GITHUB_TOKEN: str | None = None

    # RAGclear
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 20
    EMBEDDING_BATCH_SIZE: int = 32

    # Storage
    REPO_ROOT: Path = Path("data/repos")
    VECTOR_STORE_ROOT: Path = Path("data/faiss")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
