import pytest

from config import settings


@pytest.fixture(autouse=True)
def isolate_vector_store(tmp_path, monkeypatch):
    """Redirect vector store writes to a temp directory so tests never touch real data."""
    monkeypatch.setattr(settings, "VECTOR_STORE_ROOT", tmp_path / "faiss")
