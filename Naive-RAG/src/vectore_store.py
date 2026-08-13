from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from config import settings


class VectorStoreError(Exception):
    """Raised for vector store failures."""


def _store_id(repo_path: Path) -> str:
    """Return a stable, unique id for a repository (avoids folder-name collisions)."""
    resolved = repo_path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{resolved.name}-{digest}"


def _store_dir(repo_path: Path) -> Path:
    return settings.VECTOR_STORE_ROOT / _store_id(repo_path)


def index_path(repo_path: Path) -> Path:
    return _store_dir(repo_path) / "index.faiss"


def metadata_path(repo_path: Path) -> Path:
    return _store_dir(repo_path) / "metadata.json"


def _normalize_embeddings(embeddings: List[List[float]]) -> np.ndarray:
    array = np.array(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise VectorStoreError("Embeddings must be a 2D list.")
    faiss.normalize_L2(array)
    return array


def build_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    array = _normalize_embeddings(embeddings)
    index = faiss.IndexFlatIP(array.shape[1])
    index.add(array)
    return index


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexFlatIP:
    if not path.exists():
        raise VectorStoreError(f"FAISS index not found: {path}")
    return faiss.read_index(str(path))


def save_metadata(metadata: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise VectorStoreError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def vector_store_exists(repo_path: Path) -> bool:
    """Return True when a persisted index and metadata file already exist."""
    return index_path(repo_path).exists() and metadata_path(repo_path).exists()


def persist_vector_store(repo_path: Path, embeddings: List[List[float]], documents: List[Document]) -> None:
    """Persist the FAISS index and chunk metadata (including the chunk text)."""
    if len(embeddings) != len(documents):
        raise VectorStoreError(
            f"Embedding count ({len(embeddings)}) does not match document count ({len(documents)})."
        )

    index = build_index(embeddings)
    _store_dir(repo_path).mkdir(parents=True, exist_ok=True)
    save_index(index, index_path(repo_path))

    metadata = [{**doc.metadata, "text": doc.page_content} for doc in documents]
    save_metadata(metadata, metadata_path(repo_path))


def load_vector_store(repo_path: Path) -> tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    index = load_index(index_path(repo_path))
    metadata = load_metadata(metadata_path(repo_path))
    return index, metadata


def search(
    repo_path: Path,
    query_embedding: List[float],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = settings.TOP_K

    index, metadata = load_vector_store(repo_path)

    query = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    distances, indices = index.search(query, top_k)

    results: List[Dict[str, Any]] = []

    return results