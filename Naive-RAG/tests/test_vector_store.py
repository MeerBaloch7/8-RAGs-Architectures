from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from langchain_core.documents import Document

from vectore_store import (
    VectorStoreError,
    load_vector_store,
    persist_vector_store,
    search,
    vector_store_exists,
)


def test_persist_and_load_vector_store():
    with TemporaryDirectory() as tempdir:
        repo_path = Path(tempdir) / "repo"
        repo_path.mkdir(parents=True)

        documents = [
            Document(
                page_content="hello world",
                metadata={
                    "source": "sample.py",
                    "file_path": str(repo_path / "sample.py"),
                    "file_name": "sample.py",
                    "file_extension": ".py",
                },
            )
        ]
        embeddings = [[1.0, 0.0]]

        assert not vector_store_exists(repo_path)
        persist_vector_store(repo_path, embeddings, documents)
        assert vector_store_exists(repo_path)

        index, metadata = load_vector_store(repo_path)

        assert index.ntotal == 1
        assert len(metadata) == 1
        assert metadata[0]["text"] == "hello world"

        results = search(repo_path, [1.0, 0.0], top_k=1)
        assert results[0]["file_name"] == "sample.py"
        assert results[0]["text"] == "hello world"
        assert results[0]["score"] >= 0


def test_persist_vector_store_validates_counts():
    with TemporaryDirectory() as tempdir:
        repo_path = Path(tempdir) / "repo"
        repo_path.mkdir(parents=True)

        documents = [Document(page_content="hello", metadata={"source": "sample.py"})]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

        with pytest.raises(VectorStoreError):
            persist_vector_store(repo_path, embeddings, documents)
