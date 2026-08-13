from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from file_filter import iter_source_files


def load_repository(repo_path: Path) -> List[Document]:
    """Load supported source files from a repository into LangChain Documents."""
    repo_path = repo_path.resolve()
    documents: List[Document] = []

    for file_path in iter_source_files(repo_path):
        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if not content.strip():
            continue

        relative_path = file_path.relative_to(repo_path)

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(relative_path),
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix.lower(),
                },
            )
        )

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split loaded Documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    return splitter.split_documents(documents)


def chunk_repository(
    repo_path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Load repository files and return chunked Documents."""

    documents = load_repository(repo_path)

    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]