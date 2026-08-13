from __future__ import annotations

from typing import List

from langchain_core.documents import Document

MAX_SNIPPET_LENGTH = 800


def build_context(documents: List[Document], max_chunks: int = 3) -> str:
    """Build a context block from the top document chunks."""
    if not documents:
        return "(No relevant code found in the repository.)"

    chunks = []
    for document in documents[:max_chunks]:
        source = document.metadata.get("source", "unknown")
        score = document.metadata.get("score")
        snippet = document.page_content.strip()

        if len(snippet) > MAX_SNIPPET_LENGTH:
            snippet = snippet[:MAX_SNIPPET_LENGTH].rsplit(" ", 1)[0] + "..."

        header = f"[FILE: {source}]"
        if score is not None:
            header += f" (score: {score:.2f})"
        chunks.append(f"{header}\n{snippet}\n")

    return "\n---\n".join(chunks)


def build_prompt(question: str, documents: List[Document], max_chunks: int = 5) -> str:
    """Create a prompt that asks the model to answer from repository code."""
    context = build_context(documents, max_chunks=max_chunks)
    return (
        "You are a code assistant for a GitHub repository. "
        "Use only the provided code context when answering. "
        "If the context does not contain the answer, reply exactly: I don't know.\n\n"
        "Context:\n"
        f"{context}\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Answer concisely and cite the source files by path."
    )
