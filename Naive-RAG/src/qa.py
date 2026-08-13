from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from embeddings import embed_texts
from prompt_builder import build_prompt
from vectore_store import search, load_vector_store

def embed_question(question: str) -> List[float]:
    """Generate an embedding vector for a natural language query."""
    return embed_texts([question])[0]


def load_document_from_metadata(metadata: Dict[str, Any]) -> Optional[Document]:
    """Rebuild a Document from the chunk text stored in the vector store metadata."""
    text = metadata.get("text")
    if not text:
        return None
    return Document(page_content=text, metadata=metadata)


def get_relevant_documents(
    repo_path: Path,
    question: str,
    top_k: Optional[int] = None,
    debug: bool = False,
) -> List[Document]:
    """Retrieve relevant chunks using semantic + keyword matching."""

    query_embedding = embed_question(question)

    # Semantic search across the entire FAISS index
    results = search(repo_path, query_embedding, top_k=top_k)

    # Load all stored chunks so we can check exact terms across the repository
    from vectore_store import load_vector_store

    _, metadata = load_vector_store(repo_path)

    question_words = {
        word.lower()
        for word in question.split()
        if len(word) > 2
    }

    keyword_results = []

    for item in metadata:
        text = item.get("text", "").lower()

        matches = sum(
            1 for word in question_words
            if word in text
        )

        if matches:
            keyword_results.append(
                {
                    **item,
                    "keyword_score": matches,
                }
            )

    # Sort exact keyword matches first
    keyword_results.sort(
        key=lambda item: item["keyword_score"],
        reverse=True,
    )

    # Merge semantic + keyword results
    merged = {}

    for item in results:
        key = (
            item.get("source"),
            item.get("text"),
        )

        merged[key] = item

    for item in keyword_results:
        key = (
            item.get("source"),
            item.get("text"),
        )

        if key in merged:
            merged[key]["keyword_score"] = item["keyword_score"]
        else:
            merged[key] = item

    # Rank using keyword relevance first, semantic score second
    final_results = sorted(
        merged.values(),
        key=lambda item: (
            item.get("keyword_score", 0),
            item.get("score", 0.0),
        ),
        reverse=True,
    )

    if debug:
        print("\n=== HYBRID RETRIEVAL ===")
        print(f"Question: {question}")
        print(f"Semantic results: {len(results)}")
        print(f"Keyword matches: {len(keyword_results)}")
        print(f"Final results: {len(final_results)}")

        for i, item in enumerate(final_results[:10], start=1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {item.get('source')}")
            print(f"Semantic score: {item.get('score', 0):.4f}")
            print(f"Keyword score: {item.get('keyword_score', 0)}")
            print(item.get("text", "")[:700])

    documents = []

    for item in final_results:
        document = load_document_from_metadata(item)

        if document is not None:
            documents.append(document)

    return documents

def build_answer_prompt(
    question: str,
    documents: List[Document],
    max_chunks: int = 5,
) -> str:
    """Build the final prompt text for the model using retrieved documents."""
    return build_prompt(question, documents, max_chunks=max_chunks)


def prepare_qa_payload(
    repo_path: Path,
    question: str,
    top_k: Optional[int] = None,
    max_chunks: int = 3,
    debug: bool = False,
) -> Dict[str, Any]:
    documents = get_relevant_documents(
        repo_path,
        question,
        top_k=top_k,
        debug=debug,
    )

    return {
        "question": question,
        "prompt": build_answer_prompt(
            question,
            documents,
            max_chunks=max_chunks,
        ),
        "documents": documents,
    }