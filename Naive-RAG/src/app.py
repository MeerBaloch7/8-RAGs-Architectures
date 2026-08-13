from __future__ import annotations

import argparse
import contextlib
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import settings
from chunker import chunk_repository
from embeddings import embed_documents
from qa import prepare_qa_payload
from repo_crawler import clone_repo
from vectore_store import (
    load_vector_store,
    persist_vector_store,
    vector_store_exists,
)


# Keep terminal output clean.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


@contextlib.contextmanager
def silence_stderr():
    """Suppress low-level stderr output from libraries."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull_fd, 2)

    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def ensure_hf_token() -> None:
    if settings.HF_API_KEY:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HF_API_KEY


def resolve_repository_path(
    repo_ref: str,
    target_root: Path,
    branch: Optional[str],
    force: bool,
) -> Path:
    candidate = Path(repo_ref)

    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()

    return clone_repo(
        repo_ref,
        branch=branch,
        target_root=target_root,
        force=force,
    )


def build_repository_index(
    repo_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    verbose: bool = False,
    rebuild: bool = False,
) -> int:
    if vector_store_exists(repo_path) and not rebuild:
        if verbose:
            print(f"Using existing index: {repo_path.name}")
        return load_vector_store(repo_path)[0].ntotal

    if verbose:
        print(f"Indexing repository: {repo_path}")

    documents = chunk_repository(
        repo_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not documents:
        raise RuntimeError("No supported files found in the repository.")

    if verbose:
        print(f"Embedding {len(documents)} chunks...")

    embeddings = embed_documents(documents)
    persist_vector_store(repo_path, embeddings, documents)

    return len(documents)


@lru_cache(maxsize=1)
def load_generation_pipeline(model_name: str):
    """Load the generation model once and reuse it."""
    ensure_hf_token()

    token_kwargs = {}
    model_kwargs = {}

    if settings.HF_API_KEY:
        token_kwargs["token"] = settings.HF_API_KEY
        model_kwargs["token"] = settings.HF_API_KEY

    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **token_kwargs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    device = 0 if torch.cuda.is_available() else -1

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    return generator, tokenizer


def format_model_prompt(tokenizer, prompt: str) -> str:
    """Use the model's chat template when available."""
    if not tokenizer.chat_template:
        return prompt

    messages = [
        {
            "role": "system",
            "content": "You are a code assistant for a GitHub repository.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_thinking(text: str) -> str:
    """Remove thinking blocks emitted by reasoning models."""
    text = text.lstrip()

    if text.startswith("<think>"):
        end = text.find("</think>")

        if end != -1:
            text = text[end + len("</think>"):]

    return text.strip()


def generate_answer(
    prompt: str,
    model_name: str,
    max_new_tokens: int = 128,
) -> str:
    generator, tokenizer = load_generation_pipeline(model_name)

    model_prompt = format_model_prompt(
        tokenizer,
        prompt,
    )

    outputs = generator(
        model_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )

    return strip_thinking(
        outputs[0]["generated_text"]
    )


def chat(
    repo_path: Path,
    model_name: str,
    max_chunks: int,
    max_new_tokens: int,
    verbose: bool = False,
) -> None:
    """Run a continuous repository QA chat."""
    print("\nRAG Assistant ready.")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            payload = prepare_qa_payload(
                repo_path=repo_path,
                question=question,
                max_chunks=max_chunks,
                debug=verbose,
            )

            if verbose:
                print("\n=== Prompt ===")
                print(payload["prompt"])

            answer = generate_answer(
                payload["prompt"],
                model_name,
                max_new_tokens=max_new_tokens,
            )

            print(f"Assistant: {answer}\n")

        except Exception:
            print(
                "Assistant: Sorry, I couldn't process that question.\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with a GitHub repository using RAG."
    )

    parser.add_argument(
        "repo",
        help="GitHub repository (owner/repo) or local repository path.",
    )

    parser.add_argument(
        "--branch",
        default=None,
        help="Git branch or tag.",
    )

    parser.add_argument(
        "--target-root",
        default="data/repos",
        help="Directory where repositories are stored.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-clone an existing repository.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the vector index.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap.",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5,
        help="Maximum chunks passed to the LLM.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens.",
    )

    parser.add_argument(
        "--model",
        default=settings.HF_LLM_MODEL,
        help="Hugging Face generation model.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show retrieval and prompt debugging information.",
    )

    args = parser.parse_args()

    repo_path = resolve_repository_path(
        args.repo,
        Path(args.target_root),
        args.branch,
        args.force,
    )

    with silence_stderr():
        indexed_chunks = build_repository_index(
            repo_path=repo_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            verbose=args.verbose,
            rebuild=args.rebuild,
        )

        if args.verbose:
            print(f"Indexed chunks: {indexed_chunks}")

        chat(
            repo_path=repo_path,
            model_name=args.model,
            max_chunks=args.max_chunks,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()