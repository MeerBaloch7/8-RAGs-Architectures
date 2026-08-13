from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

import torch
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer

from config import settings


class EmbeddingsError(Exception):
    """Raised when embedding generation fails."""


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_embedding_model() -> tuple[AutoTokenizer, AutoModel]:
    model_name = settings.HF_EMBEDDING_MODEL
    if not model_name:
        raise EmbeddingsError("HF_EMBEDDING_MODEL is not configured.")

    if settings.HF_API_KEY:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HF_API_KEY

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average hidden states over real tokens only, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> List[List[float]]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()


def embed_texts(texts: Iterable[str], batch_size: int = 32) -> List[List[float]]:
    tokenizer, model = load_embedding_model()
    device = get_device()
    model = model.to(device)
    model.eval()

    all_embeddings: List[List[float]] = []
    batch: List[str] = []

    for text in texts:
        batch.append(text)
        if len(batch) < batch_size:
            continue
        all_embeddings.extend(_embed_batch(batch, tokenizer, model, device))
        batch = []

    if batch:
        all_embeddings.extend(_embed_batch(batch, tokenizer, model, device))

    return all_embeddings


def embed_documents(documents: list[Document], batch_size: int | None = None) -> List[List[float]]:
    texts = [doc.page_content for doc in documents]
    return embed_texts(texts, batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE)
