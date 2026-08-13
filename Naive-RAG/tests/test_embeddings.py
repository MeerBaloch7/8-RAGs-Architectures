import torch
from langchain_core.documents import Document

import embeddings


class DummyTokenizer:
    def __call__(self, texts, padding, truncation, return_tensors):
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def forward(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        return type("Output", (), {"last_hidden_state": torch.ones((batch_size, 4, 8), dtype=torch.float32)})


def test_embed_texts_monkeypatched(monkeypatch):
    def fake_load_embedding_model():
        return DummyTokenizer(), DummyModel()

    monkeypatch.setattr(embeddings, "load_embedding_model", fake_load_embedding_model)
    result = embeddings.embed_texts(["hello", "world"], batch_size=2)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(len(vector) == 8 for vector in result)
    assert all(abs(sum(x * x for x in vector) - 1.0) < 1e-5 for vector in result)


def test_embed_texts_batches(monkeypatch):
    def fake_load_embedding_model():
        return DummyTokenizer(), DummyModel()

    monkeypatch.setattr(embeddings, "load_embedding_model", fake_load_embedding_model)
    result = embeddings.embed_texts(["a", "b", "c", "d", "e"], batch_size=2)

    assert len(result) == 5
    assert all(len(vector) == 8 for vector in result)


def test_embed_documents(monkeypatch):
    def fake_load_embedding_model():
        return DummyTokenizer(), DummyModel()

    monkeypatch.setattr(embeddings, "load_embedding_model", fake_load_embedding_model)
    documents = [Document(page_content="hello world", metadata={})]
    result = embeddings.embed_documents(documents)
    assert len(result) == 1
    assert len(result[0]) == 8
