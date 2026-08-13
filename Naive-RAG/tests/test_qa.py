from pathlib import Path
from tempfile import TemporaryDirectory

import qa


def test_prepare_qa_payload(monkeypatch):
    with TemporaryDirectory() as tempdir:
        repo_path = Path(tempdir) / "repo"
        repo_path.mkdir(parents=True)
        file_path = repo_path / "sample.py"
        file_path.write_text("print('hello world')\n")

        metadata = {
            "source": "sample.py",
            "file_path": str(file_path),
            "file_name": "sample.py",
            "file_extension": ".py",
            "text": "print('hello world')\n",
        }

        monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])
        monkeypatch.setattr(qa, "search", lambda *_args, **_kwargs: [metadata])

        payload = qa.prepare_qa_payload(repo_path, "What does this file do?", top_k=1, max_chunks=1)

        assert "Question:" in payload["prompt"]
        assert payload["question"] == "What does this file do?"
        assert payload["documents"]
        assert payload["documents"][0].metadata["file_name"] == "sample.py"
        assert payload["documents"][0].page_content == "print('hello world')\n"


def test_get_relevant_documents_deduplicates_by_source(monkeypatch):
    with TemporaryDirectory() as tempdir:
        repo_path = Path(tempdir) / "repo"
        repo_path.mkdir(parents=True)

        def fake_search(*_args, **_kwargs):
            return [
                {"source": "a.py", "text": "first chunk", "score": 0.9},
                {"source": "a.py", "text": "second chunk", "score": 0.8},
                {"source": "b.py", "text": "third chunk", "score": 0.7},
            ]

        monkeypatch.setattr(qa, "embed_texts", lambda texts: [[1.0, 0.0]])
        monkeypatch.setattr(qa, "search", fake_search)

        documents = qa.get_relevant_documents(repo_path, "question", top_k=3)

        assert [doc.metadata["source"] for doc in documents] == ["a.py", "b.py"]
