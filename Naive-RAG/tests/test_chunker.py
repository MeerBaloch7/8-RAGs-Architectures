from pathlib import Path
from tempfile import TemporaryDirectory

from chunker import chunk_repository, load_repository


def test_load_repository_and_chunk_repository():
    with TemporaryDirectory() as tempdir:
        repo_path = Path(tempdir) / "repo"
        repo_path.mkdir(parents=True)
        (repo_path / "main.py").write_text("print('hello world')\n")
        (repo_path / "README.md").write_text("# Repo\nThis is a test repository.\n")
        (repo_path / "empty.py").write_text("")
        (repo_path / ".git").mkdir()
        (repo_path / ".git" / "ignored").write_text("should be ignored")

        docs = load_repository(repo_path)
        assert len(docs) == 2
        assert any(doc.metadata["file_name"] == "main.py" for doc in docs)
        assert any(doc.metadata["file_name"] == "README.md" for doc in docs)
        assert all(doc.metadata["file_name"] != "empty.py" for doc in docs)

        chunks = chunk_repository(repo_path, chunk_size=20, chunk_overlap=5)
        assert len(chunks) >= 2
        assert all(hasattr(chunk, "page_content") for chunk in chunks)
        assert all(chunk.page_content.strip() for chunk in chunks)


def test_chunk_metadata_is_preserved():
    with TemporaryDirectory() as tempdir:
        repo_path = Path(tempdir) / "repo"
        repo_path.mkdir(parents=True)
        (repo_path / "main.py").write_text("import os\n\n\nprint('hello world')\n")

        chunks = chunk_repository(repo_path, chunk_size=20, chunk_overlap=5)
        assert chunks
        assert all(chunk.metadata.get("source") == "main.py" for chunk in chunks)
        assert all(chunk.metadata.get("file_extension") == ".py" for chunk in chunks)
