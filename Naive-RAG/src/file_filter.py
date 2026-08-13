from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Set

SUPPORTED_EXTENSIONS: Set[str] = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".sql",
    ".md",
    ".rst",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
}

EXCLUDE_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".env",
}


def is_supported_file(path: Path, extensions: Set[str] = SUPPORTED_EXTENSIONS) -> bool:
    """Return True when the file extension is supported."""
    return path.is_file() and path.suffix.lower() in extensions


def iter_source_files(
    repo_path: Path,
    extensions: Set[str] = SUPPORTED_EXTENSIONS,
    exclude_dirs: Set[str] = EXCLUDE_DIRS,
) -> Iterable[Path]:
    """Yield supported files under the repository, skipping ignored directories."""
    repo_path = repo_path.resolve()

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file_name in files:
            file_path = Path(root) / file_name
            if is_supported_file(file_path, extensions):
                yield file_path


def list_source_files(
    repo_path: Path,
    extensions: Set[str] = SUPPORTED_EXTENSIONS,
    exclude_dirs: Set[str] = EXCLUDE_DIRS,
) -> List[Path]:
    """Return a sorted list of supported source files."""
    return sorted(iter_source_files(repo_path, extensions, exclude_dirs))
