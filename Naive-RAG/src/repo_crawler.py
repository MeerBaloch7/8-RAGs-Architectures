from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_REPO_ROOT = Path("data/repos")
GITHUB_URL = "https://github.com"


class RepoCrawlerError(Exception):
    """Raised when a repository cannot be fetched."""


def parse_repo(repo_ref: str) -> tuple[str, str]:
    """Parse GitHub repo references into owner and repo."""
    repo_ref = repo_ref.strip().rstrip("/")
    patterns = (
        r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?$",
        r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$",
    )
    for pattern in patterns:
        match = re.match(pattern, repo_ref)
        if match:
            return match.group(1), match.group(2)
    raise RepoCrawlerError(f"Invalid GitHub repository reference: {repo_ref}")


def clone_repo(
    repo_ref: str,
    branch: str | None = None,
    target_root: Path = DEFAULT_REPO_ROOT,
    force: bool = False,
) -> Path:
    """Shallow clone a GitHub repository and return the local path."""
    owner, repo = parse_repo(repo_ref)
    target_root.mkdir(parents=True, exist_ok=True)
    repo_dir = target_root / f"{owner}_{repo}"

    if repo_dir.exists():
        if not force:
            return repo_dir
        shutil.rmtree(repo_dir)

    env = os.environ.copy()
    token = env.get("GITHUB_TOKEN")

    command = ["git", "clone", "--depth", "1"]
    if branch:
        command.extend(["--branch", branch])
    if token:
        command.extend(["-c", f"http.extraheader=Authorization: Bearer {token}"])
    command.extend([f"{GITHUB_URL}/{owner}/{repo}.git", str(repo_dir)])

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    except FileNotFoundError as exc:
        raise RepoCrawlerError("Git is not installed or available in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stderr.strip() or exc.stdout.strip() or "Unknown Git error."
        raise RepoCrawlerError(f"Clone failed: {output}") from exc

    return repo_dir


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Clone a GitHub repository for RAG indexing.")
    parser.add_argument("repo", help="GitHub repository, e.g. owner/repo")
    parser.add_argument("--branch", default=None, help="Branch or tag to clone.")
    parser.add_argument("--target-root", default=str(DEFAULT_REPO_ROOT), help="Directory where repositories are stored.")
    parser.add_argument("--force", action="store_true", help="Delete and re-clone an existing repository.")
    args = parser.parse_args()

    path = clone_repo(args.repo, branch=args.branch, target_root=Path(args.target_root), force=args.force)
    print(f"Repository available at: {path.resolve()}")


if __name__ == "__main__":
    main()