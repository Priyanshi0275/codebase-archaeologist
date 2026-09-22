"""
Pulls commit history (message + files changed + diff) from a git repo —
either a local path or a URL to clone — and stores chunked text in Chroma.

Fully free: uses GitPython to read local git data, no GitHub API calls
required (so no rate limits, no token needed for public repos).
"""
import os
import shutil
import tempfile
import git
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore import add_chunks, get_collection

REPO_CACHE_DIR = "./repo_cache"
MAX_DIFF_CHARS = 2000  # keep individual diffs from ballooning chunk counts

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _resolve_repo(repo_path_or_url: str) -> str:
    """Return a local path to the repo, cloning it if a URL was given."""
    if os.path.isdir(repo_path_or_url):
        return repo_path_or_url

    os.makedirs(REPO_CACHE_DIR, exist_ok=True)
    repo_name = repo_path_or_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = os.path.join(REPO_CACHE_DIR, repo_name)

    if os.path.isdir(dest):
        shutil.rmtree(dest)

    git.Repo.clone_from(repo_path_or_url, dest, depth=200)  # shallow clone, keeps it fast/free
    return dest


def ingest_repo(repo_path_or_url: str, max_commits: int = 150, reset: bool = True) -> dict:
    local_path = _resolve_repo(repo_path_or_url)
    repo = git.Repo(local_path)

    ids, texts, metadatas = [], [], []
    n_commits = 0

    for commit in repo.iter_commits(max_count=max_commits):
        n_commits += 1
        sha = commit.hexsha[:10]
        message = commit.message.strip()
        author = str(commit.author)
        date = commit.committed_datetime.isoformat()

        try:
            files_changed = list(commit.stats.files.keys())
        except Exception:
            files_changed = []

        diff_text = ""
        try:
            if commit.parents:
                diff_text = commit.diff(commit.parents[0], create_patch=True)
                diff_text = "\n".join(
                    d.diff.decode("utf-8", errors="ignore")[:500] for d in diff_text[:5]
                )
        except Exception:
            diff_text = ""
        diff_text = diff_text[:MAX_DIFF_CHARS]

        header = (
            f"Commit {sha} by {author} on {date}\n"
            f"Message: {message}\n"
            f"Files changed: {', '.join(files_changed[:20])}\n"
        )
        full_text = header + (f"Diff excerpt:\n{diff_text}" if diff_text else "")

        chunks = _splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{sha}-{i}")
            texts.append(chunk)
            metadatas.append({
                "sha": sha,
                "author": author,
                "date": date,
                "files": ", ".join(files_changed[:20]),
            })

    if reset:
        get_collection(reset=True)

    if ids:
        add_chunks(ids, texts, metadatas)

    return {
        "repo": repo_path_or_url,
        "commits_ingested": n_commits,
        "chunks_stored": len(ids),
    }
