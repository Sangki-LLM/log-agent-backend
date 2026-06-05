import io
import re
from pathlib import Path

from dulwich import porcelain
from dulwich.object_store import tree_lookup_path
from dulwich.repo import Repo

from app.core.config import settings


def _repo_path(server_id: int) -> Path:
    return Path(settings.repos_path) / str(server_id)


def _auth_url(repo_url: str, token: str) -> str:
    if token and "github.com" in repo_url:
        return repo_url.replace("https://", f"https://{token}@")
    return repo_url


def clone(server_id: int, repo_url: str, branch: str, token: str = "") -> None:
    path = _repo_path(server_id)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    porcelain.clone(
        _auth_url(repo_url, token),
        str(path),
        branch=branch.encode(),
        errstream=io.BytesIO(),
    )


def fetch(server_id: int, repo_url: str = "", branch: str = "main", token: str = "") -> None:
    path = _repo_path(server_id)
    if not path.exists():
        if not repo_url:
            raise FileNotFoundError(f"Repo not cloned for server {server_id}")
        clone(server_id, repo_url, branch, token)
        return
    remote_url = _auth_url(repo_url, token) if repo_url else None
    with Repo(str(path)) as repo:
        porcelain.fetch(repo, remote_location=remote_url, errstream=io.BytesIO())


def get_remote_head(server_id: int, branch: str) -> str:
    path = _repo_path(server_id)
    with Repo(str(path)) as repo:
        ref_key = f"refs/remotes/origin/{branch}".encode()
        sha = repo.refs[ref_key]
        # dulwich refs return hex SHA as ascii bytes, not binary
        return sha.decode("ascii") if isinstance(sha, bytes) else sha


def read_files_at_commit(server_id: int, commit_hash: str, stack_trace: str) -> dict[str, str]:
    paths = _extract_source_paths(stack_trace)
    path = _repo_path(server_id)
    result: dict[str, str] = {}

    with Repo(str(path)) as repo:
        commit = repo[bytes.fromhex(commit_hash)]
        for rel_path in paths:
            try:
                _mode, blob_sha = tree_lookup_path(
                    repo.object_store.__getitem__,
                    commit.tree,
                    rel_path.encode(),
                )
                content = repo[blob_sha].data.decode("utf-8", errors="replace")
                result[rel_path] = content[:2000]
            except Exception:
                pass

    return result


def _extract_source_paths(stack_trace: str) -> list[str]:
    paths: list[str] = []

    java_pattern = re.compile(r"at ([\w$.]+)\.\w+\((\w+\.(?:java|kt)):\d+\)")
    for match in java_pattern.finditer(stack_trace):
        package = match.group(1)
        filename = match.group(2)
        package_path = package.split("$")[0].rsplit(".", 1)[0].replace(".", "/")
        paths.append(f"src/main/java/{package_path}/{filename}")
        paths.append(f"src/main/kotlin/{package_path}/{filename}")

    python_pattern = re.compile(r'File "([^"]+\.py)", line \d+')
    for match in python_pattern.finditer(stack_trace):
        raw = match.group(1)
        if "/app/" in raw:
            raw = raw.split("/app/", 1)[1]
        paths.append(raw)

    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique[:10]
