import io
import re
import shutil
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
        depth=1,
        errstream=io.BytesIO(),
    )


def fetch(server_id: int, repo_url: str = "", branch: str = "main", token: str = "") -> None:
    if not repo_url:
        raise FileNotFoundError(f"No repo URL for server {server_id}")
    path = _repo_path(server_id)
    # porcelain.fetch() updates refs but fails to register pack objects in the
    # next Repo context — always re-clone (depth=1 keeps it fast)
    if path.exists():
        shutil.rmtree(str(path))
    clone(server_id, repo_url, branch, token)


def get_remote_head(server_id: int, branch: str) -> str:
    path = _repo_path(server_id)
    with Repo(str(path)) as repo:
        ref_key = f"refs/remotes/origin/{branch}".encode()
        sha = repo.refs[ref_key]
        # dulwich refs return hex SHA as ascii bytes, not binary
        return sha.decode("ascii") if isinstance(sha, bytes) else sha


def list_all_files_at_commit(server_id: int, commit_hash: str, chunk_size: int = 1500) -> list[tuple[str, str]]:
    """커밋 시점 전체 소스 파일을 청크 리스트로 반환 (경로, 내용)."""
    SOURCE_EXTENSIONS = {".java", ".kt", ".py", ".ts", ".tsx", ".js", ".go"}
    path = _repo_path(server_id)
    chunks: list[tuple[str, str]] = []

    with Repo(str(path)) as repo:
        commit = repo[bytes.fromhex(commit_hash)]
        _walk_tree(repo, commit.tree, "", chunks, SOURCE_EXTENSIONS, chunk_size)

    return chunks


def _walk_tree(repo, tree_sha: bytes, prefix: str, chunks: list, extensions: set, chunk_size: int) -> None:
    from dulwich.objects import Blob, Tree

    tree = repo[tree_sha]
    for item in tree.items():
        name = item.path.decode("utf-8", errors="replace")
        full_path = f"{prefix}/{name}" if prefix else name
        obj = repo[item.sha]

        if isinstance(obj, Tree):
            _walk_tree(repo, item.sha, full_path, chunks, extensions, chunk_size)
        elif isinstance(obj, Blob):
            if Path(full_path).suffix.lower() in extensions:
                try:
                    content = obj.data.decode("utf-8", errors="replace")
                    for i in range(0, len(content), chunk_size):
                        chunks.append((full_path, content[i:i + chunk_size]))
                except Exception:
                    pass


def read_files_by_paths(server_id: int, commit_hash: str, paths: list[str]) -> dict[str, str]:
    """특정 경로 목록의 파일을 커밋 시점 기준으로 읽어 반환."""
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
                result[rel_path] = content[:3000]
            except Exception:
                pass

    return result


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
