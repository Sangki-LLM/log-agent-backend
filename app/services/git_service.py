import io
import re
import shutil
from pathlib import Path

from dulwich import porcelain
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
    if path.exists():
        shutil.rmtree(str(path))
    clone(server_id, repo_url, branch, token)


def get_remote_head(server_id: int, branch: str) -> str:
    path = _repo_path(server_id)
    with Repo(str(path)) as repo:
        # try remote tracking ref first, fall back to local HEAD
        ref_key = f"refs/remotes/origin/{branch}".encode()
        try:
            sha = repo.refs[ref_key]
            return sha.decode("ascii") if isinstance(sha, bytes) else sha
        except KeyError:
            pass
        sha = repo.head()
        return sha.decode("ascii") if isinstance(sha, bytes) else sha


def list_all_files_at_commit(server_id: int, commit_hash: str, chunk_size: int = 1500) -> list[tuple[str, str]]:
    """Working tree를 직접 읽어 소스 파일 청크 목록 반환."""
    SOURCE_EXTENSIONS = {".java", ".kt", ".py", ".ts", ".tsx", ".js", ".go"}
    base = _repo_path(server_id)
    chunks: list[tuple[str, str]] = []

    for file_path in sorted(base.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        # .git 디렉토리 제외
        if ".git" in file_path.parts:
            continue
        rel = str(file_path.relative_to(base)).replace("\\", "/")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            for i in range(0, len(content), chunk_size):
                chunks.append((rel, content[i:i + chunk_size]))
        except Exception:
            pass

    return chunks


def read_files_by_paths(server_id: int, commit_hash: str, paths: list[str]) -> dict[str, str]:
    """Working tree에서 경로 목록의 파일 내용 반환."""
    base = _repo_path(server_id)
    result: dict[str, str] = {}

    for rel_path in paths:
        try:
            content = (base / rel_path).read_text(encoding="utf-8", errors="replace")
            result[rel_path] = content[:3000]
        except Exception:
            pass

    return result


def read_files_at_commit(server_id: int, commit_hash: str, stack_trace: str) -> dict[str, str]:
    paths = _extract_source_paths(stack_trace)
    base = _repo_path(server_id)
    result: dict[str, str] = {}

    for rel_path in paths:
        try:
            content = (base / rel_path).read_text(encoding="utf-8", errors="replace")
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
