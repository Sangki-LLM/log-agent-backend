import os
import re
import subprocess
from pathlib import Path

from app.core.config import settings


def _repo_path(server_id: int) -> Path:
    return Path(settings.repos_path) / str(server_id)


def _auth_url(repo_url: str) -> str:
    """GitHub token을 URL에 삽입 (private repo 지원)."""
    if settings.github_token and "github.com" in repo_url:
        return repo_url.replace("https://", f"https://{settings.github_token}@")
    return repo_url


def clone(server_id: int, repo_url: str, branch: str) -> None:
    path = _repo_path(server_id)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", branch, "--depth", "1", _auth_url(repo_url), str(path)],
        check=True,
        capture_output=True,
    )


def pull(server_id: int) -> None:
    path = _repo_path(server_id)
    if not path.exists():
        raise FileNotFoundError(f"Repo not cloned for server {server_id}")
    subprocess.run(
        ["git", "-C", str(path), "pull", "--ff-only"],
        check=True,
        capture_output=True,
    )


def read_files_from_stacktrace(server_id: int, stack_trace: str) -> dict[str, str]:
    """stack trace에서 파일 경로를 추출하고 클론된 repo에서 파일 내용을 읽어 반환."""
    paths = _extract_source_paths(stack_trace)
    repo = _repo_path(server_id)
    result: dict[str, str] = {}

    for rel_path in paths:
        full_path = repo / rel_path
        if full_path.exists():
            try:
                result[rel_path] = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

    return result


def _extract_source_paths(stack_trace: str) -> list[str]:
    """
    Java/Kotlin stack trace에서 소스 파일 경로 추출.
    예: at com.puppynote.service.UserService.getUser(UserService.java:42)
        → src/main/java/com/puppynote/service/UserService.java
    Python의 경우:
    예: File "/app/services/user.py", line 42
        → services/user.py
    """
    paths: list[str] = []

    # Java / Kotlin
    java_pattern = re.compile(r'at ([\w$.]+)\.\w+\((\w+\.(?:java|kt)):\d+\)')
    for match in java_pattern.finditer(stack_trace):
        package = match.group(1)
        filename = match.group(2)
        # 내부 클래스($) 제거 후 패키지 경로로 변환
        package_path = package.split("$")[0].rsplit(".", 1)[0].replace(".", "/")
        paths.append(f"src/main/java/{package_path}/{filename}")
        paths.append(f"src/main/kotlin/{package_path}/{filename}")

    # Python
    python_pattern = re.compile(r'File "([^"]+\.py)", line \d+')
    for match in python_pattern.finditer(stack_trace):
        raw = match.group(1)
        # /app/ 이하 상대 경로로 정규화
        if "/app/" in raw:
            raw = raw.split("/app/", 1)[1]
        paths.append(raw)

    # 중복 제거, 순서 유지
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique[:10]  # 최대 10개 파일
