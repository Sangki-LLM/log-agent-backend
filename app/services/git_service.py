import re
import subprocess
from pathlib import Path

from app.core.config import settings


def _repo_path(server_id: int) -> Path:
    return Path(settings.repos_path) / str(server_id)


def _auth_url(repo_url: str, token: str) -> str:
    """서버별 GitHub token을 URL에 삽입."""
    if token and "github.com" in repo_url:
        return repo_url.replace("https://", f"https://{token}@")
    return repo_url


def clone(server_id: int, repo_url: str, branch: str, token: str = "") -> None:
    path = _repo_path(server_id)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", branch, _auth_url(repo_url, token), str(path)],
        check=True,
        capture_output=True,
    )


def fetch(server_id: int, repo_url: str = "", branch: str = "main", token: str = "") -> None:
    """원격 변경사항을 가져오되 working tree는 건드리지 않음. 미클론 시 자동 clone."""
    path = _repo_path(server_id)
    if not path.exists():
        if not repo_url:
            raise FileNotFoundError(f"Repo not cloned for server {server_id} and no repo_url provided")
        clone(server_id, repo_url, branch, token)
        return
    remote = _auth_url(repo_url, token) if repo_url and token else "--all"
    subprocess.run(
        ["git", "-C", str(path), "fetch", remote, "--quiet"],
        check=True,
        capture_output=True,
    )


def get_remote_head(server_id: int, branch: str) -> str:
    """fetch 후 원격 브랜치의 최신 커밋 해시를 반환."""
    path = _repo_path(server_id)
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", f"origin/{branch}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def read_files_at_commit(server_id: int, commit_hash: str, stack_trace: str) -> dict[str, str]:
    """특정 커밋 시점의 소스 파일을 git show로 읽어 반환."""
    paths = _extract_source_paths(stack_trace)
    path = _repo_path(server_id)
    result: dict[str, str] = {}

    for rel_path in paths:
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), "show", f"{commit_hash}:{rel_path}"],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                result[rel_path] = proc.stdout[:2000]
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
