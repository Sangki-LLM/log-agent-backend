import json

from app.models.server import AnalysisRecord, Server
from app.services import ssh_service


def apply_and_push(server: Server, record: AnalysisRecord, file_path: str, new_content: str) -> str:
    """원격 서버에 수정된 파일을 쓰고 git commit & push 후 커밋 해시를 반환."""
    ssh_service.write_source_file(server, file_path, new_content)

    commit_message = _extract_commit_message(record.llm_suggestion)
    commands = [
        f"cd {server.project_path}",
        "git add -A",
        f'git commit -m "{commit_message}"',
        f"git push origin {server.git_branch}",
    ]
    stdout, stderr = ssh_service.run_command(server, " && ".join(commands))

    if stderr and "error" in stderr.lower():
        raise RuntimeError(f"git push failed: {stderr}")

    return stdout.strip()


def get_diff(server: Server) -> str:
    stdout, _ = ssh_service.run_command(server, f"cd {server.project_path} && git diff HEAD")
    return stdout


def _extract_commit_message(llm_suggestion: str) -> str:
    try:
        data = json.loads(llm_suggestion)
        return data.get("commit_message", "fix: AI-suggested code improvement")
    except (json.JSONDecodeError, TypeError):
        return "fix: AI-suggested code improvement"
