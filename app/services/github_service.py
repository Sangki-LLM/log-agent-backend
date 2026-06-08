import base64
import json
import logging
import re

import httpx

from app.models.server import AnalysisRecord, Server

logger = logging.getLogger(__name__)

_GH_API = "https://api.github.com"


def _parse_repo(git_repo_url: str) -> tuple[str, str]:
    match = re.search(r"github\.com[:/](.+?)/(.+?)(?:\.git)?$", git_repo_url)
    if not match:
        raise ValueError(f"Cannot parse GitHub URL: {git_repo_url}")
    return match.group(1), match.group(2)


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _parse_suggestion(record: AnalysisRecord) -> dict:
    try:
        return json.loads(record.llm_suggestion or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


async def _get_file(client: httpx.AsyncClient, owner: str, repo: str, path: str, branch: str, hdrs: dict) -> tuple[str, str]:
    """파일 내용(decoded)과 blob sha를 반환합니다."""
    resp = await client.get(
        f"{_GH_API}/repos/{owner}/{repo}/contents/{path}",
        headers=hdrs,
        params={"ref": branch},
    )
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


async def _commit_file(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    path: str,
    content: str,
    blob_sha: str,
    branch: str,
    message: str,
    hdrs: dict,
) -> None:
    encoded = base64.b64encode(content.encode("utf-8")).decode()
    resp = await client.put(
        f"{_GH_API}/repos/{owner}/{repo}/contents/{path}",
        headers=hdrs,
        json={"message": message, "content": encoded, "sha": blob_sha, "branch": branch},
    )
    resp.raise_for_status()


async def _create_branch(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    branch_name: str,
    base_branch: str,
    hdrs: dict,
) -> None:
    resp = await client.get(
        f"{_GH_API}/repos/{owner}/{repo}/git/ref/heads/{base_branch}",
        headers=hdrs,
    )
    resp.raise_for_status()
    base_sha = resp.json()["object"]["sha"]

    resp = await client.post(
        f"{_GH_API}/repos/{owner}/{repo}/git/refs",
        headers=hdrs,
        json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
    )
    if resp.status_code not in (201, 422):  # 422 = 이미 존재
        resp.raise_for_status()


async def create_fix_pr(server: Server, record: AnalysisRecord) -> str:
    """승인된 레코드를 기반으로 실제 코드를 수정하여 GitHub PR을 생성합니다."""
    if not server.github_token:
        logger.warning("[github] github_token not set for server %s", server.id)
        return ""

    try:
        owner, repo = _parse_repo(server.git_repo_url)
    except ValueError as e:
        logger.warning("[github] %s", e)
        return ""

    suggestion = _parse_suggestion(record)
    file_patch = suggestion.get("file_patch") or {}
    commit_message = suggestion.get("commit_message") or f"fix: log-agent suggestion for record #{record.id}"
    error_cause = suggestion.get("error_cause", "")
    suggested_fix = suggestion.get("suggested_fix", "")

    hdrs = _headers(server.github_token)
    branch_name = f"log-agent/fix-record-{record.id}"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            await _create_branch(client, owner, repo, branch_name, server.git_branch, hdrs)

            patched = False
            if file_patch.get("file_path") and file_patch.get("before") and file_patch.get("after"):
                file_path = file_patch["file_path"]
                before = file_patch["before"]
                after = file_patch["after"]

                try:
                    original, blob_sha = await _get_file(client, owner, repo, file_path, server.git_branch, hdrs)

                    if before in original:
                        modified = original.replace(before, after, 1)
                        await _commit_file(
                            client, owner, repo, file_path, modified, blob_sha,
                            branch_name, commit_message, hdrs,
                        )
                        patched = True
                        logger.info("[github] patched file: %s", file_path)
                    else:
                        logger.warning("[github] 'before' snippet not found in %s — falling back to description-only PR", file_path)
                except Exception as e:
                    logger.warning("[github] file patch failed: %s", e)

            if not patched:
                # 파일 패치 실패 시 분석 결과 파일만 커밋
                md_path = f".log-agent/suggestions/fix-record-{record.id}.md"
                md_content = (
                    f"# Log Agent Fix Suggestion — Record #{record.id}\n\n"
                    f"## 에러 원인\n{error_cause}\n\n"
                    f"## 수정 제안\n{suggested_fix}\n"
                )
                encoded = base64.b64encode(md_content.encode()).decode()
                resp = await client.put(
                    f"{_GH_API}/repos/{owner}/{repo}/contents/{md_path}",
                    headers=hdrs,
                    json={"message": commit_message, "content": encoded, "branch": branch_name},
                )
                resp.raise_for_status()
                logger.info("[github] committed suggestion markdown (no patch applied)")

            # PR 생성
            pr_body = (
                f"## 🤖 Log Agent 자동 수정\n\n"
                f"**에러 원인**: {error_cause}\n\n"
                f"**수정 내용**: {suggested_fix}\n\n"
            )
            if patched:
                pr_body += (
                    f"### 변경 파일: `{file_patch['file_path']}`\n\n"
                    "```diff\n"
                    f"- {file_patch['before'].strip()}\n"
                    f"+ {file_patch['after'].strip()}\n"
                    "```\n\n"
                )
            pr_body += f"---\n*Record ID: #{record.id} | 트리거: `{record.trigger_line[:200]}`*"

            resp = await client.post(
                f"{_GH_API}/repos/{owner}/{repo}/pulls",
                headers=hdrs,
                json={
                    "title": f"[Log Agent] {commit_message[:80]}",
                    "body": pr_body,
                    "head": branch_name,
                    "base": server.git_branch,
                },
            )
            resp.raise_for_status()
            pr_url = resp.json().get("html_url", "")
            logger.info("[github] PR created: %s (patched=%s)", pr_url, patched)
            return pr_url

    except Exception as e:
        logger.error("[github] PR creation failed: %s", e, exc_info=True)
        return ""
