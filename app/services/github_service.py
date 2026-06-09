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



async def _get_file(
    client: httpx.AsyncClient, owner: str, repo: str, path: str, branch: str, hdrs: dict
) -> tuple[str, str]:
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
    if resp.status_code not in (201, 422):
        resp.raise_for_status()


async def _create_empty_commit(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    branch_name: str,
    message: str,
    hdrs: dict,
) -> None:
    """파일 변경 없이 빈 커밋 생성 — PR 생성을 위한 최소 커밋."""
    resp = await client.get(
        f"{_GH_API}/repos/{owner}/{repo}/git/ref/heads/{branch_name}",
        headers=hdrs,
    )
    resp.raise_for_status()
    parent_sha = resp.json()["object"]["sha"]

    resp = await client.get(
        f"{_GH_API}/repos/{owner}/{repo}/git/commits/{parent_sha}",
        headers=hdrs,
    )
    resp.raise_for_status()
    tree_sha = resp.json()["tree"]["sha"]

    resp = await client.post(
        f"{_GH_API}/repos/{owner}/{repo}/git/commits",
        headers=hdrs,
        json={"message": message, "tree": tree_sha, "parents": [parent_sha]},
    )
    resp.raise_for_status()
    new_sha = resp.json()["sha"]

    resp = await client.patch(
        f"{_GH_API}/repos/{owner}/{repo}/git/refs/heads/{branch_name}",
        headers=hdrs,
        json={"sha": new_sha},
    )
    resp.raise_for_status()


async def create_fix_pr(server: Server, record: AnalysisRecord) -> str:
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
            if file_patch.get("file_path"):
                file_path = file_patch["file_path"]
                patched_content = file_patch.get("patched_content", "")

                try:
                    _, blob_sha = await _get_file(client, owner, repo, file_path, server.git_branch, hdrs)

                    # patched_content가 없으면 original_content 기반으로 재시도
                    if not patched_content:
                        original_content = file_patch.get("original_content", "")
                        before = file_patch.get("before", "")
                        after = file_patch.get("after", "")
                        if original_content and before and after:
                            # 1차: fuzzy 매칭
                            from app.api.v1.routes.webhook import _find_and_replace
                            result = _find_and_replace(original_content, before, after)
                            if result:
                                patched_content = result
                                logger.info("[github] patched via fuzzy match at approval time: %s", file_path)
                            else:
                                # 2차: LLM에게 패치 적용 위임
                                from app.services.ollama_service import apply_patch_with_llm
                                result = await apply_patch_with_llm(original_content, before, after)
                                if result:
                                    patched_content = result
                                    logger.info("[github] patched via LLM at approval time: %s", file_path)

                    if patched_content:
                        await _commit_file(
                            client, owner, repo, file_path, patched_content, blob_sha,
                            branch_name, commit_message, hdrs,
                        )
                        patched = True
                        logger.info("[github] patched file: %s", file_path)
                    else:
                        logger.warning("[github] patch failed for %s — description-only PR", file_path)
                except Exception as e:
                    logger.warning("[github] file patch failed: %s", e)

            if not patched:
                await _create_empty_commit(client, owner, repo, branch_name, commit_message, hdrs)
                logger.info("[github] empty commit created for description-only PR")

            # PR 본문
            pr_body = (
                f"## 🤖 Log Agent 자동 분석\n\n"
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
            else:
                pr_body += (
                    f"### ⚠️ 자동 패치 실패 — 수동 적용 필요\n\n"
                    f"**파일**: `{file_patch.get('file_path', 'N/A')}`\n\n"
                    "**Before**\n"
                    f"```\n{file_patch.get('before', '').strip()}\n```\n\n"
                    "**After**\n"
                    f"```\n{file_patch.get('after', '').strip()}\n```\n\n"
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
