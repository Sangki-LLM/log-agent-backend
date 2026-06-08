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


def _build_file_content(record: AnalysisRecord) -> str:
    try:
        s = json.loads(record.llm_suggestion or "{}")
    except (json.JSONDecodeError, TypeError):
        s = {}

    error_cause = s.get("error_cause", "")
    bottleneck = s.get("bottleneck", "")
    fix_code = s.get("suggested_fix", record.llm_suggestion or "")

    return f"""# Log Agent Fix Suggestion — Record #{record.id}

## 에러 원인
{error_cause}

## 병목 지점
{bottleneck or "N/A"}

## 수정 제안
```
{fix_code}
```

---
*이 파일은 Log Agent에 의해 자동 생성되었습니다.*
*트리거: `{record.trigger_line[:300]}`*
"""


async def create_fix_pr(server: Server, record: AnalysisRecord) -> str:
    """승인된 레코드를 기반으로 GitHub PR을 생성하고 PR URL을 반환합니다."""
    if not server.github_token:
        logger.warning("[github] github_token not set for server %s", server.id)
        return ""

    try:
        owner, repo = _parse_repo(server.git_repo_url)
    except ValueError as e:
        logger.warning("[github] %s", e)
        return ""

    hdrs = _headers(server.github_token)
    branch_name = f"log-agent/fix-record-{record.id}"

    try:
        suggestion = json.loads(record.llm_suggestion or "{}")
    except (json.JSONDecodeError, TypeError):
        suggestion = {}

    error_cause = suggestion.get("error_cause", "")
    fix_code = suggestion.get("suggested_fix", record.llm_suggestion or "")[:3000]

    async with httpx.AsyncClient(timeout=20) as client:
        # 1. 베이스 브랜치 최신 SHA 조회
        resp = await client.get(
            f"{_GH_API}/repos/{owner}/{repo}/git/ref/heads/{server.git_branch}",
            headers=hdrs,
        )
        resp.raise_for_status()
        base_sha = resp.json()["object"]["sha"]

        # 2. 새 브랜치 생성
        resp = await client.post(
            f"{_GH_API}/repos/{owner}/{repo}/git/refs",
            headers=hdrs,
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
        )
        if resp.status_code not in (201, 422):  # 422 = 이미 존재
            resp.raise_for_status()

        # 3. 수정 제안 파일 커밋
        file_path = f".log-agent/suggestions/fix-record-{record.id}.md"
        encoded = base64.b64encode(_build_file_content(record).encode()).decode()
        resp = await client.put(
            f"{_GH_API}/repos/{owner}/{repo}/contents/{file_path}",
            headers=hdrs,
            json={
                "message": f"fix: log-agent suggestion for record #{record.id}\n\n{record.trigger_line[:200]}",
                "content": encoded,
                "branch": branch_name,
            },
        )
        resp.raise_for_status()

        # 4. PR 생성
        pr_body = f"""## 🤖 Log Agent 자동 분석 결과

**에러 원인**: {error_cause}

**수정 제안**:
```
{fix_code}
```

---
*Record ID: #{record.id}*
*트리거: `{record.trigger_line[:300]}`*
"""
        resp = await client.post(
            f"{_GH_API}/repos/{owner}/{repo}/pulls",
            headers=hdrs,
            json={
                "title": f"[Log Agent] Fix: {record.trigger_line[:80]}",
                "body": pr_body,
                "head": branch_name,
                "base": server.git_branch,
            },
        )
        resp.raise_for_status()
        pr_url = resp.json().get("html_url", "")
        logger.info("[github] PR created: %s", pr_url)
        return pr_url

    except Exception as e:
        logger.error("[github] PR creation failed: %s", e, exc_info=True)
        return ""
