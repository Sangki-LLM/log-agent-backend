import hashlib
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.server import AnalysisRecord, Server, ServerHost
from app.schemas.server import ErrorEventPayload
from app.services import judge_service, ollama_service, slack_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])

_recent_errors: dict[str, float] = {}


def _dedup_key(server_ip: str, stack_trace: str) -> str:
    return hashlib.md5(f"{server_ip}:{stack_trace[:100]}".encode()).hexdigest()


def _build_raw_log(payload: ErrorEventPayload) -> str:
    parts = [
        f"[서버] {payload.server_name} ({payload.server_ip})",
        f"[에러] {payload.error_type}: {payload.message}",
        f"[요청] {payload.request_method} {payload.request_url}",
    ]
    if payload.request_body:
        parts.append(f"[요청 바디]\n{payload.request_body}")
    if payload.response_status:
        parts.append(f"[응답 상태] {payload.response_status}")
    if payload.stack_trace:
        parts.append(f"[스택 트레이스]\n{payload.stack_trace}")
    return "\n".join(parts)


@router.post("/error")
async def receive_error(
    payload: ErrorEventPayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    logger.info(f"Webhook received: server_ip={payload.server_ip}, error_type={payload.error_type}")

    # 등록된 서버인지 IP로 확인 (ServerHost 테이블 경유)
    result = await db.execute(
        select(Server)
        .join(ServerHost, ServerHost.server_id == Server.id)
        .where(ServerHost.host == payload.server_ip, Server.is_active == True)
    )
    server = result.scalar_one_or_none()

    if not server:
        logger.warning(f"Webhook ignored: Server not found or inactive for IP {payload.server_ip}")
        return {"status": "ignored"}

    # 60초 내 동일 에러 중복 방지
    import asyncio
    key = _dedup_key(payload.server_ip, payload.stack_trace)
    now = asyncio.get_event_loop().time()
    if key in _recent_errors and now - _recent_errors[key] < 60:
        return {"status": "deduplicated"}
    _recent_errors[key] = now

    background_tasks.add_task(_analysis_pipeline, server, payload)
    return {"status": "received"}


def _fuzzy_replace(original: str, before: str, after: str) -> str | None:
    """라인 단위 fuzzy 매칭: 각 라인을 strip해서 순서대로 찾고, 원본 들여쓰기 유지하며 after로 교체."""
    import textwrap

    before_keys = [l.strip() for l in before.splitlines() if l.strip()]
    if not before_keys:
        return None

    orig_lines = original.splitlines()
    n, blen = len(orig_lines), len(before_keys)

    for start in range(n):
        j, end = 0, start
        while end < n and j < blen:
            stripped = orig_lines[end].strip()
            if stripped == before_keys[j]:
                j += 1
            elif stripped:  # 비어있지 않은 라인이 불일치 → 이 start 위치는 아님
                break
            end += 1

        if j == blen:
            # 매칭 성공: 첫 번째 매칭 라인의 들여쓰기 기준으로 after 재들여쓰기
            indent = len(orig_lines[start]) - len(orig_lines[start].lstrip())
            indent_str = " " * indent
            after_lines = [
                indent_str + l if l.strip() else l
                for l in textwrap.dedent(after).strip().splitlines()
            ]
            return "\n".join(orig_lines[:start] + after_lines + orig_lines[end:])

    return None


def _find_and_replace(original: str, before: str, after: str) -> str | None:
    """before를 original에서 찾아 after로 교체. 실패 시 None.
    1차: 정확한 매칭
    2차: CRLF → LF 정규화 후 매칭
    3차: 들여쓰기 보정 후 매칭
    4차: 라인 단위 fuzzy 매칭 (공백·빈 줄 무시)
    """
    import textwrap

    if before in original:
        return original.replace(before, after, 1)

    orig_lf = original.replace("\r\n", "\n")
    before_lf = before.replace("\r\n", "\n")
    after_lf = after.replace("\r\n", "\n")
    if before_lf in orig_lf:
        return orig_lf.replace(before_lf, after_lf, 1)

    before_dedented = textwrap.dedent(before_lf)
    after_dedented = textwrap.dedent(after_lf)
    for indent in ("    ", "        ", "  ", "\t", "            "):
        re_before = "\n".join(
            indent + line if line.strip() else line
            for line in before_dedented.splitlines()
        )
        if re_before in orig_lf:
            re_after = "\n".join(
                indent + line if line.strip() else line
                for line in after_dedented.splitlines()
            )
            return orig_lf.replace(re_before, re_after, 1)

    return _fuzzy_replace(orig_lf, before_lf, after_lf)


async def _enrich_with_patched_content(suggestion: str, server_id: int) -> str:
    """분석 시점에 원본 파일을 저장하고, LLM으로 완성된 수정 파일을 생성해 저장.
    승인 시에는 patched_content를 그대로 push — before/after 매칭 문제 없음.
    """
    try:
        data = json.loads(suggestion)
        fp = data.get("file_patch", {})
        file_path = fp.get("file_path", "")
        before = fp.get("before", "")
        after = fp.get("after", "")
        if not file_path:
            return suggestion

        from app.services.git_service import _repo_path
        full_path = _repo_path(server_id) / file_path
        try:
            original = full_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            logger.warning("[pipeline] file not found — %s", full_path)
            return suggestion

        data["file_patch"]["original_content"] = original

        if before and after:
            # 1차: 빠른 string 매칭
            patched = _find_and_replace(original, before, after)
            if patched is not None:
                logger.info("[pipeline] patched via string match: %s", file_path)
            else:
                # 2차: LLM이 원본 파일 전체를 보고 직접 수정 — 분석 시점에 미리 완료
                logger.info("[pipeline] string match failed, calling LLM to patch: %s", file_path)
                from app.services.ollama_service import apply_patch_with_llm
                patched = await apply_patch_with_llm(original, before, after)
                if patched:
                    logger.info("[pipeline] patched via LLM: %s", file_path)
                else:
                    logger.warning("[pipeline] LLM patch also failed: %s", file_path)

            if patched:
                data["file_patch"]["patched_content"] = patched

        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.warning("[pipeline] enrich failed: %s", e)
        return suggestion


async def _analysis_pipeline(server: Server, payload: ErrorEventPayload) -> None:
    import asyncio

    from app.core.database import AsyncSessionLocal
    from app.services import git_service, rag_service

    print(f"[pipeline] start — server={server.name} error={payload.error_type}", flush=True)
    logger.info("[pipeline] start — server=%s error=%s", server.name, payload.error_type)

    async with AsyncSessionLocal() as db:
        raw_log = _build_raw_log(payload)
        trigger_line = f"{payload.error_type}: {payload.message}"[:500]

        try:
            logger.info("[pipeline] git fetch — server_id=%s repo=%s", server.id, server.git_repo_url)
            await asyncio.to_thread(
                git_service.fetch, server.id, server.git_repo_url, server.git_branch, server.github_token or ""
            )
            commit = await asyncio.to_thread(git_service.get_remote_head, server.id, server.git_branch)
            logger.info("[pipeline] remote HEAD=%s", commit)

            # RAG: commit이 바뀐 경우 전체 소스 재인덱싱
            if not rag_service.is_indexed(server.id, commit):
                logger.info("[pipeline] RAG indexing start")
                chunks = await asyncio.to_thread(git_service.list_all_files_at_commit, server.id, commit)
                await rag_service.index_repo(server.id, commit, chunks)
        except Exception as e:
            logger.warning("[pipeline] git/rag step failed: %s", e, exc_info=True)

        logger.info("[pipeline] calling ollama (agentic)")
        try:
            suggestion = await ollama_service.analyze_log(server.id, raw_log, payload.stack_trace)
            logger.info("[pipeline] ollama response length=%d", len(suggestion or ""))
            suggestion = await _enrich_with_patched_content(suggestion, server.id)
        except Exception as e:
            logger.error("[pipeline] ollama failed: %s", e, exc_info=True)
            suggestion = ""

        # LLM Judge: Gemini로 분석 결과 품질 평가
        judge_result = None
        if suggestion:
            try:
                judge_result = await judge_service.judge_fix(raw_log, suggestion)
                if judge_result:
                    logger.info("[pipeline] judge score=%d confidence=%s",
                                judge_result["score"], judge_result["confidence"])
            except Exception as e:
                logger.warning("[pipeline] judge failed: %s", e)

        record = AnalysisRecord(
            server_id=server.id,
            trigger_line=trigger_line,
            raw_log=raw_log[:10000],
            llm_suggestion=suggestion,
            status="pending",
            judge_score=judge_result["score"] if judge_result else None,
            judge_confidence=judge_result["confidence"] if judge_result else None,
            judge_reason=judge_result["reason"] if judge_result else None,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        logger.info("[pipeline] record saved id=%s", record.id)

        try:
            await slack_service.send_analysis(server, record)
            logger.info("[pipeline] slack sent")
        except Exception as e:
            logger.error("[pipeline] slack failed: %s", e, exc_info=True)
