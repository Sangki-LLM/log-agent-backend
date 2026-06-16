import hashlib
import json
import logging

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


async def _analysis_pipeline(server: Server, payload: ErrorEventPayload) -> None:
    import asyncio

    from app.core.database import AsyncSessionLocal
    from app.services import git_service

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

        except Exception as e:
            logger.warning("[pipeline] git/rag step failed: %s", e, exc_info=True)

        logger.info("[pipeline] calling ollama (agentic)")
        try:
            suggestion = await ollama_service.analyze_log(server.id, raw_log, payload.stack_trace)
            logger.info("[pipeline] ollama response length=%d", len(suggestion or ""))
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
