import hashlib
import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.server import AnalysisRecord, Server, ServerHost
from app.schemas.server import ErrorEventPayload
from app.services import ollama_service, slack_service

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
            suggestion = await ollama_service.analyze_log(server.id, raw_log)
            logger.info("[pipeline] ollama response length=%d", len(suggestion or ""))
        except Exception as e:
            logger.error("[pipeline] ollama failed: %s", e, exc_info=True)
            suggestion = ""

        record = AnalysisRecord(
            server_id=server.id,
            trigger_line=trigger_line,
            raw_log=raw_log[:10000],
            llm_suggestion=suggestion,
            status="pending",
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        logger.info("[pipeline] record saved id=%s", record.id)

        try:
            slack_ts = await slack_service.send_analysis(server, record)
            if slack_ts:
                record.slack_ts = slack_ts
                await db.commit()
            logger.info("[pipeline] slack sent ts=%s", slack_ts)
        except Exception as e:
            logger.error("[pipeline] slack failed: %s", e, exc_info=True)
