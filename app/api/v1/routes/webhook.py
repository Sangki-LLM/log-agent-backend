import hashlib

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.server import AnalysisRecord, Server, ServerHost
from app.schemas.server import ErrorEventPayload
from app.services import ollama_service, slack_service

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
    # 등록된 서버인지 IP로 확인 (ServerHost 테이블 경유)
    result = await db.execute(
        select(Server)
        .join(ServerHost, ServerHost.server_id == Server.id)
        .where(ServerHost.host == payload.server_ip, Server.is_active == True)
    )
    server = result.scalar_one_or_none()

    if not server:
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

    async with AsyncSessionLocal() as db:
        raw_log = _build_raw_log(payload)
        trigger_line = f"{payload.error_type}: {payload.message}"[:500]

        # 최신 코드 pull 후 stack trace에서 관련 소스 파일 읽기
        source_files: dict[str, str] = {}
        try:
            await asyncio.to_thread(git_service.pull, server.id)
            if payload.stack_trace:
                source_files = await asyncio.to_thread(
                    git_service.read_files_from_stacktrace, server.id, payload.stack_trace
                )
        except Exception:
            pass

        suggestion = await ollama_service.analyze_log(raw_log, source_files)

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

        slack_ts = await slack_service.send_analysis(server, record)
        if slack_ts:
            record.slack_ts = slack_ts
            await db.commit()
