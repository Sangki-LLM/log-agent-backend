import asyncio
import hashlib

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.server import AnalysisRecord, Server
from app.schemas.server import ErrorEventPayload
from app.services import ollama_service, slack_service, ssh_service

router = APIRouter(prefix="/webhook", tags=["webhook"])

# 중복 트리거 방지: 같은 에러 60초 이내 재발 무시
_recent_errors: dict[str, float] = {}


def _dedup_key(server_id: int, trigger_line: str) -> str:
    return hashlib.md5(f"{server_id}:{trigger_line[:100]}".encode()).hexdigest()


@router.post("/error")
async def receive_error(
    payload: ErrorEventPayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    key = _dedup_key(payload.server_id, payload.trigger_line)
    now = asyncio.get_event_loop().time()

    if key in _recent_errors and now - _recent_errors[key] < 60:
        return {"status": "deduplicated"}

    _recent_errors[key] = now

    server = await db.get(Server, payload.server_id)
    if not server or not server.is_active:
        raise HTTPException(status_code=404, detail="Server not found or inactive")

    background_tasks.add_task(_analysis_pipeline, server, payload)
    return {"status": "received"}


async def _analysis_pipeline(server: Server, payload: ErrorEventPayload) -> None:
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            raw_log = ssh_service.fetch_context_logs(server, payload.trigger_line)
        except Exception as e:
            raw_log = payload.stack_trace or f"SSH fetch failed: {e}"

        suggestion = await ollama_service.analyze_log(raw_log)

        record = AnalysisRecord(
            server_id=server.id,
            trigger_line=payload.trigger_line[:500],
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
