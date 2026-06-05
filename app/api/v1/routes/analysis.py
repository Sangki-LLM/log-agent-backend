from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.server import AnalysisRecord, Server
from app.schemas.analysis import AnalysisRecordResponse
from app.services import ollama_service, ssh_service

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/records", response_model=list[AnalysisRecordResponse])
async def list_records(
    server_id: int | None = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).limit(limit)
    if server_id:
        stmt = stmt.where(AnalysisRecord.server_id == server_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/trigger/{server_id}")
async def manual_trigger(
    server_id: int,
    log_path: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    server = await db.get(Server, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    path = log_path or server.log_path
    try:
        raw_log = ssh_service.fetch_context_logs(server, "manual trigger")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SSH error: {e}")

    async def _stream():
        async for token in ollama_service.stream_analysis(raw_log):
            yield token

    return StreamingResponse(_stream(), media_type="text/plain")
