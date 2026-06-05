from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.server import AnalysisRecord, Server
from app.schemas.analysis import AnalysisRecordResponse
from app.services import ollama_service, slack_service

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


@router.get("/approve/{record_id}", response_class=HTMLResponse)
async def approve_fix(record_id: int, db: AsyncSession = Depends(get_db)):
    record = await db.get(AnalysisRecord, record_id)
    if not record:
        return _html_page("❌ 오류", "해당 분석 레코드를 찾을 수 없습니다.", error=True)
    if record.status != "pending":
        return _html_page("⚠️ 이미 처리됨", f"현재 상태: <b>{record.status}</b>")

    server = await db.get(Server, record.server_id)
    record.status = "approved"
    await db.commit()

    name = server.name if server else f"#{record.server_id}"
    await slack_service.send_result(f"✅ [{name}] 분석 결과 수락됨 (record #{record_id})")
    return _html_page("✅ 수락됨", "분석 결과가 수락되었습니다.")


@router.get("/reject/{record_id}", response_class=HTMLResponse)
async def reject_fix(record_id: int, db: AsyncSession = Depends(get_db)):
    record = await db.get(AnalysisRecord, record_id)
    if not record:
        return _html_page("❌ 오류", "해당 분석 레코드를 찾을 수 없습니다.", error=True)
    if record.status != "pending":
        return _html_page("⚠️ 이미 처리됨", f"현재 상태: <b>{record.status}</b>")

    server = await db.get(Server, record.server_id)
    record.status = "rejected"
    await db.commit()

    name = server.name if server else f"#{record.server_id}"
    await slack_service.send_result(f"❌ [{name}] 수정 제안 거절됨 (record #{record_id})")
    return _html_page("❌ 거절됨", "수정 제안이 거절되었습니다.")


def _html_page(title: str, body: str, error: bool = False) -> str:
    color = "#ef4444" if error else "#22c55e"
    frontend = settings.frontend_url
    return f"""<!DOCTYPE html>
<html lang="ko">
<head><meta charset="UTF-8"><title>{title}</title>
<style>body{{font-family:sans-serif;display:flex;flex-direction:column;align-items:center;
justify-content:center;min-height:100vh;margin:0;background:#f8fafc}}
.card{{background:white;border-radius:12px;padding:40px;box-shadow:0 4px 20px rgba(0,0,0,.1);
max-width:600px;text-align:center}}h1{{color:{color}}}
a{{color:#3b82f6;text-decoration:none}}</style></head>
<body><div class="card">
<h1>{title}</h1><p>{body}</p>
<a href="{frontend}/history">← 분석 이력으로 돌아가기</a>
</div></body></html>"""
