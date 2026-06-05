from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.server import AnalysisRecord, Server
from app.schemas.analysis import AnalysisRecordResponse
from app.services import git_service, ollama_service, slack_service, ssh_service

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
    if not server:
        return _html_page("❌ 오류", "서버 정보를 찾을 수 없습니다.", error=True)

    record.status = "approved"
    await db.commit()

    try:
        import json
        suggestion = json.loads(record.llm_suggestion or "{}")
        # git push는 수정 파일 경로가 명확해야 실행 — 여기선 상태만 approved로 변경
        # 실제 파일 경로는 프론트엔드 또는 추가 API에서 지정
        result_msg = git_service.apply_and_push(server, record, record_id)
        record.status = "applied"
        await db.commit()
        await slack_service.send_result(f"✅ [{server.name}] 코드 수정 완료\n```{result_msg[:300]}```")
        return _html_page("✅ 완료", f"코드 수정 및 git push 완료.<br><pre>{result_msg[:500]}</pre>")
    except Exception as e:
        await slack_service.send_result(f"⚠️ [{server.name}] Push 실패: {e}")
        return _html_page("⚠️ Push 실패", str(e), error=True)


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


@router.post("/trigger/{server_id}")
async def manual_trigger(
    server_id: int,
    db: AsyncSession = Depends(get_db),
):
    server = await db.get(Server, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    try:
        raw_log = ssh_service.fetch_context_logs(server, "manual trigger")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SSH error: {e}")

    async def _stream():
        async for token in ollama_service.stream_analysis(raw_log):
            yield token

    return StreamingResponse(_stream(), media_type="text/plain")


def _html_page(title: str, body: str, error: bool = False) -> str:
    color = "#ef4444" if error else "#22c55e"
    frontend = settings.frontend_url
    return f"""<!DOCTYPE html>
<html lang="ko">
<head><meta charset="UTF-8"><title>{title}</title>
<style>body{{font-family:sans-serif;display:flex;flex-direction:column;align-items:center;
justify-content:center;min-height:100vh;margin:0;background:#f8fafc}}
.card{{background:white;border-radius:12px;padding:40px;box-shadow:0 4px 20px rgba(0,0,0,.1);
max-width:600px;text-align:center}}h1{{color:{color}}}pre{{text-align:left;background:#f1f5f9;
padding:12px;border-radius:6px;font-size:12px;overflow:auto}}
a{{color:#3b82f6;text-decoration:none}}</style></head>
<body><div class="card">
<h1>{title}</h1><p>{body}</p>
<a href="{frontend}/history">← 분석 이력으로 돌아가기</a>
</div></body></html>"""
