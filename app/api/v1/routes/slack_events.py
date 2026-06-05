import json
import urllib.parse

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.server import AnalysisRecord
from app.services import git_service, slack_service

router = APIRouter(prefix="/slack", tags=["slack"])


@router.post("/actions")
async def handle_slack_action(request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.body()
    decoded = urllib.parse.unquote_plus(body.decode())
    if decoded.startswith("payload="):
        decoded = decoded[len("payload="):]

    payload = json.loads(decoded)
    actions = payload.get("actions", [])
    if not actions:
        return {"ok": True}

    action = actions[0]
    action_id = action.get("action_id")
    record_id = int(action.get("value", 0))

    record = await db.get(AnalysisRecord, record_id)
    if not record:
        return {"ok": True}

    server = await db.get(type(record.server), record.server_id)

    if action_id == "approve_fix":
        record.status = "approved"
        await db.commit()

        try:
            stdout = git_service.apply_and_push(server, record, record_id)
            record.status = "applied"
            await db.commit()
            await slack_service.update_message(record.slack_ts, f"✅ 코드 수정 완료\n```{stdout[:500]}```")
        except Exception as e:
            await slack_service.update_message(record.slack_ts, f"⚠️ Push 실패: {e}")

    elif action_id == "reject_fix":
        record.status = "rejected"
        await db.commit()
        await slack_service.update_message(record.slack_ts, "❌ 수정 제안 거절됨")

    return {"ok": True}
