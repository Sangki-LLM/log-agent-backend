import json
import logging
from urllib.parse import parse_qs

from fastapi import APIRouter, Request, Response

from app.core.database import AsyncSessionLocal
from app.models.server import AnalysisRecord, Server
from app.services import github_service, slack_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])


@router.post("/actions")
async def handle_slack_action(request: Request):
    body = await request.body()
    try:
        parsed = parse_qs(body.decode())
        payload_str = parsed.get("payload", [""])[0]
        data = json.loads(payload_str)
    except Exception:
        return Response(status_code=200)

    action_list = data.get("actions", [])
    if not action_list:
        return Response(status_code=200)

    action = action_list[0]
    action_id = action.get("action_id", "")
    record_id_str = action.get("value", "")

    if not record_id_str or action_id not in ("approve_fix", "reject_fix"):
        return Response(status_code=200)

    try:
        record_id = int(record_id_str)
    except ValueError:
        return Response(status_code=200)

    async with AsyncSessionLocal() as db:
        record = await db.get(AnalysisRecord, record_id)
        if not record or record.status != "pending":
            return Response(status_code=200)

        server = await db.get(Server, record.server_id)
        name = server.name if server else f"#{record.server_id}"

        if action_id == "approve_fix":
            record.status = "approved"
            await db.commit()

            pr_url = ""
            if server:
                pr_url = await github_service.create_fix_pr(server, record)
                if pr_url:
                    record.github_pr_url = pr_url
                    await db.commit()

            result_text = f"✅ [{name}] 분석 결과 수락됨 (record #{record_id})"
            if pr_url:
                result_text += f" — {pr_url}"
            await slack_service.send_result(result_text, server)
            logger.info("[slack] approved record=%s pr=%s", record_id, pr_url or "none")
        elif action_id == "reject_fix":
            record.status = "rejected"
            await db.commit()
            await slack_service.send_result(f"❌ [{name}] 수정 제안 거절됨 (record #{record_id})", server)
            logger.info("[slack] rejected record=%s", record_id)

    return Response(status_code=200)
