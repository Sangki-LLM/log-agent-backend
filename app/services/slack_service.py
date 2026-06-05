import json

import httpx

from app.core.config import settings
from app.models.server import AnalysisRecord, Server


def _build_blocks(server: Server, record: AnalysisRecord) -> list[dict]:
    try:
        suggestion = json.loads(record.llm_suggestion or "{}")
    except (json.JSONDecodeError, TypeError):
        suggestion = {}

    error_cause = suggestion.get("error_cause", "분석 중...")
    bottleneck = suggestion.get("bottleneck", "")
    fix_code = suggestion.get("suggested_fix", record.llm_suggestion or "")[:1200]

    approve_url = f"{settings.public_url}/api/v1/analysis/approve/{record.id}"
    reject_url = f"{settings.public_url}/api/v1/analysis/reject/{record.id}"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🔴 [{server.name}] 에러 감지"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*원인*\n{error_cause}"},
                {"type": "mrkdwn", "text": f"*병목 지점*\n{bottleneck or 'N/A'}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*트리거 로그*\n```{record.trigger_line[:300]}```",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*추천 수정 코드*\n{fix_code}",
            },
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ 수락 & Push"},
                    "style": "primary",
                    "url": approve_url,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ 거절"},
                    "style": "danger",
                    "url": reject_url,
                },
            ],
        },
    ]
    return blocks


async def send_analysis(server: Server, record: AnalysisRecord) -> str:
    if not settings.slack_webhook_url:
        return ""

    payload = {
        "text": f"[{server.name}] 에러 감지 — LLM 분석 완료",
        "blocks": _build_blocks(server, record),
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(settings.slack_webhook_url, json=payload)
        resp.raise_for_status()

    return ""


async def send_result(text: str) -> None:
    if not settings.slack_webhook_url:
        return
    payload = {"text": text}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(settings.slack_webhook_url, json=payload)
        resp.raise_for_status()
