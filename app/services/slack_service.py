import json
import logging

import httpx

from app.core.config import settings
from app.models.server import AnalysisRecord, Server

logger = logging.getLogger(__name__)


def _webhook_url(server: Server) -> str:
    return server.slack_webhook_url or settings.slack_webhook_url


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

    return [
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


async def send_analysis(server: Server, record: AnalysisRecord) -> None:
    url = _webhook_url(server)
    if not url:
        return
    payload = {
        "text": f"[{server.name}] 에러 감지 — LLM 분석 완료",
        "blocks": _build_blocks(server, record),
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()


async def send_result(text: str, server: Server | None = None) -> None:
    url = _webhook_url(server) if server else settings.slack_webhook_url
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json={"text": text})
            resp.raise_for_status()
    except Exception as e:
        logger.error("[slack] send_result failed: %s", e)
