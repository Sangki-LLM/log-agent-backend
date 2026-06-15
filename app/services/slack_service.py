import json
import logging

import httpx

from app.core.config import settings
from app.models.server import AnalysisRecord, Server

logger = logging.getLogger(__name__)


def _webhook_url(server: Server) -> str:
    return server.slack_webhook_url or settings.slack_webhook_url


def _build_judge_block(record: AnalysisRecord) -> dict:
    score = record.judge_score or 0
    filled = "⭐" * score + "☆" * (5 - score)
    confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(record.judge_confidence or "", "⚪")
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*🤖 Gemini Judge*\n"
                f"{filled} ({score}/5)  {confidence_emoji} {record.judge_confidence}\n"
                f"_{record.judge_reason}_"
            ),
        },
    }


def _build_blocks(server: Server, record: AnalysisRecord) -> list[dict]:
    try:
        suggestion = json.loads(record.llm_suggestion or "{}")
    except (json.JSONDecodeError, TypeError):
        suggestion = {}

    error_cause = suggestion.get("error_cause", "분석 중...")
    bottleneck = suggestion.get("bottleneck", "")
    fix_explanation = suggestion.get("suggested_fix", "")
    file_patch = suggestion.get("file_patch", {})
    patch_path = file_patch.get("file_path", "")
    patch_before = file_patch.get("before", "")
    patch_after = file_patch.get("after", "")
    if patch_path and patch_after:
        fix_code = (
            f"{fix_explanation}\n\n"
            f"`{patch_path}`\n"
            f"*Before*\n```\n{patch_before.strip()}\n```\n"
            f"*After*\n```\n{patch_after.strip()}\n```"
        )[:2800]
    else:
        fix_code = (fix_explanation or record.llm_suggestion or "")[:1200]

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
        *( [_build_judge_block(record)] if record.judge_score else [] ),
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ 수락 & Push"},
                    "style": "primary",
                    "action_id": "approve_fix",
                    "value": str(record.id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ 거절"},
                    "style": "danger",
                    "action_id": "reject_fix",
                    "value": str(record.id),
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
