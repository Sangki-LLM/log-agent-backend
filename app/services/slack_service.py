import json
import logging

import httpx

from app.core.config import settings
from app.models.server import AnalysisRecord, Server

logger = logging.getLogger(__name__)

_SLACK_API = "https://slack.com/api"


def _bot_headers() -> dict:
    return {"Authorization": f"Bearer {settings.slack_bot_token}"}


def _content_blocks(server: Server, record: AnalysisRecord) -> list[dict]:
    """공통 본문 블록 (버튼 제외)."""
    try:
        suggestion = json.loads(record.llm_suggestion or "{}")
    except (json.JSONDecodeError, TypeError):
        suggestion = {}

    error_cause = suggestion.get("error_cause", "분석 중...")
    bottleneck = suggestion.get("bottleneck", "")
    fix_code = suggestion.get("suggested_fix", record.llm_suggestion or "")[:1200]

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
    ]


def _action_buttons(record: AnalysisRecord) -> dict:
    approve_url = f"{settings.public_url}/api/v1/analysis/approve/{record.id}"
    reject_url = f"{settings.public_url}/api/v1/analysis/reject/{record.id}"
    return {
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
    }


def _build_blocks(server: Server, record: AnalysisRecord) -> list[dict]:
    return _content_blocks(server, record) + [{"type": "divider"}, _action_buttons(record)]


def _status_block(approved: bool, pr_url: str) -> dict:
    if approved:
        text = "✅ *승인됨*"
        if pr_url:
            text += f" — <{pr_url}|GitHub PR 보기>"
    else:
        text = "❌ *거절됨*"
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


async def send_analysis(server: Server, record: AnalysisRecord) -> tuple[str, str]:
    """분석 결과를 Slack에 전송합니다. (ts, channel) 반환."""
    blocks = _build_blocks(server, record)
    text = f"[{server.name}] 에러 감지 — LLM 분석 완료"

    if settings.slack_bot_token and settings.slack_channel_id:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{_SLACK_API}/chat.postMessage",
                headers=_bot_headers(),
                json={"channel": settings.slack_channel_id, "text": text, "blocks": blocks},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                return data.get("ts", ""), data.get("channel", "")
            logger.warning("[slack] chat.postMessage error: %s", data.get("error"))
        return "", ""

    if settings.slack_webhook_url:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(settings.slack_webhook_url, json={"text": text, "blocks": blocks})
            resp.raise_for_status()
    return "", ""


async def update_message_status(
    ts: str,
    channel: str,
    server: Server,
    record: AnalysisRecord,
    pr_url: str,
    approved: bool,
) -> None:
    """원본 Slack 메시지의 버튼을 제거하고 승인/거절 상태로 업데이트합니다."""
    if not settings.slack_bot_token or not ts or not channel:
        return

    blocks = _content_blocks(server, record) + [{"type": "divider"}, _status_block(approved, pr_url)]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{_SLACK_API}/chat.update",
                headers=_bot_headers(),
                json={"channel": channel, "ts": ts, "blocks": blocks},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                logger.warning("[slack] chat.update error: %s", data.get("error"))
    except Exception as e:
        logger.error("[slack] update_message_status failed: %s", e)


async def send_result(text: str) -> None:
    if settings.slack_bot_token and settings.slack_channel_id:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{_SLACK_API}/chat.postMessage",
                    headers=_bot_headers(),
                    json={"channel": settings.slack_channel_id, "text": text},
                )
                resp.raise_for_status()
        except Exception as e:
            logger.error("[slack] send_result (bot) failed: %s", e)
        return

    if settings.slack_webhook_url:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(settings.slack_webhook_url, json={"text": text})
                resp.raise_for_status()
        except Exception as e:
            logger.error("[slack] send_result (webhook) failed: %s", e)
