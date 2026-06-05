import json

from slack_sdk.web.async_client import AsyncWebClient

from app.core.config import settings
from app.models.server import AnalysisRecord, Server


def _client() -> AsyncWebClient:
    return AsyncWebClient(token=settings.slack_bot_token)


def _build_analysis_blocks(server: Server, record: AnalysisRecord) -> list[dict]:
    try:
        suggestion = json.loads(record.llm_suggestion or "{}")
    except json.JSONDecodeError:
        suggestion = {}

    error_cause = suggestion.get("error_cause", "분석 중...")
    fix_code = suggestion.get("suggested_fix", record.llm_suggestion or "")

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🔴 [{server.name}] 에러 감지"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*원인:* {error_cause}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*트리거 라인:*\n```{record.trigger_line[:300]}```",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*추천 수정 코드:*\n{fix_code[:1500]}",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ 수락 & Push"},
                    "action_id": "approve_fix",
                    "value": str(record.id),
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ 거절"},
                    "action_id": "reject_fix",
                    "value": str(record.id),
                    "style": "danger",
                },
            ],
        },
    ]


async def send_analysis(server: Server, record: AnalysisRecord) -> str:
    if not settings.slack_bot_token:
        return ""
    client = _client()
    response = await client.chat_postMessage(
        channel=settings.slack_channel_id,
        blocks=_build_analysis_blocks(server, record),
        text=f"[{server.name}] 에러 감지 및 LLM 분석 완료",
    )
    return response["ts"]


async def update_message(ts: str, text: str) -> None:
    if not settings.slack_bot_token or not ts:
        return
    client = _client()
    await client.chat_update(
        channel=settings.slack_channel_id,
        ts=ts,
        text=text,
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
            }
        ],
    )
