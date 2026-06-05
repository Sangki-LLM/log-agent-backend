import json
import re
from collections.abc import AsyncGenerator

import httpx

from app.core.config import settings

SYSTEM_PROMPT = """You are a senior DevSecOps engineer analyzing server error logs.
When given a log excerpt, respond ONLY in this JSON structure (no markdown, no code fences):
{
  "error_cause": "<brief root cause in Korean>",
  "bottleneck": "<suspected bottleneck or affected component>",
  "suggested_fix": "<corrected code block in markdown>",
  "commit_message": "<concise git commit message starting with fix:>"
}"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def analyze_log(raw_log: str, source_files: dict[str, str] | None = None) -> str:
    parts = [f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```"]

    if source_files:
        parts.append("\n\n관련 소스 파일:")
        for path, content in source_files.items():
            parts.append(f"\n### {path}\n```\n{content[:2000]}\n```")

    prompt = "".join(parts)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        return _strip_code_fence(raw)


async def stream_analysis(raw_log: str) -> AsyncGenerator[str, None]:
    prompt = f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```"

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    if token := chunk.get("response"):
                        yield token
                    if chunk.get("done"):
                        break
