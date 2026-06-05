import json
from collections.abc import AsyncGenerator

import httpx

from app.core.config import settings

SYSTEM_PROMPT = """You are a senior DevSecOps engineer analyzing server error logs.
When given a log excerpt, respond ONLY in this JSON structure:
{
  "error_cause": "<brief root cause in Korean>",
  "bottleneck": "<suspected bottleneck or affected component>",
  "suggested_fix": "<corrected code block in markdown>",
  "commit_message": "<concise git commit message starting with fix:>"
}"""


async def analyze_log(raw_log: str) -> str:
    prompt = f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```"

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")


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
