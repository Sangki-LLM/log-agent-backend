import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior DevSecOps engineer analyzing server error logs.
Use the provided tools to search for and read relevant source files before analyzing.
When done, respond ONLY in this JSON structure (no markdown, no code fences):
{
  "error_cause": "<brief root cause in Korean>",
  "bottleneck": "<suspected bottleneck or affected component>",
  "suggested_fix": "<corrected code block in markdown>",
  "commit_message": "<concise git commit message starting with fix:>"
}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "에러와 관련된 소스 파일 경로를 ChromaDB에서 의미 기반으로 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드 (클래스명, 메서드명, 에러 원인 등)",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "소스 파일의 내용을 읽습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "읽을 파일 경로 (예: src/main/java/com/example/Foo.java)",
                    }
                },
                "required": ["path"],
            },
        },
    },
]

_MAX_TOOL_ITERATIONS = 8


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


async def _execute_tool(name: str, args: dict, server_id: int, repo_path: Path) -> str:
    from app.services import rag_service

    if name == "search_files":
        query = args.get("query", "")
        paths = await rag_service.search_relevant_files(server_id, query, n_results=5)
        logger.info("[agent] search_files query=%r → %s", query, paths)
        return json.dumps(paths, ensure_ascii=False)

    if name == "read_file":
        path = args.get("path", "")
        try:
            content = (repo_path / path).read_text(encoding="utf-8", errors="replace")
            logger.info("[agent] read_file path=%s (%d chars)", path, len(content))
            return content[:3000]
        except FileNotFoundError:
            logger.warning("[agent] read_file not found: %s", path)
            return f"파일을 찾을 수 없습니다: {path}"

    return f"알 수 없는 도구: {name}"


async def analyze_log(server_id: int, raw_log: str) -> str:
    from app.services.git_service import _repo_path

    repo_path = _repo_path(server_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```",
        },
    ]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0)
    ) as client:
        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await client.post(
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": messages,
                    "tools": TOOLS,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                raw = message.get("content", "")
                logger.info("[agent] done after %d iterations", iteration + 1)
                return _strip_code_fence(raw)

            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = _parse_arguments(fn.get("arguments", {}))
                result = await _execute_tool(name, args, server_id, repo_path)
                messages.append({"role": "tool", "content": result})

    logger.warning("[agent] max iterations reached without final response")
    return ""


async def stream_analysis(raw_log: str) -> AsyncGenerator[str, None]:
    prompt = f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```"

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0)
    ) as client:
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
