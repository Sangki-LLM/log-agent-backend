import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior DevSecOps engineer analyzing server error logs.

The user message already includes the source file(s) directly involved in the error (extracted from the stack trace).
Use search_files to find additional related files that may be the ROOT CAUSE — such as:
- @Configuration or @Bean definitions
- injected @Service or @Component dependencies
- application properties or environment config

Once you have enough context, respond ONLY in this JSON structure (no markdown, no code fences):
{
  "error_cause": "<brief root cause in Korean>",
  "bottleneck": "<suspected bottleneck or affected component>",
  "suggested_fix": "<fix explanation in Korean>",
  "commit_message": "<concise git commit message starting with fix:>",
  "file_patch": {
    "file_path": "<relative path from repo root, e.g. src/main/java/com/example/Foo.java>",
    "before": "<exact code snippet to replace — must match the file exactly, including indentation>",
    "after": "<corrected replacement code>"
  }
}

Rules for file_patch (REQUIRED — never omit):
- 'file_path': relative path from repo root of the file to modify
- 'before': copy-paste the EXACT lines from the source file shown to you that need to change (including indentation)
- 'after': the corrected replacement lines
- provide the MINIMAL change — do not rewrite the whole file
- if multiple files need changes, pick the single most impactful one"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "에러의 근본 원인과 관련된 소스 파일을 검색하고 내용을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드 (클래스명, Bean 이름, 설정 키 등)",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_MAX_TOOL_ITERATIONS = 5

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "error_cause":    {"type": "string"},
        "bottleneck":     {"type": "string"},
        "suggested_fix":  {"type": "string"},
        "commit_message": {"type": "string"},
        "file_patch": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "before":    {"type": "string"},
                "after":     {"type": "string"},
            },
            "required": ["file_path", "before", "after"],
        },
    },
    "required": ["error_cause", "bottleneck", "suggested_fix", "commit_message", "file_patch"],
}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_from_thinking(thinking: str) -> str:
    """thinking 필드에 갇힌 JSON 응답 추출. 중괄호 쌍 매칭 방식."""
    idx = thinking.find('"error_cause"')
    if idx == -1:
        return ""
    start = thinking.rfind("{", 0, idx)
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(thinking)):
        if thinking[i] == "{":
            depth += 1
        elif thinking[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = thinking[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    break
    return ""


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _read_stack_trace_files(server_id: int, stack_trace: str) -> dict[str, str]:
    from app.services.git_service import _extract_source_paths, _repo_path

    if not stack_trace:
        return {}

    repo_path = _repo_path(server_id)
    paths = _extract_source_paths(stack_trace)
    result: dict[str, str] = {}
    for path in paths:
        try:
            content = (repo_path / path).read_text(encoding="utf-8", errors="replace")
            result[path] = content[:3000]
        except FileNotFoundError:
            pass
    return result


async def _execute_search(query: str, server_id: int, repo_path: Path) -> str:
    from app.services import rag_service

    paths = await rag_service.search_relevant_files(server_id, query, n_results=3)
    logger.info("[agent] search_files query=%r → %s", query, paths)

    results: dict[str, str] = {}
    for path in paths:
        try:
            content = (repo_path / path).read_text(encoding="utf-8", errors="replace")
            results[path] = content[:2000]
        except FileNotFoundError:
            pass

    return json.dumps(results, ensure_ascii=False)


def _build_initial_user_message(raw_log: str, stack_files: dict[str, str]) -> str:
    parts = [f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```"]
    if stack_files:
        parts.append("\n\n에러가 발생한 소스 파일:")
        for path, content in stack_files.items():
            parts.append(f"\n### {path}\n```java\n{content}\n```")
    return "".join(parts)


async def analyze_log(server_id: int, raw_log: str, stack_trace: str = "") -> str:
    from app.services.git_service import _repo_path

    repo_path = _repo_path(server_id)

    stack_files = await asyncio.to_thread(_read_stack_trace_files, server_id, stack_trace)
    if stack_files:
        logger.info("[agent] pre-loaded stack trace files: %s", list(stack_files.keys()))
    else:
        logger.info("[agent] no stack trace files found, LLM will search")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_initial_user_message(raw_log, stack_files)},
    ]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0)
    ) as client:
        for iteration in range(_MAX_TOOL_ITERATIONS):
            tools_payload = TOOLS if iteration < _MAX_TOOL_ITERATIONS - 1 else []
            if not tools_payload:
                messages.append({
                    "role": "user",
                    "content": "지금까지 수집한 정보를 바탕으로 JSON 형식으로 분석 결과를 응답해줘.",
                })

            request_body: dict = {
                "model": settings.ollama_model,
                "messages": messages,
                "tools": tools_payload,
                "stream": False,
                "think": False,
            }
            if not tools_payload:
                request_body["format"] = _ANALYSIS_SCHEMA

            response = await client.post(
                f"{settings.ollama_host}/api/chat",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                raw = message.get("content", "")
                if not raw:
                    thinking = message.get("thinking", "")
                    if thinking:
                        raw = _extract_json_from_thinking(thinking)
                        if raw:
                            logger.info("[agent] extracted %d chars from thinking field", len(raw))
                logger.info("[agent] done after %d iterations, content_len=%d", iteration + 1, len(raw))
                if not raw:
                    logger.warning("[agent] empty response from ollama, message keys=%s", list(message.keys()))
                    return await _fallback_analyze(raw_log)
                return _strip_code_fence(raw)

            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = _parse_arguments(fn.get("arguments", {}))
                if name == "search_files":
                    result = await _execute_search(args.get("query", ""), server_id, repo_path)
                else:
                    result = f"알 수 없는 도구: {name}"
                messages.append({"role": "tool", "content": result})

    logger.warning("[agent] max iterations reached without final response")
    return await _fallback_analyze(raw_log)


async def _fallback_analyze(raw_log: str) -> str:
    """Agentic 분석 실패 시 단순 one-shot 프롬프트로 재시도."""
    logger.info("[agent] fallback simple prompt")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0)
    ) as client:
        response = await client.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:3000]}\n```"},
                ],
                "stream": False,
                "think": False,
                "format": _ANALYSIS_SCHEMA,
            },
        )
        response.raise_for_status()
        msg = response.json().get("message", {})
        raw = msg.get("content", "")
        if not raw:
            thinking = msg.get("thinking", "")
            if thinking:
                raw = _extract_json_from_thinking(thinking)
                if raw:
                    logger.info("[agent] fallback extracted %d chars from thinking", len(raw))
        logger.info("[agent] fallback response length=%d", len(raw))
        return _strip_code_fence(raw)


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
