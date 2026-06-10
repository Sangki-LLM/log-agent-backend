import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
from langchain_core.tools import tool as langchain_tool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior DevSecOps engineer analyzing server error logs.

MANDATORY STEP: You MUST call search_files at least once before writing your final answer.
- Search for the exact class/file mentioned in the stack trace first.
- Never guess or invent code. The 'before' field must be copied character-for-character from the file content returned by search_files.
- If the first search does not return the right file, call search_files again with a different query.

Use search_files to find:
- The class/file mentioned in the stack trace
- @Configuration or @Bean definitions
- injected @Service or @Component dependencies
- application properties or environment config

Once you have retrieved the actual source file and confirmed the code to change, respond ONLY in this JSON structure (no markdown, no code fences):
{
  "error_cause": "<brief root cause in Korean>",
  "bottleneck": "<suspected bottleneck or affected component>",
  "suggested_fix": "<fix explanation in Korean>",
  "commit_message": "<concise git commit message starting with fix:>",
  "file_patch": {
    "file_path": "<relative path from repo root, e.g. src/main/java/com/example/Foo.java>",
    "before": "<EXACT lines copied from the file returned by search_files — including all whitespace and indentation>",
    "after": "<corrected replacement code>"
  }
}

Rules for file_patch (REQUIRED — never omit):
- 'file_path': relative path from repo root of the file to modify
- 'before': MUST be copied verbatim from the search_files result — never write code you have not seen in the file
- 'after': the corrected replacement lines
- provide the MINIMAL change — do not rewrite the whole file
- if multiple files need changes, pick the single most impactful one"""

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


def _make_search_tool(server_id: int, repo_path: Path):
    @langchain_tool
    async def search_files(query: str) -> str:
        """에러의 근본 원인과 관련된 소스 파일을 검색하고 내용을 반환합니다."""
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

    return search_files


async def analyze_log(server_id: int, raw_log: str, stack_trace: str = "") -> str:
    from app.services.git_service import _repo_path

    repo_path = _repo_path(server_id)
    tools = [_make_search_tool(server_id, repo_path)]

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        think=False,
    )

    graph = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {"messages": [("user", f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```")]},
                config={"recursion_limit": 30},
            ),
            timeout=120,
        )
        raw = _strip_code_fence(result["messages"][-1].content)
        logger.info("[agent] done, content_len=%d", len(raw))
    except GraphRecursionError:
        logger.warning("[agent] recursion limit reached, falling back")
        return await _fallback_analyze(raw_log)
    except asyncio.TimeoutError:
        logger.warning("[agent] timeout, falling back")
        return await _fallback_analyze(raw_log)

    try:
        json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[agent] output is not valid JSON, falling back")
        raw = await _fallback_analyze(raw_log)

    return await _self_reflect(raw, server_id)


async def _self_reflect(suggestion: str, server_id: int) -> str:
    """LLM이 제안한 before 코드가 실제 파일에 존재하는지 검증하고, 없으면 수정 요청."""
    try:
        data = json.loads(suggestion)
    except json.JSONDecodeError:
        return suggestion

    fp = data.get("file_patch", {})
    file_path = fp.get("file_path", "")
    before = fp.get("before", "")
    after = fp.get("after", "")
    if not file_path or not before:
        return suggestion

    from app.services.git_service import _repo_path
    try:
        file_content = (_repo_path(server_id) / file_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        logger.warning("[reflect] file not found: %s", file_path)
        return suggestion

    # before가 실제 파일에 있으면 검증 통과
    norm = lambda s: s.replace("\r\n", "\n")
    if norm(before) in norm(file_content):
        logger.info("[reflect] before verified ✓ %s", file_path)
        return suggestion

    # before가 없으면 LLM에게 올바른 before 찾기 요청
    logger.warning("[reflect] before not found — asking LLM to correct (%s)", file_path)
    prompt = (
        "아래 파일에서 수정 의도에 맞는 실제 코드를 찾아줘.\n\n"
        f"=== 파일 내용 ===\n{file_content[:3000]}\n\n"
        f"=== 수정 의도 ===\n"
        f"before (파일에 없을 수 있음):\n{before}\n\n"
        f"after:\n{after}\n\n"
        "파일에서 before에 해당하는 실제 코드를 찾아 반환해줘.\n"
        "응답은 JSON만: {\"before\": \"파일에 실제로 있는 코드\"}"
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=5.0)
    ) as client:
        resp = await client.post(
            f"{settings.ollama_host}/api/chat",
            json={"model": settings.ollama_model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "think": False},
        )
        resp.raise_for_status()
        raw = _strip_code_fence(resp.json().get("message", {}).get("content", ""))

    try:
        corrected_before = json.loads(raw).get("before", "")
        if corrected_before and norm(corrected_before) in norm(file_content):
            data["file_patch"]["before"] = corrected_before
            logger.info("[reflect] before corrected ✓")
            return json.dumps(data, ensure_ascii=False)
        logger.warning("[reflect] LLM correction also failed")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("[reflect] reflection response not valid JSON")

    return suggestion


async def _fallback_analyze(raw_log: str) -> str:
    """에이전트 결과가 유효한 JSON이 아닐 때 structured output으로 재시도."""
    logger.info("[agent] fallback with structured output")
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
        raw = response.json().get("message", {}).get("content", "")
        logger.info("[agent] fallback response length=%d", len(raw))
        return _strip_code_fence(raw)


async def apply_patch_with_llm(original: str, before: str, after: str) -> str:
    """original_content에 before→after 변경을 LLM이 직접 적용해 완성 파일 반환."""
    prompt = (
        "아래 파일에 코드 변경을 적용해줘. "
        "설명 없이 변경이 적용된 파일 전체 내용만 반환해.\n\n"
        f"=== 원본 파일 ===\n{original}\n\n"
        f"=== 변경 전 (Before) ===\n{before}\n\n"
        f"=== 변경 후 (After) ===\n{after}"
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=5.0)
    ) as client:
        response = await client.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
            },
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return _strip_code_fence(content)


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
