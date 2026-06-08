import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool as langchain_tool
from langchain_ollama import ChatOllama

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior DevSecOps engineer analyzing server error logs.

Use search_files to find source files related to the error — such as:
- The class/file mentioned in the stack trace
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=15,
        max_execution_time=120,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )

    result = await agent_executor.ainvoke({"input": f"다음 에러 로그를 분석해줘:\n\n```\n{raw_log[:4000]}\n```"})
    raw = _strip_code_fence(result.get("output", ""))

    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        logger.warning("[agent] output is not valid JSON, falling back")
        return await _fallback_analyze(raw_log)


async def _fallback_analyze(raw_log: str) -> str:
    """AgentExecutor 결과가 유효한 JSON이 아닐 때 structured output으로 재시도."""
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
