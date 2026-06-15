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

MANDATORY: You MUST read the actual source files before writing your final answer. Never guess or invent code.

Available tools:
- grep_files(pattern, path="", file_extension=""): Search for class/method names across the repo. Start here.
- read_file(path): Read a specific file. Use after finding the path via grep_files.
- list_directory(path=""): List files/directories. Use to explore project structure when needed.

Workflow:
1. Extract class or file names from the stack trace
2. Call grep_files with the class name to find the file path
3. Call read_file to read the actual source code
4. If you need config or dependencies, grep_files again (@Bean, @Configuration, etc.)
5. Once you have seen the real code, respond ONLY in this JSON structure (no markdown, no code fences):

{
  "error_cause": "<brief root cause in Korean>",
  "bottleneck": "<suspected bottleneck or affected component>",
  "suggested_fix": "<fix explanation in Korean>",
  "commit_message": "<concise git commit message starting with fix:>",
  "file_patch": {
    "file_path": "<relative path from repo root>",
    "before": "<EXACT lines copied from the file returned by read_file — including all whitespace>",
    "after": "<corrected replacement code>"
  }
}

Rules:
- 'before' MUST be copied verbatim from read_file output — never write code you have not seen
- Provide the MINIMAL change — do not rewrite the whole file
- If multiple files need changes, pick the single most impactful one"""

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


def _preprocess_log(raw_log: str) -> tuple[str, list[str]]:
    """에러 로그에서 SQL·파라미터 제거 후 핵심 메시지와 파일 경로 힌트 반환."""
    # SQL/파라미터 섹션 이전까지만 유지
    key_lines: list[str] = []
    for line in raw_log.split("\n"):
        if re.match(r"###\s*(SQL|Parameters|Cause at):", line):
            break
        key_lines.append(line)
    key_log = "\n".join(key_lines)[:2000]

    # 파일 경로 힌트 추출
    hints: list[str] = []

    # MyBatis: file [C:\...\build\resources\main\mapper\...\Foo.xml]
    for m in re.finditer(
        r"file \[.*?[/\\]((?:src|mapper|resources)[^\]]*\.(?:java|kt|xml|properties|yml|yaml))\]",
        raw_log,
    ):
        path = m.group(1).replace("\\", "/")
        path = re.sub(r"build/resources/main/", "src/main/resources/", path)
        hints.append(path)

    # Java stack trace: at com.example.Foo.method(Foo.java:123)
    for m in re.finditer(r"at ([\w$.]+)\.\w+\((\w+\.(?:java|kt)):\d+\)", raw_log):
        pkg = m.group(1).split("$")[0].rsplit(".", 1)[0].replace(".", "/")
        hints.append(f"src/main/java/{pkg}/{m.group(2)}")

    # 중복 제거, 최대 3개
    seen: set[str] = set()
    deduped: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
        if len(deduped) >= 3:
            break

    return key_log, deduped


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


_SOURCE_EXTENSIONS = {".java", ".kt", ".py", ".ts", ".tsx", ".js", ".go", ".xml", ".yml", ".yaml", ".properties"}
_MAX_FILE_CHARS = 4000
_MAX_GREP_FILES = 5
_MAX_GREP_LINES = 10


def _safe_path(repo_path: Path, rel: str) -> Path | None:
    """Path traversal 방지: repo 외부 경로는 None 반환."""
    target = (repo_path / rel).resolve()
    return target if str(target).startswith(str(repo_path.resolve())) else None


def _make_file_tools(server_id: int, repo_path: Path):
    @langchain_tool
    async def grep_files(pattern: str, path: str = "", file_extension: str = "") -> str:
        """레포지토리에서 클래스명·메서드명·패턴을 검색해 파일 경로와 매칭 라인을 반환합니다."""
        base = _safe_path(repo_path, path) if path else repo_path.resolve()
        if base is None:
            return "접근 불가: 레포지토리 외부 경로"

        exts = ({f".{file_extension.lstrip('.')}"} if file_extension else _SOURCE_EXTENSIONS)
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        results: list[str] = []
        for fp in sorted(base.rglob("*")):
            if not fp.is_file() or fp.suffix.lower() not in exts or ".git" in fp.parts:
                continue
            try:
                lines = fp.read_text(encoding="utf-8", errors="replace").split("\n")
            except Exception:
                continue
            matched = [f"  {i}: {l}" for i, l in enumerate(lines, 1) if regex.search(l)]
            if matched:
                rel = fp.relative_to(repo_path)
                results.append(f"=== {rel} ===\n" + "\n".join(matched[:_MAX_GREP_LINES]))
                logger.info("[agent] grep_files pattern=%r → %s", pattern, rel)
                if len(results) >= _MAX_GREP_FILES:
                    break

        return "\n\n".join(results) if results else f"패턴 '{pattern}'을 찾을 수 없습니다."

    @langchain_tool
    async def read_file(path: str) -> str:
        """레포지토리 내 특정 파일의 내용을 읽습니다."""
        target = _safe_path(repo_path, path)
        if target is None:
            return "접근 불가: 레포지토리 외부 경로"
        if not target.exists():
            return f"파일이 존재하지 않습니다: {path}"
        if not target.is_file():
            return f"{path}는 디렉토리입니다. list_directory를 사용하세요."
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            logger.info("[agent] read_file path=%s len=%d", path, len(content))
            if len(content) > _MAX_FILE_CHARS:
                return content[:_MAX_FILE_CHARS] + f"\n... (총 {len(content)}자, 처음 {_MAX_FILE_CHARS}자만 표시)"
            return content
        except Exception as e:
            return f"읽기 실패: {e}"

    @langchain_tool
    async def list_directory(path: str = "") -> str:
        """레포지토리 내 디렉토리 목록을 반환합니다. path가 비어있으면 루트를 탐색합니다."""
        target = _safe_path(repo_path, path) if path else repo_path.resolve()
        if target is None:
            return "접근 불가: 레포지토리 외부 경로"
        if not target.exists():
            return f"경로가 존재하지 않습니다: {path}"
        if not target.is_dir():
            return f"{path}는 파일입니다. read_file을 사용하세요."

        entries = []
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            rel = item.relative_to(repo_path)
            entries.append(f"{rel}/" if item.is_dir() else str(rel))

        logger.info("[agent] list_directory path=%r entries=%d", path, len(entries))
        return "\n".join(entries[:100]) or "(빈 디렉토리)"

    return [grep_files, read_file, list_directory]


async def analyze_log(server_id: int, raw_log: str, stack_trace: str = "") -> str:
    from app.services.git_service import _repo_path
    from app.services import rag_service

    repo_path = _repo_path(server_id)
    tools = _make_file_tools(server_id, repo_path)

    # Error Memory: 과거 유사 에러 사례 조회
    past_cases = await rag_service.search_error_memory(server_id, raw_log[:1000])
    system_prompt = SYSTEM_PROMPT
    if past_cases:
        cases_text = "\n\n".join(
            f"[과거 사례 {i+1}]\n에러: {c['error'][:300]}\n분석 요약: {c['analysis'][:500]}"
            for i, c in enumerate(past_cases)
        )
        system_prompt = f"{SYSTEM_PROMPT}\n\n=== 과거 유사 에러 사례 (참고용) ===\n{cases_text}"
        logger.info("[agent] injected %d past cases into prompt", len(past_cases))

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        think=False,
    )

    graph = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    key_log, file_hints = _preprocess_log(raw_log)
    hint_text = ""
    if file_hints:
        hint_text = "\n\n파일 경로 힌트 (grep_files 또는 read_file로 바로 확인):\n" + "\n".join(
            f"- {h}" for h in file_hints
        )
    user_message = f"다음 에러 로그를 분석해줘:\n\n```\n{key_log}\n```{hint_text}"
    logger.info("[agent] user_message len=%d file_hints=%s", len(user_message), file_hints)

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {"messages": [("user", user_message)]},
                config={"recursion_limit": 30},
            ),
            timeout=300,
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
        # Error Memory: 분석 결과 저장
        await rag_service.store_error_memory(server_id, raw_log[:1000], raw)
        return raw
    except json.JSONDecodeError:
        logger.warning("[agent] output is not valid JSON, falling back")
        return await _fallback_analyze(raw_log)


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
    key_log, file_hints = _preprocess_log(raw_log)
    hint_text = ""
    if file_hints:
        hint_text = "\n\n파일 경로 힌트:\n" + "\n".join(f"- {h}" for h in file_hints)
    user_content = f"다음 에러 로그를 분석해줘:\n\n```\n{key_log}\n```{hint_text}"

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0)
    ) as client:
        response = await client.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
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
