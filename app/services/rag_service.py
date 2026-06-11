import asyncio
import json
import logging
import re
from pathlib import Path

import chromadb
import httpx
from rank_bm25 import BM25Okapi

from app.core.config import settings

logger = logging.getLogger(__name__)

_STATE_FILE = Path(settings.repos_path) / "index_state.json"
_GRAPH_FILE = Path(settings.repos_path) / "dependency_graph.json"
_BATCH_SIZE = 50
_CONTEXT_SEMAPHORE = asyncio.Semaphore(5)


# ── ChromaDB 클라이언트 ──────────────────────────────────────────────────────

def _client():
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def _collection(server_id: int):
    return _client().get_or_create_collection(
        name=f"server_{server_id}",
        metadata={"hnsw:space": "cosine"},
    )


# ── 상태 관리 ────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state))


def is_indexed(server_id: int, commit_hash: str) -> bool:
    return _load_state().get(str(server_id)) == commit_hash


# ── 임베딩 ───────────────────────────────────────────────────────────────────

async def _embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.ollama_embed_model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# ── Contextual RAG: 청크에 컨텍스트 요약 부착 ────────────────────────────────

async def _contextualize_chunk(path: str, content: str) -> str:
    """LLM으로 청크의 역할을 한 문장으로 요약해 원본 앞에 붙인다."""
    prompt = (
        f"파일: {path}\n코드:\n{content[:800]}\n\n"
        "이 코드의 역할을 한 문장으로 설명해줘. 클래스명/메서드명/주요 기능을 포함해. "
        "설명만 출력하고 다른 말은 하지 마."
    )
    async with _CONTEXT_SEMAPHORE:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
            ) as client:
                resp = await client.post(
                    f"{settings.ollama_host}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "think": False,
                    },
                )
                resp.raise_for_status()
                summary = resp.json().get("message", {}).get("content", "").strip()
                return f"{summary}\n\n{content}"
        except Exception as e:
            logger.warning("[rag] context generation failed for %s: %s", path, e)
            return content


async def _contextualize_all(chunks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """모든 청크에 컨텍스트를 병렬로 생성한다."""
    logger.info("[rag] generating context for %d chunks", len(chunks))
    tasks = [_contextualize_chunk(path, content) for path, content in chunks]
    contextualized = await asyncio.gather(*tasks)
    logger.info("[rag] context generation complete")
    return [(path, ctx) for (path, _), ctx in zip(chunks, contextualized)]


# ── Graph RAG: 의존성 그래프 ─────────────────────────────────────────────────

_IMPORT_PATTERNS: dict[str, str] = {
    ".java": r"^import\s+[\w.]+\.(\w+);",
    ".kt":   r"^import\s+[\w.]+\.(\w+)",
    ".py":   r"^(?:from\s+[\w.]+\s+import\s+([\w, ]+)|import\s+([\w.]+))",
    ".ts":   r'from\s+[\'"]([./][\w./@-]+)[\'"]',
    ".tsx":  r'from\s+[\'"]([./][\w./@-]+)[\'"]',
    ".js":   r'from\s+[\'"]([./][\w./@-]+)[\'"]',
    ".go":   r'"([\w./\-]+)"',
}


def _extract_symbol_names(path: str, content: str) -> list[str]:
    ext = Path(path).suffix.lower()
    pattern = _IMPORT_PATTERNS.get(ext, "")
    if not pattern:
        return []
    matches = re.findall(pattern, content, re.MULTILINE)
    symbols = []
    for m in matches:
        if isinstance(m, tuple):
            symbols.extend(p.strip() for p in m if p.strip())
        else:
            symbols.append(Path(m).stem)
    return [s for s in symbols if s and len(s) > 1]


def _build_dependency_graph(chunks: list[tuple[str, str]]) -> dict[str, list[str]]:
    stem_index: dict[str, str] = {Path(path).stem.lower(): path for path, _ in chunks}
    graph: dict[str, list[str]] = {}
    for path, content in chunks:
        deps: set[str] = set()
        for sym in _extract_symbol_names(path, content):
            target = stem_index.get(sym.lower())
            if target and target != path:
                deps.add(target)
        graph[path] = list(deps)
    return graph


def _save_graph(server_id: int, graph: dict[str, list[str]]) -> None:
    data: dict = {}
    try:
        if _GRAPH_FILE.exists():
            data = json.loads(_GRAPH_FILE.read_text())
    except Exception:
        pass
    data[str(server_id)] = graph
    _GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _GRAPH_FILE.write_text(json.dumps(data))


def _load_graph(server_id: int) -> dict[str, list[str]]:
    try:
        if _GRAPH_FILE.exists():
            return json.loads(_GRAPH_FILE.read_text()).get(str(server_id), {})
    except Exception:
        pass
    return {}


def expand_with_graph(server_id: int, paths: list[str], depth: int = 1) -> list[str]:
    """검색된 파일에서 의존 파일을 depth만큼 탐색해 결과를 확장한다."""
    graph = _load_graph(server_id)
    expanded = list(paths)
    seen = set(paths)
    for _ in range(depth):
        for path in list(seen):
            for dep in graph.get(path, []):
                if dep not in seen:
                    expanded.append(dep)
                    seen.add(dep)
    return expanded


# ── 인덱싱 ───────────────────────────────────────────────────────────────────

async def index_repo(server_id: int, commit_hash: str, chunks: list[tuple[str, str]]) -> None:
    if not chunks:
        return

    logger.info("[rag] indexing %d chunks server=%s commit=%s", len(chunks), server_id, commit_hash[:8])

    # Contextual RAG: 청크에 컨텍스트 요약 부착
    ctx_chunks = await _contextualize_all(chunks)

    # Graph RAG: 의존성 그래프 구성 (원본 청크 기준)
    graph = _build_dependency_graph(chunks)
    _save_graph(server_id, graph)
    logger.info("[rag] dependency graph built: %d files", len(graph))

    all_embeddings: list[list[float]] = []
    for i in range(0, len(ctx_chunks), _BATCH_SIZE):
        batch = [content for _, content in ctx_chunks[i:i + _BATCH_SIZE]]
        embeddings = await _embed(batch)
        all_embeddings.extend(embeddings)
        logger.info("[rag] embedded %d/%d chunks", min(i + _BATCH_SIZE, len(ctx_chunks)), len(ctx_chunks))

    chroma = _client()
    collection_name = f"server_{server_id}"
    try:
        chroma.delete_collection(collection_name)
    except Exception:
        pass
    col = chroma.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"{path}__chunk{i}" for i, (path, _) in enumerate(ctx_chunks)],
        documents=[content for _, content in ctx_chunks],
        embeddings=all_embeddings,
        metadatas=[{"path": path, "commit": commit_hash} for path, _ in ctx_chunks],
    )

    state = _load_state()
    state[str(server_id)] = commit_hash
    _save_state(state)

    logger.info("[rag] index complete server=%s chunks=%d", server_id, len(chunks))


# ── 검색 ─────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z가-힣0-9]+", text.lower())


async def search_relevant_files(server_id: int, query: str, n_results: int = 5) -> list[str]:
    """에러 쿼리와 관련된 파일 경로 반환 (벡터 + BM25 하이브리드 → RRF → Graph 확장)."""
    try:
        col = _collection(server_id)
        count = col.count()
        if count == 0:
            return []

        candidates = min(n_results * 4, count)

        # --- 벡터 검색 ---
        query_embeddings = await _embed([query[:2000]])
        vec_results = col.query(
            query_embeddings=query_embeddings,
            n_results=candidates,
        )
        vec_items: list[tuple[dict, str]] = list(
            zip(vec_results["metadatas"][0], vec_results["documents"][0])
        )

        # --- BM25 검색 ---
        all_data = col.get(limit=min(count, 2000), include=["documents", "metadatas"])
        all_docs: list[str] = all_data["documents"]
        all_metas: list[dict] = all_data["metadatas"]

        tokenized_corpus = [_tokenize(doc) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(_tokenize(query))
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidates]
        bm25_items: list[tuple[dict, str]] = [(all_metas[i], all_docs[i]) for i in bm25_top_idx]

        # --- RRF 융합 ---
        K = 60
        rrf: dict[str, float] = {}
        for rank, (meta, _) in enumerate(vec_items):
            path = meta["path"]
            rrf[path] = rrf.get(path, 0.0) + 1 / (K + rank + 1)
        for rank, (meta, _) in enumerate(bm25_items):
            path = meta["path"]
            rrf[path] = rrf.get(path, 0.0) + 1 / (K + rank + 1)

        sorted_paths = sorted(rrf, key=lambda p: rrf[p], reverse=True)

        seen: set[str] = set()
        rrf_paths: list[str] = []
        for path in sorted_paths:
            if path not in seen:
                seen.add(path)
                rrf_paths.append(path)
            if len(rrf_paths) >= n_results:
                break

        # --- Graph RAG: 의존 파일 확장 ---
        paths = expand_with_graph(server_id, rrf_paths, depth=1)

        logger.info("[rag] hybrid+graph search returned %s", paths)
        return paths[:n_results + 3]
    except Exception as e:
        logger.warning("[rag] search failed: %s", e)
        return []
