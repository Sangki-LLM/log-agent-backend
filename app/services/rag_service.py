import json
import logging
from pathlib import Path

import chromadb
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# 인덱스 상태는 repos 볼륨에 저장 (컨테이너 재시작 후에도 유지)
_STATE_FILE = Path(settings.repos_path) / "index_state.json"
_BATCH_SIZE = 50


def _collection(server_id: int):
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return client.get_or_create_collection(
        name=f"server_{server_id}",
        metadata={"hnsw:space": "cosine"},
    )


def _load_state() -> dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state))


def is_indexed(server_id: int, commit_hash: str) -> bool:
    return _load_state().get(str(server_id)) == commit_hash


async def _embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.ollama_embed_model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


async def index_repo(server_id: int, commit_hash: str, chunks: list[tuple[str, str]]) -> None:
    if not chunks:
        return

    logger.info("[rag] indexing %d chunks server=%s commit=%s", len(chunks), server_id, commit_hash[:8])

    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = [content for _, content in chunks[i:i + _BATCH_SIZE]]
        embeddings = await _embed(batch)
        all_embeddings.extend(embeddings)
        logger.info("[rag] embedded %d/%d chunks", min(i + _BATCH_SIZE, len(chunks)), len(chunks))

    col = _collection(server_id)
    col.upsert(
        ids=[f"{path}__chunk{i}" for i, (path, _) in enumerate(chunks)],
        documents=[content for _, content in chunks],
        embeddings=all_embeddings,
        metadatas=[{"path": path, "commit": commit_hash} for path, _ in chunks],
    )

    state = _load_state()
    state[str(server_id)] = commit_hash
    _save_state(state)

    logger.info("[rag] index complete server=%s chunks=%d", server_id, len(chunks))


async def _rerank(query: str, documents: list[str]) -> list[int]:
    """Reranker로 후보 문서 재순위화. 관련도 높은 순 index 목록 반환."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.ollama_host}/api/rerank",
                json={
                    "model": settings.ollama_rerank_model,
                    "query": query,
                    "documents": documents,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [r["index"] for r in sorted(results, key=lambda x: x["relevance_score"], reverse=True)]
    except Exception as e:
        logger.warning("[rag] rerank failed (fallback to vector order): %s", e)
        return list(range(len(documents)))


async def search_relevant_files(server_id: int, query: str, n_results: int = 5) -> list[str]:
    """에러 쿼리와 의미적으로 유사한 파일 경로 반환 (벡터 검색 → reranker 재순위화)."""
    try:
        col = _collection(server_id)
        count = col.count()
        if count == 0:
            return []

        query_embeddings = await _embed([query[:2000]])
        # reranker 후보를 위해 넉넉하게 가져옴
        candidates = min(n_results * 5, count)
        results = col.query(
            query_embeddings=query_embeddings,
            n_results=candidates,
        )

        # (path, document) 쌍 목록 (중복 포함 — reranker가 청크 단위로 판단)
        all_docs = list(zip(results["metadatas"][0], results["documents"][0]))
        if not all_docs:
            return []

        # Reranker로 재순위화
        ranked_indices = await _rerank(query, [doc for _, doc in all_docs])

        # 파일 경로 기준 중복 제거 (가장 높은 순위 청크만 채택)
        seen: set[str] = set()
        paths: list[str] = []
        for idx in ranked_indices:
            path = all_docs[idx][0]["path"]
            if path not in seen:
                seen.add(path)
                paths.append(path)
            if len(paths) >= n_results:
                break

        logger.info("[rag] reranked search returned %s", paths)
        return paths
    except Exception as e:
        logger.warning("[rag] search failed: %s", e)
        return []
