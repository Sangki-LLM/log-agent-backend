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


async def search_relevant_files(server_id: int, query: str, n_results: int = 5) -> list[str]:
    """에러 쿼리와 의미적으로 유사한 파일 경로 반환 (벡터 검색)."""
    try:
        col = _collection(server_id)
        count = col.count()
        if count == 0:
            return []

        query_embeddings = await _embed([query[:2000]])
        results = col.query(
            query_embeddings=query_embeddings,
            n_results=min(n_results, count),
        )

        all_docs = list(zip(results["metadatas"][0], results["documents"][0]))
        if not all_docs:
            return []

        seen: set[str] = set()
        paths: list[str] = []
        for meta, _ in all_docs:
            path = meta["path"]
            if path not in seen:
                seen.add(path)
                paths.append(path)
            if len(paths) >= n_results:
                break

        logger.info("[rag] search returned %s", paths)
        return paths
    except Exception as e:
        logger.warning("[rag] search failed: %s", e)
        return []
