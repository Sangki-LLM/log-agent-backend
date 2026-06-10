import json
import logging
import re
from pathlib import Path

import chromadb
import httpx
from rank_bm25 import BM25Okapi

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


def _tokenize(text: str) -> list[str]:
    """영문·한글·숫자 토큰 추출 (소문자 정규화)."""
    return re.findall(r"[A-Za-z가-힣0-9]+", text.lower())


async def search_relevant_files(server_id: int, query: str, n_results: int = 5) -> list[str]:
    """에러 쿼리와 관련된 파일 경로 반환 (벡터 검색 + BM25 하이브리드, RRF 융합)."""
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

        # --- RRF 융합 (k=60) ---
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
        paths: list[str] = []
        for path in sorted_paths:
            if path not in seen:
                seen.add(path)
                paths.append(path)
            if len(paths) >= n_results:
                break

        logger.info("[rag] hybrid search returned %s", paths)
        return paths
    except Exception as e:
        logger.warning("[rag] search failed: %s", e)
        return []
