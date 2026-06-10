import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    name: str
    stack_trace: str
    expected_files: list[str]  # repo root 기준 상대 경로


@dataclass
class CaseResult:
    name: str
    stack_trace: str
    expected: list[str]
    retrieved: list[str]          # 파일 경로
    retrieved_previews: list[str] # 파일 내용 미리보기 (500자)
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    rr: float


def _filename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


def _hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top_k = {_filename(p) for p in retrieved[:k]}
    expected_names = {_filename(e) for e in expected}
    return 1.0 if top_k & expected_names else 0.0


def _reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    expected_names = {_filename(e) for e in expected}
    for i, path in enumerate(retrieved):
        if _filename(path) in expected_names:
            return 1.0 / (i + 1)
    return 0.0


async def _read_previews(server_id: int, paths: list[str]) -> list[str]:
    from app.services.git_service import _repo_path
    repo = _repo_path(server_id)
    previews = []
    for path in paths:
        try:
            content = (repo / path).read_text(encoding="utf-8", errors="replace")
            previews.append(content[:500])
        except Exception:
            previews.append("")
    return previews


async def evaluate(
    server_id: int,
    test_cases: list[TestCase],
    n_results: int = 5,
) -> dict:
    """RAG 검색 품질 평가 후 DB에 저장. Hit Rate@K, MRR 반환."""
    from app.core.database import AsyncSessionLocal
    from app.models.server import EvalCase, EvalRetrieved, EvalRun
    from app.services.rag_service import search_relevant_files

    case_results: list[CaseResult] = []

    for tc in test_cases:
        retrieved = await search_relevant_files(server_id, tc.stack_trace, n_results=n_results)
        previews = await _read_previews(server_id, retrieved)

        result = CaseResult(
            name=tc.name,
            stack_trace=tc.stack_trace,
            expected=tc.expected_files,
            retrieved=retrieved,
            retrieved_previews=previews,
            hit_at_1=_hit_at_k(retrieved, tc.expected_files, 1),
            hit_at_3=_hit_at_k(retrieved, tc.expected_files, 3),
            hit_at_5=_hit_at_k(retrieved, tc.expected_files, 5),
            rr=_reciprocal_rank(retrieved, tc.expected_files),
        )
        case_results.append(result)

        logger.info(
            "[eval] %-40s | HR@1=%.0f HR@3=%.0f HR@5=%.0f RR=%.2f | top3=%s",
            tc.name[:40],
            result.hit_at_1, result.hit_at_3, result.hit_at_5, result.rr,
            [_filename(p) for p in retrieved[:3]],
        )

    n = len(case_results)
    metrics = {
        "hit_rate@1": sum(r.hit_at_1 for r in case_results) / n,
        "hit_rate@3": sum(r.hit_at_3 for r in case_results) / n,
        "hit_rate@5": sum(r.hit_at_5 for r in case_results) / n,
        "mrr": sum(r.rr for r in case_results) / n,
        "n_cases": n,
        "cases": case_results,
    }

    # DB 저장
    async with AsyncSessionLocal() as db:
        run = EvalRun(
            server_id=server_id,
            n_cases=n,
            hit_rate_1=metrics["hit_rate@1"],
            hit_rate_3=metrics["hit_rate@3"],
            hit_rate_5=metrics["hit_rate@5"],
            mrr=metrics["mrr"],
        )
        db.add(run)
        await db.flush()  # run.id 확보

        for result in case_results:
            case = EvalCase(
                run_id=run.id,
                name=result.name,
                stack_trace=result.stack_trace,
                expected_files=json.dumps(result.expected, ensure_ascii=False),
                hit_at_1=result.hit_at_1,
                hit_at_3=result.hit_at_3,
                hit_at_5=result.hit_at_5,
                rr=result.rr,
            )
            db.add(case)
            await db.flush()  # case.id 확보

            for rank, (path, preview) in enumerate(
                zip(result.retrieved, result.retrieved_previews), start=1
            ):
                db.add(EvalRetrieved(
                    case_id=case.id,
                    rank=rank,
                    file_path=path,
                    content_preview=preview,
                ))

        await db.commit()
        logger.info("[eval] saved run_id=%s to DB", run.id)
        metrics["run_id"] = run.id

    return metrics
