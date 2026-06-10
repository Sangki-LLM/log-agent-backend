import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.evaluation.rag_eval import TestCase, evaluate
from app.models.server import EvalCase, EvalRetrieved, EvalRun

router = APIRouter(prefix="/eval", tags=["eval"])

_TEST_CASES = [
    TestCase(
        name="NullPointerException in TestErrorController",
        stack_trace=(
            "java.lang.NullPointerException: Cannot invoke \"String.toUpperCase()\" because \"value\" is null\n"
            "\tat com.puppynoteserver.global.TestErrorController.triggerNpe(TestErrorController.java:21)"
        ),
        expected_files=["src/main/java/com/puppynoteserver/global/TestErrorController.java"],
    ),
    TestCase(
        name="NoSuchElementException in UserService",
        stack_trace=(
            "java.util.NoSuchElementException: User not found\n"
            "\tat com.puppynoteserver.domain.user.service.UserService.findById(UserService.java:34)"
        ),
        expected_files=["src/main/java/com/puppynoteserver/domain/user/service/UserService.java"],
    ),
    TestCase(
        name="DataIntegrityViolationException in PostService",
        stack_trace=(
            "org.springframework.dao.DataIntegrityViolationException: could not execute statement\n"
            "\tat com.puppynoteserver.domain.post.service.PostService.create(PostService.java:52)"
        ),
        expected_files=["src/main/java/com/puppynoteserver/domain/post/service/PostService.java"],
    ),
]


@router.post("/rag/{server_id}")
async def run_rag_eval(server_id: int, n_results: int = 5):
    """RAG 검색 품질 평가 실행 후 DB에 저장."""
    metrics = await evaluate(server_id, _TEST_CASES, n_results=n_results)

    return {
        "run_id": metrics["run_id"],
        "server_id": server_id,
        "n_cases": metrics["n_cases"],
        "hit_rate@1": round(metrics["hit_rate@1"], 3),
        "hit_rate@3": round(metrics["hit_rate@3"], 3),
        "hit_rate@5": round(metrics["hit_rate@5"], 3),
        "mrr": round(metrics["mrr"], 3),
        "cases": [
            {
                "name": r.name,
                "expected": r.expected,
                "retrieved": [
                    {"rank": i + 1, "path": p, "preview": r.retrieved_previews[i][:200]}
                    for i, p in enumerate(r.retrieved)
                ],
                "hit@1": r.hit_at_1,
                "hit@3": r.hit_at_3,
                "hit@5": r.hit_at_5,
                "rr": round(r.rr, 3),
            }
            for r in metrics["cases"]
        ],
    }


@router.get("/rag/{server_id}/history")
async def get_eval_history(server_id: int, db: AsyncSession = Depends(get_db)):
    """서버의 평가 실행 이력 조회."""
    result = await db.execute(
        select(EvalRun)
        .where(EvalRun.server_id == server_id)
        .order_by(EvalRun.evaluated_at.desc())
    )
    runs = result.scalars().all()

    return [
        {
            "run_id": r.id,
            "evaluated_at": r.evaluated_at,
            "n_cases": r.n_cases,
            "hit_rate@1": round(r.hit_rate_1, 3),
            "hit_rate@3": round(r.hit_rate_3, 3),
            "hit_rate@5": round(r.hit_rate_5, 3),
            "mrr": round(r.mrr, 3),
        }
        for r in runs
    ]


@router.get("/rag/run/{run_id}")
async def get_eval_run_detail(run_id: int, db: AsyncSession = Depends(get_db)):
    """특정 평가 실행의 케이스별 상세 결과 조회 (검색된 파일 내용 포함)."""
    result = await db.execute(
        select(EvalRun)
        .where(EvalRun.id == run_id)
        .options(
            selectinload(EvalRun.cases).selectinload(EvalCase.retrieved)
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    return {
        "run_id": run.id,
        "server_id": run.server_id,
        "evaluated_at": run.evaluated_at,
        "hit_rate@1": round(run.hit_rate_1, 3),
        "hit_rate@3": round(run.hit_rate_3, 3),
        "hit_rate@5": round(run.hit_rate_5, 3),
        "mrr": round(run.mrr, 3),
        "cases": [
            {
                "name": c.name,
                "stack_trace": c.stack_trace,
                "expected_files": json.loads(c.expected_files),
                "hit@1": c.hit_at_1,
                "hit@3": c.hit_at_3,
                "hit@5": c.hit_at_5,
                "rr": round(c.rr, 3),
                "retrieved": [
                    {
                        "rank": r.rank,
                        "file_path": r.file_path,
                        "content_preview": r.content_preview,
                    }
                    for r in sorted(c.retrieved, key=lambda x: x.rank)
                ],
            }
            for c in run.cases
        ],
    }
