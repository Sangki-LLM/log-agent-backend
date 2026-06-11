from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={
        "ssl": None,          # useSSL=false
        "auth_plugin": "",    # allowPublicKeyRetrieval (mysql_native_password fallback)
    },
    pool_pre_ping=True,
    pool_recycle=1800,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate(conn)


async def _migrate(conn) -> None:
    """기존 테이블에 새 컬럼을 추가합니다 (이미 존재하면 무시)."""
    new_columns = [
        "ALTER TABLE analysis_records ADD COLUMN github_pr_url VARCHAR(500)",
        "ALTER TABLE servers ADD COLUMN slack_webhook_url VARCHAR(500)",
        "ALTER TABLE analysis_records ADD COLUMN judge_score INT",
        "ALTER TABLE analysis_records ADD COLUMN judge_confidence VARCHAR(20)",
        "ALTER TABLE analysis_records ADD COLUMN judge_reason TEXT",
    ]
    for stmt in new_columns:
        try:
            await conn.execute(text(stmt))
        except Exception:
            pass
