from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Server(Base):
    __tablename__ = "servers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    git_repo_url: Mapped[str] = mapped_column(String(500), nullable=False)
    git_branch: Mapped[str] = mapped_column(String(100), default="main")
    github_token: Mapped[str] = mapped_column(String(200), nullable=True)
    slack_webhook_url: Mapped[str] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    hosts: Mapped[list["ServerHost"]] = relationship(back_populates="server", cascade="all, delete-orphan")
    records: Mapped[list["AnalysisRecord"]] = relationship(back_populates="server")


class ServerHost(Base):
    __tablename__ = "server_hosts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    server_id: Mapped[int] = mapped_column(ForeignKey("servers.id"), nullable=False)
    host: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    server: Mapped["Server"] = relationship(back_populates="hosts")


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    server_id: Mapped[int] = mapped_column(ForeignKey("servers.id"), nullable=False)
    trigger_line: Mapped[str] = mapped_column(Text, nullable=False)
    raw_log: Mapped[str] = mapped_column(Text, nullable=False)
    llm_suggestion: Mapped[str] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    slack_ts: Mapped[str] = mapped_column(String(50), nullable=True)
    github_pr_url: Mapped[str] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    server: Mapped["Server"] = relationship(back_populates="records")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    server_id: Mapped[int] = mapped_column(ForeignKey("servers.id"), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    n_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    hit_rate_1: Mapped[float] = mapped_column(Float, nullable=False)
    hit_rate_3: Mapped[float] = mapped_column(Float, nullable=False)
    hit_rate_5: Mapped[float] = mapped_column(Float, nullable=False)
    mrr: Mapped[float] = mapped_column(Float, nullable=False)

    cases: Mapped[list["EvalCase"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    stack_trace: Mapped[str] = mapped_column(Text, nullable=False)
    expected_files: Mapped[str] = mapped_column(Text, nullable=False)   # JSON
    hit_at_1: Mapped[float] = mapped_column(Float, nullable=False)
    hit_at_3: Mapped[float] = mapped_column(Float, nullable=False)
    hit_at_5: Mapped[float] = mapped_column(Float, nullable=False)
    rr: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped["EvalRun"] = relationship(back_populates="cases")
    retrieved: Mapped[list["EvalRetrieved"]] = relationship(back_populates="case", cascade="all, delete-orphan")


class EvalRetrieved(Base):
    __tablename__ = "eval_retrieved"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("eval_cases.id"), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)          # 1-based
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    content_preview: Mapped[str] = mapped_column(Text, nullable=True)   # 파일 앞 500자

    case: Mapped["EvalCase"] = relationship(back_populates="retrieved")
