from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Server(Base):
    __tablename__ = "servers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    git_repo_url: Mapped[str] = mapped_column(String(500), nullable=False)
    git_branch: Mapped[str] = mapped_column(String(100), default="main")
    github_token: Mapped[str] = mapped_column(String(200), nullable=True)
    current_commit: Mapped[str] = mapped_column(String(40), nullable=True)
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
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    server: Mapped["Server"] = relationship(back_populates="records")
