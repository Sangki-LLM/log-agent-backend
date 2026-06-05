import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import analysis, servers, webhook
from app.core.config import settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="AI Log Analysis Agent",
    description="Event-driven DevSecOps agent: SSH → LLM → Slack → git push",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://log-agent.sangkihan.co.kr", "http://localhost:6000", "http://localhost:6001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(servers.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(webhook.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "ollama_host": settings.ollama_host,
    }
