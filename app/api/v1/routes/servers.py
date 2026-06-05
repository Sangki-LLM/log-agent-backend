import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.models.server import Server, ServerHost
from app.schemas.server import ServerCreate, ServerResponse, ServerUpdate
from app.services import git_service

router = APIRouter(prefix="/servers", tags=["servers"])


@router.post("", response_model=ServerResponse, status_code=201)
async def register_server(body: ServerCreate, db: AsyncSession = Depends(get_db)):
    server = Server(
        name=body.name,
        git_repo_url=body.git_repo_url,
        git_branch=body.git_branch,
        github_token=body.github_token or None,
    )
    db.add(server)
    await db.flush()

    for ip in body.hosts:
        db.add(ServerHost(server_id=server.id, host=ip.strip()))

    await db.commit()

    result = await db.execute(
        select(Server).where(Server.id == server.id).options(selectinload(Server.hosts))
    )
    saved = result.scalar_one()

    # 백그라운드에서 git clone (실패해도 등록은 완료)
    asyncio.create_task(_clone_repo(saved.id, saved.git_repo_url, saved.git_branch, saved.github_token or ""))

    return ServerResponse.from_orm_with_hosts(saved)


async def _clone_repo(server_id: int, repo_url: str, branch: str, token: str) -> None:
    try:
        await asyncio.to_thread(git_service.clone, server_id, repo_url, branch, token)
    except Exception:
        pass


@router.get("", response_model=list[ServerResponse])
async def list_servers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Server).where(Server.is_active == True).options(selectinload(Server.hosts))
    )
    return [ServerResponse.from_orm_with_hosts(s) for s in result.scalars().all()]


@router.put("/{server_id}", response_model=ServerResponse)
async def update_server(server_id: int, body: ServerUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Server).where(Server.id == server_id).options(selectinload(Server.hosts))
    )
    server = result.scalar_one_or_none()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    repo_changed = body.git_repo_url and body.git_repo_url != server.git_repo_url

    if body.name is not None:
        server.name = body.name
    if body.git_repo_url is not None:
        server.git_repo_url = body.git_repo_url
    if body.git_branch is not None:
        server.git_branch = body.git_branch
    if body.github_token is not None:
        server.github_token = body.github_token or None

    if body.hosts is not None:
        for host in server.hosts:
            await db.delete(host)
        for ip in body.hosts:
            db.add(ServerHost(server_id=server_id, host=ip.strip()))

    await db.commit()

    result = await db.execute(
        select(Server).where(Server.id == server_id).options(selectinload(Server.hosts))
    )
    saved = result.scalar_one()

    # repo URL이 바뀌면 기존 clone 삭제 후 재클론
    if repo_changed:
        import shutil
        from pathlib import Path
        repo_path = Path(settings.repos_path) / str(server_id)
        if repo_path.exists():
            shutil.rmtree(repo_path)
        asyncio.create_task(_clone_repo(saved.id, saved.git_repo_url, saved.git_branch, saved.github_token or ""))

    return ServerResponse.from_orm_with_hosts(saved)


@router.post("/{server_id}/hosts", response_model=ServerResponse)
async def add_host(server_id: int, host: str, db: AsyncSession = Depends(get_db)):
    server = await db.get(Server, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    db.add(ServerHost(server_id=server_id, host=host.strip()))
    await db.commit()

    result = await db.execute(
        select(Server).where(Server.id == server_id).options(selectinload(Server.hosts))
    )
    return ServerResponse.from_orm_with_hosts(result.scalar_one())


@router.delete("/{server_id}", status_code=204)
async def deactivate_server(server_id: int, db: AsyncSession = Depends(get_db)):
    server = await db.get(Server, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    server.is_active = False
    await db.commit()
