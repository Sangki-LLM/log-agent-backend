import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.server import Server, ServerHost
from app.schemas.server import ServerCreate, ServerResponse
from app.services import git_service

router = APIRouter(prefix="/servers", tags=["servers"])


@router.post("", response_model=ServerResponse, status_code=201)
async def register_server(body: ServerCreate, db: AsyncSession = Depends(get_db)):
    server = Server(name=body.name, git_repo_url=body.git_repo_url, git_branch=body.git_branch)
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
    asyncio.create_task(_clone_repo(saved.id, saved.git_repo_url, saved.git_branch))

    return ServerResponse.from_orm_with_hosts(saved)


async def _clone_repo(server_id: int, repo_url: str, branch: str) -> None:
    try:
        await asyncio.to_thread(git_service.clone, server_id, repo_url, branch)
    except Exception:
        pass


@router.get("", response_model=list[ServerResponse])
async def list_servers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Server).where(Server.is_active == True).options(selectinload(Server.hosts))
    )
    return [ServerResponse.from_orm_with_hosts(s) for s in result.scalars().all()]


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
