from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.server import Server
from app.schemas.server import ServerCreate, ServerResponse

router = APIRouter(prefix="/servers", tags=["servers"])


@router.post("", response_model=ServerResponse, status_code=201)
async def register_server(body: ServerCreate, db: AsyncSession = Depends(get_db)):
    server = Server(name=body.name, host=body.host)
    db.add(server)
    await db.commit()
    await db.refresh(server)
    return server


@router.get("", response_model=list[ServerResponse])
async def list_servers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Server).where(Server.is_active == True))
    return result.scalars().all()


@router.delete("/{server_id}", status_code=204)
async def deactivate_server(server_id: int, db: AsyncSession = Depends(get_db)):
    server = await db.get(Server, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    server.is_active = False
    await db.commit()
