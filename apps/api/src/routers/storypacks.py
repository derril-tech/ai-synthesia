"""Story pack generation endpoints."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database import get_db
from ..models.auth import User
from ..models.content import StoryPack, StoryPackStatus
from ..routers.auth import get_current_user

router = APIRouter()


class StoryPackCreate(BaseModel):
    """Story pack creation schema."""
    project_id: str
    name: str
    prompt: str
    generation_config: Optional[dict] = None


class StoryPackResponse(BaseModel):
    """Story pack response schema."""
    id: str
    project_id: str
    name: str
    prompt: str
    status: StoryPackStatus
    script: Optional[str]
    cover_image_url: Optional[str]
    scene_images: Optional[List[str]]
    audio_url: Optional[str]
    video_url: Optional[str]
    created_at: str
    updated_at: Optional[str]


@router.post("/generate", response_model=StoryPackResponse)
async def generate_story_pack(
    story_data: StoryPackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a new story pack."""
    # TODO: Verify user has access to project
    
    story_pack = StoryPack(
        project_id=UUID(story_data.project_id),
        name=story_data.name,
        prompt=story_data.prompt,
        generation_config=story_data.generation_config,
        status=StoryPackStatus.PENDING,
    )
    
    db.add(story_pack)
    await db.commit()
    await db.refresh(story_pack)
    
    # TODO: Trigger async generation via NATS/CrewAI
    
    return StoryPackResponse(
        id=str(story_pack.id),
        project_id=str(story_pack.project_id),
        name=story_pack.name,
        prompt=story_pack.prompt,
        status=story_pack.status,
        script=story_pack.script,
        cover_image_url=story_pack.cover_image_url,
        scene_images=story_pack.scene_images,
        audio_url=story_pack.audio_url,
        video_url=story_pack.video_url,
        created_at=story_pack.created_at.isoformat(),
        updated_at=story_pack.updated_at.isoformat() if story_pack.updated_at else None,
    )


@router.get("/", response_model=List[StoryPackResponse])
async def list_story_packs(
    project_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List story packs."""
    query = select(StoryPack)
    if project_id:
        query = query.where(StoryPack.project_id == UUID(project_id))
    
    result = await db.execute(query)
    story_packs = result.scalars().all()
    
    return [
        StoryPackResponse(
            id=str(sp.id),
            project_id=str(sp.project_id),
            name=sp.name,
            prompt=sp.prompt,
            status=sp.status,
            script=sp.script,
            cover_image_url=sp.cover_image_url,
            scene_images=sp.scene_images,
            audio_url=sp.audio_url,
            video_url=sp.video_url,
            created_at=sp.created_at.isoformat(),
            updated_at=sp.updated_at.isoformat() if sp.updated_at else None,
        )
        for sp in story_packs
    ]


@router.get("/{story_pack_id}", response_model=StoryPackResponse)
async def get_story_pack(
    story_pack_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific story pack."""
    result = await db.execute(
        select(StoryPack).where(StoryPack.id == UUID(story_pack_id))
    )
    story_pack = result.scalar_one_or_none()
    
    if not story_pack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story pack not found"
        )
    
    return StoryPackResponse(
        id=str(story_pack.id),
        project_id=str(story_pack.project_id),
        name=story_pack.name,
        prompt=story_pack.prompt,
        status=story_pack.status,
        script=story_pack.script,
        cover_image_url=story_pack.cover_image_url,
        scene_images=story_pack.scene_images,
        audio_url=story_pack.audio_url,
        video_url=story_pack.video_url,
        created_at=story_pack.created_at.isoformat(),
        updated_at=story_pack.updated_at.isoformat() if story_pack.updated_at else None,
    )
