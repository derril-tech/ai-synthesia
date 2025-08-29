"""Brand kit management endpoints."""

from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database import get_db
from ..models.auth import User
from ..models.content import BrandKit
from ..routers.auth import get_current_user

router = APIRouter()


class ColorPalette(BaseModel):
    """Color palette schema."""
    primary: str
    secondary: str
    accent: str
    background: str
    text: str
    muted: str


class Typography(BaseModel):
    """Typography schema."""
    heading_font: str
    body_font: str
    mono_font: str
    heading_sizes: Dict[str, str]
    line_heights: Dict[str, float]


class Lexicon(BaseModel):
    """Brand lexicon schema."""
    preferred_terms: List[str]
    avoid_terms: List[str]
    tone_keywords: List[str]
    brand_voice: str


class SSMLPresets(BaseModel):
    """SSML voice presets."""
    default_voice: str
    speaking_rate: float
    pitch: str
    volume: str
    emphasis_style: str


class BrandKitCreate(BaseModel):
    """Brand kit creation schema."""
    project_id: str
    name: str
    color_palette: Optional[ColorPalette] = None
    typography: Optional[Typography] = None
    tone_guidelines: Optional[str] = None
    lexicon: Optional[Lexicon] = None
    logo_url: Optional[str] = None
    ssml_presets: Optional[SSMLPresets] = None


class BrandKitUpdate(BaseModel):
    """Brand kit update schema."""
    name: Optional[str] = None
    color_palette: Optional[ColorPalette] = None
    typography: Optional[Typography] = None
    tone_guidelines: Optional[str] = None
    lexicon: Optional[Lexicon] = None
    logo_url: Optional[str] = None
    ssml_presets: Optional[SSMLPresets] = None


class BrandKitResponse(BaseModel):
    """Brand kit response schema."""
    id: str
    project_id: str
    name: str
    color_palette: Optional[Dict[str, Any]]
    typography: Optional[Dict[str, Any]]
    tone_guidelines: Optional[str]
    lexicon: Optional[Dict[str, Any]]
    logo_url: Optional[str]
    ssml_presets: Optional[Dict[str, Any]]
    created_at: str
    updated_at: Optional[str]


@router.post("/", response_model=BrandKitResponse)
async def create_brand_kit(
    brand_kit_data: BrandKitCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new brand kit."""
    # TODO: Verify user has access to project
    
    brand_kit = BrandKit(
        project_id=UUID(brand_kit_data.project_id),
        name=brand_kit_data.name,
        color_palette=brand_kit_data.color_palette.dict() if brand_kit_data.color_palette else None,
        typography=brand_kit_data.typography.dict() if brand_kit_data.typography else None,
        tone_guidelines=brand_kit_data.tone_guidelines,
        lexicon=brand_kit_data.lexicon.dict() if brand_kit_data.lexicon else None,
        logo_url=brand_kit_data.logo_url,
    )
    
    db.add(brand_kit)
    await db.commit()
    await db.refresh(brand_kit)
    
    return BrandKitResponse(
        id=str(brand_kit.id),
        project_id=str(brand_kit.project_id),
        name=brand_kit.name,
        color_palette=brand_kit.color_palette,
        typography=brand_kit.typography,
        tone_guidelines=brand_kit.tone_guidelines,
        lexicon=brand_kit.lexicon,
        logo_url=brand_kit.logo_url,
        ssml_presets=None,  # TODO: Add SSML presets to model
        created_at=brand_kit.created_at.isoformat(),
        updated_at=brand_kit.updated_at.isoformat() if brand_kit.updated_at else None,
    )


@router.get("/", response_model=List[BrandKitResponse])
async def list_brand_kits(
    project_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List brand kits."""
    query = select(BrandKit)
    if project_id:
        query = query.where(BrandKit.project_id == UUID(project_id))
    
    result = await db.execute(query)
    brand_kits = result.scalars().all()
    
    return [
        BrandKitResponse(
            id=str(bk.id),
            project_id=str(bk.project_id),
            name=bk.name,
            color_palette=bk.color_palette,
            typography=bk.typography,
            tone_guidelines=bk.tone_guidelines,
            lexicon=bk.lexicon,
            logo_url=bk.logo_url,
            ssml_presets=None,
            created_at=bk.created_at.isoformat(),
            updated_at=bk.updated_at.isoformat() if bk.updated_at else None,
        )
        for bk in brand_kits
    ]


@router.get("/{brand_kit_id}", response_model=BrandKitResponse)
async def get_brand_kit(
    brand_kit_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific brand kit."""
    result = await db.execute(
        select(BrandKit).where(BrandKit.id == UUID(brand_kit_id))
    )
    brand_kit = result.scalar_one_or_none()
    
    if not brand_kit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brand kit not found"
        )
    
    return BrandKitResponse(
        id=str(brand_kit.id),
        project_id=str(brand_kit.project_id),
        name=brand_kit.name,
        color_palette=brand_kit.color_palette,
        typography=brand_kit.typography,
        tone_guidelines=brand_kit.tone_guidelines,
        lexicon=brand_kit.lexicon,
        logo_url=brand_kit.logo_url,
        ssml_presets=None,
        created_at=brand_kit.created_at.isoformat(),
        updated_at=brand_kit.updated_at.isoformat() if brand_kit.updated_at else None,
    )


@router.put("/{brand_kit_id}", response_model=BrandKitResponse)
async def update_brand_kit(
    brand_kit_id: str,
    brand_kit_data: BrandKitUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a brand kit."""
    result = await db.execute(
        select(BrandKit).where(BrandKit.id == UUID(brand_kit_id))
    )
    brand_kit = result.scalar_one_or_none()
    
    if not brand_kit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brand kit not found"
        )
    
    # Update fields
    if brand_kit_data.name is not None:
        brand_kit.name = brand_kit_data.name
    if brand_kit_data.color_palette is not None:
        brand_kit.color_palette = brand_kit_data.color_palette.dict()
    if brand_kit_data.typography is not None:
        brand_kit.typography = brand_kit_data.typography.dict()
    if brand_kit_data.tone_guidelines is not None:
        brand_kit.tone_guidelines = brand_kit_data.tone_guidelines
    if brand_kit_data.lexicon is not None:
        brand_kit.lexicon = brand_kit_data.lexicon.dict()
    if brand_kit_data.logo_url is not None:
        brand_kit.logo_url = brand_kit_data.logo_url
    
    await db.commit()
    await db.refresh(brand_kit)
    
    return BrandKitResponse(
        id=str(brand_kit.id),
        project_id=str(brand_kit.project_id),
        name=brand_kit.name,
        color_palette=brand_kit.color_palette,
        typography=brand_kit.typography,
        tone_guidelines=brand_kit.tone_guidelines,
        lexicon=brand_kit.lexicon,
        logo_url=brand_kit.logo_url,
        ssml_presets=None,
        created_at=brand_kit.created_at.isoformat(),
        updated_at=brand_kit.updated_at.isoformat() if brand_kit.updated_at else None,
    )


@router.delete("/{brand_kit_id}")
async def delete_brand_kit(
    brand_kit_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a brand kit."""
    result = await db.execute(
        select(BrandKit).where(BrandKit.id == UUID(brand_kit_id))
    )
    brand_kit = result.scalar_one_or_none()
    
    if not brand_kit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brand kit not found"
        )
    
    await db.delete(brand_kit)
    await db.commit()
    
    return {"message": "Brand kit deleted successfully"}
