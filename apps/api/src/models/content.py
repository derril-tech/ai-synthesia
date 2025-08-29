"""Content and generation models."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from ..database import Base


class ProjectStatus(str, Enum):
    """Project status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class AssetType(str, Enum):
    """Asset type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class StoryPackStatus(str, Enum):
    """Story pack generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Project(Base):
    """Project model for organizing story packs."""
    
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default=ProjectStatus.DRAFT)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    workspace = relationship("Workspace", back_populates="projects")
    story_packs = relationship("StoryPack", back_populates="project")
    brand_kits = relationship("BrandKit", back_populates="project")


class BrandKit(Base):
    """Brand kit for consistent styling and tone."""
    
    __tablename__ = "brand_kits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    
    # Brand elements
    color_palette = Column(JSON, nullable=True)  # {"primary": "#hex", "secondary": "#hex", ...}
    typography = Column(JSON, nullable=True)     # {"heading": "font-family", "body": "font-family", ...}
    tone_guidelines = Column(Text, nullable=True)
    lexicon = Column(JSON, nullable=True)        # {"preferred": ["word1"], "avoid": ["word2"]}
    logo_url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="brand_kits")


class Asset(Base):
    """Asset storage with embeddings."""
    
    __tablename__ = "assets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    type = Column(String(20), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Embeddings for similarity search
    embedding = Column(Vector(1536), nullable=True)  # OpenAI ada-002 dimension
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class StoryPack(Base):
    """Generated story pack with multi-modal content."""
    
    __tablename__ = "story_packs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default=StoryPackStatus.PENDING)
    
    # Generated content
    script = Column(Text, nullable=True)
    cover_image_url = Column(String(500), nullable=True)
    scene_images = Column(JSON, nullable=True)  # List of image URLs
    audio_url = Column(String(500), nullable=True)
    video_url = Column(String(500), nullable=True)
    
    # Generation metadata
    generation_config = Column(JSON, nullable=True)
    generation_log = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="story_packs")
    evaluations = relationship("Evaluation", back_populates="story_pack")


class Evaluation(Base):
    """Evaluation scores for story packs."""
    
    __tablename__ = "evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_pack_id = Column(UUID(as_uuid=True), ForeignKey("story_packs.id"), nullable=False)
    
    # Alignment scores
    text_image_alignment = Column(Float, nullable=True)
    text_audio_alignment = Column(Float, nullable=True)
    brand_consistency = Column(Float, nullable=True)
    overall_quality = Column(Float, nullable=True)
    
    # Safety scores
    safety_score = Column(Float, nullable=True)
    toxicity_score = Column(Float, nullable=True)
    
    # Detailed metrics
    metrics = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    story_pack = relationship("StoryPack", back_populates="evaluations")
