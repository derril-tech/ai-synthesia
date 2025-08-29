"""Asset management endpoints with signed URL uploads."""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..config import get_settings
from ..database import get_db
from ..models.auth import User
from ..models.content import Asset, AssetType
from ..routers.auth import get_current_user

router = APIRouter()
settings = get_settings()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    region_name=settings.s3_region,
)


class AssetUploadRequest(BaseModel):
    """Asset upload request schema."""
    filename: str
    content_type: str
    workspace_id: str


class AssetUploadResponse(BaseModel):
    """Asset upload response schema."""
    upload_url: str
    asset_id: str
    fields: dict


class AssetResponse(BaseModel):
    """Asset response schema."""
    id: str
    workspace_id: str
    type: AssetType
    filename: str
    file_path: str
    file_size: Optional[int]
    mime_type: Optional[str]
    metadata: Optional[dict]
    created_at: str
    updated_at: Optional[str]


@router.post("/upload-url", response_model=AssetUploadResponse)
async def create_upload_url(
    upload_request: AssetUploadRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a signed URL for direct S3 upload."""
    # TODO: Verify user has access to workspace
    
    # Generate unique asset ID and file path
    asset_id = str(uuid.uuid4())
    file_extension = upload_request.filename.split('.')[-1] if '.' in upload_request.filename else ''
    s3_key = f"workspaces/{upload_request.workspace_id}/assets/{asset_id}.{file_extension}"
    
    # Create asset record in database
    asset_type = get_asset_type_from_mime(upload_request.content_type)
    
    asset = Asset(
        id=UUID(asset_id),
        workspace_id=UUID(upload_request.workspace_id),
        type=asset_type,
        filename=upload_request.filename,
        file_path=s3_key,
        mime_type=upload_request.content_type,
    )
    
    db.add(asset)
    await db.commit()
    
    try:
        # Generate presigned POST URL
        presigned_post = s3_client.generate_presigned_post(
            Bucket=settings.s3_bucket,
            Key=s3_key,
            Fields={
                'Content-Type': upload_request.content_type,
                'x-amz-meta-asset-id': asset_id,
                'x-amz-meta-workspace-id': upload_request.workspace_id,
            },
            Conditions=[
                {'Content-Type': upload_request.content_type},
                ['content-length-range', 1, 100 * 1024 * 1024],  # 1 byte to 100MB
            ],
            ExpiresIn=3600,  # 1 hour
        )
        
        return AssetUploadResponse(
            upload_url=presigned_post['url'],
            asset_id=asset_id,
            fields=presigned_post['fields'],
        )
        
    except ClientError as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.post("/upload-complete/{asset_id}")
async def complete_upload(
    asset_id: str,
    file_size: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Mark upload as complete and update asset metadata."""
    result = await db.execute(
        select(Asset).where(Asset.id == UUID(asset_id))
    )
    asset = result.scalar_one_or_none()
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    # Update asset with file size
    asset.file_size = file_size
    
    # TODO: Trigger processing pipeline (normalization, embedding generation)
    
    await db.commit()
    await db.refresh(asset)
    
    return {"message": "Upload completed successfully"}


@router.get("/", response_model=List[AssetResponse])
async def list_assets(
    workspace_id: Optional[str] = None,
    asset_type: Optional[AssetType] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List assets with optional filtering."""
    query = select(Asset).offset(offset).limit(limit)
    
    if workspace_id:
        query = query.where(Asset.workspace_id == UUID(workspace_id))
    
    if asset_type:
        query = query.where(Asset.type == asset_type)
    
    result = await db.execute(query)
    assets = result.scalars().all()
    
    return [
        AssetResponse(
            id=str(asset.id),
            workspace_id=str(asset.workspace_id),
            type=asset.type,
            filename=asset.filename,
            file_path=asset.file_path,
            file_size=asset.file_size,
            mime_type=asset.mime_type,
            metadata=asset.metadata,
            created_at=asset.created_at.isoformat(),
            updated_at=asset.updated_at.isoformat() if asset.updated_at else None,
        )
        for asset in assets
    ]


@router.get("/{asset_id}", response_model=AssetResponse)
async def get_asset(
    asset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific asset."""
    result = await db.execute(
        select(Asset).where(Asset.id == UUID(asset_id))
    )
    asset = result.scalar_one_or_none()
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    return AssetResponse(
        id=str(asset.id),
        workspace_id=str(asset.workspace_id),
        type=asset.type,
        filename=asset.filename,
        file_path=asset.file_path,
        file_size=asset.file_size,
        mime_type=asset.mime_type,
        metadata=asset.metadata,
        created_at=asset.created_at.isoformat(),
        updated_at=asset.updated_at.isoformat() if asset.updated_at else None,
    )


@router.get("/{asset_id}/download-url")
async def get_download_url(
    asset_id: str,
    expires_in: int = 3600,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a signed URL for asset download."""
    result = await db.execute(
        select(Asset).where(Asset.id == UUID(asset_id))
    )
    asset = result.scalar_one_or_none()
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    try:
        download_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.s3_bucket, 'Key': asset.file_path},
            ExpiresIn=expires_in,
        )
        
        return {"download_url": download_url, "expires_in": expires_in}
        
    except ClientError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate download URL: {str(e)}"
        )


@router.delete("/{asset_id}")
async def delete_asset(
    asset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete an asset."""
    result = await db.execute(
        select(Asset).where(Asset.id == UUID(asset_id))
    )
    asset = result.scalar_one_or_none()
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    # Delete from S3
    try:
        s3_client.delete_object(Bucket=settings.s3_bucket, Key=asset.file_path)
    except ClientError as e:
        # Log error but continue with database deletion
        print(f"Failed to delete S3 object: {e}")
    
    # Delete from database
    await db.delete(asset)
    await db.commit()
    
    return {"message": "Asset deleted successfully"}


def get_asset_type_from_mime(mime_type: str) -> AssetType:
    """Determine asset type from MIME type."""
    if mime_type.startswith('text/'):
        return AssetType.TEXT
    elif mime_type.startswith('image/'):
        return AssetType.IMAGE
    elif mime_type.startswith('audio/'):
        return AssetType.AUDIO
    elif mime_type.startswith('video/'):
        return AssetType.VIDEO
    else:
        return AssetType.TEXT  # Default fallback
