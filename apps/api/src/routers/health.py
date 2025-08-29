"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..database import get_db

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "synesthesia-ai-api"}


@router.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """Detailed health check with database connectivity."""
    try:
        # Test database connection
        result = await db.execute(text("SELECT 1"))
        db_healthy = result.scalar() == 1
    except Exception:
        db_healthy = False
    
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "service": "synesthesia-ai-api",
        "checks": {
            "database": "healthy" if db_healthy else "unhealthy"
        }
    }
