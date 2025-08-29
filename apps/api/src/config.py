"""Configuration settings for Synesthesia AI API."""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="CORS allowed origins"
    )
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/synthesia",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=10, description="Database pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow")
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # S3/Object Storage
    s3_bucket: str = Field(default="synthesia-assets", description="S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="S3 region")
    aws_access_key_id: str = Field(default="", description="AWS access key")
    aws_secret_access_key: str = Field(default="", description="AWS secret key")
    
    # Authentication
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )
    
    # External APIs
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    
    # Observability
    sentry_dsn: str = Field(default="", description="Sentry DSN")
    
    # NATS
    nats_url: str = Field(
        default="nats://localhost:4222",
        description="NATS server URL"
    )
    
    # CrewAI
    crew_max_iterations: int = Field(default=5, description="Max CrewAI iterations")
    crew_timeout_seconds: int = Field(default=300, description="CrewAI timeout")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
