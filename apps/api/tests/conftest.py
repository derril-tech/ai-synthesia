"""
Pytest configuration and shared fixtures for Synesthesia AI API tests
"""

import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock

from src.main import app
from src.database import get_db, Base
from src.models.auth import User, Organization, Workspace
from src.models.content import Project, StoryPack, BrandKit


# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = sessionmaker(
    test_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with TestingSessionLocal() as session:
        yield session
    
    # Drop tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def override_get_db(db_session: AsyncSession):
    """Override the get_db dependency."""
    async def _override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client(override_get_db) -> AsyncGenerator[AsyncClient, None]:
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def test_organization(db_session: AsyncSession) -> Organization:
    """Create a test organization."""
    org = Organization(
        name="Test Organization",
        slug="test-org",
        description="Test organization for testing"
    )
    db_session.add(org)
    await db_session.commit()
    await db_session.refresh(org)
    return org


@pytest.fixture
async def test_workspace(db_session: AsyncSession, test_organization: Organization) -> Workspace:
    """Create a test workspace."""
    workspace = Workspace(
        organization_id=test_organization.id,
        name="Test Workspace",
        slug="test-workspace",
        description="Test workspace for testing"
    )
    db_session.add(workspace)
    await db_session.commit()
    await db_session.refresh(workspace)
    return workspace


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        full_name="Test User",
        is_active=True,
        is_verified=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_project(db_session: AsyncSession, test_workspace: Workspace) -> Project:
    """Create a test project."""
    project = Project(
        workspace_id=test_workspace.id,
        name="Test Project",
        description="Test project for testing"
    )
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.fixture
async def test_brand_kit(db_session: AsyncSession, test_project: Project) -> BrandKit:
    """Create a test brand kit."""
    brand_kit = BrandKit(
        project_id=test_project.id,
        name="Test Brand Kit",
        color_palette={
            "primary": "#3B82F6",
            "secondary": "#6B7280",
            "accent": "#F59E0B",
            "background": "#FFFFFF",
            "text": "#1F2937"
        },
        typography={
            "heading_font": "Inter",
            "body_font": "Inter",
            "heading_sizes": {"h1": "2.5rem", "h2": "2rem"}
        }
    )
    db_session.add(brand_kit)
    await db_session.commit()
    await db_session.refresh(brand_kit)
    return brand_kit


@pytest.fixture
async def test_story_pack(db_session: AsyncSession, test_project: Project) -> StoryPack:
    """Create a test story pack."""
    story_pack = StoryPack(
        project_id=test_project.id,
        name="Test Story Pack",
        prompt="Create a test story",
        status="completed",
        generated_content={
            "text": "This is a test story about innovation and creativity.",
            "images": ["base64_image_data_1", "base64_image_data_2"],
            "audio": {"mixed_audio": "base64_audio_data"}
        }
    )
    db_session.add(story_pack)
    await db_session.commit()
    await db_session.refresh(story_pack)
    return story_pack


@pytest.fixture
def mock_current_user(test_user: User):
    """Mock current user for authentication."""
    return Mock(
        id=test_user.id,
        email=test_user.email,
        full_name=test_user.full_name,
        is_active=test_user.is_active
    )


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Create authentication headers for test user."""
    # In a real implementation, this would be a valid JWT token
    token = f"test_jwt_token_for_{test_user.id}"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    
    # Mock chat completion
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock AI response"
    mock_client.chat.completions.create.return_value = mock_response
    
    # Mock image generation
    mock_image_response = Mock()
    mock_image_response.data = [Mock()]
    mock_image_response.data[0].url = "https://example.com/mock_image.png"
    mock_client.images.generate.return_value = mock_image_response
    
    return mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = Mock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    return mock_client


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    mock_client = Mock()
    
    # Mock upload
    mock_client.upload_fileobj.return_value = None
    
    # Mock generate presigned URL
    mock_client.generate_presigned_url.return_value = "https://example.com/presigned_url"
    
    # Mock download
    mock_client.download_fileobj.return_value = None
    
    return mock_client


# Test data fixtures

@pytest.fixture
def sample_story_pack_data():
    """Sample story pack data for testing."""
    return {
        "name": "Sample Story Pack",
        "prompt": "Create an engaging story about artificial intelligence and human creativity",
        "settings": {
            "narrative_style": "engaging",
            "target_audience": "general",
            "include_images": True,
            "include_audio": True,
            "image_count": 3
        }
    }


@pytest.fixture
def sample_brand_kit_data():
    """Sample brand kit data for testing."""
    return {
        "name": "Sample Brand Kit",
        "color_palette": {
            "primary": "#3B82F6",
            "secondary": "#6B7280",
            "accent": "#F59E0B",
            "background": "#FFFFFF",
            "text": "#1F2937",
            "muted": "#9CA3AF"
        },
        "typography": {
            "heading_font": "Inter",
            "body_font": "Inter",
            "mono_font": "JetBrains Mono",
            "heading_sizes": {
                "h1": "2.5rem",
                "h2": "2rem",
                "h3": "1.5rem"
            },
            "line_heights": {
                "heading": 1.2,
                "body": 1.6
            }
        },
        "lexicon": {
            "brand_voice": "professional",
            "tone": "friendly",
            "preferred_terms": ["innovative", "creative", "efficient"],
            "avoided_terms": ["cheap", "basic", "simple"]
        }
    }


@pytest.fixture
def sample_automation_preset_data():
    """Sample automation preset data for testing."""
    return {
        "name": "Sample Automation Preset",
        "description": "A sample preset for testing automation",
        "preset_type": "story_pack",
        "configuration": {
            "narrative_style": "engaging",
            "image_count": 3,
            "include_audio": True,
            "quality_threshold": 0.8,
            "max_retries": 3
        },
        "tags": ["sample", "test", "automation"]
    }


# Performance testing fixtures

@pytest.fixture
def performance_test_data():
    """Data for performance testing."""
    return {
        "concurrent_users": 10,
        "requests_per_user": 5,
        "max_response_time": 5.0,  # seconds
        "success_rate_threshold": 0.95
    }


# Mock external services

@pytest.fixture
def mock_external_services(mock_openai_client, mock_redis_client, mock_s3_client):
    """Mock all external services."""
    return {
        "openai": mock_openai_client,
        "redis": mock_redis_client,
        "s3": mock_s3_client
    }


# Database utilities

async def create_test_data(db_session: AsyncSession, count: int = 10):
    """Create test data for performance testing."""
    organizations = []
    workspaces = []
    projects = []
    
    for i in range(count):
        # Create organization
        org = Organization(
            name=f"Test Org {i}",
            slug=f"test-org-{i}",
            description=f"Test organization {i}"
        )
        db_session.add(org)
        organizations.append(org)
        
        # Create workspace
        workspace = Workspace(
            organization_id=org.id,
            name=f"Test Workspace {i}",
            slug=f"test-workspace-{i}",
            description=f"Test workspace {i}"
        )
        db_session.add(workspace)
        workspaces.append(workspace)
        
        # Create project
        project = Project(
            workspace_id=workspace.id,
            name=f"Test Project {i}",
            description=f"Test project {i}"
        )
        db_session.add(project)
        projects.append(project)
    
    await db_session.commit()
    
    return {
        "organizations": organizations,
        "workspaces": workspaces,
        "projects": projects
    }


@pytest.fixture
async def test_data_set(db_session: AsyncSession):
    """Create a set of test data."""
    return await create_test_data(db_session, count=5)


# Cleanup utilities

@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    # Cleanup code would go here if needed
    # For now, we're using a fresh database for each test


# Test markers

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "external: mark test as requiring external services")
