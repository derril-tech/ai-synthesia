"""
Test suite for Synesthesia AI API
Comprehensive testing including unit, integration, and E2E tests
"""

import pytest
import asyncio
from datetime import datetime
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch, AsyncMock

from src.main import app
from src.database import get_db, Base
from src.models.auth import User, Organization, Workspace
from src.models.content import Project, StoryPack, BrandKit
from src.config import get_settings


# Test configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True
)

TestingSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture
async def db_session():
    """Create a test database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def override_get_db(db_session):
    """Override the get_db dependency."""
    def _override_get_db():
        return db_session
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client(override_get_db):
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password="hashed_password",
        full_name="Test User",
        is_active=True,
        is_verified=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_organization(db_session):
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
async def test_workspace(db_session, test_organization):
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
async def test_project(db_session, test_workspace):
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
async def auth_headers(test_user):
    """Create authentication headers for test user."""
    # Mock JWT token for testing
    token = "test_jwt_token"
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    async def test_basic_health_check(self, client):
        """Test basic health check."""
        response = await client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "synesthesia-ai-api"
    
    async def test_detailed_health_check(self, client):
        """Test detailed health check."""
        response = await client.get("/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "database" in data["checks"]


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    async def test_user_registration(self, client):
        """Test user registration."""
        user_data = {
            "email": "newuser@example.com",
            "password": "testpassword123",
            "full_name": "New User"
        }
        
        with patch('src.routers.auth.get_password_hash') as mock_hash:
            mock_hash.return_value = "hashed_password"
            
            response = await client.post("/v1/auth/register", json=user_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["email"] == user_data["email"]
            assert data["full_name"] == user_data["full_name"]
            assert data["is_active"] is True
    
    async def test_user_login(self, client, test_user):
        """Test user login."""
        login_data = {
            "email": test_user.email,
            "password": "testpassword123"
        }
        
        with patch('src.routers.auth.verify_password') as mock_verify:
            mock_verify.return_value = True
            
            with patch('src.routers.auth.create_access_token') as mock_token:
                mock_token.return_value = "test_jwt_token"
                
                response = await client.post("/v1/auth/login", json=login_data)
                assert response.status_code == 200
                
                data = response.json()
                assert data["access_token"] == "test_jwt_token"
                assert data["token_type"] == "bearer"
    
    async def test_invalid_login(self, client):
        """Test login with invalid credentials."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        
        response = await client.post("/v1/auth/login", json=login_data)
        assert response.status_code == 401


class TestProjectEndpoints:
    """Test project management endpoints."""
    
    async def test_create_project(self, client, auth_headers, test_workspace):
        """Test project creation."""
        project_data = {
            "name": "New Test Project",
            "description": "A new test project",
            "workspace_id": str(test_workspace.id)
        }
        
        with patch('src.routers.projects.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.post(
                "/v1/projects/",
                json=project_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["name"] == project_data["name"]
            assert data["workspace_id"] == project_data["workspace_id"]
    
    async def test_list_projects(self, client, auth_headers, test_project):
        """Test project listing."""
        with patch('src.routers.projects.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.get("/v1/projects/", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
    
    async def test_get_project(self, client, auth_headers, test_project):
        """Test getting a specific project."""
        with patch('src.routers.projects.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.get(
                f"/v1/projects/{test_project.id}",
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["name"] == test_project.name


class TestStoryPackEndpoints:
    """Test story pack generation endpoints."""
    
    async def test_generate_story_pack(self, client, auth_headers, test_project):
        """Test story pack generation."""
        story_data = {
            "project_id": str(test_project.id),
            "name": "Test Story Pack",
            "prompt": "Create a story about AI and creativity"
        }
        
        with patch('src.routers.storypacks.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.post(
                "/v1/storypacks/generate",
                json=story_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["name"] == story_data["name"]
            assert data["prompt"] == story_data["prompt"]
            assert data["status"] == "pending"
    
    async def test_list_story_packs(self, client, auth_headers):
        """Test story pack listing."""
        with patch('src.routers.storypacks.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.get("/v1/storypacks/", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)


class TestBrandKitEndpoints:
    """Test brand kit management endpoints."""
    
    async def test_create_brand_kit(self, client, auth_headers, test_project):
        """Test brand kit creation."""
        brand_kit_data = {
            "project_id": str(test_project.id),
            "name": "Test Brand Kit",
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
                "heading_sizes": {"h1": "2.5rem", "h2": "2rem"},
                "line_heights": {"heading": 1.2, "body": 1.6}
            }
        }
        
        with patch('src.routers.brand_kits.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.post(
                "/v1/brand-kits/",
                json=brand_kit_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["name"] == brand_kit_data["name"]
            assert data["color_palette"] == brand_kit_data["color_palette"]


class TestReportsEndpoints:
    """Test reporting endpoints."""
    
    async def test_dashboard_metrics(self, client, auth_headers):
        """Test dashboard metrics endpoint."""
        with patch('src.routers.reports.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.get(
                "/v1/reports/dashboard-metrics",
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "summary" in data
            assert "quality_metrics" in data
            assert "trends" in data
    
    async def test_brand_fit_report(self, client, auth_headers, test_project):
        """Test brand fit report generation."""
        report_filter = {
            "project_id": str(test_project.id),
            "date_from": "2024-01-01T00:00:00",
            "date_to": "2024-12-31T23:59:59"
        }
        
        with patch('src.routers.reports.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            # Mock empty story packs to avoid 404
            with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
                mock_result = Mock()
                mock_result.scalars.return_value.all.return_value = []
                mock_execute.return_value = mock_result
                
                response = await client.post(
                    "/v1/reports/brand-fit",
                    json=report_filter,
                    headers=auth_headers
                )
                # Expect 404 since no story packs exist
                assert response.status_code == 404


class TestGovernanceEndpoints:
    """Test governance and review endpoints."""
    
    async def test_create_review(self, client, auth_headers):
        """Test review creation."""
        review_data = {
            "story_pack_id": "test_story_pack_id",
            "reviewers": ["reviewer1", "reviewer2"],
            "instructions": "Please review for quality and brand alignment",
            "priority": "high"
        }
        
        with patch('src.routers.governance.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id", full_name="Test User")
            
            # Mock story pack exists
            with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
                mock_result = Mock()
                mock_result.scalar_one_or_none.return_value = Mock(id="test_story_pack_id")
                mock_execute.return_value = mock_result
                
                response = await client.post(
                    "/v1/governance/reviews",
                    json=review_data,
                    headers=auth_headers
                )
                assert response.status_code == 200
                
                data = response.json()
                assert data["story_pack_id"] == review_data["story_pack_id"]
                assert len(data["reviewers"]) == 2
    
    async def test_governance_dashboard(self, client, auth_headers):
        """Test governance dashboard."""
        with patch('src.routers.governance.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.get(
                "/v1/governance/governance-dashboard",
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "summary" in data
            assert "review_breakdown" in data
            assert "activity_breakdown" in data


class TestAutomationEndpoints:
    """Test automation endpoints."""
    
    async def test_create_preset(self, client, auth_headers):
        """Test automation preset creation."""
        preset_data = {
            "name": "Test Preset",
            "description": "A test automation preset",
            "preset_type": "story_pack",
            "configuration": {
                "narrative_style": "engaging",
                "image_count": 3,
                "include_audio": True
            },
            "tags": ["test", "automation"]
        }
        
        with patch('src.routers.automation.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.post(
                "/v1/automation/presets",
                json=preset_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["name"] == preset_data["name"]
            assert data["preset_type"] == preset_data["preset_type"]
    
    async def test_monitoring_dashboard(self, client, auth_headers):
        """Test monitoring dashboard."""
        with patch('src.routers.automation.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.get(
                "/v1/automation/monitoring/dashboard",
                headers=auth_headers
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "health_metrics" in data
            assert "alert_summary" in data
            assert "automation_status" in data


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    async def test_complete_story_pack_workflow(self, client, auth_headers, test_project):
        """Test complete story pack creation workflow."""
        
        with patch('src.routers.storypacks.get_current_user') as mock_user, \
             patch('src.routers.brand_kits.get_current_user') as mock_user2, \
             patch('src.routers.governance.get_current_user') as mock_user3:
            
            mock_user.return_value = Mock(id="user_id")
            mock_user2.return_value = Mock(id="user_id")
            mock_user3.return_value = Mock(id="user_id", full_name="Test User")
            
            # Step 1: Create brand kit
            brand_kit_data = {
                "project_id": str(test_project.id),
                "name": "Workflow Brand Kit",
                "color_palette": {"primary": "#3B82F6", "secondary": "#6B7280"}
            }
            
            brand_response = await client.post(
                "/v1/brand-kits/",
                json=brand_kit_data,
                headers=auth_headers
            )
            assert brand_response.status_code == 200
            brand_kit = brand_response.json()
            
            # Step 2: Generate story pack
            story_data = {
                "project_id": str(test_project.id),
                "name": "Workflow Story Pack",
                "prompt": "Create an engaging story about innovation"
            }
            
            story_response = await client.post(
                "/v1/storypacks/generate",
                json=story_data,
                headers=auth_headers
            )
            assert story_response.status_code == 200
            story_pack = story_response.json()
            
            # Step 3: Submit for review
            with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
                mock_result = Mock()
                mock_result.scalar_one_or_none.return_value = Mock(id=story_pack["id"])
                mock_execute.return_value = mock_result
                
                review_response = await client.post(
                    f"/v1/governance/story-packs/{story_pack['id']}/submit-for-review",
                    json={"reviewers": ["reviewer1"], "instructions": "Please review"},
                    headers=auth_headers
                )
                assert review_response.status_code == 200
    
    async def test_automation_preset_execution(self, client, auth_headers):
        """Test automation preset creation and execution."""
        
        with patch('src.routers.automation.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            # Step 1: Create preset
            preset_data = {
                "name": "Integration Test Preset",
                "preset_type": "story_pack",
                "configuration": {"narrative_style": "professional"}
            }
            
            preset_response = await client.post(
                "/v1/automation/presets",
                json=preset_data,
                headers=auth_headers
            )
            assert preset_response.status_code == 200
            preset = preset_response.json()
            
            # Step 2: Execute preset
            exec_response = await client.post(
                f"/v1/automation/presets/{preset['id']}/execute",
                json={"parameters": {"test": True}},
                headers=auth_headers
            )
            assert exec_response.status_code == 200
            
            execution = exec_response.json()
            assert "execution_id" in execution


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    async def test_404_endpoints(self, client, auth_headers):
        """Test 404 responses for non-existent resources."""
        
        with patch('src.routers.projects.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            response = await client.get(
                "/v1/projects/non-existent-id",
                headers=auth_headers
            )
            assert response.status_code == 404
    
    async def test_validation_errors(self, client, auth_headers):
        """Test validation error responses."""
        
        with patch('src.routers.projects.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            # Missing required fields
            response = await client.post(
                "/v1/projects/",
                json={"description": "Missing name field"},
                headers=auth_headers
            )
            assert response.status_code == 422
    
    async def test_unauthorized_access(self, client):
        """Test unauthorized access responses."""
        
        response = await client.get("/v1/projects/")
        assert response.status_code == 401 or response.status_code == 403


class TestPerformance:
    """Test performance characteristics."""
    
    async def test_concurrent_requests(self, client, auth_headers):
        """Test handling of concurrent requests."""
        
        with patch('src.routers.reports.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            # Create multiple concurrent requests
            tasks = []
            for _ in range(10):
                task = client.get("/v1/reports/dashboard-metrics", headers=auth_headers)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
    
    async def test_large_payload_handling(self, client, auth_headers, test_project):
        """Test handling of large payloads."""
        
        with patch('src.routers.storypacks.get_current_user') as mock_user:
            mock_user.return_value = Mock(id="user_id")
            
            # Create a large prompt
            large_prompt = "Create a story about " + "innovation " * 1000
            
            story_data = {
                "project_id": str(test_project.id),
                "name": "Large Prompt Test",
                "prompt": large_prompt
            }
            
            response = await client.post(
                "/v1/storypacks/generate",
                json=story_data,
                headers=auth_headers
            )
            assert response.status_code == 200


# Test configuration and fixtures for running tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
