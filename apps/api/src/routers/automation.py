"""Automation endpoints for presets, schedules, A/B testing, and monitoring."""

from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from ..database import get_db
from ..models.auth import User
from ..models.content import StoryPack, Project
from ..routers.auth import get_current_user

router = APIRouter()


class PresetType(str, Enum):
    """Automation preset types."""
    STORY_PACK = "story_pack"
    CAMPAIGN_KIT = "campaign_kit"
    LESSON_MODULE = "lesson_module"
    BRAND_SET = "brand_set"


class ScheduleFrequency(str, Enum):
    """Schedule frequency options."""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ABTestStatus(str, Enum):
    """A/B test status."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class PresetCreate(BaseModel):
    """Automation preset creation schema."""
    name: str
    description: Optional[str] = None
    preset_type: PresetType
    configuration: Dict[str, Any]
    brand_kit_id: Optional[str] = None
    tags: Optional[List[str]] = None


class PresetResponse(BaseModel):
    """Automation preset response schema."""
    id: str
    name: str
    description: Optional[str]
    preset_type: PresetType
    configuration: Dict[str, Any]
    brand_kit_id: Optional[str]
    tags: Optional[List[str]]
    usage_count: int
    success_rate: float
    avg_quality_score: float
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime]


class ScheduleCreate(BaseModel):
    """Schedule creation schema."""
    name: str
    preset_id: str
    frequency: ScheduleFrequency
    start_date: datetime
    end_date: Optional[datetime] = None
    cron_expression: Optional[str] = None  # For custom frequency
    parameters: Optional[Dict[str, Any]] = None
    enabled: bool = True


class ScheduleResponse(BaseModel):
    """Schedule response schema."""
    id: str
    name: str
    preset_id: str
    preset_name: str
    frequency: ScheduleFrequency
    start_date: datetime
    end_date: Optional[datetime]
    cron_expression: Optional[str]
    parameters: Optional[Dict[str, Any]]
    enabled: bool
    next_run: Optional[datetime]
    last_run: Optional[datetime]
    run_count: int
    success_count: int
    created_at: datetime


class ABTestCreate(BaseModel):
    """A/B test creation schema."""
    name: str
    description: Optional[str] = None
    variants: List[Dict[str, Any]] = Field(..., min_items=2, max_items=5)
    traffic_split: List[float] = Field(..., min_items=2, max_items=5)
    success_metrics: List[str]
    duration_days: int = Field(default=7, ge=1, le=30)
    min_sample_size: int = Field(default=100, ge=10)


class ABTestResponse(BaseModel):
    """A/B test response schema."""
    id: str
    name: str
    description: Optional[str]
    status: ABTestStatus
    variants: List[Dict[str, Any]]
    traffic_split: List[float]
    success_metrics: List[str]
    duration_days: int
    min_sample_size: int
    current_results: Optional[Dict[str, Any]]
    winner: Optional[str]
    confidence_level: Optional[float]
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]


class MonitoringAlert(BaseModel):
    """Monitoring alert schema."""
    id: str
    alert_type: str  # "quality_drift", "cost_spike", "failure_rate", "performance"
    severity: str    # "low", "medium", "high", "critical"
    title: str
    description: str
    entity_type: str
    entity_id: str
    metrics: Dict[str, Any]
    triggered_at: datetime
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]


# In-memory storage for demo (in production, use proper database tables)
presets_db: Dict[str, Dict] = {}
schedules_db: Dict[str, Dict] = {}
ab_tests_db: Dict[str, Dict] = {}
monitoring_alerts_db: List[Dict] = []


@router.post("/presets", response_model=PresetResponse)
async def create_preset(
    preset_create: PresetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create an automation preset."""
    
    preset_id = str(uuid4())
    
    preset_data = {
        'id': preset_id,
        'name': preset_create.name,
        'description': preset_create.description,
        'preset_type': preset_create.preset_type,
        'configuration': preset_create.configuration,
        'brand_kit_id': preset_create.brand_kit_id,
        'tags': preset_create.tags or [],
        'usage_count': 0,
        'success_rate': 0.0,
        'avg_quality_score': 0.0,
        'created_by': str(current_user.id),
        'created_at': datetime.now(),
        'updated_at': None
    }
    
    presets_db[preset_id] = preset_data
    
    return PresetResponse(**preset_data)


@router.get("/presets", response_model=List[PresetResponse])
async def list_presets(
    preset_type: Optional[PresetType] = None,
    tags: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List automation presets with optional filtering."""
    
    filtered_presets = []
    
    for preset_data in presets_db.values():
        # Apply filters
        if preset_type and preset_data['preset_type'] != preset_type:
            continue
        
        if tags:
            preset_tags = preset_data.get('tags', [])
            if not any(tag in preset_tags for tag in tags):
                continue
        
        filtered_presets.append(PresetResponse(**preset_data))
    
    # Sort by usage count and quality score
    filtered_presets.sort(key=lambda x: (x.usage_count, x.avg_quality_score), reverse=True)
    
    return filtered_presets


@router.get("/presets/{preset_id}", response_model=PresetResponse)
async def get_preset(
    preset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific automation preset."""
    
    if preset_id not in presets_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preset not found"
        )
    
    return PresetResponse(**presets_db[preset_id])


@router.post("/presets/{preset_id}/execute")
async def execute_preset(
    preset_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Execute an automation preset."""
    
    if preset_id not in presets_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preset not found"
        )
    
    preset_data = presets_db[preset_id]
    
    # Add background task to execute preset
    background_tasks.add_task(
        _execute_preset_background,
        preset_id,
        preset_data,
        parameters or {},
        str(current_user.id)
    )
    
    # Update usage count
    preset_data['usage_count'] += 1
    preset_data['updated_at'] = datetime.now()
    
    return {
        "message": "Preset execution started",
        "preset_id": preset_id,
        "execution_id": f"exec_{int(datetime.now().timestamp())}"
    }


@router.post("/schedules", response_model=ScheduleResponse)
async def create_schedule(
    schedule_create: ScheduleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create an automation schedule."""
    
    # Verify preset exists
    if schedule_create.preset_id not in presets_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preset not found"
        )
    
    schedule_id = str(uuid4())
    preset_data = presets_db[schedule_create.preset_id]
    
    # Calculate next run time
    next_run = _calculate_next_run(schedule_create.frequency, schedule_create.start_date, schedule_create.cron_expression)
    
    schedule_data = {
        'id': schedule_id,
        'name': schedule_create.name,
        'preset_id': schedule_create.preset_id,
        'preset_name': preset_data['name'],
        'frequency': schedule_create.frequency,
        'start_date': schedule_create.start_date,
        'end_date': schedule_create.end_date,
        'cron_expression': schedule_create.cron_expression,
        'parameters': schedule_create.parameters,
        'enabled': schedule_create.enabled,
        'next_run': next_run,
        'last_run': None,
        'run_count': 0,
        'success_count': 0,
        'created_at': datetime.now()
    }
    
    schedules_db[schedule_id] = schedule_data
    
    return ScheduleResponse(**schedule_data)


@router.get("/schedules", response_model=List[ScheduleResponse])
async def list_schedules(
    enabled_only: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List automation schedules."""
    
    filtered_schedules = []
    
    for schedule_data in schedules_db.values():
        if enabled_only and not schedule_data['enabled']:
            continue
        
        filtered_schedules.append(ScheduleResponse(**schedule_data))
    
    # Sort by next run time
    filtered_schedules.sort(key=lambda x: x.next_run or datetime.max)
    
    return filtered_schedules


@router.put("/schedules/{schedule_id}/toggle")
async def toggle_schedule(
    schedule_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Enable or disable a schedule."""
    
    if schedule_id not in schedules_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found"
        )
    
    schedule_data = schedules_db[schedule_id]
    schedule_data['enabled'] = not schedule_data['enabled']
    
    return {
        "message": f"Schedule {'enabled' if schedule_data['enabled'] else 'disabled'}",
        "enabled": schedule_data['enabled']
    }


@router.post("/ab-tests", response_model=ABTestResponse)
async def create_ab_test(
    ab_test_create: ABTestCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create an A/B test."""
    
    # Validate traffic split
    if abs(sum(ab_test_create.traffic_split) - 1.0) > 0.01:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Traffic split must sum to 1.0"
        )
    
    if len(ab_test_create.variants) != len(ab_test_create.traffic_split):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of variants must match traffic split entries"
        )
    
    test_id = str(uuid4())
    
    test_data = {
        'id': test_id,
        'name': ab_test_create.name,
        'description': ab_test_create.description,
        'status': ABTestStatus.DRAFT,
        'variants': ab_test_create.variants,
        'traffic_split': ab_test_create.traffic_split,
        'success_metrics': ab_test_create.success_metrics,
        'duration_days': ab_test_create.duration_days,
        'min_sample_size': ab_test_create.min_sample_size,
        'current_results': None,
        'winner': None,
        'confidence_level': None,
        'created_at': datetime.now(),
        'started_at': None,
        'ended_at': None
    }
    
    ab_tests_db[test_id] = test_data
    
    return ABTestResponse(**test_data)


@router.post("/ab-tests/{test_id}/start")
async def start_ab_test(
    test_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start an A/B test."""
    
    if test_id not in ab_tests_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="A/B test not found"
        )
    
    test_data = ab_tests_db[test_id]
    
    if test_data['status'] != ABTestStatus.DRAFT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A/B test can only be started from draft status"
        )
    
    test_data['status'] = ABTestStatus.RUNNING
    test_data['started_at'] = datetime.now()
    
    # Add background task to monitor test
    background_tasks.add_task(_monitor_ab_test, test_id)
    
    return {"message": "A/B test started successfully"}


@router.get("/ab-tests", response_model=List[ABTestResponse])
async def list_ab_tests(
    status: Optional[ABTestStatus] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List A/B tests."""
    
    filtered_tests = []
    
    for test_data in ab_tests_db.values():
        if status and test_data['status'] != status:
            continue
        
        filtered_tests.append(ABTestResponse(**test_data))
    
    # Sort by created date descending
    filtered_tests.sort(key=lambda x: x.created_at, reverse=True)
    
    return filtered_tests


@router.get("/ab-tests/{test_id}/results")
async def get_ab_test_results(
    test_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed A/B test results."""
    
    if test_id not in ab_tests_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="A/B test not found"
        )
    
    test_data = ab_tests_db[test_id]
    
    # Generate mock results for demo
    results = {
        'test_id': test_id,
        'status': test_data['status'],
        'duration_elapsed': (datetime.now() - test_data['started_at']).days if test_data['started_at'] else 0,
        'total_samples': 1250,  # Mock data
        'variant_results': [
            {
                'variant_id': f"variant_{i}",
                'name': f"Variant {chr(65+i)}",
                'samples': int(1250 * split),
                'conversion_rate': 0.15 + (i * 0.02),  # Mock conversion rates
                'avg_quality_score': 0.75 + (i * 0.05),
                'cost_per_conversion': 2.50 - (i * 0.20),
                'confidence_interval': [0.12 + (i * 0.02), 0.18 + (i * 0.02)]
            }
            for i, split in enumerate(test_data['traffic_split'])
        ],
        'statistical_significance': 0.95 if test_data['status'] == ABTestStatus.COMPLETED else 0.78,
        'recommended_action': 'Continue test' if test_data['status'] == ABTestStatus.RUNNING else 'Deploy variant B'
    }
    
    return results


@router.get("/monitoring/alerts", response_model=List[MonitoringAlert])
async def get_monitoring_alerts(
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get monitoring alerts."""
    
    filtered_alerts = []
    
    for alert_data in monitoring_alerts_db:
        # Apply filters
        if severity and alert_data['severity'] != severity:
            continue
        
        if resolved is not None:
            is_resolved = alert_data['resolved_at'] is not None
            if resolved != is_resolved:
                continue
        
        filtered_alerts.append(MonitoringAlert(**alert_data))
    
    # Sort by triggered time descending
    filtered_alerts.sort(key=lambda x: x.triggered_at, reverse=True)
    
    return filtered_alerts[:limit]


@router.post("/monitoring/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Acknowledge a monitoring alert."""
    
    alert = next((a for a in monitoring_alerts_db if a['id'] == alert_id), None)
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    alert['acknowledged_by'] = str(current_user.id)
    
    return {"message": "Alert acknowledged successfully"}


@router.get("/monitoring/dashboard")
async def get_monitoring_dashboard(
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get monitoring dashboard data."""
    
    # Calculate time range
    from_time = datetime.now() - timedelta(hours=hours)
    
    # Filter recent alerts
    recent_alerts = [
        alert for alert in monitoring_alerts_db
        if alert['triggered_at'] >= from_time
    ]
    
    # System health metrics (mock data)
    health_metrics = {
        'overall_health': 'healthy',  # healthy, degraded, critical
        'api_response_time': 145,  # ms
        'success_rate': 99.2,  # percentage
        'active_generations': 12,
        'queue_length': 3,
        'error_rate': 0.8,  # percentage
        'cost_per_hour': 15.75,  # dollars
    }
    
    # Alert breakdown
    alert_breakdown = {
        'critical': len([a for a in recent_alerts if a['severity'] == 'critical']),
        'high': len([a for a in recent_alerts if a['severity'] == 'high']),
        'medium': len([a for a in recent_alerts if a['severity'] == 'medium']),
        'low': len([a for a in recent_alerts if a['severity'] == 'low']),
    }
    
    # Performance trends (mock data)
    performance_trends = []
    for i in range(24):  # Last 24 hours
        hour = datetime.now() - timedelta(hours=i)
        performance_trends.append({
            'timestamp': hour.isoformat(),
            'response_time': 145 + (i % 5) * 10,  # Mock variation
            'success_rate': 99.2 - (i % 3) * 0.1,
            'throughput': 50 + (i % 7) * 5,
            'cost': 15.75 + (i % 4) * 0.5
        })
    
    # Active automations
    active_schedules = len([s for s in schedules_db.values() if s['enabled']])
    running_ab_tests = len([t for t in ab_tests_db.values() if t['status'] == ABTestStatus.RUNNING])
    
    return {
        'health_metrics': health_metrics,
        'alert_summary': {
            'total_alerts': len(recent_alerts),
            'unresolved_alerts': len([a for a in recent_alerts if not a['resolved_at']]),
            'breakdown': alert_breakdown
        },
        'automation_status': {
            'active_schedules': active_schedules,
            'running_ab_tests': running_ab_tests,
            'total_presets': len(presets_db),
            'executions_today': 45  # Mock data
        },
        'performance_trends': list(reversed(performance_trends)),  # Most recent first
        'period': {
            'hours': hours,
            'from': from_time.isoformat(),
            'to': datetime.now().isoformat()
        }
    }


# Background tasks and helper functions

async def _execute_preset_background(
    preset_id: str,
    preset_data: Dict,
    parameters: Dict[str, Any],
    user_id: str
):
    """Execute preset in background."""
    
    try:
        # Simulate preset execution
        await asyncio.sleep(2)  # Simulate processing time
        
        # Update success metrics
        preset_data['success_rate'] = min(100.0, preset_data['success_rate'] + 1.0)
        preset_data['avg_quality_score'] = min(1.0, preset_data['avg_quality_score'] + 0.01)
        
        # Create monitoring alert if needed
        if preset_data['success_rate'] < 80:
            _create_monitoring_alert(
                alert_type="quality_drift",
                severity="medium",
                title=f"Preset {preset_data['name']} quality declining",
                description=f"Success rate dropped to {preset_data['success_rate']:.1f}%",
                entity_type="preset",
                entity_id=preset_id,
                metrics={'success_rate': preset_data['success_rate']}
            )
        
    except Exception as e:
        # Handle execution failure
        _create_monitoring_alert(
            alert_type="failure_rate",
            severity="high",
            title=f"Preset execution failed",
            description=f"Preset {preset_data['name']} failed to execute: {str(e)}",
            entity_type="preset",
            entity_id=preset_id,
            metrics={'error': str(e)}
        )


def _calculate_next_run(frequency: ScheduleFrequency, start_date: datetime, cron_expression: Optional[str]) -> Optional[datetime]:
    """Calculate next run time for a schedule."""
    
    now = datetime.now()
    
    if frequency == ScheduleFrequency.ONCE:
        return start_date if start_date > now else None
    elif frequency == ScheduleFrequency.DAILY:
        next_run = start_date
        while next_run <= now:
            next_run += timedelta(days=1)
        return next_run
    elif frequency == ScheduleFrequency.WEEKLY:
        next_run = start_date
        while next_run <= now:
            next_run += timedelta(weeks=1)
        return next_run
    elif frequency == ScheduleFrequency.MONTHLY:
        next_run = start_date
        while next_run <= now:
            # Simple monthly increment (doesn't handle month boundaries perfectly)
            next_run += timedelta(days=30)
        return next_run
    elif frequency == ScheduleFrequency.CUSTOM and cron_expression:
        # Simplified cron parsing (in production, use proper cron library)
        return start_date + timedelta(hours=1)  # Default to 1 hour
    
    return None


async def _monitor_ab_test(test_id: str):
    """Monitor A/B test progress."""
    
    try:
        test_data = ab_tests_db.get(test_id)
        if not test_data:
            return
        
        # Simulate test monitoring
        await asyncio.sleep(5)  # Simulate monitoring delay
        
        # Check if test should be completed
        if test_data['started_at']:
            elapsed_days = (datetime.now() - test_data['started_at']).days
            if elapsed_days >= test_data['duration_days']:
                test_data['status'] = ABTestStatus.COMPLETED
                test_data['ended_at'] = datetime.now()
                test_data['winner'] = 'variant_1'  # Mock winner
                test_data['confidence_level'] = 0.95
        
    except Exception as e:
        print(f"A/B test monitoring failed: {e}")


def _create_monitoring_alert(
    alert_type: str,
    severity: str,
    title: str,
    description: str,
    entity_type: str,
    entity_id: str,
    metrics: Dict[str, Any]
):
    """Create a monitoring alert."""
    
    alert = {
        'id': f"alert_{int(datetime.now().timestamp())}",
        'alert_type': alert_type,
        'severity': severity,
        'title': title,
        'description': description,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'metrics': metrics,
        'triggered_at': datetime.now(),
        'resolved_at': None,
        'acknowledged_by': None
    }
    
    monitoring_alerts_db.append(alert)
    
    # Keep only last 1000 alerts
    if len(monitoring_alerts_db) > 1000:
        monitoring_alerts_db[:] = monitoring_alerts_db[-1000:]


# Initialize some demo data
def _initialize_demo_data():
    """Initialize demo presets and alerts."""
    
    # Demo presets
    demo_presets = [
        {
            'id': 'preset_story_pack_default',
            'name': 'Default Story Pack',
            'description': 'Standard story pack generation with balanced settings',
            'preset_type': PresetType.STORY_PACK,
            'configuration': {
                'narrative_style': 'engaging',
                'image_count': 3,
                'include_audio': True,
                'quality_threshold': 0.8
            },
            'brand_kit_id': None,
            'tags': ['default', 'story', 'balanced'],
            'usage_count': 45,
            'success_rate': 92.5,
            'avg_quality_score': 0.85,
            'created_by': 'system',
            'created_at': datetime.now() - timedelta(days=30),
            'updated_at': None
        },
        {
            'id': 'preset_commercial_fast',
            'name': 'Fast Commercial',
            'description': 'Quick commercial generation for marketing campaigns',
            'preset_type': PresetType.CAMPAIGN_KIT,
            'configuration': {
                'narrative_style': 'punchy',
                'image_count': 2,
                'include_audio': True,
                'duration_target': 30,
                'quality_threshold': 0.75
            },
            'brand_kit_id': None,
            'tags': ['commercial', 'fast', 'marketing'],
            'usage_count': 23,
            'success_rate': 88.0,
            'avg_quality_score': 0.78,
            'created_by': 'system',
            'created_at': datetime.now() - timedelta(days=15),
            'updated_at': None
        }
    ]
    
    for preset in demo_presets:
        presets_db[preset['id']] = preset
    
    # Demo monitoring alerts
    demo_alerts = [
        {
            'id': 'alert_quality_drift_001',
            'alert_type': 'quality_drift',
            'severity': 'medium',
            'title': 'Quality score declining',
            'description': 'Average quality score has dropped below 0.8 threshold',
            'entity_type': 'system',
            'entity_id': 'global',
            'metrics': {'avg_quality': 0.78, 'threshold': 0.8},
            'triggered_at': datetime.now() - timedelta(hours=2),
            'resolved_at': None,
            'acknowledged_by': None
        },
        {
            'id': 'alert_cost_spike_001',
            'alert_type': 'cost_spike',
            'severity': 'high',
            'title': 'Cost increase detected',
            'description': 'Hourly costs increased by 35% above baseline',
            'entity_type': 'system',
            'entity_id': 'global',
            'metrics': {'current_cost': 21.25, 'baseline': 15.75, 'increase_pct': 35},
            'triggered_at': datetime.now() - timedelta(minutes=30),
            'resolved_at': None,
            'acknowledged_by': None
        }
    ]
    
    monitoring_alerts_db.extend(demo_alerts)


# Initialize demo data on module load
_initialize_demo_data()
