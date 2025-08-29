"""Governance endpoints for review workflows, comments, and audit logs."""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models.auth import User
from ..models.content import StoryPack, Project
from ..routers.auth import get_current_user

router = APIRouter()


class ReviewStatus(str, Enum):
    """Review status enumeration."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class CommentType(str, Enum):
    """Comment type enumeration."""
    GENERAL = "general"
    SUGGESTION = "suggestion"
    ISSUE = "issue"
    APPROVAL = "approval"
    REJECTION = "rejection"


class AuditAction(str, Enum):
    """Audit action enumeration."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPORTED = "exported"
    SHARED = "shared"


class ReviewRequest(BaseModel):
    """Review request schema."""
    story_pack_id: str
    reviewers: List[str]  # User IDs
    deadline: Optional[datetime] = None
    instructions: Optional[str] = None
    priority: str = "medium"  # low, medium, high, urgent


class ReviewUpdate(BaseModel):
    """Review update schema."""
    status: ReviewStatus
    comments: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None


class CommentCreate(BaseModel):
    """Comment creation schema."""
    content: str
    comment_type: CommentType = CommentType.GENERAL
    target_element: Optional[str] = None  # "text", "image_1", "audio", etc.
    position: Optional[Dict[str, Any]] = None  # For precise positioning
    mentions: Optional[List[str]] = None  # User IDs


class CommentResponse(BaseModel):
    """Comment response schema."""
    id: str
    story_pack_id: str
    author_id: str
    author_name: str
    content: str
    comment_type: CommentType
    target_element: Optional[str]
    position: Optional[Dict[str, Any]]
    mentions: Optional[List[str]]
    created_at: datetime
    updated_at: Optional[datetime]
    resolved: bool
    resolved_by: Optional[str]
    resolved_at: Optional[datetime]


class ReviewResponse(BaseModel):
    """Review response schema."""
    id: str
    story_pack_id: str
    status: ReviewStatus
    created_by: str
    created_by_name: str
    reviewers: List[Dict[str, str]]  # [{id, name, status}]
    deadline: Optional[datetime]
    instructions: Optional[str]
    priority: str
    created_at: datetime
    updated_at: Optional[datetime]
    completed_at: Optional[datetime]
    comments_count: int


class AuditLogEntry(BaseModel):
    """Audit log entry schema."""
    id: str
    entity_type: str  # "story_pack", "project", "brand_kit", etc.
    entity_id: str
    action: AuditAction
    user_id: str
    user_name: str
    details: Optional[Dict[str, Any]]
    timestamp: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]


# In-memory storage for demo (in production, use proper database tables)
reviews_db: Dict[str, Dict] = {}
comments_db: Dict[str, Dict] = {}
audit_logs_db: List[Dict] = []


@router.post("/reviews", response_model=ReviewResponse)
async def create_review(
    review_request: ReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new review request."""
    
    # Verify story pack exists
    result = await db.execute(
        select(StoryPack).where(StoryPack.id == UUID(review_request.story_pack_id))
    )
    story_pack = result.scalar_one_or_none()
    
    if not story_pack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story pack not found"
        )
    
    # TODO: Verify user has permission to request reviews
    # TODO: Verify reviewers exist and have appropriate permissions
    
    review_id = f"review_{int(datetime.now().timestamp())}"
    
    review_data = {
        'id': review_id,
        'story_pack_id': review_request.story_pack_id,
        'status': ReviewStatus.PENDING_REVIEW,
        'created_by': str(current_user.id),
        'created_by_name': current_user.full_name or current_user.email,
        'reviewers': [
            {'id': reviewer_id, 'name': f'User {reviewer_id}', 'status': 'pending'}
            for reviewer_id in review_request.reviewers
        ],
        'deadline': review_request.deadline,
        'instructions': review_request.instructions,
        'priority': review_request.priority,
        'created_at': datetime.now(),
        'updated_at': None,
        'completed_at': None,
        'comments_count': 0
    }
    
    reviews_db[review_id] = review_data
    
    # Log audit event
    await _log_audit_event(
        entity_type="story_pack",
        entity_id=review_request.story_pack_id,
        action=AuditAction.REVIEWED,
        user_id=str(current_user.id),
        user_name=current_user.full_name or current_user.email,
        details={
            'review_id': review_id,
            'reviewers': review_request.reviewers,
            'priority': review_request.priority
        }
    )
    
    return ReviewResponse(**review_data)


@router.get("/reviews", response_model=List[ReviewResponse])
async def list_reviews(
    story_pack_id: Optional[str] = Query(None),
    status: Optional[ReviewStatus] = Query(None),
    assigned_to_me: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List reviews with optional filtering."""
    
    filtered_reviews = []
    
    for review_data in reviews_db.values():
        # Apply filters
        if story_pack_id and review_data['story_pack_id'] != story_pack_id:
            continue
        
        if status and review_data['status'] != status:
            continue
        
        if assigned_to_me:
            reviewer_ids = [r['id'] for r in review_data['reviewers']]
            if str(current_user.id) not in reviewer_ids:
                continue
        
        filtered_reviews.append(ReviewResponse(**review_data))
    
    # Sort by created_at descending
    filtered_reviews.sort(key=lambda x: x.created_at, reverse=True)
    
    return filtered_reviews


@router.get("/reviews/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific review."""
    
    if review_id not in reviews_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    review_data = reviews_db[review_id]
    
    # TODO: Check if user has permission to view this review
    
    return ReviewResponse(**review_data)


@router.put("/reviews/{review_id}", response_model=ReviewResponse)
async def update_review(
    review_id: str,
    review_update: ReviewUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update review status and add feedback."""
    
    if review_id not in reviews_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    review_data = reviews_db[review_id]
    
    # TODO: Check if user has permission to update this review
    
    # Update review data
    review_data['status'] = review_update.status
    review_data['updated_at'] = datetime.now()
    
    if review_update.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED]:
        review_data['completed_at'] = datetime.now()
    
    # Update reviewer status
    for reviewer in review_data['reviewers']:
        if reviewer['id'] == str(current_user.id):
            reviewer['status'] = review_update.status.value
            break
    
    # Add comment if provided
    if review_update.comments:
        comment_data = {
            'id': f"comment_{int(datetime.now().timestamp())}",
            'story_pack_id': review_data['story_pack_id'],
            'review_id': review_id,
            'author_id': str(current_user.id),
            'author_name': current_user.full_name or current_user.email,
            'content': review_update.comments,
            'comment_type': CommentType.APPROVAL if review_update.status == ReviewStatus.APPROVED else CommentType.GENERAL,
            'target_element': None,
            'position': None,
            'mentions': None,
            'created_at': datetime.now(),
            'updated_at': None,
            'resolved': False,
            'resolved_by': None,
            'resolved_at': None
        }
        
        comments_db[comment_data['id']] = comment_data
        review_data['comments_count'] += 1
    
    # Log audit event
    await _log_audit_event(
        entity_type="story_pack",
        entity_id=review_data['story_pack_id'],
        action=AuditAction.APPROVED if review_update.status == ReviewStatus.APPROVED else AuditAction.UPDATED,
        user_id=str(current_user.id),
        user_name=current_user.full_name or current_user.email,
        details={
            'review_id': review_id,
            'new_status': review_update.status.value,
            'feedback': review_update.feedback
        }
    )
    
    return ReviewResponse(**review_data)


@router.post("/comments", response_model=CommentResponse)
async def create_comment(
    comment_create: CommentCreate,
    story_pack_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new comment on a story pack."""
    
    # Verify story pack exists
    result = await db.execute(
        select(StoryPack).where(StoryPack.id == UUID(story_pack_id))
    )
    story_pack = result.scalar_one_or_none()
    
    if not story_pack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story pack not found"
        )
    
    comment_id = f"comment_{int(datetime.now().timestamp())}"
    
    comment_data = {
        'id': comment_id,
        'story_pack_id': story_pack_id,
        'author_id': str(current_user.id),
        'author_name': current_user.full_name or current_user.email,
        'content': comment_create.content,
        'comment_type': comment_create.comment_type,
        'target_element': comment_create.target_element,
        'position': comment_create.position,
        'mentions': comment_create.mentions,
        'created_at': datetime.now(),
        'updated_at': None,
        'resolved': False,
        'resolved_by': None,
        'resolved_at': None
    }
    
    comments_db[comment_id] = comment_data
    
    # Log audit event
    await _log_audit_event(
        entity_type="story_pack",
        entity_id=story_pack_id,
        action=AuditAction.UPDATED,
        user_id=str(current_user.id),
        user_name=current_user.full_name or current_user.email,
        details={
            'comment_id': comment_id,
            'comment_type': comment_create.comment_type.value,
            'target_element': comment_create.target_element
        }
    )
    
    return CommentResponse(**comment_data)


@router.get("/comments", response_model=List[CommentResponse])
async def list_comments(
    story_pack_id: str = Query(...),
    comment_type: Optional[CommentType] = Query(None),
    resolved: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List comments for a story pack."""
    
    filtered_comments = []
    
    for comment_data in comments_db.values():
        # Filter by story pack
        if comment_data['story_pack_id'] != story_pack_id:
            continue
        
        # Apply optional filters
        if comment_type and comment_data['comment_type'] != comment_type:
            continue
        
        if resolved is not None and comment_data['resolved'] != resolved:
            continue
        
        filtered_comments.append(CommentResponse(**comment_data))
    
    # Sort by created_at descending
    filtered_comments.sort(key=lambda x: x.created_at, reverse=True)
    
    return filtered_comments


@router.put("/comments/{comment_id}/resolve")
async def resolve_comment(
    comment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Mark a comment as resolved."""
    
    if comment_id not in comments_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found"
        )
    
    comment_data = comments_db[comment_id]
    
    # TODO: Check if user has permission to resolve this comment
    
    comment_data['resolved'] = True
    comment_data['resolved_by'] = str(current_user.id)
    comment_data['resolved_at'] = datetime.now()
    comment_data['updated_at'] = datetime.now()
    
    # Log audit event
    await _log_audit_event(
        entity_type="story_pack",
        entity_id=comment_data['story_pack_id'],
        action=AuditAction.UPDATED,
        user_id=str(current_user.id),
        user_name=current_user.full_name or current_user.email,
        details={
            'comment_id': comment_id,
            'action': 'resolved'
        }
    )
    
    return {"message": "Comment resolved successfully"}


@router.get("/audit-logs", response_model=List[AuditLogEntry])
async def get_audit_logs(
    entity_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[AuditAction] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get audit logs with optional filtering."""
    
    # TODO: Check if user has permission to view audit logs
    
    filtered_logs = []
    
    for log_entry in audit_logs_db:
        # Apply filters
        if entity_type and log_entry['entity_type'] != entity_type:
            continue
        
        if entity_id and log_entry['entity_id'] != entity_id:
            continue
        
        if user_id and log_entry['user_id'] != user_id:
            continue
        
        if action and log_entry['action'] != action:
            continue
        
        filtered_logs.append(AuditLogEntry(**log_entry))
    
    # Sort by timestamp descending
    filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Apply pagination
    return filtered_logs[offset:offset + limit]


@router.get("/governance-dashboard")
async def get_governance_dashboard(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get governance dashboard metrics."""
    
    # Calculate date range
    from_date = datetime.now() - timedelta(days=days)
    
    # Filter logs by date range
    recent_logs = [
        log for log in audit_logs_db 
        if log['timestamp'] >= from_date
    ]
    
    # Filter reviews by date range
    recent_reviews = [
        review for review in reviews_db.values()
        if review['created_at'] >= from_date
    ]
    
    # Calculate metrics
    total_reviews = len(recent_reviews)
    pending_reviews = len([r for r in recent_reviews if r['status'] == ReviewStatus.PENDING_REVIEW])
    approved_reviews = len([r for r in recent_reviews if r['status'] == ReviewStatus.APPROVED])
    rejected_reviews = len([r for r in recent_reviews if r['status'] == ReviewStatus.REJECTED])
    
    # Review completion rate
    completed_reviews = approved_reviews + rejected_reviews
    completion_rate = (completed_reviews / total_reviews * 100) if total_reviews > 0 else 0
    
    # Average review time (simplified)
    completed_review_times = []
    for review in recent_reviews:
        if review['completed_at'] and review['created_at']:
            time_diff = (review['completed_at'] - review['created_at']).total_seconds() / 3600  # hours
            completed_review_times.append(time_diff)
    
    avg_review_time = sum(completed_review_times) / len(completed_review_times) if completed_review_times else 0
    
    # Activity breakdown
    activity_breakdown = {}
    for log in recent_logs:
        action = log['action']
        activity_breakdown[action] = activity_breakdown.get(action, 0) + 1
    
    # Top reviewers
    reviewer_activity = {}
    for review in recent_reviews:
        for reviewer in review['reviewers']:
            if reviewer['status'] != 'pending':
                reviewer_id = reviewer['id']
                reviewer_activity[reviewer_id] = reviewer_activity.get(reviewer_id, 0) + 1
    
    top_reviewers = sorted(
        [{'id': k, 'reviews': v} for k, v in reviewer_activity.items()],
        key=lambda x: x['reviews'],
        reverse=True
    )[:5]
    
    # Recent activity timeline
    activity_timeline = []
    for i in range(7):  # Last 7 days
        day = datetime.now() - timedelta(days=i)
        day_logs = [log for log in recent_logs if log['timestamp'].date() == day.date()]
        
        activity_timeline.append({
            'date': day.strftime('%Y-%m-%d'),
            'activities': len(day_logs),
            'reviews_created': len([log for log in day_logs if log['action'] == AuditAction.REVIEWED]),
            'approvals': len([log for log in day_logs if log['action'] == AuditAction.APPROVED])
        })
    
    return {
        'summary': {
            'total_reviews': total_reviews,
            'pending_reviews': pending_reviews,
            'completion_rate': round(completion_rate, 1),
            'avg_review_time_hours': round(avg_review_time, 1),
            'total_activities': len(recent_logs)
        },
        'review_breakdown': {
            'pending': pending_reviews,
            'approved': approved_reviews,
            'rejected': rejected_reviews,
            'in_review': len([r for r in recent_reviews if r['status'] == ReviewStatus.IN_REVIEW])
        },
        'activity_breakdown': activity_breakdown,
        'top_reviewers': top_reviewers,
        'activity_timeline': list(reversed(activity_timeline)),  # Most recent first
        'period': {
            'days': days,
            'from': from_date.isoformat(),
            'to': datetime.now().isoformat()
        }
    }


# Helper functions

async def _log_audit_event(
    entity_type: str,
    entity_id: str,
    action: AuditAction,
    user_id: str,
    user_name: str,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    """Log an audit event."""
    
    audit_entry = {
        'id': f"audit_{int(datetime.now().timestamp())}_{len(audit_logs_db)}",
        'entity_type': entity_type,
        'entity_id': entity_id,
        'action': action,
        'user_id': user_id,
        'user_name': user_name,
        'details': details or {},
        'timestamp': datetime.now(),
        'ip_address': ip_address,
        'user_agent': user_agent
    }
    
    audit_logs_db.append(audit_entry)
    
    # Keep only last 10000 entries to prevent memory issues
    if len(audit_logs_db) > 10000:
        audit_logs_db[:] = audit_logs_db[-10000:]


@router.post("/story-packs/{story_pack_id}/submit-for-review")
async def submit_for_review(
    story_pack_id: str,
    reviewers: List[str],
    instructions: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit a story pack for review."""
    
    # Create review request
    review_request = ReviewRequest(
        story_pack_id=story_pack_id,
        reviewers=reviewers,
        instructions=instructions,
        priority="medium"
    )
    
    review = await create_review(review_request, db, current_user)
    
    return {
        "message": "Story pack submitted for review successfully",
        "review_id": review.id
    }


@router.post("/story-packs/{story_pack_id}/approve")
async def approve_story_pack(
    story_pack_id: str,
    comments: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Approve a story pack (shortcut for review approval)."""
    
    # Log approval
    await _log_audit_event(
        entity_type="story_pack",
        entity_id=story_pack_id,
        action=AuditAction.APPROVED,
        user_id=str(current_user.id),
        user_name=current_user.full_name or current_user.email,
        details={'comments': comments}
    )
    
    return {"message": "Story pack approved successfully"}


@router.post("/story-packs/{story_pack_id}/reject")
async def reject_story_pack(
    story_pack_id: str,
    reason: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Reject a story pack with reason."""
    
    # Log rejection
    await _log_audit_event(
        entity_type="story_pack",
        entity_id=story_pack_id,
        action=AuditAction.REJECTED,
        user_id=str(current_user.id),
        user_name=current_user.full_name or current_user.email,
        details={'reason': reason}
    )
    
    return {"message": "Story pack rejected", "reason": reason}
