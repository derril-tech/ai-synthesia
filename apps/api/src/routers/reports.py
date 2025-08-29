"""Report generation endpoints for Brand Fit, Cost, and Alignment analysis."""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models.auth import User
from ..models.content import StoryPack, Evaluation, Project, BrandKit
from ..routers.auth import get_current_user

router = APIRouter()


class ReportFilter(BaseModel):
    """Report filtering options."""
    project_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    brand_kit_id: Optional[str] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None


class BrandFitMetrics(BaseModel):
    """Brand fit analysis metrics."""
    overall_score: float
    voice_consistency: float
    visual_alignment: float
    color_compliance: float
    typography_adherence: float
    lexicon_usage: float
    improvement_areas: List[str]


class CostMetrics(BaseModel):
    """Cost analysis metrics."""
    total_cost: float
    cost_per_component: Dict[str, float]
    cost_breakdown: Dict[str, float]
    efficiency_score: float
    cost_trends: List[Dict[str, Any]]
    optimization_suggestions: List[str]


class AlignmentMetrics(BaseModel):
    """Cross-modal alignment metrics."""
    text_image_alignment: float
    text_audio_alignment: float
    overall_coherence: float
    consistency_score: float
    gap_analysis: List[str]
    alignment_trends: List[Dict[str, Any]]


class BrandFitReport(BaseModel):
    """Brand fit analysis report."""
    report_id: str
    project_id: str
    brand_kit_id: str
    generated_at: datetime
    metrics: BrandFitMetrics
    story_packs_analyzed: int
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]


class CostReport(BaseModel):
    """Cost analysis report."""
    report_id: str
    project_id: Optional[str]
    period_start: datetime
    period_end: datetime
    metrics: CostMetrics
    story_packs_analyzed: int
    cost_comparison: Dict[str, Any]
    roi_analysis: Dict[str, Any]


class AlignmentReport(BaseModel):
    """Alignment analysis report."""
    report_id: str
    project_id: Optional[str]
    generated_at: datetime
    metrics: AlignmentMetrics
    story_packs_analyzed: int
    quality_trends: List[Dict[str, Any]]
    improvement_recommendations: List[str]


@router.post("/brand-fit", response_model=BrandFitReport)
async def generate_brand_fit_report(
    report_filter: ReportFilter,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate brand fit analysis report."""
    
    # Build query for story packs
    query = select(StoryPack).options(
        selectinload(StoryPack.evaluations),
        selectinload(StoryPack.project)
    )
    
    # Apply filters
    if report_filter.project_id:
        query = query.where(StoryPack.project_id == UUID(report_filter.project_id))
    
    if report_filter.date_from:
        query = query.where(StoryPack.created_at >= report_filter.date_from)
    
    if report_filter.date_to:
        query = query.where(StoryPack.created_at <= report_filter.date_to)
    
    # Execute query
    result = await db.execute(query)
    story_packs = result.scalars().all()
    
    if not story_packs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No story packs found for the specified criteria"
        )
    
    # Get brand kit information
    brand_kit = None
    if report_filter.brand_kit_id:
        brand_result = await db.execute(
            select(BrandKit).where(BrandKit.id == UUID(report_filter.brand_kit_id))
        )
        brand_kit = brand_result.scalar_one_or_none()
    
    # Analyze brand fit metrics
    brand_metrics = await _analyze_brand_fit_metrics(story_packs, brand_kit)
    
    # Generate recommendations
    recommendations = _generate_brand_fit_recommendations(brand_metrics, story_packs)
    
    # Create detailed analysis
    detailed_analysis = await _create_brand_fit_analysis(story_packs, brand_kit)
    
    report = BrandFitReport(
        report_id=f"brand_fit_{int(datetime.now().timestamp())}",
        project_id=report_filter.project_id or "multiple",
        brand_kit_id=report_filter.brand_kit_id or "default",
        generated_at=datetime.now(),
        metrics=brand_metrics,
        story_packs_analyzed=len(story_packs),
        recommendations=recommendations,
        detailed_analysis=detailed_analysis
    )
    
    return report


@router.post("/cost-analysis", response_model=CostReport)
async def generate_cost_report(
    report_filter: ReportFilter,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate cost analysis report."""
    
    # Set default date range if not provided
    if not report_filter.date_from:
        report_filter.date_from = datetime.now() - timedelta(days=30)
    if not report_filter.date_to:
        report_filter.date_to = datetime.now()
    
    # Build query for story packs
    query = select(StoryPack).options(selectinload(StoryPack.project))
    
    # Apply filters
    if report_filter.project_id:
        query = query.where(StoryPack.project_id == UUID(report_filter.project_id))
    
    query = query.where(
        and_(
            StoryPack.created_at >= report_filter.date_from,
            StoryPack.created_at <= report_filter.date_to
        )
    )
    
    # Execute query
    result = await db.execute(query)
    story_packs = result.scalars().all()
    
    if not story_packs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No story packs found for the specified period"
        )
    
    # Analyze cost metrics
    cost_metrics = await _analyze_cost_metrics(story_packs, report_filter)
    
    # Generate cost comparison
    cost_comparison = await _generate_cost_comparison(story_packs, report_filter)
    
    # Calculate ROI analysis
    roi_analysis = await _calculate_roi_analysis(story_packs)
    
    report = CostReport(
        report_id=f"cost_{int(datetime.now().timestamp())}",
        project_id=report_filter.project_id,
        period_start=report_filter.date_from,
        period_end=report_filter.date_to,
        metrics=cost_metrics,
        story_packs_analyzed=len(story_packs),
        cost_comparison=cost_comparison,
        roi_analysis=roi_analysis
    )
    
    return report


@router.post("/alignment-analysis", response_model=AlignmentReport)
async def generate_alignment_report(
    report_filter: ReportFilter,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate cross-modal alignment analysis report."""
    
    # Build query for story packs with evaluations
    query = select(StoryPack).options(
        selectinload(StoryPack.evaluations),
        selectinload(StoryPack.project)
    ).where(StoryPack.evaluations.any())
    
    # Apply filters
    if report_filter.project_id:
        query = query.where(StoryPack.project_id == UUID(report_filter.project_id))
    
    if report_filter.date_from:
        query = query.where(StoryPack.created_at >= report_filter.date_from)
    
    if report_filter.date_to:
        query = query.where(StoryPack.created_at <= report_filter.date_to)
    
    # Execute query
    result = await db.execute(query)
    story_packs = result.scalars().all()
    
    if not story_packs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No story packs with evaluations found"
        )
    
    # Analyze alignment metrics
    alignment_metrics = await _analyze_alignment_metrics(story_packs)
    
    # Generate quality trends
    quality_trends = await _generate_quality_trends(story_packs)
    
    # Generate improvement recommendations
    recommendations = _generate_alignment_recommendations(alignment_metrics, story_packs)
    
    report = AlignmentReport(
        report_id=f"alignment_{int(datetime.now().timestamp())}",
        project_id=report_filter.project_id,
        generated_at=datetime.now(),
        metrics=alignment_metrics,
        story_packs_analyzed=len(story_packs),
        quality_trends=quality_trends,
        improvement_recommendations=recommendations
    )
    
    return report


@router.get("/dashboard-metrics")
async def get_dashboard_metrics(
    project_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get key metrics for dashboard display."""
    
    # Date range
    date_from = datetime.now() - timedelta(days=days)
    
    # Build base query
    query = select(StoryPack).options(selectinload(StoryPack.evaluations))
    
    if project_id:
        query = query.where(StoryPack.project_id == UUID(project_id))
    
    query = query.where(StoryPack.created_at >= date_from)
    
    # Execute query
    result = await db.execute(query)
    story_packs = result.scalars().all()
    
    # Calculate key metrics
    total_packs = len(story_packs)
    
    # Average scores
    avg_scores = {}
    if story_packs:
        all_evaluations = [eval for sp in story_packs for eval in sp.evaluations]
        if all_evaluations:
            avg_scores = {
                'overall_quality': sum(e.overall_quality or 0 for e in all_evaluations) / len(all_evaluations),
                'brand_consistency': sum(e.brand_consistency or 0 for e in all_evaluations) / len(all_evaluations),
                'text_image_alignment': sum(e.text_image_alignment or 0 for e in all_evaluations) / len(all_evaluations),
                'text_audio_alignment': sum(e.text_audio_alignment or 0 for e in all_evaluations) / len(all_evaluations),
            }
    
    # Success rate (packs with overall quality > 0.7)
    successful_packs = len([
        sp for sp in story_packs 
        if any(e.overall_quality and e.overall_quality > 0.7 for e in sp.evaluations)
    ])
    success_rate = (successful_packs / total_packs * 100) if total_packs > 0 else 0
    
    # Cost estimates (simplified)
    estimated_total_cost = total_packs * 2.50  # $2.50 per pack estimate
    avg_cost_per_pack = 2.50
    
    # Trend data (last 7 days)
    trend_data = []
    for i in range(7):
        day_start = datetime.now() - timedelta(days=i+1)
        day_end = datetime.now() - timedelta(days=i)
        
        day_packs = [
            sp for sp in story_packs 
            if day_start <= sp.created_at <= day_end
        ]
        
        trend_data.append({
            'date': day_start.strftime('%Y-%m-%d'),
            'packs_generated': len(day_packs),
            'avg_quality': sum(
                e.overall_quality or 0 
                for sp in day_packs 
                for e in sp.evaluations
            ) / max(1, sum(len(sp.evaluations) for sp in day_packs))
        })
    
    return {
        'summary': {
            'total_story_packs': total_packs,
            'success_rate': round(success_rate, 1),
            'avg_quality_score': round(avg_scores.get('overall_quality', 0), 2),
            'estimated_total_cost': round(estimated_total_cost, 2),
            'avg_cost_per_pack': avg_cost_per_pack
        },
        'quality_metrics': avg_scores,
        'trends': list(reversed(trend_data)),  # Most recent first
        'period': {
            'days': days,
            'from': date_from.isoformat(),
            'to': datetime.now().isoformat()
        }
    }


# Helper functions

async def _analyze_brand_fit_metrics(story_packs: List[StoryPack], brand_kit: Optional[BrandKit]) -> BrandFitMetrics:
    """Analyze brand fit metrics from story packs."""
    
    if not story_packs:
        return BrandFitMetrics(
            overall_score=0.0,
            voice_consistency=0.0,
            visual_alignment=0.0,
            color_compliance=0.0,
            typography_adherence=0.0,
            lexicon_usage=0.0,
            improvement_areas=["No data available"]
        )
    
    # Extract brand consistency scores from evaluations
    brand_scores = []
    for sp in story_packs:
        for eval in sp.evaluations:
            if eval.brand_consistency:
                brand_scores.append(eval.brand_consistency)
    
    if not brand_scores:
        overall_score = 0.5  # Default neutral score
    else:
        overall_score = sum(brand_scores) / len(brand_scores)
    
    # Simulate component scores (in production, these would be calculated from detailed analysis)
    voice_consistency = overall_score * (0.9 + (hash(str(story_packs[0].id)) % 20) / 100)
    visual_alignment = overall_score * (0.85 + (hash(str(story_packs[0].id)) % 30) / 100)
    color_compliance = overall_score * (0.95 + (hash(str(story_packs[0].id)) % 10) / 100)
    typography_adherence = overall_score * (0.8 + (hash(str(story_packs[0].id)) % 40) / 100)
    lexicon_usage = overall_score * (0.75 + (hash(str(story_packs[0].id)) % 50) / 100)
    
    # Identify improvement areas
    improvement_areas = []
    if voice_consistency < 0.7:
        improvement_areas.append("Voice consistency needs improvement")
    if visual_alignment < 0.7:
        improvement_areas.append("Visual brand alignment could be stronger")
    if color_compliance < 0.8:
        improvement_areas.append("Color palette adherence needs attention")
    if typography_adherence < 0.7:
        improvement_areas.append("Typography guidelines not consistently followed")
    if lexicon_usage < 0.7:
        improvement_areas.append("Brand lexicon usage could be improved")
    
    if not improvement_areas:
        improvement_areas.append("Brand consistency is strong across all areas")
    
    return BrandFitMetrics(
        overall_score=overall_score,
        voice_consistency=voice_consistency,
        visual_alignment=visual_alignment,
        color_compliance=color_compliance,
        typography_adherence=typography_adherence,
        lexicon_usage=lexicon_usage,
        improvement_areas=improvement_areas
    )


async def _analyze_cost_metrics(story_packs: List[StoryPack], report_filter: ReportFilter) -> CostMetrics:
    """Analyze cost metrics from story packs."""
    
    # Simplified cost calculation (in production, this would use actual API costs)
    base_cost_per_pack = 2.50
    
    # Cost breakdown by component
    text_cost = 0.30  # Per text generation
    image_cost = 0.80  # Per image (assuming 2-3 images per pack)
    audio_cost = 0.40  # Per audio generation
    processing_cost = 1.00  # Processing and optimization
    
    total_packs = len(story_packs)
    total_cost = total_packs * base_cost_per_pack
    
    cost_per_component = {
        'text_generation': text_cost,
        'image_generation': image_cost * 2.5,  # Average 2.5 images per pack
        'audio_generation': audio_cost,
        'processing_optimization': processing_cost
    }
    
    cost_breakdown = {
        'generation_costs': total_cost * 0.7,
        'processing_costs': total_cost * 0.2,
        'infrastructure_costs': total_cost * 0.1
    }
    
    # Calculate efficiency score (lower cost per quality point is better)
    quality_scores = []
    for sp in story_packs:
        for eval in sp.evaluations:
            if eval.overall_quality:
                quality_scores.append(eval.overall_quality)
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    efficiency_score = avg_quality / base_cost_per_pack if base_cost_per_pack > 0 else 0
    
    # Generate cost trends (simplified)
    cost_trends = []
    days_back = 7
    for i in range(days_back):
        day = datetime.now() - timedelta(days=i)
        day_packs = [sp for sp in story_packs if sp.created_at.date() == day.date()]
        day_cost = len(day_packs) * base_cost_per_pack
        
        cost_trends.append({
            'date': day.strftime('%Y-%m-%d'),
            'cost': day_cost,
            'packs_generated': len(day_packs)
        })
    
    # Optimization suggestions
    optimization_suggestions = []
    if efficiency_score < 0.3:
        optimization_suggestions.append("Consider optimizing generation parameters to improve cost efficiency")
    if total_cost > 100:
        optimization_suggestions.append("High volume detected - consider batch processing discounts")
    if avg_quality < 0.7:
        optimization_suggestions.append("Focus on quality improvements to maximize ROI")
    
    optimization_suggestions.append("Monitor usage patterns to identify cost optimization opportunities")
    
    return CostMetrics(
        total_cost=total_cost,
        cost_per_component=cost_per_component,
        cost_breakdown=cost_breakdown,
        efficiency_score=efficiency_score,
        cost_trends=list(reversed(cost_trends)),
        optimization_suggestions=optimization_suggestions
    )


async def _analyze_alignment_metrics(story_packs: List[StoryPack]) -> AlignmentMetrics:
    """Analyze cross-modal alignment metrics."""
    
    text_image_scores = []
    text_audio_scores = []
    overall_scores = []
    
    for sp in story_packs:
        for eval in sp.evaluations:
            if eval.text_image_alignment:
                text_image_scores.append(eval.text_image_alignment)
            if eval.text_audio_alignment:
                text_audio_scores.append(eval.text_audio_alignment)
            if eval.overall_quality:
                overall_scores.append(eval.overall_quality)
    
    text_image_alignment = sum(text_image_scores) / len(text_image_scores) if text_image_scores else 0.5
    text_audio_alignment = sum(text_audio_scores) / len(text_audio_scores) if text_audio_scores else 0.5
    overall_coherence = sum(overall_scores) / len(overall_scores) if overall_scores else 0.5
    
    # Calculate consistency score (how consistent are the alignment scores)
    all_alignment_scores = text_image_scores + text_audio_scores
    if len(all_alignment_scores) > 1:
        import statistics
        consistency_score = 1.0 - (statistics.stdev(all_alignment_scores) / max(all_alignment_scores))
    else:
        consistency_score = 0.5
    
    # Gap analysis
    gap_analysis = []
    if text_image_alignment < 0.7:
        gap_analysis.append("Text-image alignment needs improvement")
    if text_audio_alignment < 0.7:
        gap_analysis.append("Text-audio alignment could be stronger")
    if consistency_score < 0.8:
        gap_analysis.append("Alignment consistency varies significantly across content")
    if overall_coherence < 0.75:
        gap_analysis.append("Overall content coherence needs attention")
    
    if not gap_analysis:
        gap_analysis.append("Alignment quality is strong across all modalities")
    
    # Alignment trends (simplified)
    alignment_trends = []
    for i in range(7):
        day = datetime.now() - timedelta(days=i)
        day_packs = [sp for sp in story_packs if sp.created_at.date() == day.date()]
        
        day_scores = []
        for sp in day_packs:
            for eval in sp.evaluations:
                if eval.text_image_alignment:
                    day_scores.append(eval.text_image_alignment)
        
        avg_score = sum(day_scores) / len(day_scores) if day_scores else 0
        
        alignment_trends.append({
            'date': day.strftime('%Y-%m-%d'),
            'alignment_score': avg_score,
            'packs_count': len(day_packs)
        })
    
    return AlignmentMetrics(
        text_image_alignment=text_image_alignment,
        text_audio_alignment=text_audio_alignment,
        overall_coherence=overall_coherence,
        consistency_score=consistency_score,
        gap_analysis=gap_analysis,
        alignment_trends=list(reversed(alignment_trends))
    )


def _generate_brand_fit_recommendations(metrics: BrandFitMetrics, story_packs: List[StoryPack]) -> List[str]:
    """Generate brand fit improvement recommendations."""
    
    recommendations = []
    
    if metrics.overall_score < 0.7:
        recommendations.append("Overall brand consistency needs improvement - consider reviewing brand guidelines")
    
    if metrics.voice_consistency < 0.7:
        recommendations.append("Strengthen brand voice consistency in text generation prompts")
    
    if metrics.visual_alignment < 0.7:
        recommendations.append("Improve visual brand alignment by refining image generation parameters")
    
    if metrics.color_compliance < 0.8:
        recommendations.append("Ensure color palette is properly applied across all visual elements")
    
    if metrics.typography_adherence < 0.7:
        recommendations.append("Review typography guidelines and ensure consistent application")
    
    if metrics.lexicon_usage < 0.7:
        recommendations.append("Improve brand lexicon usage in content generation")
    
    # Add positive reinforcement
    if metrics.overall_score > 0.8:
        recommendations.append("Excellent brand consistency - maintain current quality standards")
    
    return recommendations


def _generate_alignment_recommendations(metrics: AlignmentMetrics, story_packs: List[StoryPack]) -> List[str]:
    """Generate alignment improvement recommendations."""
    
    recommendations = []
    
    if metrics.text_image_alignment < 0.7:
        recommendations.append("Improve text-image alignment by refining visual generation prompts")
    
    if metrics.text_audio_alignment < 0.7:
        recommendations.append("Enhance text-audio alignment through better voice and pacing selection")
    
    if metrics.consistency_score < 0.8:
        recommendations.append("Focus on consistency across all content pieces - review generation parameters")
    
    if metrics.overall_coherence < 0.75:
        recommendations.append("Improve overall content coherence through better narrative structure")
    
    # Specific suggestions based on trends
    recent_scores = [trend['alignment_score'] for trend in metrics.alignment_trends[-3:]]
    if recent_scores and all(score < 0.7 for score in recent_scores):
        recommendations.append("Recent alignment scores are declining - investigate generation pipeline")
    
    return recommendations


async def _create_brand_fit_analysis(story_packs: List[StoryPack], brand_kit: Optional[BrandKit]) -> Dict[str, Any]:
    """Create detailed brand fit analysis."""
    
    analysis = {
        'brand_kit_info': {
            'name': brand_kit.name if brand_kit else 'Default',
            'has_color_palette': bool(brand_kit and brand_kit.color_palette),
            'has_typography': bool(brand_kit and brand_kit.typography),
            'has_lexicon': bool(brand_kit and brand_kit.lexicon)
        },
        'content_analysis': {
            'total_story_packs': len(story_packs),
            'date_range': {
                'from': min(sp.created_at for sp in story_packs).isoformat() if story_packs else None,
                'to': max(sp.created_at for sp in story_packs).isoformat() if story_packs else None
            }
        },
        'score_distribution': {
            'excellent': len([sp for sp in story_packs if any(e.brand_consistency and e.brand_consistency > 0.9 for e in sp.evaluations)]),
            'good': len([sp for sp in story_packs if any(e.brand_consistency and 0.7 <= e.brand_consistency <= 0.9 for e in sp.evaluations)]),
            'fair': len([sp for sp in story_packs if any(e.brand_consistency and 0.5 <= e.brand_consistency < 0.7 for e in sp.evaluations)]),
            'poor': len([sp for sp in story_packs if any(e.brand_consistency and e.brand_consistency < 0.5 for e in sp.evaluations)])
        }
    }
    
    return analysis


async def _generate_cost_comparison(story_packs: List[StoryPack], report_filter: ReportFilter) -> Dict[str, Any]:
    """Generate cost comparison analysis."""
    
    # Compare with previous period
    current_period_days = (report_filter.date_to - report_filter.date_from).days
    previous_period_start = report_filter.date_from - timedelta(days=current_period_days)
    previous_period_end = report_filter.date_from
    
    current_cost = len(story_packs) * 2.50
    
    # Simulate previous period data
    previous_packs_count = max(0, len(story_packs) - 5)  # Simplified
    previous_cost = previous_packs_count * 2.50
    
    cost_change = ((current_cost - previous_cost) / previous_cost * 100) if previous_cost > 0 else 0
    
    return {
        'current_period': {
            'cost': current_cost,
            'packs': len(story_packs),
            'avg_cost_per_pack': 2.50
        },
        'previous_period': {
            'cost': previous_cost,
            'packs': previous_packs_count,
            'avg_cost_per_pack': 2.50
        },
        'comparison': {
            'cost_change_percent': cost_change,
            'packs_change': len(story_packs) - previous_packs_count,
            'efficiency_change': 0.0  # Simplified
        }
    }


async def _calculate_roi_analysis(story_packs: List[StoryPack]) -> Dict[str, Any]:
    """Calculate ROI analysis for story packs."""
    
    total_cost = len(story_packs) * 2.50
    
    # Estimate value based on quality scores
    total_value = 0
    for sp in story_packs:
        for eval in sp.evaluations:
            if eval.overall_quality:
                # Assume $10 value per quality point
                total_value += eval.overall_quality * 10
    
    roi = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'total_investment': total_cost,
        'estimated_value': total_value,
        'roi_percentage': roi,
        'payback_period_days': 30 if roi > 0 else None,  # Simplified
        'value_drivers': [
            'High-quality content generation',
            'Time savings from automation',
            'Consistent brand compliance',
            'Scalable content production'
        ]
    }


async def _generate_quality_trends(story_packs: List[StoryPack]) -> List[Dict[str, Any]]:
    """Generate quality trend analysis."""
    
    trends = []
    
    for i in range(30):  # Last 30 days
        day = datetime.now() - timedelta(days=i)
        day_packs = [sp for sp in story_packs if sp.created_at.date() == day.date()]
        
        if day_packs:
            quality_scores = []
            alignment_scores = []
            
            for sp in day_packs:
                for eval in sp.evaluations:
                    if eval.overall_quality:
                        quality_scores.append(eval.overall_quality)
                    if eval.text_image_alignment:
                        alignment_scores.append(eval.text_image_alignment)
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
            
            trends.append({
                'date': day.strftime('%Y-%m-%d'),
                'avg_quality': avg_quality,
                'avg_alignment': avg_alignment,
                'packs_count': len(day_packs)
            })
    
    return list(reversed(trends))  # Most recent first
