"""
Consistency Worker - Ensures cross-modal alignment and brand consistency
"""

import asyncio
import json
import base64
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from PIL import Image
import io
import openai
from crewai import Agent, Task, Crew
from langchain.llms import OpenAI


class ConsistencyType(Enum):
    """Types of consistency checks."""
    TEXT_IMAGE = "text_image"
    TEXT_AUDIO = "text_audio"
    IMAGE_AUDIO = "image_audio"
    BRAND_ALIGNMENT = "brand_alignment"
    TONE_CONSISTENCY = "tone_consistency"
    STYLE_CONSISTENCY = "style_consistency"


@dataclass
class ConsistencyRequest:
    """Request for consistency analysis."""
    content_items: Dict[str, any]  # {type: content} - text, images, audio
    brand_guidelines: Optional[Dict] = None
    consistency_thresholds: Optional[Dict[str, float]] = None
    reference_content: Optional[Dict] = None


@dataclass
class ConsistencyIssue:
    """Identified consistency issue."""
    issue_type: ConsistencyType
    severity: float  # 0.0 to 1.0
    description: str
    affected_content: List[str]
    suggested_fixes: List[str]


@dataclass
class ConsistencyResult:
    """Result of consistency analysis."""
    overall_score: float
    individual_scores: Dict[str, float]
    issues: List[ConsistencyIssue]
    recommendations: List[str]
    analysis_log: List[Dict]


class ConsistencyWorker:
    """AI-powered consistency analysis and alignment worker."""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.2,  # Low temperature for consistent analysis
        )
        
        # Define consistency analysis agents
        self.text_analyst = Agent(
            role='Text Consistency Analyst',
            goal='Analyze text content for tone, style, and brand consistency',
            backstory="""You are an expert in content analysis and brand consistency. 
            You excel at identifying inconsistencies in tone, style, messaging, 
            and brand alignment across text content.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.visual_analyst = Agent(
            role='Visual Consistency Analyst',
            goal='Analyze visual content for style, color, and brand consistency',
            backstory="""You are a visual design expert who specializes in brand 
            consistency and visual alignment. You can identify style inconsistencies, 
            color palette deviations, and visual brand guideline violations.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.cross_modal_analyst = Agent(
            role='Cross-Modal Alignment Specialist',
            goal='Analyze alignment between different content modalities',
            backstory="""You are an expert in multimodal content analysis. You 
            understand how text, images, and audio should work together cohesively 
            and can identify misalignments between different content types.""",
            llm=self.llm,
            verbose=True,
        )
        
        # Default consistency thresholds
        self.default_thresholds = {
            'text_image_alignment': 0.7,
            'text_audio_alignment': 0.7,
            'brand_consistency': 0.8,
            'tone_consistency': 0.75,
            'style_consistency': 0.7,
            'overall_minimum': 0.7
        }

    async def analyze_consistency(
        self,
        request: ConsistencyRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Analyze content consistency across modalities."""
        
        analysis_log = []
        start_time = datetime.now()
        thresholds = request.consistency_thresholds or self.default_thresholds
        
        try:
            # Step 1: Individual content analysis
            yield {'stage': 'individual_analysis', 'status': 'starting', 'progress': 0.1}
            
            individual_scores = {}
            content_analyses = {}
            
            # Analyze text content
            if 'text' in request.content_items:
                text_analysis = await self._analyze_text_consistency(
                    request.content_items['text'],
                    request.brand_guidelines
                )
                content_analyses['text'] = text_analysis
                individual_scores['text_consistency'] = text_analysis.get('consistency_score', 0.5)
            
            # Analyze visual content
            if 'images' in request.content_items:
                visual_analysis = await self._analyze_visual_consistency(
                    request.content_items['images'],
                    request.brand_guidelines
                )
                content_analyses['visual'] = visual_analysis
                individual_scores['visual_consistency'] = visual_analysis.get('consistency_score', 0.5)
            
            analysis_log.append({
                'stage': 'individual_analysis',
                'timestamp': datetime.now().isoformat(),
                'scores': individual_scores
            })
            
            yield {
                'stage': 'individual_analysis',
                'status': 'completed',
                'progress': 0.4,
                'individual_scores': individual_scores
            }
            
            # Step 2: Cross-modal alignment analysis
            yield {'stage': 'cross_modal_analysis', 'status': 'starting', 'progress': 0.45}
            
            cross_modal_scores = {}
            cross_modal_issues = []
            
            # Text-Image alignment
            if 'text' in request.content_items and 'images' in request.content_items:
                text_image_result = await self._analyze_text_image_alignment(
                    request.content_items['text'],
                    request.content_items['images']
                )
                cross_modal_scores['text_image_alignment'] = text_image_result['score']
                cross_modal_issues.extend(text_image_result.get('issues', []))
            
            # Text-Audio alignment
            if 'text' in request.content_items and 'audio' in request.content_items:
                text_audio_result = await self._analyze_text_audio_alignment(
                    request.content_items['text'],
                    request.content_items['audio']
                )
                cross_modal_scores['text_audio_alignment'] = text_audio_result['score']
                cross_modal_issues.extend(text_audio_result.get('issues', []))
            
            analysis_log.append({
                'stage': 'cross_modal_analysis',
                'timestamp': datetime.now().isoformat(),
                'scores': cross_modal_scores,
                'issues_found': len(cross_modal_issues)
            })
            
            yield {
                'stage': 'cross_modal_analysis',
                'status': 'completed',
                'progress': 0.7,
                'cross_modal_scores': cross_modal_scores,
                'issues_count': len(cross_modal_issues)
            }
            
            # Step 3: Brand consistency analysis
            yield {'stage': 'brand_analysis', 'status': 'starting', 'progress': 0.75}
            
            brand_analysis = await self._analyze_brand_consistency(
                request.content_items,
                request.brand_guidelines
            )
            
            brand_score = brand_analysis.get('consistency_score', 0.5)
            brand_issues = brand_analysis.get('issues', [])
            
            analysis_log.append({
                'stage': 'brand_analysis',
                'timestamp': datetime.now().isoformat(),
                'brand_score': brand_score,
                'brand_issues': len(brand_issues)
            })
            
            # Step 4: Generate recommendations
            yield {'stage': 'recommendations', 'status': 'starting', 'progress': 0.85}
            
            all_scores = {**individual_scores, **cross_modal_scores, 'brand_consistency': brand_score}
            all_issues = cross_modal_issues + brand_issues
            
            recommendations = await self._generate_recommendations(
                all_scores,
                all_issues,
                thresholds,
                request.brand_guidelines
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_consistency_score(all_scores)
            
            # Final result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = ConsistencyResult(
                overall_score=overall_score,
                individual_scores=all_scores,
                issues=all_issues,
                recommendations=recommendations,
                analysis_log=analysis_log
            )
            
            yield {
                'stage': 'complete',
                'status': 'success',
                'progress': 1.0,
                'result': result,
                'duration': duration
            }
            
        except Exception as e:
            analysis_log.append({
                'stage': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            yield {
                'stage': 'error',
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'analysis_log': analysis_log
            }

    async def _analyze_text_consistency(
        self,
        text_content: str,
        brand_guidelines: Optional[Dict]
    ) -> Dict:
        """Analyze text content for internal consistency and brand alignment."""
        
        brand_info = json.dumps(brand_guidelines, indent=2) if brand_guidelines else "No specific guidelines"
        
        text_analysis_task = Task(
            description=f"""
            Analyze this text content for consistency and brand alignment:
            
            Text Content: {text_content}
            
            Brand Guidelines: {brand_info}
            
            Evaluate:
            1. Tone consistency throughout the text
            2. Style and voice consistency
            3. Message clarity and coherence
            4. Brand alignment (if guidelines provided)
            5. Terminology consistency
            6. Overall readability and flow
            
            Provide scores (0.0 to 1.0) for each aspect and identify specific issues.
            
            Return as JSON with:
            - consistency_score (overall)
            - tone_score
            - style_score
            - brand_alignment_score
            - issues (list of specific problems)
            - strengths (list of positive aspects)
            """,
            agent=self.text_analyst,
            expected_output="JSON analysis of text consistency with scores and issues"
        )
        
        crew = Crew(
            agents=[self.text_analyst],
            tasks=[text_analysis_task],
            verbose=True
        )
        
        analysis_result = crew.kickoff()
        
        try:
            return json.loads(str(analysis_result))
        except json.JSONDecodeError:
            return {
                "consistency_score": 0.6,
                "tone_score": 0.6,
                "style_score": 0.6,
                "brand_alignment_score": 0.5,
                "issues": ["Unable to parse detailed analysis"],
                "strengths": []
            }

    async def _analyze_visual_consistency(
        self,
        images: List[str],  # Base64 encoded images
        brand_guidelines: Optional[Dict]
    ) -> Dict:
        """Analyze visual content for style and brand consistency."""
        
        # Extract visual features from images
        visual_features = []
        for i, image_b64 in enumerate(images):
            features = await self._extract_visual_features(image_b64)
            visual_features.append(features)
        
        brand_info = json.dumps(brand_guidelines, indent=2) if brand_guidelines else "No specific guidelines"
        features_summary = json.dumps(visual_features, indent=2)
        
        visual_analysis_task = Task(
            description=f"""
            Analyze visual content consistency across {len(images)} images:
            
            Visual Features Summary: {features_summary}
            
            Brand Guidelines: {brand_info}
            
            Evaluate:
            1. Color palette consistency across images
            2. Style and aesthetic consistency
            3. Composition and layout consistency
            4. Brand guideline adherence (colors, fonts, logos)
            5. Overall visual cohesion
            6. Quality and technical consistency
            
            Consider:
            - Are colors consistent with brand palette?
            - Do images share a cohesive visual style?
            - Are there jarring inconsistencies?
            - Do images work well together as a set?
            
            Return as JSON with:
            - consistency_score (overall)
            - color_consistency_score
            - style_consistency_score
            - brand_adherence_score
            - issues (list of specific problems)
            - recommendations (list of improvements)
            """,
            agent=self.visual_analyst,
            expected_output="JSON analysis of visual consistency with scores and recommendations"
        )
        
        crew = Crew(
            agents=[self.visual_analyst],
            tasks=[visual_analysis_task],
            verbose=True
        )
        
        analysis_result = crew.kickoff()
        
        try:
            return json.loads(str(analysis_result))
        except json.JSONDecodeError:
            return {
                "consistency_score": 0.6,
                "color_consistency_score": 0.6,
                "style_consistency_score": 0.6,
                "brand_adherence_score": 0.5,
                "issues": ["Unable to parse detailed visual analysis"],
                "recommendations": []
            }

    async def _analyze_text_image_alignment(
        self,
        text_content: str,
        images: List[str]
    ) -> Dict:
        """Analyze alignment between text and visual content."""
        
        # Extract key themes from text
        text_themes = await self._extract_text_themes(text_content)
        
        # Extract visual elements from images
        visual_elements = []
        for image_b64 in images:
            elements = await self._describe_image_content(image_b64)
            visual_elements.append(elements)
        
        alignment_task = Task(
            description=f"""
            Analyze alignment between text content and visual elements:
            
            Text Themes: {json.dumps(text_themes, indent=2)}
            
            Visual Elements: {json.dumps(visual_elements, indent=2)}
            
            Evaluate:
            1. Do images support and reinforce the text message?
            2. Are visual elements relevant to the text content?
            3. Is there good thematic alignment?
            4. Do images enhance understanding of the text?
            5. Are there any conflicting messages?
            
            Consider:
            - Semantic alignment (do visuals match text meaning?)
            - Emotional alignment (do visuals match text tone?)
            - Contextual relevance
            - Narrative coherence
            
            Return as JSON with:
            - score (0.0 to 1.0 alignment score)
            - semantic_alignment
            - emotional_alignment
            - contextual_relevance
            - issues (list of misalignments)
            - suggestions (list of improvements)
            """,
            agent=self.cross_modal_analyst,
            expected_output="JSON analysis of text-image alignment with detailed scores"
        )
        
        crew = Crew(
            agents=[self.cross_modal_analyst],
            tasks=[alignment_task],
            verbose=True
        )
        
        alignment_result = crew.kickoff()
        
        try:
            result = json.loads(str(alignment_result))
            # Convert issues to ConsistencyIssue objects
            issues = []
            for issue_text in result.get('issues', []):
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyType.TEXT_IMAGE,
                    severity=0.7,  # Default severity
                    description=issue_text,
                    affected_content=['text', 'images'],
                    suggested_fixes=result.get('suggestions', [])
                ))
            result['issues'] = issues
            return result
        except json.JSONDecodeError:
            return {
                "score": 0.6,
                "semantic_alignment": 0.6,
                "emotional_alignment": 0.6,
                "contextual_relevance": 0.6,
                "issues": [],
                "suggestions": []
            }

    async def _analyze_text_audio_alignment(
        self,
        text_content: str,
        audio_metadata: Dict
    ) -> Dict:
        """Analyze alignment between text and audio content."""
        
        # For now, analyze based on metadata since we don't have audio processing
        # In production, you'd analyze actual audio content
        
        alignment_task = Task(
            description=f"""
            Analyze alignment between text content and audio characteristics:
            
            Text Content: {text_content[:500]}...
            
            Audio Metadata: {json.dumps(audio_metadata, indent=2)}
            
            Evaluate based on available information:
            1. Does the audio duration match text length appropriately?
            2. Does the voice style match the text tone?
            3. Is the pacing appropriate for the content?
            4. Are there any obvious mismatches?
            
            Consider:
            - Text complexity vs. audio pacing
            - Emotional tone alignment
            - Content appropriateness
            - Technical quality consistency
            
            Return as JSON with:
            - score (0.0 to 1.0 alignment score)
            - duration_appropriateness
            - tone_alignment
            - pacing_suitability
            - issues (list of problems)
            - suggestions (list of improvements)
            """,
            agent=self.cross_modal_analyst,
            expected_output="JSON analysis of text-audio alignment"
        )
        
        crew = Crew(
            agents=[self.cross_modal_analyst],
            tasks=[alignment_task],
            verbose=True
        )
        
        alignment_result = crew.kickoff()
        
        try:
            result = json.loads(str(alignment_result))
            # Convert issues to ConsistencyIssue objects
            issues = []
            for issue_text in result.get('issues', []):
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyType.TEXT_AUDIO,
                    severity=0.6,
                    description=issue_text,
                    affected_content=['text', 'audio'],
                    suggested_fixes=result.get('suggestions', [])
                ))
            result['issues'] = issues
            return result
        except json.JSONDecodeError:
            return {
                "score": 0.7,
                "duration_appropriateness": 0.7,
                "tone_alignment": 0.7,
                "pacing_suitability": 0.7,
                "issues": [],
                "suggestions": []
            }

    async def _analyze_brand_consistency(
        self,
        content_items: Dict,
        brand_guidelines: Optional[Dict]
    ) -> Dict:
        """Analyze overall brand consistency across all content."""
        
        if not brand_guidelines:
            return {
                "consistency_score": 0.8,  # Assume good if no guidelines to check
                "issues": [],
                "recommendations": []
            }
        
        content_summary = {}
        for content_type, content in content_items.items():
            if content_type == 'text':
                content_summary[content_type] = content[:200] + "..." if len(content) > 200 else content
            elif content_type == 'images':
                content_summary[content_type] = f"{len(content)} images provided"
            else:
                content_summary[content_type] = str(type(content))
        
        brand_task = Task(
            description=f"""
            Analyze brand consistency across all content types:
            
            Content Summary: {json.dumps(content_summary, indent=2)}
            
            Brand Guidelines: {json.dumps(brand_guidelines, indent=2)}
            
            Check for:
            1. Adherence to brand voice and tone guidelines
            2. Consistent use of brand terminology
            3. Alignment with brand values and messaging
            4. Proper use of brand colors (if specified)
            5. Consistency with brand personality
            6. Overall brand coherence across modalities
            
            Identify:
            - Specific guideline violations
            - Inconsistencies between content types
            - Areas where brand could be strengthened
            - Positive brand alignment examples
            
            Return as JSON with:
            - consistency_score (0.0 to 1.0)
            - voice_alignment
            - terminology_consistency
            - value_alignment
            - issues (list of specific violations)
            - recommendations (list of improvements)
            """,
            agent=self.text_analyst,  # Use text analyst for brand analysis
            expected_output="JSON analysis of brand consistency with specific issues and recommendations"
        )
        
        crew = Crew(
            agents=[self.text_analyst],
            tasks=[brand_task],
            verbose=True
        )
        
        brand_result = crew.kickoff()
        
        try:
            result = json.loads(str(brand_result))
            # Convert issues to ConsistencyIssue objects
            issues = []
            for issue_text in result.get('issues', []):
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyType.BRAND_ALIGNMENT,
                    severity=0.8,  # Brand issues are typically high severity
                    description=issue_text,
                    affected_content=list(content_items.keys()),
                    suggested_fixes=result.get('recommendations', [])
                ))
            result['issues'] = issues
            return result
        except json.JSONDecodeError:
            return {
                "consistency_score": 0.7,
                "voice_alignment": 0.7,
                "terminology_consistency": 0.7,
                "value_alignment": 0.7,
                "issues": [],
                "recommendations": []
            }

    async def _extract_visual_features(self, image_b64: str) -> Dict:
        """Extract visual features from an image."""
        
        try:
            # Decode image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Extract basic features
            features = {
                "dimensions": image.size,
                "mode": image.mode,
                "format": getattr(image, 'format', 'Unknown')
            }
            
            # Extract dominant colors (simplified)
            if image.mode == 'RGB':
                # Convert to numpy array and find dominant colors
                img_array = np.array(image)
                # Reshape to list of pixels
                pixels = img_array.reshape(-1, 3)
                # Simple dominant color extraction (would use clustering in production)
                mean_color = np.mean(pixels, axis=0)
                features["dominant_color"] = mean_color.tolist()
            
            return features
            
        except Exception as e:
            return {"error": str(e), "dimensions": [0, 0]}

    async def _extract_text_themes(self, text: str) -> List[str]:
        """Extract key themes from text content."""
        
        # Simple keyword extraction (in production, use more sophisticated NLP)
        words = text.lower().split()
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        meaningful_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Return top themes (simplified)
        return meaningful_words[:10]

    async def _describe_image_content(self, image_b64: str) -> str:
        """Get description of image content using GPT-4V."""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe the key visual elements, colors, style, and mood of this image in a concise way."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Unable to analyze image: {str(e)}"

    async def _generate_recommendations(
        self,
        scores: Dict[str, float],
        issues: List[ConsistencyIssue],
        thresholds: Dict[str, float],
        brand_guidelines: Optional[Dict]
    ) -> List[str]:
        """Generate actionable recommendations for improving consistency."""
        
        recommendations = []
        
        # Check scores against thresholds
        for metric, score in scores.items():
            threshold = thresholds.get(metric, 0.7)
            if score < threshold:
                gap = threshold - score
                if gap > 0.2:
                    recommendations.append(f"Critical improvement needed in {metric.replace('_', ' ')}: current score {score:.2f}, target {threshold:.2f}")
                else:
                    recommendations.append(f"Minor improvement needed in {metric.replace('_', ' ')}: current score {score:.2f}, target {threshold:.2f}")
        
        # Add specific recommendations based on issues
        high_severity_issues = [issue for issue in issues if issue.severity > 0.7]
        if high_severity_issues:
            recommendations.append(f"Address {len(high_severity_issues)} high-severity consistency issues")
        
        # Add general recommendations
        if scores.get('overall_score', 0) < 0.8:
            recommendations.append("Consider reviewing all content for better cross-modal alignment")
        
        if brand_guidelines and scores.get('brand_consistency', 0) < 0.8:
            recommendations.append("Strengthen adherence to brand guidelines across all content types")
        
        return recommendations

    def _calculate_overall_consistency_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall consistency score from individual metrics."""
        
        if not scores:
            return 0.0
        
        # Weighted scoring
        weights = {
            'text_image_alignment': 0.25,
            'text_audio_alignment': 0.20,
            'brand_consistency': 0.25,
            'text_consistency': 0.15,
            'visual_consistency': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def fix_consistency_issues(
        self,
        issues: List[ConsistencyIssue],
        content_items: Dict,
        brand_guidelines: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Generate specific fixes for identified consistency issues."""
        
        fixes = {}
        
        for issue in issues:
            if issue.issue_type == ConsistencyType.TEXT_IMAGE:
                # Generate text or image modification suggestions
                fixes[f"text_image_fix_{len(fixes)}"] = f"To address '{issue.description}': {'; '.join(issue.suggested_fixes)}"
            
            elif issue.issue_type == ConsistencyType.BRAND_ALIGNMENT:
                # Generate brand alignment fixes
                fixes[f"brand_fix_{len(fixes)}"] = f"Brand consistency issue: {issue.description}. Suggested fixes: {'; '.join(issue.suggested_fixes)}"
        
        return fixes
