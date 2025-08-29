"""
Evaluation Worker - Comprehensive quality assessment across all modalities
"""

import asyncio
import json
import base64
import re
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
import textstat
from transformers import pipeline


class EvaluationMetric(Enum):
    """Evaluation metrics for content quality."""
    ALIGNMENT_SCORE = "alignment_score"
    READABILITY = "readability"
    AUDIO_MOS = "audio_mos"  # Mean Opinion Score proxy
    IMAGE_QUALITY = "image_quality"
    BRAND_CONSISTENCY = "brand_consistency"
    ENGAGEMENT_POTENTIAL = "engagement_potential"
    TECHNICAL_QUALITY = "technical_quality"
    COHERENCE = "coherence"


@dataclass
class EvaluationRequest:
    """Request for comprehensive evaluation."""
    story_pack_id: str
    content: Dict[str, any]  # {text, images, audio, metadata}
    brand_guidelines: Optional[Dict] = None
    target_audience: Optional[str] = None
    evaluation_criteria: Optional[List[EvaluationMetric]] = None


@dataclass
class EvaluationScore:
    """Individual evaluation score."""
    metric: EvaluationMetric
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    details: Dict
    recommendations: List[str]


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    overall_score: float
    individual_scores: List[EvaluationScore]
    summary_report: str
    detailed_analysis: Dict
    improvement_suggestions: List[str]
    evaluation_metadata: Dict


class EvaluationWorker:
    """Comprehensive content evaluation system."""
    
    def __init__(self, openai_api_key: str, huggingface_token: Optional[str] = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1,  # Low temperature for consistent evaluation
        )
        
        # Initialize evaluation agents
        self.text_evaluator = Agent(
            role='Text Quality Evaluator',
            goal='Assess text content for readability, coherence, and engagement',
            backstory="""You are an expert in content quality assessment with deep 
            knowledge of readability metrics, engagement factors, and text coherence. 
            You provide detailed, objective evaluations of written content.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.visual_evaluator = Agent(
            role='Visual Quality Evaluator',
            goal='Assess image quality, composition, and visual appeal',
            backstory="""You are a visual design expert who evaluates image quality, 
            composition, color harmony, and overall visual appeal. You understand 
            technical image metrics and aesthetic principles.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.alignment_evaluator = Agent(
            role='Cross-Modal Alignment Evaluator',
            goal='Assess alignment and coherence across different content modalities',
            backstory="""You specialize in evaluating how well different content 
            types work together. You assess semantic alignment, emotional coherence, 
            and narrative consistency across text, images, and audio.""",
            llm=self.llm,
            verbose=True,
        )
        
        # Initialize ML models for technical evaluation
        try:
            if huggingface_token:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    use_auth_token=huggingface_token
                )
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    use_auth_token=huggingface_token
                )
            else:
                self.sentiment_analyzer = None
                self.toxicity_classifier = None
        except Exception as e:
            print(f"Warning: Could not load ML models: {e}")
            self.sentiment_analyzer = None
            self.toxicity_classifier = None

    async def evaluate_content(
        self,
        request: EvaluationRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Perform comprehensive content evaluation."""
        
        evaluation_log = []
        start_time = datetime.now()
        
        # Default evaluation criteria if not specified
        criteria = request.evaluation_criteria or [
            EvaluationMetric.ALIGNMENT_SCORE,
            EvaluationMetric.READABILITY,
            EvaluationMetric.IMAGE_QUALITY,
            EvaluationMetric.BRAND_CONSISTENCY,
            EvaluationMetric.ENGAGEMENT_POTENTIAL
        ]
        
        try:
            # Step 1: Text evaluation
            yield {'stage': 'text_evaluation', 'status': 'starting', 'progress': 0.1}
            
            text_scores = []
            if 'text' in request.content and EvaluationMetric.READABILITY in criteria:
                readability_score = await self._evaluate_text_readability(
                    request.content['text'],
                    request.target_audience
                )
                text_scores.append(readability_score)
            
            if 'text' in request.content and EvaluationMetric.ENGAGEMENT_POTENTIAL in criteria:
                engagement_score = await self._evaluate_text_engagement(
                    request.content['text'],
                    request.target_audience
                )
                text_scores.append(engagement_score)
            
            evaluation_log.append({
                'stage': 'text_evaluation',
                'timestamp': datetime.now().isoformat(),
                'scores_generated': len(text_scores)
            })
            
            yield {
                'stage': 'text_evaluation',
                'status': 'completed',
                'progress': 0.3,
                'text_scores': [score.score for score in text_scores]
            }
            
            # Step 2: Visual evaluation
            yield {'stage': 'visual_evaluation', 'status': 'starting', 'progress': 0.35}
            
            visual_scores = []
            if 'images' in request.content and EvaluationMetric.IMAGE_QUALITY in criteria:
                for i, image_data in enumerate(request.content['images']):
                    image_score = await self._evaluate_image_quality(
                        image_data,
                        f"image_{i}"
                    )
                    visual_scores.append(image_score)
            
            evaluation_log.append({
                'stage': 'visual_evaluation',
                'timestamp': datetime.now().isoformat(),
                'images_evaluated': len(visual_scores)
            })
            
            yield {
                'stage': 'visual_evaluation',
                'status': 'completed',
                'progress': 0.5,
                'visual_scores': [score.score for score in visual_scores]
            }
            
            # Step 3: Audio evaluation
            yield {'stage': 'audio_evaluation', 'status': 'starting', 'progress': 0.55}
            
            audio_scores = []
            if 'audio' in request.content and EvaluationMetric.AUDIO_MOS in criteria:
                audio_score = await self._evaluate_audio_quality(
                    request.content['audio']
                )
                audio_scores.append(audio_score)
            
            yield {
                'stage': 'audio_evaluation',
                'status': 'completed',
                'progress': 0.7,
                'audio_scores': [score.score for score in audio_scores]
            }
            
            # Step 4: Cross-modal alignment
            yield {'stage': 'alignment_evaluation', 'status': 'starting', 'progress': 0.75}
            
            alignment_scores = []
            if EvaluationMetric.ALIGNMENT_SCORE in criteria:
                alignment_score = await self._evaluate_cross_modal_alignment(
                    request.content,
                    request.brand_guidelines
                )
                alignment_scores.append(alignment_score)
            
            # Step 5: Brand consistency
            brand_scores = []
            if EvaluationMetric.BRAND_CONSISTENCY in criteria and request.brand_guidelines:
                brand_score = await self._evaluate_brand_consistency(
                    request.content,
                    request.brand_guidelines
                )
                brand_scores.append(brand_score)
            
            evaluation_log.append({
                'stage': 'alignment_and_brand',
                'timestamp': datetime.now().isoformat(),
                'alignment_scores': len(alignment_scores),
                'brand_scores': len(brand_scores)
            })
            
            yield {
                'stage': 'alignment_evaluation',
                'status': 'completed',
                'progress': 0.9,
                'alignment_score': alignment_scores[0].score if alignment_scores else None,
                'brand_score': brand_scores[0].score if brand_scores else None
            }
            
            # Step 6: Generate comprehensive report
            yield {'stage': 'report_generation', 'status': 'starting', 'progress': 0.95}
            
            all_scores = text_scores + visual_scores + audio_scores + alignment_scores + brand_scores
            overall_score = self._calculate_overall_score(all_scores)
            
            summary_report = await self._generate_summary_report(
                all_scores,
                overall_score,
                request.content,
                request.brand_guidelines
            )
            
            improvement_suggestions = await self._generate_improvement_suggestions(
                all_scores,
                request.content
            )
            
            # Final result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = EvaluationResult(
                overall_score=overall_score,
                individual_scores=all_scores,
                summary_report=summary_report,
                detailed_analysis={
                    'text_analysis': [score.details for score in text_scores],
                    'visual_analysis': [score.details for score in visual_scores],
                    'audio_analysis': [score.details for score in audio_scores],
                    'alignment_analysis': [score.details for score in alignment_scores],
                    'brand_analysis': [score.details for score in brand_scores]
                },
                improvement_suggestions=improvement_suggestions,
                evaluation_metadata={
                    'evaluation_time': duration,
                    'criteria_evaluated': [metric.value for metric in criteria],
                    'total_scores': len(all_scores),
                    'story_pack_id': request.story_pack_id
                }
            )
            
            yield {
                'stage': 'complete',
                'status': 'success',
                'progress': 1.0,
                'result': result
            }
            
        except Exception as e:
            evaluation_log.append({
                'stage': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            yield {
                'stage': 'error',
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'evaluation_log': evaluation_log
            }

    async def _evaluate_text_readability(
        self,
        text: str,
        target_audience: Optional[str] = None
    ) -> EvaluationScore:
        """Evaluate text readability using multiple metrics."""
        
        # Calculate readability metrics
        flesch_score = textstat.flesch_reading_ease(text)
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        smog_index = textstat.smog_index(text)
        
        # Normalize scores to 0-1 scale
        # Flesch Reading Ease: 0-100, higher is better
        flesch_normalized = min(1.0, max(0.0, flesch_score / 100.0))
        
        # Grade levels: lower is generally better for readability
        # Target around 8th grade (8.0) for general audience
        target_grade = 8.0 if not target_audience else self._get_target_grade(target_audience)
        grade_score = max(0.0, 1.0 - abs(flesch_kincaid - target_grade) / 10.0)
        
        # Combine metrics
        readability_score = (flesch_normalized * 0.4 + grade_score * 0.6)
        
        # AI-powered readability assessment
        ai_assessment = await self._ai_readability_assessment(text, target_audience)
        
        # Combine technical and AI scores
        final_score = (readability_score * 0.6 + ai_assessment['score'] * 0.4)
        
        details = {
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': flesch_kincaid,
            'gunning_fog_index': gunning_fog,
            'smog_index': smog_index,
            'target_grade': target_grade,
            'ai_assessment': ai_assessment,
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text))
        }
        
        recommendations = []
        if flesch_score < 60:
            recommendations.append("Consider using shorter sentences and simpler words")
        if flesch_kincaid > target_grade + 2:
            recommendations.append(f"Text grade level ({flesch_kincaid:.1f}) is above target ({target_grade})")
        if gunning_fog > 12:
            recommendations.append("Reduce complex sentences and technical jargon")
        
        return EvaluationScore(
            metric=EvaluationMetric.READABILITY,
            score=final_score,
            confidence=0.85,
            details=details,
            recommendations=recommendations
        )

    async def _evaluate_text_engagement(
        self,
        text: str,
        target_audience: Optional[str] = None
    ) -> EvaluationScore:
        """Evaluate text engagement potential."""
        
        engagement_task = Task(
            description=f"""
            Evaluate the engagement potential of this text content:
            
            Text: {text[:1000]}...
            Target Audience: {target_audience or 'General'}
            
            Assess:
            1. Hook strength - Does it grab attention immediately?
            2. Emotional appeal - Does it evoke emotions?
            3. Clarity and flow - Is it easy to follow?
            4. Call to action - Does it motivate action/response?
            5. Storytelling elements - Does it use narrative techniques?
            6. Relevance to audience - Is it appropriate for the target audience?
            
            Consider factors like:
            - Opening strength
            - Use of active voice
            - Emotional language
            - Concrete examples
            - Rhythm and pacing
            - Memorable phrases
            
            Return a JSON with:
            - engagement_score (0.0 to 1.0)
            - hook_strength (0.0 to 1.0)
            - emotional_appeal (0.0 to 1.0)
            - clarity_flow (0.0 to 1.0)
            - call_to_action (0.0 to 1.0)
            - storytelling (0.0 to 1.0)
            - audience_relevance (0.0 to 1.0)
            - strengths (list of positive aspects)
            - weaknesses (list of areas for improvement)
            """,
            agent=self.text_evaluator,
            expected_output="JSON evaluation of text engagement with detailed scores"
        )
        
        crew = Crew(
            agents=[self.text_evaluator],
            tasks=[engagement_task],
            verbose=True
        )
        
        engagement_result = crew.kickoff()
        
        try:
            result_data = json.loads(str(engagement_result))
            
            # Technical engagement metrics
            technical_metrics = self._calculate_technical_engagement_metrics(text)
            
            # Combine AI and technical scores
            ai_score = result_data.get('engagement_score', 0.6)
            technical_score = technical_metrics['overall_score']
            final_score = (ai_score * 0.7 + technical_score * 0.3)
            
            details = {
                **result_data,
                'technical_metrics': technical_metrics,
                'combined_score': final_score
            }
            
            recommendations = result_data.get('weaknesses', [])
            
            return EvaluationScore(
                metric=EvaluationMetric.ENGAGEMENT_POTENTIAL,
                score=final_score,
                confidence=0.75,
                details=details,
                recommendations=recommendations
            )
            
        except json.JSONDecodeError:
            # Fallback to technical metrics only
            technical_metrics = self._calculate_technical_engagement_metrics(text)
            
            return EvaluationScore(
                metric=EvaluationMetric.ENGAGEMENT_POTENTIAL,
                score=technical_metrics['overall_score'],
                confidence=0.6,
                details=technical_metrics,
                recommendations=["Consider adding more engaging elements"]
            )

    async def _evaluate_image_quality(
        self,
        image_data: str,  # Base64 encoded
        image_id: str
    ) -> EvaluationScore:
        """Evaluate image quality using technical and aesthetic metrics."""
        
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Technical quality metrics
            technical_metrics = self._calculate_image_technical_metrics(image)
            
            # AI-powered aesthetic evaluation
            aesthetic_assessment = await self._ai_image_assessment(image_data)
            
            # Combine scores
            technical_score = technical_metrics['overall_score']
            aesthetic_score = aesthetic_assessment.get('aesthetic_score', 0.6)
            final_score = (technical_score * 0.4 + aesthetic_score * 0.6)
            
            details = {
                'technical_metrics': technical_metrics,
                'aesthetic_assessment': aesthetic_assessment,
                'image_id': image_id,
                'dimensions': image.size,
                'format': image.format,
                'mode': image.mode
            }
            
            recommendations = []
            if technical_metrics['resolution_score'] < 0.7:
                recommendations.append("Consider using higher resolution images")
            if technical_metrics['aspect_ratio_score'] < 0.8:
                recommendations.append("Check aspect ratio for optimal display")
            if aesthetic_score < 0.6:
                recommendations.append("Improve composition and visual appeal")
            
            return EvaluationScore(
                metric=EvaluationMetric.IMAGE_QUALITY,
                score=final_score,
                confidence=0.8,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            return EvaluationScore(
                metric=EvaluationMetric.IMAGE_QUALITY,
                score=0.5,
                confidence=0.3,
                details={'error': str(e), 'image_id': image_id},
                recommendations=["Unable to evaluate image quality"]
            )

    async def _evaluate_audio_quality(
        self,
        audio_metadata: Dict
    ) -> EvaluationScore:
        """Evaluate audio quality using metadata and proxy metrics."""
        
        # Extract audio characteristics from metadata
        duration = audio_metadata.get('total_duration', 0)
        sample_rate = audio_metadata.get('sample_rate', 44100)
        bit_depth = audio_metadata.get('bit_depth', 16)
        voice_used = audio_metadata.get('voice_used', 'unknown')
        
        # Calculate technical quality score
        technical_score = 0.0
        
        # Sample rate scoring (44.1kHz is standard)
        if sample_rate >= 44100:
            sample_rate_score = 1.0
        elif sample_rate >= 22050:
            sample_rate_score = 0.7
        else:
            sample_rate_score = 0.4
        
        # Bit depth scoring (16-bit minimum)
        if bit_depth >= 24:
            bit_depth_score = 1.0
        elif bit_depth >= 16:
            bit_depth_score = 0.8
        else:
            bit_depth_score = 0.5
        
        # Duration appropriateness (assume 1-5 minutes is optimal)
        if 60 <= duration <= 300:  # 1-5 minutes
            duration_score = 1.0
        elif 30 <= duration <= 600:  # 30s-10 minutes
            duration_score = 0.8
        else:
            duration_score = 0.6
        
        # Voice quality (based on known good voices)
        good_voices = ['nova', 'shimmer', 'echo', 'alloy']
        voice_score = 0.9 if voice_used in good_voices else 0.7
        
        technical_score = (
            sample_rate_score * 0.25 +
            bit_depth_score * 0.25 +
            duration_score * 0.25 +
            voice_score * 0.25
        )
        
        # Estimate MOS (Mean Opinion Score) proxy
        # Based on technical parameters and generation settings
        mos_proxy = min(5.0, max(1.0, technical_score * 4 + 1))
        
        details = {
            'duration_seconds': duration,
            'sample_rate': sample_rate,
            'bit_depth': bit_depth,
            'voice_used': voice_used,
            'technical_score': technical_score,
            'mos_proxy': mos_proxy,
            'sample_rate_score': sample_rate_score,
            'bit_depth_score': bit_depth_score,
            'duration_score': duration_score,
            'voice_score': voice_score
        }
        
        recommendations = []
        if sample_rate < 44100:
            recommendations.append("Use higher sample rate (44.1kHz or above)")
        if bit_depth < 16:
            recommendations.append("Use at least 16-bit audio depth")
        if duration < 30:
            recommendations.append("Content may be too short for effective engagement")
        elif duration > 600:
            recommendations.append("Consider breaking into shorter segments")
        
        return EvaluationScore(
            metric=EvaluationMetric.AUDIO_MOS,
            score=technical_score,
            confidence=0.7,
            details=details,
            recommendations=recommendations
        )

    async def _evaluate_cross_modal_alignment(
        self,
        content: Dict,
        brand_guidelines: Optional[Dict]
    ) -> EvaluationScore:
        """Evaluate alignment across different content modalities."""
        
        # Prepare content summary for analysis
        content_summary = {}
        if 'text' in content:
            content_summary['text'] = content['text'][:500] + "..." if len(content['text']) > 500 else content['text']
        if 'images' in content:
            content_summary['images'] = f"{len(content['images'])} images provided"
        if 'audio' in content:
            content_summary['audio'] = content['audio'].get('metadata', {})
        
        alignment_task = Task(
            description=f"""
            Evaluate cross-modal alignment across all content types:
            
            Content Summary: {json.dumps(content_summary, indent=2)}
            Brand Guidelines: {json.dumps(brand_guidelines, indent=2) if brand_guidelines else 'None'}
            
            Assess alignment across:
            1. Semantic consistency - Do all modalities convey the same message?
            2. Emotional coherence - Do they evoke consistent emotions?
            3. Tone alignment - Is the tone consistent across text, visuals, and audio?
            4. Narrative flow - Do they work together to tell a cohesive story?
            5. Brand consistency - Do they align with brand guidelines?
            6. Technical harmony - Are technical qualities consistent?
            
            Consider:
            - Message consistency across modalities
            - Emotional impact alignment
            - Style and aesthetic coherence
            - Pacing and timing coordination
            - Brand voice consistency
            
            Return JSON with:
            - alignment_score (0.0 to 1.0 overall)
            - semantic_consistency (0.0 to 1.0)
            - emotional_coherence (0.0 to 1.0)
            - tone_alignment (0.0 to 1.0)
            - narrative_flow (0.0 to 1.0)
            - brand_consistency (0.0 to 1.0)
            - technical_harmony (0.0 to 1.0)
            - strengths (list of well-aligned aspects)
            - misalignments (list of inconsistencies)
            - suggestions (list of improvements)
            """,
            agent=self.alignment_evaluator,
            expected_output="JSON evaluation of cross-modal alignment with detailed scores"
        )
        
        crew = Crew(
            agents=[self.alignment_evaluator],
            tasks=[alignment_task],
            verbose=True
        )
        
        alignment_result = crew.kickoff()
        
        try:
            result_data = json.loads(str(alignment_result))
            
            alignment_score = result_data.get('alignment_score', 0.6)
            
            return EvaluationScore(
                metric=EvaluationMetric.ALIGNMENT_SCORE,
                score=alignment_score,
                confidence=0.8,
                details=result_data,
                recommendations=result_data.get('suggestions', [])
            )
            
        except json.JSONDecodeError:
            return EvaluationScore(
                metric=EvaluationMetric.ALIGNMENT_SCORE,
                score=0.6,
                confidence=0.5,
                details={'error': 'Could not parse alignment evaluation'},
                recommendations=["Manual review recommended for alignment assessment"]
            )

    async def _evaluate_brand_consistency(
        self,
        content: Dict,
        brand_guidelines: Dict
    ) -> EvaluationScore:
        """Evaluate consistency with brand guidelines."""
        
        content_summary = self._prepare_content_summary(content)
        
        brand_task = Task(
            description=f"""
            Evaluate brand consistency across all content:
            
            Content: {json.dumps(content_summary, indent=2)}
            Brand Guidelines: {json.dumps(brand_guidelines, indent=2)}
            
            Check adherence to:
            1. Brand voice and tone guidelines
            2. Color palette usage (if applicable)
            3. Typography guidelines (if applicable)
            4. Messaging consistency
            5. Brand values alignment
            6. Lexicon and terminology usage
            7. Overall brand personality expression
            
            Identify:
            - Specific guideline adherences
            - Violations or inconsistencies
            - Opportunities for stronger brand expression
            - Areas where brand could be enhanced
            
            Return JSON with:
            - brand_score (0.0 to 1.0 overall)
            - voice_adherence (0.0 to 1.0)
            - visual_adherence (0.0 to 1.0)
            - messaging_consistency (0.0 to 1.0)
            - values_alignment (0.0 to 1.0)
            - lexicon_usage (0.0 to 1.0)
            - violations (list of specific issues)
            - strengths (list of good brand expressions)
            - enhancement_opportunities (list of improvements)
            """,
            agent=self.text_evaluator,
            expected_output="JSON evaluation of brand consistency with detailed analysis"
        )
        
        crew = Crew(
            agents=[self.text_evaluator],
            tasks=[brand_task],
            verbose=True
        )
        
        brand_result = crew.kickoff()
        
        try:
            result_data = json.loads(str(brand_result))
            
            brand_score = result_data.get('brand_score', 0.7)
            
            recommendations = []
            recommendations.extend(result_data.get('violations', []))
            recommendations.extend(result_data.get('enhancement_opportunities', []))
            
            return EvaluationScore(
                metric=EvaluationMetric.BRAND_CONSISTENCY,
                score=brand_score,
                confidence=0.85,
                details=result_data,
                recommendations=recommendations
            )
            
        except json.JSONDecodeError:
            return EvaluationScore(
                metric=EvaluationMetric.BRAND_CONSISTENCY,
                score=0.7,
                confidence=0.6,
                details={'error': 'Could not parse brand evaluation'},
                recommendations=["Manual brand review recommended"]
            )

    # Helper methods
    
    def _get_target_grade(self, audience: str) -> float:
        """Get target reading grade level for audience."""
        audience_grades = {
            'elementary': 4.0,
            'middle_school': 6.0,
            'high_school': 9.0,
            'college': 12.0,
            'general': 8.0,
            'professional': 10.0,
            'technical': 12.0
        }
        return audience_grades.get(audience.lower(), 8.0)

    async def _ai_readability_assessment(self, text: str, audience: Optional[str]) -> Dict:
        """AI-powered readability assessment."""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in readability assessment. Evaluate text readability and provide a score from 0.0 to 1.0."
                    },
                    {
                        "role": "user",
                        "content": f"Evaluate the readability of this text for {audience or 'general'} audience:\n\n{text[:1000]}\n\nProvide a JSON response with 'score' (0.0-1.0), 'clarity', 'complexity', and 'suggestions'."
                    }
                ],
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            return {'score': 0.6, 'error': str(e)}

    def _calculate_technical_engagement_metrics(self, text: str) -> Dict:
        """Calculate technical engagement metrics."""
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Metrics
        avg_sentence_length = len(words) / max(1, len(sentences))
        question_ratio = len(re.findall(r'\?', text)) / max(1, len(sentences))
        exclamation_ratio = len(re.findall(r'!', text)) / max(1, len(sentences))
        
        # Engagement indicators
        personal_pronouns = len(re.findall(r'\b(you|your|we|our|I|my)\b', text, re.IGNORECASE))
        action_words = len(re.findall(r'\b(discover|learn|create|build|achieve|transform)\b', text, re.IGNORECASE))
        
        # Scoring
        sentence_length_score = max(0.0, 1.0 - abs(avg_sentence_length - 15) / 20)  # Optimal ~15 words
        interaction_score = min(1.0, (question_ratio + exclamation_ratio) * 2)
        personal_score = min(1.0, personal_pronouns / len(words) * 20)
        action_score = min(1.0, action_words / len(words) * 30)
        
        overall_score = (sentence_length_score + interaction_score + personal_score + action_score) / 4
        
        return {
            'overall_score': overall_score,
            'sentence_length_score': sentence_length_score,
            'interaction_score': interaction_score,
            'personal_score': personal_score,
            'action_score': action_score,
            'avg_sentence_length': avg_sentence_length,
            'question_ratio': question_ratio,
            'exclamation_ratio': exclamation_ratio,
            'personal_pronouns': personal_pronouns,
            'action_words': action_words
        }

    def _calculate_image_technical_metrics(self, image: Image.Image) -> Dict:
        """Calculate technical image quality metrics."""
        
        width, height = image.size
        total_pixels = width * height
        
        # Resolution scoring
        if total_pixels >= 2073600:  # 1920x1080 or equivalent
            resolution_score = 1.0
        elif total_pixels >= 921600:  # 1280x720 or equivalent
            resolution_score = 0.8
        elif total_pixels >= 307200:  # 640x480 or equivalent
            resolution_score = 0.6
        else:
            resolution_score = 0.4
        
        # Aspect ratio scoring (common ratios score higher)
        aspect_ratio = width / height
        common_ratios = [16/9, 4/3, 3/2, 1/1, 9/16]
        aspect_ratio_score = max([1.0 - abs(aspect_ratio - ratio) for ratio in common_ratios])
        
        # Format scoring
        format_scores = {'JPEG': 0.8, 'PNG': 1.0, 'WEBP': 0.9}
        format_score = format_scores.get(image.format, 0.6)
        
        overall_score = (resolution_score * 0.5 + aspect_ratio_score * 0.3 + format_score * 0.2)
        
        return {
            'overall_score': overall_score,
            'resolution_score': resolution_score,
            'aspect_ratio_score': aspect_ratio_score,
            'format_score': format_score,
            'width': width,
            'height': height,
            'total_pixels': total_pixels,
            'aspect_ratio': aspect_ratio,
            'format': image.format
        }

    async def _ai_image_assessment(self, image_b64: str) -> Dict:
        """AI-powered image aesthetic assessment."""
        
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
                                "text": "Evaluate this image's aesthetic quality, composition, and visual appeal. Provide a JSON response with 'aesthetic_score' (0.0-1.0), 'composition', 'color_harmony', 'visual_impact', and 'suggestions'."
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
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            return {'aesthetic_score': 0.6, 'error': str(e)}

    def _prepare_content_summary(self, content: Dict) -> Dict:
        """Prepare content summary for evaluation."""
        
        summary = {}
        
        if 'text' in content:
            text = content['text']
            summary['text'] = {
                'preview': text[:200] + "..." if len(text) > 200 else text,
                'word_count': len(text.split()),
                'character_count': len(text)
            }
        
        if 'images' in content:
            summary['images'] = {
                'count': len(content['images']),
                'description': f"{len(content['images'])} images provided"
            }
        
        if 'audio' in content:
            summary['audio'] = content['audio'].get('metadata', {})
        
        return summary

    def _calculate_overall_score(self, scores: List[EvaluationScore]) -> float:
        """Calculate weighted overall score."""
        
        if not scores:
            return 0.0
        
        # Weights for different metrics
        weights = {
            EvaluationMetric.ALIGNMENT_SCORE: 0.25,
            EvaluationMetric.BRAND_CONSISTENCY: 0.20,
            EvaluationMetric.READABILITY: 0.15,
            EvaluationMetric.IMAGE_QUALITY: 0.15,
            EvaluationMetric.AUDIO_MOS: 0.10,
            EvaluationMetric.ENGAGEMENT_POTENTIAL: 0.10,
            EvaluationMetric.TECHNICAL_QUALITY: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score in scores:
            weight = weights.get(score.metric, 0.1)
            weighted_sum += score.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def _generate_summary_report(
        self,
        scores: List[EvaluationScore],
        overall_score: float,
        content: Dict,
        brand_guidelines: Optional[Dict]
    ) -> str:
        """Generate comprehensive summary report."""
        
        scores_summary = {}
        for score in scores:
            scores_summary[score.metric.value] = {
                'score': score.score,
                'confidence': score.confidence
            }
        
        report_task = Task(
            description=f"""
            Generate a comprehensive evaluation summary report:
            
            Overall Score: {overall_score:.2f}
            Individual Scores: {json.dumps(scores_summary, indent=2)}
            
            Content Summary: {json.dumps(self._prepare_content_summary(content), indent=2)}
            Brand Guidelines Present: {'Yes' if brand_guidelines else 'No'}
            
            Create a professional summary report that includes:
            1. Executive summary of overall quality
            2. Key strengths identified
            3. Primary areas for improvement
            4. Specific recommendations
            5. Overall assessment and next steps
            
            Write in a clear, professional tone suitable for stakeholders.
            Focus on actionable insights and specific improvements.
            """,
            agent=self.text_evaluator,
            expected_output="Professional evaluation summary report with actionable insights"
        )
        
        crew = Crew(
            agents=[self.text_evaluator],
            tasks=[report_task],
            verbose=True
        )
        
        report_result = crew.kickoff()
        return str(report_result)

    async def _generate_improvement_suggestions(
        self,
        scores: List[EvaluationScore],
        content: Dict
    ) -> List[str]:
        """Generate prioritized improvement suggestions."""
        
        suggestions = []
        
        # Collect all recommendations
        for score in scores:
            suggestions.extend(score.recommendations)
        
        # Add priority-based suggestions
        low_scores = [score for score in scores if score.score < 0.6]
        for score in low_scores:
            suggestions.append(f"Priority: Improve {score.metric.value} (current: {score.score:.2f})")
        
        # Remove duplicates and return top suggestions
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:10]  # Top 10 suggestions
