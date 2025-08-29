"""
Safety Worker - Content moderation, NSFW detection, toxicity filtering, and copyright compliance
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
from transformers import pipeline
import hashlib


class SafetyViolationType(Enum):
    """Types of safety violations."""
    NSFW_CONTENT = "nsfw_content"
    TOXICITY = "toxicity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    COPYRIGHT_INFRINGEMENT = "copyright_infringement"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    INAPPROPRIATE_LANGUAGE = "inappropriate_language"


class SafetySeverity(Enum):
    """Severity levels for safety violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyViolation:
    """Detected safety violation."""
    violation_type: SafetyViolationType
    severity: SafetySeverity
    confidence: float  # 0.0 to 1.0
    description: str
    affected_content: str  # text, image_1, audio, etc.
    evidence: Dict
    recommended_action: str


@dataclass
class SafetyRequest:
    """Request for safety analysis."""
    content: Dict[str, any]  # {text, images, audio, metadata}
    safety_policies: Optional[Dict] = None
    target_audience: Optional[str] = None
    content_category: Optional[str] = None  # educational, commercial, entertainment, etc.


@dataclass
class SafetyResult:
    """Result of safety analysis."""
    overall_safe: bool
    safety_score: float  # 0.0 to 1.0 (1.0 = completely safe)
    violations: List[SafetyViolation]
    content_ratings: Dict[str, str]  # G, PG, PG-13, R, etc.
    recommendations: List[str]
    analysis_metadata: Dict


class SafetyWorker:
    """Comprehensive content safety and moderation system."""
    
    def __init__(self, openai_api_key: str, huggingface_token: Optional[str] = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1,  # Low temperature for consistent safety analysis
        )
        
        # Initialize safety analysis agents
        self.text_safety_agent = Agent(
            role='Text Safety Analyst',
            goal='Identify harmful, toxic, or inappropriate text content',
            backstory="""You are an expert content moderator specializing in text 
            safety analysis. You identify toxicity, hate speech, inappropriate 
            content, and policy violations with high accuracy and nuanced judgment.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.visual_safety_agent = Agent(
            role='Visual Content Safety Analyst',
            goal='Identify inappropriate, NSFW, or harmful visual content',
            backstory="""You are a visual content moderation expert who identifies 
            NSFW content, violence, inappropriate imagery, and visual policy 
            violations while understanding context and artistic expression.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.copyright_agent = Agent(
            role='Copyright Compliance Specialist',
            goal='Identify potential copyright infringement and IP violations',
            backstory="""You are a copyright and intellectual property expert who 
            identifies potential infringement, trademark violations, and helps 
            ensure content compliance with IP laws and fair use principles.""",
            llm=self.llm,
            verbose=True,
        )
        
        # Initialize ML models for safety detection
        try:
            if huggingface_token:
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    use_auth_token=huggingface_token
                )
                self.hate_speech_detector = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-hate-latest",
                    use_auth_token=huggingface_token
                )
            else:
                self.toxicity_classifier = None
                self.hate_speech_detector = None
        except Exception as e:
            print(f"Warning: Could not load safety ML models: {e}")
            self.toxicity_classifier = None
            self.hate_speech_detector = None
        
        # Safety thresholds
        self.safety_thresholds = {
            'toxicity': 0.7,
            'hate_speech': 0.8,
            'nsfw': 0.8,
            'violence': 0.75,
            'copyright_risk': 0.6
        }
        
        # Known problematic patterns
        self.problematic_patterns = {
            'personal_info': [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone number
            ],
            'spam_indicators': [
                r'click here now',
                r'limited time offer',
                r'act now',
                r'guaranteed',
                r'risk free'
            ]
        }

    async def analyze_safety(
        self,
        request: SafetyRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Perform comprehensive safety analysis."""
        
        analysis_log = []
        start_time = datetime.now()
        violations = []
        
        try:
            # Step 1: Text safety analysis
            yield {'stage': 'text_safety', 'status': 'starting', 'progress': 0.1}
            
            if 'text' in request.content:
                text_violations = await self._analyze_text_safety(
                    request.content['text'],
                    request.safety_policies,
                    request.target_audience
                )
                violations.extend(text_violations)
            
            analysis_log.append({
                'stage': 'text_safety',
                'timestamp': datetime.now().isoformat(),
                'violations_found': len([v for v in violations if 'text' in v.affected_content])
            })
            
            yield {
                'stage': 'text_safety',
                'status': 'completed',
                'progress': 0.3,
                'text_violations': len([v for v in violations if 'text' in v.affected_content])
            }
            
            # Step 2: Visual safety analysis
            yield {'stage': 'visual_safety', 'status': 'starting', 'progress': 0.35}
            
            if 'images' in request.content:
                for i, image_data in enumerate(request.content['images']):
                    image_violations = await self._analyze_image_safety(
                        image_data,
                        f"image_{i}",
                        request.safety_policies
                    )
                    violations.extend(image_violations)
            
            analysis_log.append({
                'stage': 'visual_safety',
                'timestamp': datetime.now().isoformat(),
                'images_analyzed': len(request.content.get('images', [])),
                'violations_found': len([v for v in violations if 'image' in v.affected_content])
            })
            
            yield {
                'stage': 'visual_safety',
                'status': 'completed',
                'progress': 0.6,
                'visual_violations': len([v for v in violations if 'image' in v.affected_content])
            }
            
            # Step 3: Copyright analysis
            yield {'stage': 'copyright_analysis', 'status': 'starting', 'progress': 0.65}
            
            copyright_violations = await self._analyze_copyright_compliance(
                request.content,
                request.content_category
            )
            violations.extend(copyright_violations)
            
            analysis_log.append({
                'stage': 'copyright_analysis',
                'timestamp': datetime.now().isoformat(),
                'copyright_violations': len(copyright_violations)
            })
            
            # Step 4: Audio safety analysis (if applicable)
            if 'audio' in request.content:
                audio_violations = await self._analyze_audio_safety(
                    request.content['audio']
                )
                violations.extend(audio_violations)
            
            # Step 5: Overall safety assessment
            yield {'stage': 'overall_assessment', 'status': 'starting', 'progress': 0.8}
            
            safety_score = self._calculate_safety_score(violations)
            overall_safe = self._determine_overall_safety(violations, safety_score)
            content_ratings = self._assign_content_ratings(violations, request.content)
            recommendations = self._generate_safety_recommendations(violations)
            
            # Final result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = SafetyResult(
                overall_safe=overall_safe,
                safety_score=safety_score,
                violations=violations,
                content_ratings=content_ratings,
                recommendations=recommendations,
                analysis_metadata={
                    'analysis_time': duration,
                    'total_violations': len(violations),
                    'critical_violations': len([v for v in violations if v.severity == SafetySeverity.CRITICAL]),
                    'high_violations': len([v for v in violations if v.severity == SafetySeverity.HIGH]),
                    'content_analyzed': list(request.content.keys())
                }
            )
            
            yield {
                'stage': 'complete',
                'status': 'success',
                'progress': 1.0,
                'result': result
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

    async def _analyze_text_safety(
        self,
        text: str,
        safety_policies: Optional[Dict],
        target_audience: Optional[str]
    ) -> List[SafetyViolation]:
        """Analyze text content for safety violations."""
        
        violations = []
        
        # 1. ML-based toxicity detection
        if self.toxicity_classifier:
            try:
                toxicity_result = self.toxicity_classifier(text[:512])  # Limit length
                toxicity_score = max([r['score'] for r in toxicity_result if r['label'] == 'TOXIC'])
                
                if toxicity_score > self.safety_thresholds['toxicity']:
                    violations.append(SafetyViolation(
                        violation_type=SafetyViolationType.TOXICITY,
                        severity=SafetySeverity.HIGH if toxicity_score > 0.9 else SafetySeverity.MEDIUM,
                        confidence=toxicity_score,
                        description=f"Toxic content detected with {toxicity_score:.2f} confidence",
                        affected_content="text",
                        evidence={'toxicity_score': toxicity_score, 'model': 'toxic-bert'},
                        recommended_action="Review and revise toxic language"
                    ))
            except Exception as e:
                print(f"Toxicity detection failed: {e}")
        
        # 2. Hate speech detection
        if self.hate_speech_detector:
            try:
                hate_result = self.hate_speech_detector(text[:512])
                hate_score = max([r['score'] for r in hate_result if 'hate' in r['label'].lower()])
                
                if hate_score > self.safety_thresholds['hate_speech']:
                    violations.append(SafetyViolation(
                        violation_type=SafetyViolationType.HATE_SPEECH,
                        severity=SafetySeverity.CRITICAL,
                        confidence=hate_score,
                        description=f"Hate speech detected with {hate_score:.2f} confidence",
                        affected_content="text",
                        evidence={'hate_score': hate_score, 'model': 'hate-speech-detector'},
                        recommended_action="Remove hate speech content immediately"
                    ))
            except Exception as e:
                print(f"Hate speech detection failed: {e}")
        
        # 3. Pattern-based detection
        pattern_violations = self._detect_pattern_violations(text)
        violations.extend(pattern_violations)
        
        # 4. AI-powered comprehensive analysis
        ai_violations = await self._ai_text_safety_analysis(text, target_audience, safety_policies)
        violations.extend(ai_violations)
        
        return violations

    async def _analyze_image_safety(
        self,
        image_data: str,  # Base64 encoded
        image_id: str,
        safety_policies: Optional[Dict]
    ) -> List[SafetyViolation]:
        """Analyze image content for safety violations."""
        
        violations = []
        
        try:
            # 1. AI-powered visual content analysis
            visual_analysis = await self._ai_visual_safety_analysis(image_data, image_id)
            
            # 2. Technical image analysis
            technical_analysis = self._analyze_image_technical_safety(image_data, image_id)
            
            # Combine results
            violations.extend(visual_analysis)
            violations.extend(technical_analysis)
            
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.NSFW_CONTENT,
                severity=SafetySeverity.MEDIUM,
                confidence=0.5,
                description=f"Could not analyze image safety: {str(e)}",
                affected_content=image_id,
                evidence={'error': str(e)},
                recommended_action="Manual review required"
            ))
        
        return violations

    async def _analyze_copyright_compliance(
        self,
        content: Dict,
        content_category: Optional[str]
    ) -> List[SafetyViolation]:
        """Analyze content for potential copyright issues."""
        
        violations = []
        
        # Prepare content summary for analysis
        content_summary = {}
        if 'text' in content:
            content_summary['text'] = content['text'][:1000] + "..." if len(content['text']) > 1000 else content['text']
        if 'images' in content:
            content_summary['images'] = f"{len(content['images'])} images"
        if 'audio' in content:
            content_summary['audio'] = "Audio content present"
        
        copyright_task = Task(
            description=f"""
            Analyze content for potential copyright and intellectual property issues:
            
            Content Summary: {json.dumps(content_summary, indent=2)}
            Content Category: {content_category or 'General'}
            
            Check for:
            1. Potential trademark violations
            2. Copyrighted character or brand references
            3. Quoted copyrighted text without attribution
            4. References to copyrighted works
            5. Use of branded terms or slogans
            6. Potential fair use considerations
            
            Consider:
            - Direct copying vs. inspiration
            - Commercial vs. educational use
            - Transformative nature of content
            - Attribution and fair use factors
            
            Return JSON with:
            - copyright_risk_score (0.0 to 1.0)
            - potential_violations (list of specific issues)
            - trademark_concerns (list of trademark issues)
            - fair_use_factors (analysis of fair use applicability)
            - recommendations (list of actions to reduce risk)
            """,
            agent=self.copyright_agent,
            expected_output="JSON analysis of copyright compliance with risk assessment"
        )
        
        crew = Crew(
            agents=[self.copyright_agent],
            tasks=[copyright_task],
            verbose=True
        )
        
        copyright_result = crew.kickoff()
        
        try:
            result_data = json.loads(str(copyright_result))
            
            copyright_risk = result_data.get('copyright_risk_score', 0.0)
            
            if copyright_risk > self.safety_thresholds['copyright_risk']:
                severity = SafetySeverity.HIGH if copyright_risk > 0.8 else SafetySeverity.MEDIUM
                
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.COPYRIGHT_INFRINGEMENT,
                    severity=severity,
                    confidence=copyright_risk,
                    description=f"Potential copyright issues detected (risk: {copyright_risk:.2f})",
                    affected_content="multiple",
                    evidence=result_data,
                    recommended_action="Review copyright compliance and consider modifications"
                ))
            
        except json.JSONDecodeError:
            # Fallback: basic keyword-based copyright detection
            copyright_keywords = [
                'disney', 'marvel', 'star wars', 'pokemon', 'nintendo',
                'coca cola', 'pepsi', 'mcdonalds', 'apple inc', 'microsoft'
            ]
            
            text_content = content.get('text', '').lower()
            found_keywords = [kw for kw in copyright_keywords if kw in text_content]
            
            if found_keywords:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.COPYRIGHT_INFRINGEMENT,
                    severity=SafetySeverity.MEDIUM,
                    confidence=0.6,
                    description=f"Potential trademark references found: {', '.join(found_keywords)}",
                    affected_content="text",
                    evidence={'keywords_found': found_keywords},
                    recommended_action="Review trademark usage and consider alternatives"
                ))
        
        return violations

    async def _analyze_audio_safety(
        self,
        audio_metadata: Dict
    ) -> List[SafetyViolation]:
        """Analyze audio content safety based on metadata."""
        
        violations = []
        
        # Check audio characteristics
        duration = audio_metadata.get('total_duration', 0)
        voice_used = audio_metadata.get('voice_used', 'unknown')
        
        # Check for excessively long content (potential spam)
        if duration > 1800:  # 30 minutes
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.SPAM,
                severity=SafetySeverity.LOW,
                confidence=0.7,
                description=f"Unusually long audio content ({duration/60:.1f} minutes)",
                affected_content="audio",
                evidence={'duration': duration},
                recommended_action="Consider breaking into shorter segments"
            ))
        
        # Check for suspicious voice settings (if available)
        if 'generation_log' in audio_metadata:
            log = audio_metadata['generation_log']
            # Look for signs of manipulation or inappropriate content generation
            # This would be expanded with more sophisticated analysis
        
        return violations

    def _detect_pattern_violations(self, text: str) -> List[SafetyViolation]:
        """Detect violations using pattern matching."""
        
        violations = []
        
        # Check for personal information
        for pattern in self.problematic_patterns['personal_info']:
            matches = re.findall(pattern, text)
            if matches:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.PRIVACY_VIOLATION,
                    severity=SafetySeverity.HIGH,
                    confidence=0.9,
                    description=f"Personal information detected: {len(matches)} instances",
                    affected_content="text",
                    evidence={'matches': matches[:3], 'pattern_type': 'personal_info'},
                    recommended_action="Remove or redact personal information"
                ))
        
        # Check for spam indicators
        spam_count = 0
        for pattern in self.problematic_patterns['spam_indicators']:
            if re.search(pattern, text, re.IGNORECASE):
                spam_count += 1
        
        if spam_count >= 2:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.SPAM,
                severity=SafetySeverity.MEDIUM,
                confidence=min(0.9, spam_count * 0.3),
                description=f"Spam indicators detected: {spam_count} patterns",
                affected_content="text",
                evidence={'spam_indicators': spam_count},
                recommended_action="Review and remove spam-like language"
            ))
        
        return violations

    async def _ai_text_safety_analysis(
        self,
        text: str,
        target_audience: Optional[str],
        safety_policies: Optional[Dict]
    ) -> List[SafetyViolation]:
        """AI-powered comprehensive text safety analysis."""
        
        safety_task = Task(
            description=f"""
            Perform comprehensive safety analysis of this text content:
            
            Text: {text[:2000]}...
            Target Audience: {target_audience or 'General'}
            Safety Policies: {json.dumps(safety_policies, indent=2) if safety_policies else 'Standard policies'}
            
            Analyze for:
            1. Inappropriate language or profanity
            2. Violence or threatening content
            3. Discriminatory or biased language
            4. Misinformation or false claims
            5. Age-inappropriate content
            6. Harassment or bullying language
            7. Self-harm or dangerous activities
            8. Political bias or controversial content
            
            Consider context, intent, and target audience.
            Distinguish between legitimate discussion and harmful content.
            
            Return JSON with:
            - safety_violations (list of issues found)
            - severity_levels (low/medium/high/critical for each)
            - confidence_scores (0.0 to 1.0 for each detection)
            - context_considerations (factors that affect severity)
            - recommendations (specific actions to address issues)
            """,
            agent=self.text_safety_agent,
            expected_output="JSON safety analysis with detailed violation assessment"
        )
        
        crew = Crew(
            agents=[self.text_safety_agent],
            tasks=[safety_task],
            verbose=True
        )
        
        safety_result = crew.kickoff()
        
        violations = []
        
        try:
            result_data = json.loads(str(safety_result))
            
            safety_violations = result_data.get('safety_violations', [])
            severity_levels = result_data.get('severity_levels', [])
            confidence_scores = result_data.get('confidence_scores', [])
            recommendations = result_data.get('recommendations', [])
            
            for i, violation_desc in enumerate(safety_violations):
                severity_str = severity_levels[i] if i < len(severity_levels) else 'medium'
                confidence = confidence_scores[i] if i < len(confidence_scores) else 0.6
                
                # Map severity string to enum
                severity_map = {
                    'low': SafetySeverity.LOW,
                    'medium': SafetySeverity.MEDIUM,
                    'high': SafetySeverity.HIGH,
                    'critical': SafetySeverity.CRITICAL
                }
                severity = severity_map.get(severity_str.lower(), SafetySeverity.MEDIUM)
                
                # Determine violation type based on description
                violation_type = self._classify_violation_type(violation_desc)
                
                violations.append(SafetyViolation(
                    violation_type=violation_type,
                    severity=severity,
                    confidence=confidence,
                    description=violation_desc,
                    affected_content="text",
                    evidence=result_data,
                    recommended_action=recommendations[i] if i < len(recommendations) else "Review and revise content"
                ))
            
        except json.JSONDecodeError:
            # Fallback: basic safety check
            if any(word in text.lower() for word in ['violence', 'hate', 'harm', 'kill', 'death']):
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.VIOLENCE,
                    severity=SafetySeverity.MEDIUM,
                    confidence=0.6,
                    description="Potentially violent content detected",
                    affected_content="text",
                    evidence={'method': 'keyword_detection'},
                    recommended_action="Review content for violent themes"
                ))
        
        return violations

    async def _ai_visual_safety_analysis(
        self,
        image_b64: str,
        image_id: str
    ) -> List[SafetyViolation]:
        """AI-powered visual content safety analysis."""
        
        violations = []
        
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
                                "text": """Analyze this image for safety concerns. Check for:
                                1. NSFW or sexually explicit content
                                2. Violence or graphic content
                                3. Inappropriate or offensive imagery
                                4. Hate symbols or discriminatory content
                                5. Dangerous activities or self-harm
                                6. Age-inappropriate content
                                
                                Provide a JSON response with:
                                - is_safe (boolean)
                                - safety_score (0.0 to 1.0, 1.0 = completely safe)
                                - violations_found (list of issues)
                                - severity_levels (list of severity for each violation)
                                - recommendations (list of actions)"""
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
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            result_data = json.loads(content)
            
            if not result_data.get('is_safe', True):
                violations_found = result_data.get('violations_found', [])
                severity_levels = result_data.get('severity_levels', [])
                recommendations = result_data.get('recommendations', [])
                
                for i, violation_desc in enumerate(violations_found):
                    severity_str = severity_levels[i] if i < len(severity_levels) else 'medium'
                    severity = getattr(SafetySeverity, severity_str.upper(), SafetySeverity.MEDIUM)
                    
                    violation_type = self._classify_visual_violation_type(violation_desc)
                    
                    violations.append(SafetyViolation(
                        violation_type=violation_type,
                        severity=severity,
                        confidence=1.0 - result_data.get('safety_score', 0.5),
                        description=violation_desc,
                        affected_content=image_id,
                        evidence=result_data,
                        recommended_action=recommendations[i] if i < len(recommendations) else "Review image content"
                    ))
            
        except Exception as e:
            # Fallback: flag for manual review
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.NSFW_CONTENT,
                severity=SafetySeverity.LOW,
                confidence=0.3,
                description=f"Could not analyze image safety: {str(e)}",
                affected_content=image_id,
                evidence={'error': str(e)},
                recommended_action="Manual safety review required"
            ))
        
        return violations

    def _analyze_image_technical_safety(
        self,
        image_data: str,
        image_id: str
    ) -> List[SafetyViolation]:
        """Analyze image for technical safety issues."""
        
        violations = []
        
        try:
            # Decode and analyze image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check for suspicious image characteristics
            width, height = image.size
            
            # Check for extremely large images (potential DoS)
            if width * height > 50000000:  # 50MP
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.SPAM,
                    severity=SafetySeverity.MEDIUM,
                    confidence=0.8,
                    description=f"Extremely large image ({width}x{height})",
                    affected_content=image_id,
                    evidence={'dimensions': [width, height], 'total_pixels': width * height},
                    recommended_action="Resize image to reasonable dimensions"
                ))
            
            # Check image hash against known problematic content
            image_hash = hashlib.md5(image_bytes).hexdigest()
            # In production, you'd check against a database of known harmful image hashes
            
        except Exception as e:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.NSFW_CONTENT,
                severity=SafetySeverity.LOW,
                confidence=0.3,
                description=f"Technical image analysis failed: {str(e)}",
                affected_content=image_id,
                evidence={'error': str(e)},
                recommended_action="Manual technical review required"
            ))
        
        return violations

    def _classify_violation_type(self, description: str) -> SafetyViolationType:
        """Classify violation type based on description."""
        
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['toxic', 'offensive', 'inappropriate']):
            return SafetyViolationType.TOXICITY
        elif any(word in description_lower for word in ['hate', 'discriminat', 'bias']):
            return SafetyViolationType.HATE_SPEECH
        elif any(word in description_lower for word in ['violence', 'violent', 'harm', 'threat']):
            return SafetyViolationType.VIOLENCE
        elif any(word in description_lower for word in ['nsfw', 'sexual', 'explicit']):
            return SafetyViolationType.NSFW_CONTENT
        elif any(word in description_lower for word in ['spam', 'promotional']):
            return SafetyViolationType.SPAM
        elif any(word in description_lower for word in ['privacy', 'personal', 'information']):
            return SafetyViolationType.PRIVACY_VIOLATION
        else:
            return SafetyViolationType.INAPPROPRIATE_LANGUAGE

    def _classify_visual_violation_type(self, description: str) -> SafetyViolationType:
        """Classify visual violation type based on description."""
        
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['nsfw', 'sexual', 'explicit', 'nude']):
            return SafetyViolationType.NSFW_CONTENT
        elif any(word in description_lower for word in ['violence', 'violent', 'graphic', 'blood']):
            return SafetyViolationType.VIOLENCE
        elif any(word in description_lower for word in ['hate', 'symbol', 'discriminat']):
            return SafetyViolationType.HATE_SPEECH
        else:
            return SafetyViolationType.INAPPROPRIATE_LANGUAGE

    def _calculate_safety_score(self, violations: List[SafetyViolation]) -> float:
        """Calculate overall safety score based on violations."""
        
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            SafetySeverity.LOW: 0.1,
            SafetySeverity.MEDIUM: 0.3,
            SafetySeverity.HIGH: 0.6,
            SafetySeverity.CRITICAL: 1.0
        }
        
        total_penalty = 0.0
        for violation in violations:
            weight = severity_weights[violation.severity]
            penalty = weight * violation.confidence
            total_penalty += penalty
        
        # Calculate safety score (1.0 = completely safe, 0.0 = completely unsafe)
        safety_score = max(0.0, 1.0 - (total_penalty / len(violations)))
        
        return safety_score

    def _determine_overall_safety(
        self,
        violations: List[SafetyViolation],
        safety_score: float
    ) -> bool:
        """Determine if content is overall safe for publication."""
        
        # Content is unsafe if:
        # 1. Any critical violations exist
        # 2. Multiple high-severity violations exist
        # 3. Safety score is below threshold
        
        critical_violations = [v for v in violations if v.severity == SafetySeverity.CRITICAL]
        high_violations = [v for v in violations if v.severity == SafetySeverity.HIGH]
        
        if critical_violations:
            return False
        
        if len(high_violations) >= 2:
            return False
        
        if safety_score < 0.7:
            return False
        
        return True

    def _assign_content_ratings(
        self,
        violations: List[SafetyViolation],
        content: Dict
    ) -> Dict[str, str]:
        """Assign content ratings based on violations."""
        
        ratings = {}
        
        # Default rating
        base_rating = "G"  # General audiences
        
        # Adjust rating based on violations
        has_mild_language = any(
            v.violation_type == SafetyViolationType.INAPPROPRIATE_LANGUAGE 
            and v.severity == SafetySeverity.LOW 
            for v in violations
        )
        
        has_moderate_content = any(
            v.severity == SafetySeverity.MEDIUM 
            for v in violations
        )
        
        has_mature_content = any(
            v.severity in [SafetySeverity.HIGH, SafetySeverity.CRITICAL]
            for v in violations
        )
        
        if has_mature_content:
            base_rating = "R"  # Restricted
        elif has_moderate_content:
            base_rating = "PG-13"  # Parents strongly cautioned
        elif has_mild_language:
            base_rating = "PG"  # Parental guidance suggested
        
        ratings['overall'] = base_rating
        
        # Specific content type ratings
        if 'text' in content:
            ratings['text'] = base_rating
        if 'images' in content:
            ratings['visual'] = base_rating
        if 'audio' in content:
            ratings['audio'] = base_rating
        
        return ratings

    def _generate_safety_recommendations(
        self,
        violations: List[SafetyViolation]
    ) -> List[str]:
        """Generate actionable safety recommendations."""
        
        recommendations = []
        
        # Collect all violation-specific recommendations
        for violation in violations:
            if violation.recommended_action not in recommendations:
                recommendations.append(violation.recommended_action)
        
        # Add general recommendations based on violation patterns
        critical_count = len([v for v in violations if v.severity == SafetySeverity.CRITICAL])
        high_count = len([v for v in violations if v.severity == SafetySeverity.HIGH])
        
        if critical_count > 0:
            recommendations.insert(0, f"URGENT: Address {critical_count} critical safety violation(s) before publication")
        
        if high_count > 0:
            recommendations.append(f"Review and resolve {high_count} high-severity safety issue(s)")
        
        # Add preventive recommendations
        if len(violations) > 3:
            recommendations.append("Consider implementing additional content review processes")
        
        return recommendations[:10]  # Limit to top 10 recommendations

    async def create_safety_report(
        self,
        safety_result: SafetyResult,
        content_metadata: Dict
    ) -> str:
        """Generate comprehensive safety report."""
        
        report_data = {
            'overall_safe': safety_result.overall_safe,
            'safety_score': safety_result.safety_score,
            'total_violations': len(safety_result.violations),
            'violation_breakdown': {},
            'content_ratings': safety_result.content_ratings,
            'recommendations': safety_result.recommendations
        }
        
        # Breakdown violations by type and severity
        for violation in safety_result.violations:
            vtype = violation.violation_type.value
            if vtype not in report_data['violation_breakdown']:
                report_data['violation_breakdown'][vtype] = []
            report_data['violation_breakdown'][vtype].append({
                'severity': violation.severity.value,
                'confidence': violation.confidence,
                'description': violation.description
            })
        
        report_task = Task(
            description=f"""
            Generate a comprehensive safety analysis report:
            
            Safety Analysis Results: {json.dumps(report_data, indent=2)}
            Content Metadata: {json.dumps(content_metadata, indent=2)}
            
            Create a professional safety report that includes:
            1. Executive summary of safety status
            2. Detailed violation analysis
            3. Risk assessment and implications
            4. Specific remediation steps
            5. Content rating justification
            6. Compliance recommendations
            
            Write in a clear, professional tone suitable for content reviewers and stakeholders.
            Focus on actionable insights and specific safety improvements.
            """,
            agent=self.text_safety_agent,
            expected_output="Professional safety analysis report with actionable recommendations"
        )
        
        crew = Crew(
            agents=[self.text_safety_agent],
            tasks=[report_task],
            verbose=True
        )
        
        report_result = crew.kickoff()
        return str(report_result)
