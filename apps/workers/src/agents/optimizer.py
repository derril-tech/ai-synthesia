"""
Prompt Optimizer - Iteratively improves prompts based on evaluation signals
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
from langchain.schema import BaseMessage


class OptimizationObjective(Enum):
    """Optimization objectives for prompt improvement."""
    ALIGNMENT = "alignment"  # Cross-modal alignment
    QUALITY = "quality"      # Overall content quality
    BRAND_FIT = "brand_fit"  # Brand consistency
    ENGAGEMENT = "engagement" # Audience engagement
    EFFICIENCY = "efficiency" # Generation efficiency


@dataclass
class OptimizationRequest:
    """Request for prompt optimization."""
    original_prompt: str
    content_type: str  # narrative, visual, audio
    evaluation_scores: Dict[str, float]
    feedback: Optional[str] = None
    objectives: List[OptimizationObjective] = None
    constraints: Optional[Dict] = None
    max_iterations: int = 5


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""
    optimized_prompt: str
    improvement_score: float
    iterations_performed: int
    optimization_log: List[Dict]
    final_scores: Dict[str, float]


class PromptOptimizer:
    """AI-powered prompt optimization worker."""
    
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.3,  # Lower temperature for more consistent optimization
        )
        
        # Define optimization agents
        self.analysis_agent = Agent(
            role='Prompt Analysis Specialist',
            goal='Analyze prompts and identify improvement opportunities',
            backstory="""You are an expert in prompt engineering and content analysis. 
            You excel at identifying weaknesses in prompts and understanding how 
            different prompt elements affect output quality.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.optimization_agent = Agent(
            role='Prompt Optimization Engineer',
            goal='Rewrite prompts to improve specific metrics',
            backstory="""You are a master prompt engineer who specializes in 
            iterative improvement. You understand how to modify prompts to achieve 
            specific objectives while maintaining core intent.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.evaluation_agent = Agent(
            role='Quality Evaluation Specialist',
            goal='Predict prompt performance and quality scores',
            backstory="""You are an expert evaluator who can predict how well 
            a prompt will perform across different quality metrics. You understand 
            the relationship between prompt structure and output quality.""",
            llm=self.llm,
            verbose=True,
        )

    async def optimize_prompt(
        self,
        request: OptimizationRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Optimize prompt through iterative improvement."""
        
        optimization_log = []
        start_time = datetime.now()
        current_prompt = request.original_prompt
        best_prompt = current_prompt
        best_score = self._calculate_composite_score(request.evaluation_scores)
        
        try:
            # Step 1: Initial analysis
            yield {'stage': 'analysis', 'status': 'starting', 'progress': 0.1}
            
            analysis_result = await self._analyze_prompt(
                current_prompt,
                request.content_type,
                request.evaluation_scores,
                request.objectives or [OptimizationObjective.QUALITY]
            )
            
            optimization_log.append({
                'iteration': 0,
                'stage': 'analysis',
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis_result,
                'score': best_score
            })
            
            yield {
                'stage': 'analysis',
                'status': 'completed',
                'progress': 0.2,
                'analysis': analysis_result,
                'baseline_score': best_score
            }
            
            # Step 2: Iterative optimization
            for iteration in range(1, request.max_iterations + 1):
                progress = 0.2 + (0.7 * iteration / request.max_iterations)
                
                yield {
                    'stage': 'optimization',
                    'status': 'iterating',
                    'progress': progress,
                    'iteration': iteration,
                    'current_score': best_score
                }
                
                # Generate improved prompt
                improved_prompt = await self._generate_improved_prompt(
                    current_prompt,
                    analysis_result,
                    request.objectives or [OptimizationObjective.QUALITY],
                    request.constraints,
                    iteration
                )
                
                # Evaluate improved prompt
                predicted_scores = await self._evaluate_prompt(
                    improved_prompt,
                    request.content_type,
                    request.objectives or [OptimizationObjective.QUALITY]
                )
                
                predicted_composite = self._calculate_composite_score(predicted_scores)
                
                optimization_log.append({
                    'iteration': iteration,
                    'stage': 'optimization',
                    'timestamp': datetime.now().isoformat(),
                    'prompt': improved_prompt,
                    'predicted_scores': predicted_scores,
                    'composite_score': predicted_composite,
                    'improvement': predicted_composite - best_score
                })
                
                # Check if this is an improvement
                if predicted_composite > best_score:
                    best_prompt = improved_prompt
                    best_score = predicted_composite
                    current_prompt = improved_prompt
                    
                    # Re-analyze for next iteration
                    analysis_result = await self._analyze_prompt(
                        current_prompt,
                        request.content_type,
                        predicted_scores,
                        request.objectives or [OptimizationObjective.QUALITY]
                    )
                    
                    yield {
                        'stage': 'optimization',
                        'status': 'improved',
                        'progress': progress,
                        'iteration': iteration,
                        'new_score': best_score,
                        'improvement': predicted_composite - request.evaluation_scores.get('overall', 0)
                    }
                else:
                    # No improvement, try different approach
                    yield {
                        'stage': 'optimization',
                        'status': 'no_improvement',
                        'progress': progress,
                        'iteration': iteration,
                        'score': predicted_composite
                    }
                
                # Early stopping if we've achieved good improvement
                improvement_threshold = 0.1  # 10% improvement
                if (best_score - self._calculate_composite_score(request.evaluation_scores)) > improvement_threshold:
                    break
            
            # Step 3: Final validation
            yield {'stage': 'validation', 'status': 'starting', 'progress': 0.9}
            
            final_scores = await self._evaluate_prompt(
                best_prompt,
                request.content_type,
                request.objectives or [OptimizationObjective.QUALITY]
            )
            
            # Calculate final improvement
            original_composite = self._calculate_composite_score(request.evaluation_scores)
            final_composite = self._calculate_composite_score(final_scores)
            improvement = final_composite - original_composite
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = OptimizationResult(
                optimized_prompt=best_prompt,
                improvement_score=improvement,
                iterations_performed=len([log for log in optimization_log if log['stage'] == 'optimization']),
                optimization_log=optimization_log,
                final_scores=final_scores
            )
            
            yield {
                'stage': 'complete',
                'status': 'success',
                'progress': 1.0,
                'result': result,
                'improvement': improvement,
                'duration': duration
            }
            
        except Exception as e:
            optimization_log.append({
                'stage': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            yield {
                'stage': 'error',
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'optimization_log': optimization_log
            }

    async def _analyze_prompt(
        self,
        prompt: str,
        content_type: str,
        scores: Dict[str, float],
        objectives: List[OptimizationObjective]
    ) -> Dict:
        """Analyze prompt to identify improvement opportunities."""
        
        objectives_str = ", ".join([obj.value for obj in objectives])
        scores_str = json.dumps(scores, indent=2)
        
        analysis_task = Task(
            description=f"""
            Analyze this {content_type} prompt for optimization opportunities:
            
            Prompt: {prompt}
            
            Current Performance Scores:
            {scores_str}
            
            Optimization Objectives: {objectives_str}
            
            Provide a detailed analysis including:
            1. Strengths of the current prompt
            2. Specific weaknesses affecting the target objectives
            3. Concrete improvement suggestions
            4. Potential risks of modifications
            5. Priority areas for optimization
            
            Format as JSON with clear categories.
            """,
            agent=self.analysis_agent,
            expected_output="JSON analysis with strengths, weaknesses, suggestions, and priorities"
        )
        
        crew = Crew(
            agents=[self.analysis_agent],
            tasks=[analysis_task],
            verbose=True
        )
        
        analysis_result = crew.kickoff()
        
        try:
            return json.loads(str(analysis_result))
        except json.JSONDecodeError:
            # Fallback to structured text analysis
            return {
                "analysis_text": str(analysis_result),
                "suggestions": ["Improve clarity", "Add specificity", "Enhance structure"]
            }

    async def _generate_improved_prompt(
        self,
        current_prompt: str,
        analysis: Dict,
        objectives: List[OptimizationObjective],
        constraints: Optional[Dict],
        iteration: int
    ) -> str:
        """Generate an improved version of the prompt."""
        
        objectives_str = ", ".join([obj.value for obj in objectives])
        analysis_str = json.dumps(analysis, indent=2)
        constraints_str = json.dumps(constraints, indent=2) if constraints else "None"
        
        optimization_task = Task(
            description=f"""
            Improve this prompt based on the analysis and objectives:
            
            Current Prompt: {current_prompt}
            
            Analysis: {analysis_str}
            
            Optimization Objectives: {objectives_str}
            Constraints: {constraints_str}
            Iteration: {iteration}
            
            Create an improved version that:
            1. Addresses the identified weaknesses
            2. Maintains the core intent and strengths
            3. Optimizes for the specified objectives
            4. Respects any constraints
            5. Uses proven prompt engineering techniques
            
            For iteration {iteration}, focus on {'fundamental structure' if iteration <= 2 else 'fine-tuning details'}.
            
            Return ONLY the improved prompt text, no explanations.
            """,
            agent=self.optimization_agent,
            expected_output="An improved prompt that addresses the analysis findings"
        )
        
        crew = Crew(
            agents=[self.optimization_agent],
            tasks=[optimization_task],
            verbose=True
        )
        
        improved_prompt = crew.kickoff()
        return str(improved_prompt).strip()

    async def _evaluate_prompt(
        self,
        prompt: str,
        content_type: str,
        objectives: List[OptimizationObjective]
    ) -> Dict[str, float]:
        """Evaluate prompt and predict performance scores."""
        
        objectives_str = ", ".join([obj.value for obj in objectives])
        
        evaluation_task = Task(
            description=f"""
            Evaluate this {content_type} prompt and predict performance scores:
            
            Prompt: {prompt}
            
            Evaluation Objectives: {objectives_str}
            
            Predict scores (0.0 to 1.0) for:
            1. clarity - How clear and unambiguous is the prompt?
            2. specificity - How specific and detailed are the instructions?
            3. coherence - How well-structured and logical is the prompt?
            4. completeness - Does it cover all necessary aspects?
            5. effectiveness - How likely is it to produce good results?
            6. alignment - How well does it align with objectives?
            7. overall - Overall predicted quality score
            
            Consider factors like:
            - Prompt structure and organization
            - Clarity of instructions
            - Specificity of requirements
            - Potential for misinterpretation
            - Alignment with best practices
            
            Return ONLY a JSON object with the scores, no explanations.
            Example: {{"clarity": 0.85, "specificity": 0.75, ...}}
            """,
            agent=self.evaluation_agent,
            expected_output="JSON object with predicted performance scores"
        )
        
        crew = Crew(
            agents=[self.evaluation_agent],
            tasks=[evaluation_task],
            verbose=True
        )
        
        evaluation_result = crew.kickoff()
        
        try:
            scores = json.loads(str(evaluation_result))
            # Ensure all scores are floats between 0 and 1
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    scores[key] = max(0.0, min(1.0, float(value)))
                else:
                    scores[key] = 0.5  # Default neutral score
            return scores
        except (json.JSONDecodeError, TypeError):
            # Fallback scores
            return {
                "clarity": 0.7,
                "specificity": 0.6,
                "coherence": 0.7,
                "completeness": 0.6,
                "effectiveness": 0.7,
                "alignment": 0.6,
                "overall": 0.65
            }

    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate a composite score from individual metrics."""
        
        if not scores:
            return 0.0
        
        # Weighted scoring - adjust weights based on importance
        weights = {
            "overall": 0.3,
            "alignment": 0.2,
            "effectiveness": 0.2,
            "clarity": 0.1,
            "coherence": 0.1,
            "specificity": 0.05,
            "completeness": 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0.1)  # Default weight for unknown metrics
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def batch_optimize_prompts(
        self,
        prompts: List[Dict],  # List of {prompt, content_type, scores, objectives}
        max_concurrent: int = 3
    ) -> List[OptimizationResult]:
        """Optimize multiple prompts concurrently."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def optimize_single(prompt_data: Dict) -> OptimizationResult:
            async with semaphore:
                request = OptimizationRequest(
                    original_prompt=prompt_data['prompt'],
                    content_type=prompt_data['content_type'],
                    evaluation_scores=prompt_data['scores'],
                    objectives=prompt_data.get('objectives', [OptimizationObjective.QUALITY])
                )
                
                # Collect results from generator
                result = None
                async for update in self.optimize_prompt(request):
                    if update.get('stage') == 'complete' and update.get('status') == 'success':
                        result = update['result']
                        break
                
                return result
        
        # Execute optimizations concurrently
        tasks = [optimize_single(prompt_data) for prompt_data in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        return [result for result in results if isinstance(result, OptimizationResult)]

    async def suggest_prompt_templates(
        self,
        content_type: str,
        objectives: List[OptimizationObjective],
        domain: Optional[str] = None
    ) -> List[str]:
        """Suggest optimized prompt templates for specific use cases."""
        
        objectives_str = ", ".join([obj.value for obj in objectives])
        domain_str = f" in the {domain} domain" if domain else ""
        
        template_task = Task(
            description=f"""
            Create optimized prompt templates for {content_type} generation{domain_str}.
            
            Optimization Objectives: {objectives_str}
            
            Generate 3-5 different prompt templates that:
            1. Follow prompt engineering best practices
            2. Are optimized for the specified objectives
            3. Include placeholders for customization
            4. Are structured for clarity and effectiveness
            5. Address common pitfalls in {content_type} generation
            
            Each template should be different in approach but optimized for quality.
            Include brief comments explaining the template's strengths.
            
            Format as a JSON list of template objects with 'template' and 'description' fields.
            """,
            agent=self.optimization_agent,
            expected_output="JSON list of optimized prompt templates with descriptions"
        )
        
        crew = Crew(
            agents=[self.optimization_agent],
            tasks=[template_task],
            verbose=True
        )
        
        templates_result = crew.kickoff()
        
        try:
            templates_data = json.loads(str(templates_result))
            return [template['template'] for template in templates_data if 'template' in template]
        except (json.JSONDecodeError, TypeError, KeyError):
            # Fallback templates
            return [
                f"Create a high-quality {content_type} that [SPECIFIC_REQUIREMENTS]. Focus on [KEY_OBJECTIVES] while maintaining [CONSTRAINTS]. The output should be [DESIRED_CHARACTERISTICS].",
                f"Generate {content_type} content with the following specifications: [DETAILED_SPECS]. Ensure the result [QUALITY_CRITERIA] and aligns with [BRAND_GUIDELINES].",
                f"Develop {content_type} that effectively [PRIMARY_GOAL]. Consider [CONTEXT_FACTORS] and optimize for [TARGET_METRICS]. Include [REQUIRED_ELEMENTS]."
            ]
