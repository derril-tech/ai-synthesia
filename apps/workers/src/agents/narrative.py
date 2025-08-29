"""
Narrative Worker - Generates story scripts from ideas with streaming support
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
from langchain.schema import BaseMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


@dataclass
class NarrativeRequest:
    """Request for narrative generation."""
    idea: str
    target_length: int  # in words
    tone: str
    audience: str
    brand_guidelines: Optional[Dict] = None
    existing_outline: Optional[str] = None


@dataclass
class NarrativeResult:
    """Result of narrative generation."""
    idea: str
    outline: str
    script: str
    captions: List[str]
    metadata: Dict
    generation_log: List[Dict]


class StreamingCallback(StreamingStdOutCallbackHandler):
    """Custom streaming callback for real-time updates."""
    
    def __init__(self, callback_func):
        super().__init__()
        self.callback_func = callback_func
        self.current_text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.current_text += token
        if self.callback_func:
            asyncio.create_task(self.callback_func({
                'type': 'token',
                'content': token,
                'full_text': self.current_text
            }))


class NarrativeWorker:
    """CrewAI-based narrative generation worker."""
    
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.7,
            streaming=True,
        )
        
        # Define agents
        self.ideation_agent = Agent(
            role='Creative Ideation Specialist',
            goal='Transform raw ideas into compelling story concepts',
            backstory="""You are a master storyteller with expertise in narrative 
            structure and audience engagement. You excel at taking rough ideas and 
            developing them into rich, engaging story concepts.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.outline_agent = Agent(
            role='Story Structure Architect',
            goal='Create detailed story outlines with proper pacing and flow',
            backstory="""You are an expert in story structure, pacing, and narrative 
            flow. You create detailed outlines that serve as blueprints for 
            compelling stories.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.script_agent = Agent(
            role='Script Writer',
            goal='Write engaging scripts from outlines',
            backstory="""You are a professional scriptwriter who transforms 
            outlines into compelling, well-paced scripts with natural dialogue 
            and vivid descriptions.""",
            llm=self.llm,
            verbose=True,
        )
        
        self.caption_agent = Agent(
            role='Caption Specialist',
            goal='Create accessible captions and scene descriptions',
            backstory="""You specialize in creating clear, engaging captions 
            and scene descriptions that enhance accessibility and engagement.""",
            llm=self.llm,
            verbose=True,
        )

    async def generate_narrative(
        self, 
        request: NarrativeRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Generate narrative content with streaming updates."""
        
        generation_log = []
        start_time = datetime.now()
        
        try:
            # Step 1: Idea Development
            yield {'stage': 'ideation', 'status': 'starting', 'progress': 0.1}
            
            idea_task = Task(
                description=f"""
                Develop the following idea into a compelling story concept:
                
                Idea: {request.idea}
                Target Length: {request.target_length} words
                Tone: {request.tone}
                Audience: {request.audience}
                
                Brand Guidelines: {json.dumps(request.brand_guidelines) if request.brand_guidelines else 'None'}
                
                Create a refined story concept that:
                1. Expands on the core idea
                2. Defines key themes and messages
                3. Identifies the target emotional impact
                4. Suggests visual and audio elements
                """,
                agent=self.ideation_agent,
                expected_output="A detailed story concept with themes, emotional arc, and multimedia suggestions"
            )
            
            # Execute idea development
            crew = Crew(
                agents=[self.ideation_agent],
                tasks=[idea_task],
                verbose=True
            )
            
            idea_result = crew.kickoff()
            generation_log.append({
                'stage': 'ideation',
                'timestamp': datetime.now().isoformat(),
                'result': str(idea_result)
            })
            
            yield {
                'stage': 'ideation', 
                'status': 'completed', 
                'progress': 0.25,
                'result': str(idea_result)
            }
            
            # Step 2: Outline Creation
            yield {'stage': 'outline', 'status': 'starting', 'progress': 0.3}
            
            outline_task = Task(
                description=f"""
                Create a detailed story outline based on this concept:
                
                {idea_result}
                
                The outline should:
                1. Break the story into clear scenes/sections
                2. Define the narrative arc (setup, conflict, resolution)
                3. Specify pacing and transitions
                4. Include notes for visual and audio cues
                5. Ensure the total length targets {request.target_length} words
                
                Existing outline to build upon: {request.existing_outline or 'None'}
                """,
                agent=self.outline_agent,
                expected_output="A structured outline with scenes, pacing notes, and multimedia cues"
            )
            
            crew = Crew(
                agents=[self.outline_agent],
                tasks=[outline_task],
                verbose=True
            )
            
            outline_result = crew.kickoff()
            generation_log.append({
                'stage': 'outline',
                'timestamp': datetime.now().isoformat(),
                'result': str(outline_result)
            })
            
            yield {
                'stage': 'outline', 
                'status': 'completed', 
                'progress': 0.5,
                'result': str(outline_result)
            }
            
            # Step 3: Script Writing
            yield {'stage': 'script', 'status': 'starting', 'progress': 0.55}
            
            script_task = Task(
                description=f"""
                Write a complete script based on this outline:
                
                {outline_result}
                
                The script should:
                1. Follow the outline structure
                2. Include natural dialogue and narration
                3. Provide clear scene descriptions
                4. Maintain the {request.tone} tone throughout
                5. Be appropriate for {request.audience} audience
                6. Target approximately {request.target_length} words
                
                Brand Guidelines to follow: {json.dumps(request.brand_guidelines) if request.brand_guidelines else 'None'}
                """,
                agent=self.script_agent,
                expected_output="A complete, well-formatted script with dialogue, narration, and scene descriptions"
            )
            
            crew = Crew(
                agents=[self.script_agent],
                tasks=[script_task],
                verbose=True
            )
            
            script_result = crew.kickoff()
            generation_log.append({
                'stage': 'script',
                'timestamp': datetime.now().isoformat(),
                'result': str(script_result)
            })
            
            yield {
                'stage': 'script', 
                'status': 'completed', 
                'progress': 0.8,
                'result': str(script_result)
            }
            
            # Step 4: Caption Generation
            yield {'stage': 'captions', 'status': 'starting', 'progress': 0.85}
            
            caption_task = Task(
                description=f"""
                Create captions and scene descriptions for this script:
                
                {script_result}
                
                Generate:
                1. Accessible captions for all dialogue and narration
                2. Scene descriptions for visual elements
                3. Audio cue descriptions
                4. Timing suggestions for each caption
                
                Format as a JSON list of caption objects with text, timing, and type.
                """,
                agent=self.caption_agent,
                expected_output="JSON-formatted list of captions with timing and type information"
            )
            
            crew = Crew(
                agents=[self.caption_agent],
                tasks=[caption_task],
                verbose=True
            )
            
            caption_result = crew.kickoff()
            generation_log.append({
                'stage': 'captions',
                'timestamp': datetime.now().isoformat(),
                'result': str(caption_result)
            })
            
            # Parse captions
            try:
                captions_data = json.loads(str(caption_result))
                captions = [item.get('text', '') for item in captions_data if isinstance(item, dict)]
            except (json.JSONDecodeError, TypeError):
                # Fallback: split script into sentences for captions
                captions = str(script_result).split('. ')
            
            # Final result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = NarrativeResult(
                idea=str(idea_result),
                outline=str(outline_result),
                script=str(script_result),
                captions=captions,
                metadata={
                    'target_length': request.target_length,
                    'actual_length': len(str(script_result).split()),
                    'tone': request.tone,
                    'audience': request.audience,
                    'generation_time': duration,
                    'stages_completed': ['ideation', 'outline', 'script', 'captions']
                },
                generation_log=generation_log
            )
            
            yield {
                'stage': 'complete', 
                'status': 'success', 
                'progress': 1.0,
                'result': result
            }
            
        except Exception as e:
            generation_log.append({
                'stage': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            yield {
                'stage': 'error',
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'generation_log': generation_log
            }

    async def refine_narrative(
        self, 
        existing_script: str,
        feedback: str,
        brand_guidelines: Optional[Dict] = None
    ) -> str:
        """Refine existing narrative based on feedback."""
        
        refinement_task = Task(
            description=f"""
            Refine this existing script based on the feedback provided:
            
            Existing Script:
            {existing_script}
            
            Feedback:
            {feedback}
            
            Brand Guidelines: {json.dumps(brand_guidelines) if brand_guidelines else 'None'}
            
            Make targeted improvements while maintaining the overall structure and flow.
            """,
            agent=self.script_agent,
            expected_output="An improved version of the script addressing the feedback"
        )
        
        crew = Crew(
            agents=[self.script_agent],
            tasks=[refinement_task],
            verbose=True
        )
        
        refined_result = crew.kickoff()
        return str(refined_result)

    def extract_key_scenes(self, script: str) -> List[Dict]:
        """Extract key scenes from script for visual generation."""
        # Simple scene extraction - in production, use more sophisticated NLP
        lines = script.split('\n')
        scenes = []
        current_scene = ""
        scene_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('SCENE') or line.startswith('INT.') or line.startswith('EXT.'):
                if current_scene:
                    scenes.append({
                        'id': scene_count,
                        'description': current_scene.strip(),
                        'type': 'scene'
                    })
                    scene_count += 1
                current_scene = line
            else:
                current_scene += f" {line}"
        
        # Add final scene
        if current_scene:
            scenes.append({
                'id': scene_count,
                'description': current_scene.strip(),
                'type': 'scene'
            })
        
        return scenes[:5]  # Limit to 5 key scenes
