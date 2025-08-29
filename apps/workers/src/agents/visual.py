"""
Visual Worker - Generates images for story scenes using SDXL/DALL-E
"""

import asyncio
import base64
import io
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter

import openai
from diffusers import StableDiffusionXLPipeline
import torch


@dataclass
class VisualRequest:
    """Request for visual generation."""
    scenes: List[Dict]  # Scene descriptions from narrative
    style_guidelines: Optional[Dict] = None
    brand_colors: Optional[Dict] = None
    aspect_ratio: str = "16:9"
    quality: str = "high"  # high, medium, low
    consistency_reference: Optional[str] = None  # Reference image for consistency


@dataclass
class VisualResult:
    """Result of visual generation."""
    cover_image: str  # Base64 encoded
    scene_images: List[Dict]  # List of {scene_id, image_b64, prompt, metadata}
    style_metadata: Dict
    generation_log: List[Dict]


class VisualWorker:
    """AI-powered visual generation worker."""
    
    def __init__(self, openai_api_key: str, use_local_sdxl: bool = False):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.use_local_sdxl = use_local_sdxl
        
        # Initialize SDXL pipeline if using local generation
        if use_local_sdxl:
            self.sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            if torch.cuda.is_available():
                self.sdxl_pipeline = self.sdxl_pipeline.to("cuda")
        else:
            self.sdxl_pipeline = None

    async def generate_visuals(
        self,
        request: VisualRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Generate visual content with streaming updates."""
        
        generation_log = []
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze scenes and create visual prompts
            yield {'stage': 'analysis', 'status': 'starting', 'progress': 0.1}
            
            visual_prompts = await self._create_visual_prompts(
                request.scenes, 
                request.style_guidelines,
                request.brand_colors
            )
            
            generation_log.append({
                'stage': 'analysis',
                'timestamp': datetime.now().isoformat(),
                'prompts_created': len(visual_prompts)
            })
            
            yield {
                'stage': 'analysis',
                'status': 'completed',
                'progress': 0.2,
                'prompts': visual_prompts
            }
            
            # Step 2: Generate cover image
            yield {'stage': 'cover', 'status': 'starting', 'progress': 0.25}
            
            cover_prompt = self._create_cover_prompt(request.scenes, request.style_guidelines)
            cover_image = await self._generate_single_image(
                cover_prompt,
                request.aspect_ratio,
                request.quality,
                is_cover=True
            )
            
            generation_log.append({
                'stage': 'cover',
                'timestamp': datetime.now().isoformat(),
                'prompt': cover_prompt,
                'success': cover_image is not None
            })
            
            yield {
                'stage': 'cover',
                'status': 'completed',
                'progress': 0.4,
                'image': cover_image,
                'prompt': cover_prompt
            }
            
            # Step 3: Generate scene images
            scene_images = []
            total_scenes = len(visual_prompts)
            
            for i, prompt_data in enumerate(visual_prompts):
                scene_progress = 0.4 + (0.5 * (i + 1) / total_scenes)
                
                yield {
                    'stage': 'scenes',
                    'status': 'generating',
                    'progress': scene_progress,
                    'current_scene': i + 1,
                    'total_scenes': total_scenes
                }
                
                scene_image = await self._generate_single_image(
                    prompt_data['prompt'],
                    request.aspect_ratio,
                    request.quality,
                    consistency_reference=request.consistency_reference
                )
                
                if scene_image:
                    scene_images.append({
                        'scene_id': prompt_data['scene_id'],
                        'image_b64': scene_image,
                        'prompt': prompt_data['prompt'],
                        'metadata': {
                            'generation_time': datetime.now().isoformat(),
                            'style': prompt_data.get('style', 'default'),
                            'aspect_ratio': request.aspect_ratio
                        }
                    })
                
                generation_log.append({
                    'stage': 'scene_generation',
                    'timestamp': datetime.now().isoformat(),
                    'scene_id': prompt_data['scene_id'],
                    'prompt': prompt_data['prompt'],
                    'success': scene_image is not None
                })
            
            # Step 4: Post-processing and consistency checks
            yield {'stage': 'processing', 'status': 'starting', 'progress': 0.9}
            
            # Apply brand colors and style consistency
            if request.brand_colors:
                cover_image = await self._apply_brand_styling(cover_image, request.brand_colors)
                for scene_data in scene_images:
                    scene_data['image_b64'] = await self._apply_brand_styling(
                        scene_data['image_b64'], 
                        request.brand_colors
                    )
            
            # Final result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = VisualResult(
                cover_image=cover_image,
                scene_images=scene_images,
                style_metadata={
                    'total_images': len(scene_images) + 1,
                    'generation_time': duration,
                    'style_guidelines': request.style_guidelines,
                    'brand_colors': request.brand_colors,
                    'aspect_ratio': request.aspect_ratio,
                    'quality': request.quality
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

    async def _create_visual_prompts(
        self,
        scenes: List[Dict],
        style_guidelines: Optional[Dict],
        brand_colors: Optional[Dict]
    ) -> List[Dict]:
        """Create optimized prompts for visual generation."""
        
        prompts = []
        base_style = self._build_base_style(style_guidelines, brand_colors)
        
        for scene in scenes:
            # Extract key visual elements from scene description
            scene_description = scene.get('description', '')
            
            # Build prompt with style consistency
            prompt = f"{scene_description}, {base_style}"
            
            # Add quality and technical parameters
            prompt += ", highly detailed, professional photography, cinematic lighting"
            
            # Remove unwanted elements
            negative_prompt = "blurry, low quality, distorted, text, watermark, signature"
            
            prompts.append({
                'scene_id': scene.get('id', len(prompts)),
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'style': base_style,
                'original_description': scene_description
            })
        
        return prompts

    def _create_cover_prompt(self, scenes: List[Dict], style_guidelines: Optional[Dict]) -> str:
        """Create a cover image prompt that represents the overall story."""
        
        # Extract key themes and elements from all scenes
        key_elements = []
        for scene in scenes[:3]:  # Use first 3 scenes for cover
            description = scene.get('description', '')
            # Simple keyword extraction (in production, use more sophisticated NLP)
            words = description.split()
            key_elements.extend([w for w in words if len(w) > 4])
        
        # Build cover prompt
        base_style = self._build_base_style(style_guidelines, None)
        cover_elements = ', '.join(key_elements[:5])  # Top 5 elements
        
        prompt = f"Movie poster style image featuring {cover_elements}, {base_style}"
        prompt += ", dramatic composition, eye-catching, professional design"
        
        return prompt

    def _build_base_style(self, style_guidelines: Optional[Dict], brand_colors: Optional[Dict]) -> str:
        """Build base style string from guidelines and brand colors."""
        
        style_parts = []
        
        if style_guidelines:
            if 'mood' in style_guidelines:
                style_parts.append(f"{style_guidelines['mood']} mood")
            if 'art_style' in style_guidelines:
                style_parts.append(f"{style_guidelines['art_style']} style")
            if 'lighting' in style_guidelines:
                style_parts.append(f"{style_guidelines['lighting']} lighting")
        
        if brand_colors:
            color_names = []
            for color_name, color_value in brand_colors.items():
                if color_name in ['primary', 'secondary', 'accent']:
                    color_names.append(self._hex_to_color_name(color_value))
            if color_names:
                style_parts.append(f"color palette: {', '.join(color_names)}")
        
        return ', '.join(style_parts) if style_parts else "cinematic, professional"

    async def _generate_single_image(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        quality: str = "high",
        is_cover: bool = False,
        consistency_reference: Optional[str] = None
    ) -> Optional[str]:
        """Generate a single image using the configured pipeline."""
        
        try:
            if self.use_local_sdxl and self.sdxl_pipeline:
                return await self._generate_with_sdxl(prompt, aspect_ratio, quality)
            else:
                return await self._generate_with_dalle(prompt, aspect_ratio, quality, is_cover)
        except Exception as e:
            print(f"Image generation failed: {e}")
            return None

    async def _generate_with_dalle(
        self,
        prompt: str,
        aspect_ratio: str,
        quality: str,
        is_cover: bool = False
    ) -> Optional[str]:
        """Generate image using DALL-E 3."""
        
        # Map aspect ratio to DALL-E sizes
        size_map = {
            "1:1": "1024x1024",
            "16:9": "1792x1024",
            "9:16": "1024x1792"
        }
        size = size_map.get(aspect_ratio, "1024x1024")
        
        # Map quality
        dalle_quality = "hd" if quality == "high" else "standard"
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.images.generate,
                model="dall-e-3",
                prompt=prompt[:4000],  # DALL-E prompt limit
                size=size,
                quality=dalle_quality,
                n=1,
                response_format="b64_json"
            )
            
            return response.data[0].b64_json
            
        except Exception as e:
            print(f"DALL-E generation failed: {e}")
            return None

    async def _generate_with_sdxl(
        self,
        prompt: str,
        aspect_ratio: str,
        quality: str
    ) -> Optional[str]:
        """Generate image using local SDXL pipeline."""
        
        # Map aspect ratio to dimensions
        dimension_map = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344)
        }
        width, height = dimension_map.get(aspect_ratio, (1024, 1024))
        
        # Map quality to steps
        steps_map = {"high": 50, "medium": 30, "low": 20}
        num_steps = steps_map.get(quality, 30)
        
        try:
            # Generate image
            image = await asyncio.to_thread(
                self.sdxl_pipeline,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=7.5
            )
            
            # Convert to base64
            img_buffer = io.BytesIO()
            image.images[0].save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"SDXL generation failed: {e}")
            return None

    async def _apply_brand_styling(self, image_b64: str, brand_colors: Dict) -> str:
        """Apply brand color styling to generated images."""
        
        try:
            # Decode image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Apply subtle color adjustments based on brand colors
            # This is a simplified version - in production, use more sophisticated color matching
            
            # Enhance colors based on brand palette
            if 'primary' in brand_colors:
                # Slightly adjust color balance towards brand primary
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.1)
            
            # Convert back to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            print(f"Brand styling failed: {e}")
            return image_b64  # Return original if styling fails

    def _hex_to_color_name(self, hex_color: str) -> str:
        """Convert hex color to approximate color name."""
        # Simplified color name mapping
        color_map = {
            '#FF0000': 'red', '#00FF00': 'green', '#0000FF': 'blue',
            '#FFFF00': 'yellow', '#FF00FF': 'magenta', '#00FFFF': 'cyan',
            '#FFA500': 'orange', '#800080': 'purple', '#FFC0CB': 'pink',
            '#A52A2A': 'brown', '#808080': 'gray', '#000000': 'black',
            '#FFFFFF': 'white'
        }
        
        # Find closest color (simplified)
        return color_map.get(hex_color.upper(), 'neutral')

    async def upscale_image(self, image_b64: str, scale_factor: int = 2) -> str:
        """Upscale image for higher resolution output."""
        
        try:
            # Decode image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Simple upscaling using PIL (in production, use AI upscaler)
            new_size = (image.width * scale_factor, image.height * scale_factor)
            upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply sharpening
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            # Convert back to base64
            img_buffer = io.BytesIO()
            upscaled.save(img_buffer, format='PNG', quality=95)
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            print(f"Image upscaling failed: {e}")
            return image_b64  # Return original if upscaling fails
