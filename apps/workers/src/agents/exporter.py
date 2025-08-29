"""
Export Worker - Generate MP4 videos, ZIP bundles, JSON metadata, and PDF reports
"""

import asyncio
import json
import base64
import tempfile
import os
import zipfile
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, 
    concatenate_videoclips, TextClip, ColorClip
)


@dataclass
class ExportRequest:
    """Request for content export."""
    story_pack_id: str
    content: Dict[str, any]  # {text, images, audio, metadata}
    export_formats: List[str]  # ['mp4', 'zip', 'json', 'pdf']
    export_settings: Optional[Dict] = None
    brand_kit: Optional[Dict] = None
    template_options: Optional[Dict] = None


@dataclass
class ExportResult:
    """Result of export operation."""
    export_id: str
    exported_files: Dict[str, str]  # {format: file_path or base64}
    export_metadata: Dict
    generation_log: List[Dict]


class ExportWorker:
    """Comprehensive content export system."""
    
    def __init__(self):
        # Default export settings
        self.default_video_settings = {
            'resolution': '1920x1080',
            'fps': 30,
            'bitrate': '5000k',
            'codec': 'libx264',
            'audio_codec': 'aac',
            'audio_bitrate': '128k'
        }
        
        self.default_image_settings = {
            'format': 'PNG',
            'quality': 95,
            'dpi': 300
        }
        
        # Video templates
        self.video_templates = {
            'story_pack': {
                'intro_duration': 2.0,
                'scene_duration': 4.0,
                'outro_duration': 2.0,
                'transition_duration': 0.5,
                'text_overlay': True,
                'background_music': True
            },
            'commercial': {
                'intro_duration': 1.0,
                'scene_duration': 3.0,
                'outro_duration': 1.5,
                'transition_duration': 0.3,
                'text_overlay': True,
                'background_music': True
            },
            'educational': {
                'intro_duration': 3.0,
                'scene_duration': 6.0,
                'outro_duration': 2.0,
                'transition_duration': 0.8,
                'text_overlay': True,
                'background_music': False
            }
        }

    async def export_content(
        self,
        request: ExportRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Export content in requested formats."""
        
        export_log = []
        start_time = datetime.now()
        exported_files = {}
        export_id = f"export_{int(start_time.timestamp())}"
        
        try:
            # Step 1: Prepare export workspace
            yield {'stage': 'preparation', 'status': 'starting', 'progress': 0.05}
            
            temp_dir = tempfile.mkdtemp(prefix=f"export_{export_id}_")
            export_settings = request.export_settings or {}
            
            export_log.append({
                'stage': 'preparation',
                'timestamp': datetime.now().isoformat(),
                'temp_dir': temp_dir,
                'formats_requested': request.export_formats
            })
            
            yield {
                'stage': 'preparation',
                'status': 'completed',
                'progress': 0.1,
                'export_id': export_id,
                'temp_dir': temp_dir
            }
            
            # Step 2: Export JSON metadata (always first)
            if 'json' in request.export_formats:
                yield {'stage': 'json_export', 'status': 'starting', 'progress': 0.15}
                
                json_file = await self._export_json_metadata(
                    request.content,
                    request.story_pack_id,
                    temp_dir,
                    export_settings
                )
                exported_files['json'] = json_file
                
                export_log.append({
                    'stage': 'json_export',
                    'timestamp': datetime.now().isoformat(),
                    'file_path': json_file
                })
                
                yield {
                    'stage': 'json_export',
                    'status': 'completed',
                    'progress': 0.2,
                    'file_path': json_file
                }
            
            # Step 3: Export PDF report
            if 'pdf' in request.export_formats:
                yield {'stage': 'pdf_export', 'status': 'starting', 'progress': 0.25}
                
                pdf_file = await self._export_pdf_report(
                    request.content,
                    request.story_pack_id,
                    request.brand_kit,
                    temp_dir,
                    export_settings
                )
                exported_files['pdf'] = pdf_file
                
                export_log.append({
                    'stage': 'pdf_export',
                    'timestamp': datetime.now().isoformat(),
                    'file_path': pdf_file
                })
                
                yield {
                    'stage': 'pdf_export',
                    'status': 'completed',
                    'progress': 0.4,
                    'file_path': pdf_file
                }
            
            # Step 4: Export MP4 video
            if 'mp4' in request.export_formats:
                yield {'stage': 'mp4_export', 'status': 'starting', 'progress': 0.45}
                
                mp4_file = await self._export_mp4_video(
                    request.content,
                    request.brand_kit,
                    request.template_options,
                    temp_dir,
                    export_settings,
                    progress_callback
                )
                exported_files['mp4'] = mp4_file
                
                export_log.append({
                    'stage': 'mp4_export',
                    'timestamp': datetime.now().isoformat(),
                    'file_path': mp4_file
                })
                
                yield {
                    'stage': 'mp4_export',
                    'status': 'completed',
                    'progress': 0.8,
                    'file_path': mp4_file
                }
            
            # Step 5: Create ZIP bundle
            if 'zip' in request.export_formats:
                yield {'stage': 'zip_export', 'status': 'starting', 'progress': 0.85}
                
                zip_file = await self._create_zip_bundle(
                    request.content,
                    exported_files,
                    temp_dir,
                    export_id
                )
                exported_files['zip'] = zip_file
                
                export_log.append({
                    'stage': 'zip_export',
                    'timestamp': datetime.now().isoformat(),
                    'file_path': zip_file
                })
                
                yield {
                    'stage': 'zip_export',
                    'status': 'completed',
                    'progress': 0.95,
                    'file_path': zip_file
                }
            
            # Step 6: Finalize and cleanup
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Convert files to base64 for API response
            base64_files = {}
            for format_name, file_path in exported_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        base64_files[format_name] = base64.b64encode(file_content).decode('utf-8')
            
            result = ExportResult(
                export_id=export_id,
                exported_files=base64_files,
                export_metadata={
                    'export_time': duration,
                    'formats_exported': list(exported_files.keys()),
                    'total_file_size': sum(
                        os.path.getsize(path) for path in exported_files.values() 
                        if os.path.exists(path)
                    ),
                    'story_pack_id': request.story_pack_id,
                    'export_settings': export_settings
                },
                generation_log=export_log
            )
            
            yield {
                'stage': 'complete',
                'status': 'success',
                'progress': 1.0,
                'result': result
            }
            
        except Exception as e:
            export_log.append({
                'stage': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            yield {
                'stage': 'error',
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'export_log': export_log
            }
        
        finally:
            # Cleanup temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

    async def _export_json_metadata(
        self,
        content: Dict,
        story_pack_id: str,
        temp_dir: str,
        export_settings: Dict
    ) -> str:
        """Export comprehensive JSON metadata."""
        
        metadata = {
            'story_pack_id': story_pack_id,
            'export_timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'content': {
                'text': {
                    'script': content.get('text', ''),
                    'word_count': len(content.get('text', '').split()),
                    'character_count': len(content.get('text', '')),
                    'estimated_reading_time': len(content.get('text', '').split()) / 200  # 200 WPM
                },
                'images': {
                    'count': len(content.get('images', [])),
                    'formats': ['base64_encoded'] * len(content.get('images', [])),
                    'total_size_estimate': len(content.get('images', [])) * 500000  # Rough estimate
                },
                'audio': content.get('audio', {}).get('metadata', {}),
                'captions': content.get('captions', [])
            },
            'generation_metadata': content.get('metadata', {}),
            'export_settings': export_settings,
            'schema_version': '1.0'
        }
        
        # Add evaluation scores if available
        if 'evaluation_scores' in content:
            metadata['quality_scores'] = content['evaluation_scores']
        
        # Add safety analysis if available
        if 'safety_analysis' in content:
            metadata['safety_analysis'] = content['safety_analysis']
        
        json_file_path = os.path.join(temp_dir, f"{story_pack_id}_metadata.json")
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return json_file_path

    async def _export_pdf_report(
        self,
        content: Dict,
        story_pack_id: str,
        brand_kit: Optional[Dict],
        temp_dir: str,
        export_settings: Dict
    ) -> str:
        """Export comprehensive PDF report."""
        
        pdf_file_path = os.path.join(temp_dir, f"{story_pack_id}_report.pdf")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_file_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB') if not brand_kit else colors.HexColor(brand_kit.get('color_palette', {}).get('primary', '#2E86AB'))
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph(f"Story Pack Report: {story_pack_id}", title_style))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        summary_text = f"""
        This report provides a comprehensive overview of the generated story pack content, 
        including narrative script, visual assets, audio components, and quality metrics.
        
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Content Type: Multi-modal Story Pack
        Total Components: {len(content.get('images', [])) + (1 if content.get('text') else 0) + (1 if content.get('audio') else 0)}
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Content Overview
        story.append(Paragraph("Content Overview", styles['Heading2']))
        
        # Text content
        if content.get('text'):
            story.append(Paragraph("Narrative Script", styles['Heading3']))
            
            # Truncate long text for PDF
            script_text = content['text']
            if len(script_text) > 2000:
                script_text = script_text[:2000] + "... [Content truncated for report]"
            
            story.append(Paragraph(script_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Text statistics
            word_count = len(content['text'].split())
            char_count = len(content['text'])
            reading_time = word_count / 200  # 200 WPM average
            
            stats_data = [
                ['Metric', 'Value'],
                ['Word Count', str(word_count)],
                ['Character Count', str(char_count)],
                ['Estimated Reading Time', f"{reading_time:.1f} minutes"],
            ]
            
            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 20))
        
        # Visual content
        if content.get('images'):
            story.append(Paragraph("Visual Assets", styles['Heading3']))
            story.append(Paragraph(f"Generated {len(content['images'])} images for this story pack.", styles['Normal']))
            
            # Add first image as sample (if not too large)
            try:
                if content['images']:
                    first_image_b64 = content['images'][0]
                    image_data = base64.b64decode(first_image_b64)
                    
                    # Save temporary image for PDF
                    temp_img_path = os.path.join(temp_dir, "sample_image.png")
                    with open(temp_img_path, 'wb') as f:
                        f.write(image_data)
                    
                    # Add to PDF (scaled down)
                    img = RLImage(temp_img_path, width=4*inch, height=3*inch)
                    story.append(img)
                    story.append(Paragraph("Sample Image from Story Pack", styles['Caption']))
            except Exception as e:
                story.append(Paragraph(f"[Image preview unavailable: {str(e)}]", styles['Normal']))
            
            story.append(Spacer(1, 20))
        
        # Audio content
        if content.get('audio'):
            story.append(Paragraph("Audio Components", styles['Heading3']))
            
            audio_metadata = content['audio'].get('metadata', {})
            audio_info = f"""
            Duration: {audio_metadata.get('total_duration', 0):.1f} seconds
            Voice: {audio_metadata.get('voice_used', 'Unknown')}
            Sample Rate: {audio_metadata.get('sample_rate', 'Unknown')} Hz
            Quality: {audio_metadata.get('bit_depth', 'Unknown')} bit
            """
            
            story.append(Paragraph(audio_info, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Quality Metrics (if available)
        if content.get('evaluation_scores'):
            story.append(Paragraph("Quality Assessment", styles['Heading2']))
            
            scores = content['evaluation_scores']
            quality_data = [['Metric', 'Score', 'Rating']]
            
            for metric, score in scores.items():
                rating = self._score_to_rating(score)
                quality_data.append([metric.replace('_', ' ').title(), f"{score:.2f}", rating])
            
            quality_table = Table(quality_data)
            quality_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(quality_table)
            story.append(Spacer(1, 20))
        
        # Brand Kit Information (if available)
        if brand_kit:
            story.append(Paragraph("Brand Guidelines Applied", styles['Heading2']))
            
            brand_info = f"""
            Brand Kit: {brand_kit.get('name', 'Unnamed')}
            Primary Color: {brand_kit.get('color_palette', {}).get('primary', 'Not specified')}
            Typography: {brand_kit.get('typography', {}).get('heading_font', 'Default')}
            Brand Voice: {brand_kit.get('lexicon', {}).get('brand_voice', 'Not specified')}
            """
            
            story.append(Paragraph(brand_info, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return pdf_file_path

    async def _export_mp4_video(
        self,
        content: Dict,
        brand_kit: Optional[Dict],
        template_options: Optional[Dict],
        temp_dir: str,
        export_settings: Dict,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Export MP4 video from story pack content."""
        
        mp4_file_path = os.path.join(temp_dir, "story_pack_video.mp4")
        
        # Get template settings
        template_name = template_options.get('template', 'story_pack') if template_options else 'story_pack'
        template = self.video_templates.get(template_name, self.video_templates['story_pack'])
        
        # Video settings
        video_settings = {**self.default_video_settings, **export_settings.get('video', {})}
        width, height = map(int, video_settings['resolution'].split('x'))
        fps = video_settings['fps']
        
        try:
            clips = []
            
            # 1. Create intro clip
            if progress_callback:
                await progress_callback({'stage': 'video_intro', 'progress': 0.1})
            
            intro_clip = self._create_intro_clip(
                content.get('text', ''),
                brand_kit,
                template['intro_duration'],
                (width, height)
            )
            clips.append(intro_clip)
            
            # 2. Create scene clips from images and text
            if content.get('images') and progress_callback:
                await progress_callback({'stage': 'video_scenes', 'progress': 0.3})
            
            scene_clips = await self._create_scene_clips(
                content.get('images', []),
                content.get('text', ''),
                content.get('captions', []),
                template,
                (width, height),
                brand_kit
            )
            clips.extend(scene_clips)
            
            # 3. Create outro clip
            if progress_callback:
                await progress_callback({'stage': 'video_outro', 'progress': 0.7})
            
            outro_clip = self._create_outro_clip(
                brand_kit,
                template['outro_duration'],
                (width, height)
            )
            clips.append(outro_clip)
            
            # 4. Concatenate all clips
            if progress_callback:
                await progress_callback({'stage': 'video_assembly', 'progress': 0.8})
            
            final_video = concatenate_videoclips(clips, method="compose")
            
            # 5. Add audio if available
            if content.get('audio') and content['audio'].get('mixed_audio'):
                if progress_callback:
                    await progress_callback({'stage': 'video_audio', 'progress': 0.9})
                
                # Decode audio from base64
                audio_b64 = content['audio']['mixed_audio']
                audio_data = base64.b64decode(audio_b64)
                
                # Save temporary audio file
                temp_audio_path = os.path.join(temp_dir, "temp_audio.mp3")
                with open(temp_audio_path, 'wb') as f:
                    f.write(audio_data)
                
                # Load audio and set to video
                audio_clip = AudioFileClip(temp_audio_path)
                
                # Adjust audio duration to match video
                if audio_clip.duration > final_video.duration:
                    audio_clip = audio_clip.subclip(0, final_video.duration)
                elif audio_clip.duration < final_video.duration:
                    # Loop audio if needed
                    loops_needed = int(final_video.duration / audio_clip.duration) + 1
                    audio_clip = concatenate_videoclips([audio_clip] * loops_needed).subclip(0, final_video.duration)
                
                final_video = final_video.set_audio(audio_clip)
            
            # 6. Export video
            if progress_callback:
                await progress_callback({'stage': 'video_export', 'progress': 0.95})
            
            # Export in separate thread to avoid blocking
            await asyncio.to_thread(
                final_video.write_videofile,
                mp4_file_path,
                fps=fps,
                codec=video_settings['codec'],
                bitrate=video_settings['bitrate'],
                audio_codec=video_settings['audio_codec'],
                verbose=False,
                logger=None
            )
            
            # Cleanup clips
            for clip in clips:
                clip.close()
            final_video.close()
            
            return mp4_file_path
            
        except Exception as e:
            # Fallback: create simple slideshow video
            print(f"Advanced video creation failed: {e}. Creating simple slideshow.")
            return await self._create_simple_slideshow(content, temp_dir, (width, height), fps)

    def _create_intro_clip(
        self,
        title_text: str,
        brand_kit: Optional[Dict],
        duration: float,
        size: Tuple[int, int]
    ) -> VideoFileClip:
        """Create intro clip with title."""
        
        width, height = size
        
        # Extract title from text (first sentence or first 50 chars)
        title = title_text.split('.')[0][:50] + "..." if len(title_text) > 50 else title_text.split('.')[0]
        
        # Brand colors
        bg_color = brand_kit.get('color_palette', {}).get('background', '#FFFFFF') if brand_kit else '#FFFFFF'
        text_color = brand_kit.get('color_palette', {}).get('text', '#000000') if brand_kit else '#000000'
        
        # Create background
        bg_clip = ColorClip(size=(width, height), color=bg_color, duration=duration)
        
        # Create title text
        try:
            title_clip = TextClip(
                title,
                fontsize=60,
                color=text_color,
                font='Arial-Bold'
            ).set_position('center').set_duration(duration)
            
            # Composite
            intro = CompositeVideoClip([bg_clip, title_clip])
            
        except Exception as e:
            print(f"Text clip creation failed: {e}. Using background only.")
            intro = bg_clip
        
        return intro

    async def _create_scene_clips(
        self,
        images: List[str],
        text: str,
        captions: List[str],
        template: Dict,
        size: Tuple[int, int],
        brand_kit: Optional[Dict]
    ) -> List[VideoFileClip]:
        """Create scene clips from images and text."""
        
        clips = []
        scene_duration = template['scene_duration']
        
        for i, image_b64 in enumerate(images):
            try:
                # Decode image
                image_data = base64.b64decode(image_b64)
                
                # Create temporary image file
                temp_img_path = f"/tmp/scene_{i}.png"
                with open(temp_img_path, 'wb') as f:
                    f.write(image_data)
                
                # Create image clip
                img_clip = ImageClip(temp_img_path, duration=scene_duration).resize(size)
                
                # Add caption if available and template allows
                if template.get('text_overlay') and i < len(captions) and captions[i]:
                    try:
                        caption_clip = TextClip(
                            captions[i][:100],  # Limit caption length
                            fontsize=24,
                            color='white',
                            stroke_color='black',
                            stroke_width=2
                        ).set_position(('center', 'bottom')).set_duration(scene_duration)
                        
                        scene_clip = CompositeVideoClip([img_clip, caption_clip])
                    except:
                        scene_clip = img_clip
                else:
                    scene_clip = img_clip
                
                clips.append(scene_clip)
                
                # Cleanup temp file
                try:
                    os.remove(temp_img_path)
                except:
                    pass
                
            except Exception as e:
                print(f"Failed to create scene clip {i}: {e}")
                # Create placeholder clip
                placeholder = ColorClip(size=size, color='#CCCCCC', duration=scene_duration)
                clips.append(placeholder)
        
        return clips

    def _create_outro_clip(
        self,
        brand_kit: Optional[Dict],
        duration: float,
        size: Tuple[int, int]
    ) -> VideoFileClip:
        """Create outro clip."""
        
        width, height = size
        
        # Brand colors
        bg_color = brand_kit.get('color_palette', {}).get('primary', '#2E86AB') if brand_kit else '#2E86AB'
        text_color = brand_kit.get('color_palette', {}).get('background', '#FFFFFF') if brand_kit else '#FFFFFF'
        
        # Create background
        bg_clip = ColorClip(size=(width, height), color=bg_color, duration=duration)
        
        # Create outro text
        outro_text = "Created with Synesthesia AI"
        if brand_kit and brand_kit.get('name'):
            outro_text = f"Created with {brand_kit['name']}"
        
        try:
            text_clip = TextClip(
                outro_text,
                fontsize=40,
                color=text_color,
                font='Arial'
            ).set_position('center').set_duration(duration)
            
            outro = CompositeVideoClip([bg_clip, text_clip])
            
        except Exception as e:
            print(f"Outro text creation failed: {e}. Using background only.")
            outro = bg_clip
        
        return outro

    async def _create_simple_slideshow(
        self,
        content: Dict,
        temp_dir: str,
        size: Tuple[int, int],
        fps: int
    ) -> str:
        """Create simple slideshow video as fallback."""
        
        mp4_file_path = os.path.join(temp_dir, "simple_slideshow.mp4")
        
        try:
            # Use FFmpeg to create slideshow
            images = content.get('images', [])
            if not images:
                # Create placeholder video
                cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', f'color=c=blue:s={size[0]}x{size[1]}:d=5',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4_file_path, '-y'
                ]
            else:
                # Save images temporarily
                image_paths = []
                for i, img_b64 in enumerate(images[:5]):  # Limit to 5 images
                    img_data = base64.b64decode(img_b64)
                    img_path = os.path.join(temp_dir, f"slide_{i:03d}.png")
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    image_paths.append(img_path)
                
                # Create slideshow with FFmpeg
                cmd = [
                    'ffmpeg', '-framerate', f'1/{3}',  # 3 seconds per image
                    '-pattern_type', 'glob', '-i', os.path.join(temp_dir, 'slide_*.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-vf', f'scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2',
                    mp4_file_path, '-y'
                ]
            
            # Run FFmpeg
            await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)
            
            return mp4_file_path
            
        except Exception as e:
            print(f"Simple slideshow creation failed: {e}")
            # Create minimal video file
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', f'color=c=black:s={size[0]}x{size[1]}:d=5',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4_file_path, '-y'
            ]
            try:
                await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)
            except:
                # Create empty file as last resort
                with open(mp4_file_path, 'w') as f:
                    f.write("")
            
            return mp4_file_path

    async def _create_zip_bundle(
        self,
        content: Dict,
        exported_files: Dict[str, str],
        temp_dir: str,
        export_id: str
    ) -> str:
        """Create ZIP bundle with all content and exports."""
        
        zip_file_path = os.path.join(temp_dir, f"{export_id}_bundle.zip")
        
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add exported files
            for format_name, file_path in exported_files.items():
                if os.path.exists(file_path):
                    arcname = f"exports/{os.path.basename(file_path)}"
                    zipf.write(file_path, arcname)
            
            # Add raw content files
            
            # 1. Save script as text file
            if content.get('text'):
                script_path = os.path.join(temp_dir, "script.txt")
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(content['text'])
                zipf.write(script_path, "content/script.txt")
            
            # 2. Save images
            if content.get('images'):
                for i, img_b64 in enumerate(content['images']):
                    img_data = base64.b64decode(img_b64)
                    img_path = os.path.join(temp_dir, f"image_{i+1:03d}.png")
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    zipf.write(img_path, f"content/images/image_{i+1:03d}.png")
            
            # 3. Save audio
            if content.get('audio') and content['audio'].get('mixed_audio'):
                audio_data = base64.b64decode(content['audio']['mixed_audio'])
                audio_path = os.path.join(temp_dir, "audio.mp3")
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                zipf.write(audio_path, "content/audio.mp3")
            
            # 4. Save captions
            if content.get('captions'):
                captions_path = os.path.join(temp_dir, "captions.srt")
                with open(captions_path, 'w', encoding='utf-8') as f:
                    for i, caption in enumerate(content['captions']):
                        # Simple SRT format
                        start_time = i * 3  # 3 seconds per caption
                        end_time = start_time + 3
                        f.write(f"{i+1}\n")
                        f.write(f"{self._seconds_to_srt_time(start_time)} --> {self._seconds_to_srt_time(end_time)}\n")
                        f.write(f"{caption}\n\n")
                zipf.write(captions_path, "content/captions.srt")
            
            # 5. Add README
            readme_content = f"""
# Story Pack Bundle - {export_id}

This bundle contains all generated content and exports for your story pack.

## Contents:

### Exports/
- JSON metadata with comprehensive information
- PDF report with formatted content overview
- MP4 video compilation (if generated)

### Content/
- script.txt: The generated narrative script
- images/: All generated images in PNG format
- audio.mp3: Mixed audio with narration and background (if available)
- captions.srt: Subtitle file for video content

## Usage:
- Use the MP4 video for direct sharing or embedding
- Use individual content files for custom editing
- Refer to the JSON metadata for technical details
- Use the PDF report for stakeholder reviews

Generated by Synesthesia AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            zipf.write(readme_path, "README.md")
        
        return zip_file_path

    def _seconds_to_srt_time(self, seconds: int) -> str:
        """Convert seconds to SRT time format."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d},000"

    def _score_to_rating(self, score: float) -> str:
        """Convert numeric score to rating."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Needs Improvement"

    async def get_export_templates(self) -> Dict[str, Dict]:
        """Get available export templates."""
        return {
            'video_templates': self.video_templates,
            'export_formats': ['mp4', 'zip', 'json', 'pdf'],
            'video_resolutions': ['1920x1080', '1280x720', '854x480'],
            'video_quality_presets': {
                'high': {'bitrate': '8000k', 'fps': 30},
                'medium': {'bitrate': '5000k', 'fps': 30},
                'low': {'bitrate': '2000k', 'fps': 24}
            }
        }

    async def estimate_export_time(
        self,
        content: Dict,
        export_formats: List[str],
        export_settings: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Estimate export time for different formats."""
        
        estimates = {}
        
        # Base estimates in seconds
        base_estimates = {
            'json': 1.0,
            'pdf': 5.0,
            'zip': 3.0,
            'mp4': 30.0  # Base estimate
        }
        
        for format_name in export_formats:
            estimate = base_estimates.get(format_name, 5.0)
            
            # Adjust based on content
            if format_name == 'mp4':
                num_images = len(content.get('images', []))
                has_audio = bool(content.get('audio'))
                
                # More images = longer processing
                estimate += num_images * 2.0
                
                # Audio processing adds time
                if has_audio:
                    estimate += 10.0
                
                # High quality settings add time
                if export_settings and export_settings.get('video', {}).get('quality') == 'high':
                    estimate *= 1.5
            
            elif format_name == 'pdf':
                # More content = longer PDF generation
                text_length = len(content.get('text', ''))
                num_images = len(content.get('images', []))
                
                estimate += (text_length / 1000) * 0.5  # 0.5s per 1000 chars
                estimate += num_images * 1.0  # 1s per image
            
            estimates[format_name] = estimate
        
        return estimates
