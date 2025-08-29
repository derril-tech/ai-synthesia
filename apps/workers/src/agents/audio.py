"""
Audio Worker - Generates TTS audio and soundbeds with mixdown
"""

import asyncio
import base64
import io
import json
import tempfile
import os
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import openai
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import numpy as np


@dataclass
class AudioRequest:
    """Request for audio generation."""
    script: str
    captions: List[str]
    voice_settings: Optional[Dict] = None
    background_music: Optional[str] = None  # URL or style
    audio_style: str = "narration"  # narration, podcast, commercial, etc.
    target_duration: Optional[int] = None  # seconds
    ssml_presets: Optional[Dict] = None


@dataclass
class AudioResult:
    """Result of audio generation."""
    narration_audio: str  # Base64 encoded audio
    background_audio: Optional[str]  # Base64 encoded background
    mixed_audio: str  # Final mixed audio
    audio_metadata: Dict
    generation_log: List[Dict]


class AudioWorker:
    """AI-powered audio generation and mixing worker."""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Default voice settings
        self.default_voices = {
            "narration": "nova",  # Clear, professional
            "podcast": "shimmer",  # Conversational
            "commercial": "alloy",  # Energetic
            "storytelling": "echo",  # Warm, engaging
        }
        
        # Audio processing settings
        self.sample_rate = 44100
        self.bit_depth = 16

    async def generate_audio(
        self,
        request: AudioRequest,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """Generate audio content with streaming updates."""
        
        generation_log = []
        start_time = datetime.now()
        
        try:
            # Step 1: Prepare SSML and voice settings
            yield {'stage': 'preparation', 'status': 'starting', 'progress': 0.1}
            
            ssml_content = await self._create_ssml(
                request.script,
                request.captions,
                request.ssml_presets,
                request.voice_settings
            )
            
            voice_name = self._select_voice(request.audio_style, request.voice_settings)
            
            generation_log.append({
                'stage': 'preparation',
                'timestamp': datetime.now().isoformat(),
                'voice_selected': voice_name,
                'ssml_length': len(ssml_content)
            })
            
            yield {
                'stage': 'preparation',
                'status': 'completed',
                'progress': 0.2,
                'voice': voice_name,
                'ssml_preview': ssml_content[:200] + "..." if len(ssml_content) > 200 else ssml_content
            }
            
            # Step 2: Generate narration audio
            yield {'stage': 'narration', 'status': 'starting', 'progress': 0.25}
            
            narration_segments = await self._generate_narration_segments(
                ssml_content,
                voice_name,
                request.captions
            )
            
            # Combine narration segments
            narration_audio = await self._combine_audio_segments(narration_segments)
            
            generation_log.append({
                'stage': 'narration',
                'timestamp': datetime.now().isoformat(),
                'segments_generated': len(narration_segments),
                'total_duration': len(narration_audio) / 1000.0  # Convert to seconds
            })
            
            yield {
                'stage': 'narration',
                'status': 'completed',
                'progress': 0.6,
                'duration': len(narration_audio) / 1000.0,
                'segments': len(narration_segments)
            }
            
            # Step 3: Generate or select background music
            background_audio = None
            if request.background_music:
                yield {'stage': 'background', 'status': 'starting', 'progress': 0.65}
                
                background_audio = await self._generate_background_music(
                    request.background_music,
                    len(narration_audio) / 1000.0,  # Duration in seconds
                    request.audio_style
                )
                
                generation_log.append({
                    'stage': 'background',
                    'timestamp': datetime.now().isoformat(),
                    'background_type': request.background_music,
                    'success': background_audio is not None
                })
                
                yield {
                    'stage': 'background',
                    'status': 'completed',
                    'progress': 0.8,
                    'has_background': background_audio is not None
                }
            
            # Step 4: Audio mixing and mastering
            yield {'stage': 'mixing', 'status': 'starting', 'progress': 0.85}
            
            mixed_audio = await self._mix_audio(
                narration_audio,
                background_audio,
                request.audio_style
            )
            
            # Apply final mastering
            mixed_audio = await self._master_audio(mixed_audio, request.audio_style)
            
            generation_log.append({
                'stage': 'mixing',
                'timestamp': datetime.now().isoformat(),
                'final_duration': len(mixed_audio) / 1000.0,
                'has_background': background_audio is not None
            })
            
            # Convert to base64 for response
            narration_b64 = await self._audio_to_base64(narration_audio)
            background_b64 = await self._audio_to_base64(background_audio) if background_audio else None
            mixed_b64 = await self._audio_to_base64(mixed_audio)
            
            # Final result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = AudioResult(
                narration_audio=narration_b64,
                background_audio=background_b64,
                mixed_audio=mixed_b64,
                audio_metadata={
                    'total_duration': len(mixed_audio) / 1000.0,
                    'sample_rate': self.sample_rate,
                    'bit_depth': self.bit_depth,
                    'voice_used': voice_name,
                    'has_background': background_audio is not None,
                    'audio_style': request.audio_style,
                    'generation_time': duration,
                    'segments_count': len(narration_segments)
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

    async def _create_ssml(
        self,
        script: str,
        captions: List[str],
        ssml_presets: Optional[Dict],
        voice_settings: Optional[Dict]
    ) -> str:
        """Create SSML markup from script and captions."""
        
        # Default SSML settings
        default_rate = "medium"
        default_pitch = "medium"
        default_volume = "medium"
        
        if ssml_presets:
            default_rate = ssml_presets.get('speaking_rate', default_rate)
            default_pitch = ssml_presets.get('pitch', default_pitch)
            default_volume = ssml_presets.get('volume', default_volume)
        
        if voice_settings:
            default_rate = voice_settings.get('rate', default_rate)
            default_pitch = voice_settings.get('pitch', default_pitch)
            default_volume = voice_settings.get('volume', default_volume)
        
        # Build SSML
        ssml_parts = [
            f'<speak>',
            f'<prosody rate="{default_rate}" pitch="{default_pitch}" volume="{default_volume}">'
        ]
        
        # Split script into sentences and add appropriate SSML tags
        sentences = script.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence with appropriate pauses
            ssml_parts.append(f'{sentence}.')
            
            # Add pause between sentences
            ssml_parts.append('<break time="0.5s"/>')
        
        ssml_parts.extend(['</prosody>', '</speak>'])
        
        return ' '.join(ssml_parts)

    def _select_voice(self, audio_style: str, voice_settings: Optional[Dict]) -> str:
        """Select appropriate voice for the audio style."""
        
        if voice_settings and 'voice' in voice_settings:
            return voice_settings['voice']
        
        return self.default_voices.get(audio_style, "nova")

    async def _generate_narration_segments(
        self,
        ssml_content: str,
        voice_name: str,
        captions: List[str]
    ) -> List[AudioSegment]:
        """Generate narration audio segments using OpenAI TTS."""
        
        segments = []
        
        # Split SSML into chunks if too long (OpenAI has limits)
        max_chunk_size = 4000  # characters
        chunks = self._split_ssml_into_chunks(ssml_content, max_chunk_size)
        
        for chunk in chunks:
            try:
                # Generate audio for chunk
                response = await asyncio.to_thread(
                    self.openai_client.audio.speech.create,
                    model="tts-1-hd",  # High quality model
                    voice=voice_name,
                    input=chunk,
                    response_format="mp3"
                )
                
                # Convert to AudioSegment
                audio_data = response.content
                segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                segments.append(segment)
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"TTS generation failed for chunk: {e}")
                # Create silence as fallback
                silence = AudioSegment.silent(duration=2000)  # 2 seconds
                segments.append(silence)
        
        return segments

    async def _combine_audio_segments(self, segments: List[AudioSegment]) -> AudioSegment:
        """Combine multiple audio segments into one."""
        
        if not segments:
            return AudioSegment.silent(duration=1000)  # 1 second silence
        
        combined = segments[0]
        
        for segment in segments[1:]:
            # Add small gap between segments
            gap = AudioSegment.silent(duration=200)  # 0.2 seconds
            combined = combined + gap + segment
        
        return combined

    async def _generate_background_music(
        self,
        background_spec: str,
        duration_seconds: float,
        audio_style: str
    ) -> Optional[AudioSegment]:
        """Generate or retrieve background music."""
        
        try:
            # For now, create simple ambient background
            # In production, you'd use music generation APIs or libraries
            
            # Generate simple ambient tone based on style
            if audio_style == "commercial":
                # Upbeat background
                frequency = 440  # A4
                volume = -20  # dB
            elif audio_style == "storytelling":
                # Warm, subtle background
                frequency = 220  # A3
                volume = -25  # dB
            else:
                # Neutral background
                frequency = 330  # E4
                volume = -30  # dB
            
            # Create simple sine wave background
            duration_ms = int(duration_seconds * 1000)
            background = self._generate_sine_wave(frequency, duration_ms, volume)
            
            # Apply fade in/out
            background = background.fade_in(2000).fade_out(2000)
            
            return background
            
        except Exception as e:
            print(f"Background music generation failed: {e}")
            return None

    def _generate_sine_wave(self, frequency: float, duration_ms: int, volume_db: float) -> AudioSegment:
        """Generate a sine wave audio segment."""
        
        # Generate sine wave
        sample_rate = self.sample_rate
        duration_s = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply volume
        wave = wave * (10 ** (volume_db / 20))
        
        # Convert to 16-bit integers
        wave = (wave * 32767).astype(np.int16)
        
        # Create AudioSegment
        audio = AudioSegment(
            wave.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1
        )
        
        return audio

    async def _mix_audio(
        self,
        narration: AudioSegment,
        background: Optional[AudioSegment],
        audio_style: str
    ) -> AudioSegment:
        """Mix narration with background music."""
        
        if not background:
            return narration
        
        # Adjust background volume based on style
        volume_adjustments = {
            "narration": -15,  # Quiet background
            "podcast": -12,    # Subtle background
            "commercial": -8,  # More prominent background
            "storytelling": -10  # Moderate background
        }
        
        bg_volume_db = volume_adjustments.get(audio_style, -12)
        
        # Adjust background volume
        background_adjusted = background + bg_volume_db
        
        # Ensure background matches narration length
        if len(background_adjusted) < len(narration):
            # Loop background if too short
            loops_needed = (len(narration) // len(background_adjusted)) + 1
            background_adjusted = background_adjusted * loops_needed
        
        # Trim background to match narration length
        background_adjusted = background_adjusted[:len(narration)]
        
        # Mix the audio
        mixed = narration.overlay(background_adjusted)
        
        return mixed

    async def _master_audio(self, audio: AudioSegment, audio_style: str) -> AudioSegment:
        """Apply mastering effects to final audio."""
        
        # Normalize audio levels
        mastered = normalize(audio)
        
        # Apply compression based on style
        if audio_style in ["commercial", "podcast"]:
            # More aggressive compression for broadcast styles
            mastered = compress_dynamic_range(mastered, threshold=-20.0, ratio=4.0)
        else:
            # Gentle compression for narration/storytelling
            mastered = compress_dynamic_range(mastered, threshold=-25.0, ratio=2.0)
        
        # Apply EQ (simplified - boost presence frequencies)
        # In production, use more sophisticated audio processing
        
        # Final limiting to prevent clipping
        if mastered.max_dBFS > -1.0:
            mastered = mastered.apply_gain(-1.0 - mastered.max_dBFS)
        
        return mastered

    def _split_ssml_into_chunks(self, ssml: str, max_size: int) -> List[str]:
        """Split SSML content into manageable chunks."""
        
        if len(ssml) <= max_size:
            return [ssml]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences to maintain natural breaks
        sentences = ssml.split('.')
        
        for sentence in sentences:
            if len(current_chunk + sentence + '.') <= max_size:
                current_chunk += sentence + '.'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '.'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    async def _audio_to_base64(self, audio: Optional[AudioSegment]) -> Optional[str]:
        """Convert AudioSegment to base64 string."""
        
        if not audio:
            return None
        
        try:
            # Export to MP3 format
            buffer = io.BytesIO()
            audio.export(buffer, format="mp3", bitrate="128k")
            
            # Encode to base64
            audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return audio_b64
            
        except Exception as e:
            print(f"Audio encoding failed: {e}")
            return None

    async def adjust_audio_timing(
        self,
        audio: AudioSegment,
        target_duration: float,
        preserve_pitch: bool = True
    ) -> AudioSegment:
        """Adjust audio timing to match target duration."""
        
        current_duration = len(audio) / 1000.0  # Convert to seconds
        speed_ratio = current_duration / target_duration
        
        if abs(speed_ratio - 1.0) < 0.05:  # Within 5%, no adjustment needed
            return audio
        
        try:
            if preserve_pitch:
                # Use time-stretching to preserve pitch
                # This is a simplified version - in production, use librosa or similar
                new_frame_rate = int(audio.frame_rate * speed_ratio)
                adjusted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
                adjusted = adjusted.set_frame_rate(audio.frame_rate)
            else:
                # Simple speed adjustment (changes pitch)
                new_frame_rate = int(audio.frame_rate * speed_ratio)
                adjusted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
            
            return adjusted
            
        except Exception as e:
            print(f"Audio timing adjustment failed: {e}")
            return audio  # Return original if adjustment fails
