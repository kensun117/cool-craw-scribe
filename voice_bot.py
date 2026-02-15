import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import discord
import mlx_whisper
import numpy as np
import requests
import torch
from discord.ext import commands

LOGGER = logging.getLogger(__name__)

# Configuration - Discord Bot Token for voice bot
DISCORD_TOKEN = os.getenv("VOICE_DISCORD_TOKEN", "")

# OpenClaw Gateway configuration
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

# Audio settings
SAMPLE_RATE = 48000
TARGET_RATE = 16000
CHUNK_DURATION = 3.0  # Process in 3-second chunks
HF_TOKEN = os.getenv("HF_TOKEN", "")


@dataclass
class AudioBuffer:
    """Buffer for accumulating audio data per user."""
    user_id: int
    username: str
    data: bytes = field(default_factory=bytes)
    last_activity: float = field(default_factory=time.time)


class SimpleVAD:
    """Simple energy-based voice activity detection."""
    
    def has_speech(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if audio contains speech based on energy."""
        # Normalize
        audio = audio.astype(np.float32) / 32768.0
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > threshold


class VoiceBot(commands.Bot):
    """Discord bot for real-time voice transcription."""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.messages = True
        intents.message_content = True
        
        super().__init__(intents=intents)
        
        self.active_sessions: dict[int, dict] = {}
        self.audio_buffers: dict[int, AudioBuffer] = {}
        self.vad = SimpleVAD()
        self.processing_task = None
        
    async def on_ready(self):
        """Called when bot is ready."""
        LOGGER.info(f"Voice Bot logged in as {self.user}")
        
        # Start audio processing loop
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._audio_processing_loop())
    
    async def setup_hook(self):
        """Set up slash commands."""
        guilds = [discord.Object(id=g.id) for g in self.guilds]
        
        @self.tree.command(name="voice_on", description="开启实时语音转文字")
        async def voice_on(interaction: discord.Interaction, language: str = "zh"):
            """Start voice transcription."""
            if not interaction.user.voice or not interaction.user.voice.channel:
                await interaction.response.send_message("❌ 你需要先进入语音频道！", ephemeral=True)
                return
            
            guild_id = interaction.guild_id
            if guild_id in self.active_sessions:
                await interaction.response.send_message("⚠️ 已经在转录中了！", ephemeral=True)
                return
            
            voice_channel = interaction.user.voice.channel
            
            # Connect to voice
            voice_client = await voice_channel.connect()
            
            self.active_sessions[guild_id] = {
                "voice_client": voice_client,
                "text_channel": interaction.channel,
                "language": language,
                "guild_id": guild_id
            }
            
            # Start recording
            sink = discord.sinks.WaveSink()
            
            def on_recording(sink, *args):
                asyncio.create_task(self._handle_recording(sink, guild_id))
            
            voice_client.start_recording(sink, on_recording, interaction.channel)
            
            await interaction.response.send_message(
                f"🎙️ **实时语音转录已开启！**\n"
                f"频道: {voice_channel.mention}\n"
                f"语言: `{language}`\n"
                f"使用 `/voice_off` 停止",
                ephemeral=False
            )
        
        @self.tree.command(name="voice_off", description="关闭实时语音转文字")
        async def voice_off(interaction: discord.Interaction):
            """Stop voice transcription."""
            guild_id = interaction.guild_id
            
            if guild_id not in self.active_sessions:
                await interaction.response.send_message("⚠️ 没有在运行的转录。", ephemeral=True)
                return
            
            session = self.active_sessions.pop(guild_id)
            voice_client = session["voice_client"]
            
            if voice_client.is_recording():
                voice_client.stop_recording()
            
            if voice_client.is_connected():
                await voice_client.disconnect(force=True)
            
            await interaction.response.send_message("🛑 语音转录已停止。", ephemeral=False)
        
        # Sync commands
        for guild in guilds:
            try:
                await self.tree.sync(guild=guild)
            except Exception as e:
                LOGGER.warning(f"Failed to sync commands for guild {guild.id}: {e}")
    
    async def _handle_recording(self, sink: discord.sinks.Sink, guild_id: int):
        """Handle incoming audio data."""
        audio_data = getattr(sink, "audio_data", {})
        
        for user_id, data in audio_data.items():
            file_obj = getattr(data, "file", None)
            if file_obj is None:
                continue
            
            file_obj.seek(0)
            raw_audio = file_obj.read()
            
            if not raw_audio:
                continue
            
            # Get username
            username = f"User_{user_id}"
            guild = self.get_guild(guild_id)
            if guild:
                member = guild.get_member(user_id)
                if member:
                    username = member.display_name
            
            # Accumulate in buffer
            if user_id not in self.audio_buffers:
                self.audio_buffers[user_id] = AudioBuffer(user_id=user_id, username=username)
            
            buffer = self.audio_buffers[user_id]
            buffer.data += raw_audio
            buffer.username = username
            buffer.last_activity = time.time()
    
    async def _audio_processing_loop(self):
        """Main processing loop."""
        while True:
            try:
                await asyncio.sleep(1.0)  # Process every second
                
                current_time = time.time()
                
                for user_id, buffer in list(self.audio_buffers.items()):
                    # Check if we have enough data (3+ seconds)
                    min_bytes = int(CHUNK_DURATION * SAMPLE_RATE * 2 * 2)
                    
                    if len(buffer.data) >= min_bytes:
                        await self._process_chunk(user_id, buffer)
                    
                    # Clean up old buffers
                    elif current_time - buffer.last_activity > 15:
                        del self.audio_buffers[user_id]
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.exception("Processing error")
    
    async def _process_chunk(self, user_id: int, buffer: AudioBuffer):
        """Process audio chunk."""
        # Take chunk
        chunk_size = int(CHUNK_DURATION * SAMPLE_RATE * 2 * 2)
        audio_chunk = buffer.data[:chunk_size]
        buffer.data = buffer.data[chunk_size:]
        
        # Convert to numpy
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Stereo to mono
        audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        # Resample 48kHz -> 16kHz
        audio_array = self._resample(audio_array, SAMPLE_RATE, TARGET_RATE)
        
        # Check for speech
        if not self.vad.has_speech(audio_array):
            return
        
        # Save temp file
        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"chunk_{user_id}.wav"
        
        try:
            import soundfile as sf
            audio_float = audio_array.astype(np.float32) / 32768.0
            sf.write(str(temp_wav), audio_float, TARGET_RATE)
            
            # Find language setting
            language = "zh"
            for session in self.active_sessions.values():
                language = session.get("language", "zh")
                break
            
            # Transcribe
            result = mlx_whisper.transcribe(
                str(temp_wav),
                path_or_hf_repo="mlx-community/whisper-small-mlx",
                language=language,
                word_timestamps=False
            )
            
            text = result.get("text", "").strip()
            
            if text and len(text) > 2:  # Filter out very short/noise results
                LOGGER.info(f"[{buffer.username}] {text}")
                
                # Send to text channels
                for session in self.active_sessions.values():
                    text_channel = session.get("text_channel")
                    if text_channel:
                        await text_channel.send(f"🎙️ **{buffer.username}**: {text}")
                
                # Send to OpenClaw Gateway
                await self._send_to_openclaw(text, buffer.username, user_id)
                        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _send_to_openclaw(self, text: str, username: str, user_id: int):
        """Send transcription to OpenClaw via Gateway."""
        try:
            headers = {}
            if OPENCLAW_GATEWAY_TOKEN:
                headers["Authorization"] = f"Bearer {OPENCLAW_GATEWAY_TOKEN}"
            
            payload = {
                "type": "voice_transcription",
                "content": text,
                "speaker": username,
                "user_id": str(user_id),
                "timestamp": time.time(),
                "channel": "discord"
            }
            
            # Use asyncio.to_thread for sync requests
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{OPENCLAW_GATEWAY_URL}/api/v1/events",
                    json=payload,
                    headers=headers,
                    timeout=10
                )
            )
            
            if response.status_code == 200:
                LOGGER.debug(f"Sent to OpenClaw: {text[:50]}...")
            else:
                LOGGER.warning(f"Failed to send to OpenClaw: {response.status_code}")
                
        except Exception as e:
            LOGGER.error(f"Error sending to OpenClaw: {e}")
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio."""
        if orig_sr == target_sr:
            return audio
        
        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        
        indices = np.linspace(0, len(audio) - 1, new_length)
        indices_floor = np.floor(indices).astype(np.int64)
        indices_ceil = np.minimum(indices_floor + 1, len(audio) - 1)
        fractions = indices - indices_floor
        
        return audio[indices_floor] * (1 - fractions) + audio[indices_ceil] * fractions


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    if not DISCORD_TOKEN:
        LOGGER.error("Missing VOICE_DISCORD_TOKEN environment variable!")
        LOGGER.error("Set it with: export VOICE_DISCORD_TOKEN=your_token_here")
        return
    
    bot = VoiceBot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
