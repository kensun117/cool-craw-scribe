import asyncio
import json
import logging
import os
import struct
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from queue import Queue
import threading

import discord
import mlx_whisper
import numpy as np
import torch
from discord.ext import commands
from pyannote.audio import Model
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

LOGGER = logging.getLogger(__name__)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "ws://localhost:8080/ws")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

# Audio settings
SAMPLE_RATE = 48000  # Discord voice sample rate
TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 2.0  # Process audio in 2-second chunks
VAD_THRESHOLD = 0.5  # Voice activity detection threshold
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to process


@dataclass
class SpeakerProfile:
    """Speaker profile with embedding and Discord user mapping."""
    user_id: int
    username: str
    embedding: Optional[np.ndarray] = None
    registered_at: float = field(default_factory=time.time)


@dataclass
class AudioBuffer:
    """Buffer for accumulating audio data per user."""
    user_id: int
    data: bytes = field(default_factory=bytes)
    last_activity: float = field(default_factory=time.time)
    is_speaking: bool = False


class SileroVAD:
    """Voice Activity Detection using Silero VAD."""
    
    def __init__(self):
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.model.eval()
        
    def get_speech_timestamps(self, audio: np.ndarray, threshold: float = 0.5) -> list[dict]:
        """Get speech timestamps from audio array."""
        import torchaudio.transforms as T
        
        # Convert to tensor
        wav = torch.from_numpy(audio).float()
        
        # Get speech probabilities
        speech_probs = []
        window_size_samples = 512
        
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i:i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
            
            with torch.no_grad():
                speech_prob = self.model(chunk, SAMPLE_RATE).item()
                speech_probs.append(speech_prob)
        
        # Find speech regions
        timestamps = []
        start = None
        
        for i, prob in enumerate(speech_probs):
            time_ms = (i * window_size_samples / SAMPLE_RATE) * 1000
            
            if prob >= threshold and start is None:
                start = time_ms
            elif prob < threshold and start is not None:
                timestamps.append({'start': start, 'end': time_ms})
                start = None
        
        if start is not None:
            timestamps.append({'start': start, 'end': time_ms})
            
        return timestamps


class SpeakerRecognizer:
    """Speaker recognition using pyannote embeddings."""
    
    def __init__(self):
        self.embedding_model = None
        self.speaker_profiles: dict[int, SpeakerProfile] = {}  # user_id -> profile
        self.similarity_threshold = 0.7
        
    def load_model(self):
        """Load the speaker embedding model."""
        if self.embedding_model is None and HF_TOKEN:
            self.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/ecapa-tdnn",
                device=torch.device("cpu")
            )
            LOGGER.info("Speaker embedding model loaded")
    
    def extract_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio file."""
        if self.embedding_model is None:
            return None
            
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model(waveform)
                return embedding.numpy()
        except Exception as e:
            LOGGER.error(f"Failed to extract embedding: {e}")
            return None
    
    def register_speaker(self, user_id: int, username: str, audio_path: Path) -> bool:
        """Register a new speaker profile."""
        self.load_model()
        embedding = self.extract_embedding(audio_path)
        
        if embedding is not None:
            self.speaker_profiles[user_id] = SpeakerProfile(
                user_id=user_id,
                username=username,
                embedding=embedding
            )
            LOGGER.info(f"Registered speaker: {username} (ID: {user_id})")
            return True
        return False
    
    def identify_speaker(self, audio_path: Path) -> tuple[Optional[int], float]:
        """Identify speaker from audio. Returns (user_id, confidence)."""
        self.load_model()
        embedding = self.extract_embedding(audio_path)
        
        if embedding is None or not self.speaker_profiles:
            return None, 0.0
        
        best_match = None
        best_score = -1.0
        
        for user_id, profile in self.speaker_profiles.items():
            # Cosine similarity
            similarity = np.dot(embedding, profile.embedding.T) / (
                np.linalg.norm(embedding) * np.linalg.norm(profile.embedding)
            )
            score = float(similarity)
            
            if score > best_score:
                best_score = score
                best_match = user_id
        
        if best_score >= self.similarity_threshold:
            return best_match, best_score
        
        return None, best_score


class OpenClawBridge:
    """Bridge to connect Discord voice chat with OpenClaw Gateway."""
    
    def __init__(self, gateway_url: str, token: str):
        self.gateway_url = gateway_url
        self.token = token
        self.ws = None
        self.connected = False
        self.reconnect_delay = 5
        self.message_queue = asyncio.Queue()
        self.response_callbacks: dict[str, asyncio.Future] = {}
        
    async def connect(self):
        """Connect to OpenClaw Gateway WebSocket."""
        import aiohttp
        
        try:
            self.session = aiohttp.ClientSession()
            self.ws = await self.session.ws_connect(
                self.gateway_url,
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None
            )
            self.connected = True
            LOGGER.info("Connected to OpenClaw Gateway")
            
            # Start listener task
            asyncio.create_task(self._listen())
            asyncio.create_task(self._process_queue())
            
        except Exception as e:
            LOGGER.error(f"Failed to connect to OpenClaw: {e}")
            self.connected = False
            
    async def disconnect(self):
        """Disconnect from OpenClaw Gateway."""
        self.connected = False
        if self.ws:
            await self.ws.close()
        if hasattr(self, 'session'):
            await self.session.close()
        LOGGER.info("Disconnected from OpenClaw Gateway")
        
    async def _listen(self):
        """Listen for messages from OpenClaw."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception as e:
            LOGGER.error(f"WebSocket error: {e}")
        finally:
            self.connected = False
            # Attempt reconnect
            await asyncio.sleep(self.reconnect_delay)
            await self.connect()
            
    async def _handle_message(self, data: dict):
        """Handle incoming message from OpenClaw."""
        msg_type = data.get("type")
        
        if msg_type == "response":
            request_id = data.get("request_id")
            if request_id in self.response_callbacks:
                self.response_callbacks[request_id].set_result(data.get("content", ""))
                del self.response_callbacks[request_id]
                
    async def _process_queue(self):
        """Process outgoing message queue."""
        while self.connected:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                if self.ws and self.connected:
                    await self.ws.send_str(json.dumps(message))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                LOGGER.error(f"Error processing queue: {e}")
                
    async def send_transcription(
        self, 
        text: str, 
        speaker_name: str, 
        user_id: Optional[int] = None,
        channel_id: Optional[int] = None
    ) -> str:
        """Send transcription to OpenClaw and get response."""
        request_id = f"{int(time.time() * 1000)}_{user_id or 'unknown'}"
        
        message = {
            "type": "voice_transcription",
            "request_id": request_id,
            "content": text,
            "speaker": speaker_name,
            "user_id": user_id,
            "channel_id": channel_id,
            "timestamp": time.time()
        }
        
        # Create future for response
        response_future = asyncio.Future()
        self.response_callbacks[request_id] = response_future
        
        await self.message_queue.put(message)
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(response_future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            LOGGER.warning(f"Timeout waiting for OpenClaw response for {request_id}")
            if request_id in self.response_callbacks:
                del self.response_callbacks[request_id]
            return ""


class RealtimeVoiceCog(commands.Cog):
    """Real-time voice transcription and OpenClaw integration cog."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.active_sessions: dict[int, dict] = {}  # guild_id -> session
        self.audio_buffers: dict[int, AudioBuffer] = {}  # user_id -> buffer
        self.speaker_recognizer = SpeakerRecognizer()
        self.vad = SileroVAD()
        self.bridge = OpenClawBridge(OPENCLAW_GATEWAY_URL, OPENCLAW_GATEWAY_TOKEN)
        
        # Processing settings
        self.chunk_duration = CHUNK_DURATION
        self.sample_rate = SAMPLE_RATE
        self.target_rate = TARGET_SAMPLE_RATE
        
        # Start processing loop
        self.processing_task = None
        
    def cog_unload(self):
        """Clean up when cog is unloaded."""
        if self.processing_task:
            self.processing_task.cancel()
            
    @commands.slash_command(
        name="start_voice_chat",
        description="启动实时语音聊天模式（语音转文字并连接 OpenClaw）"
    )
    async def start_voice_chat(
        self,
        ctx: discord.ApplicationContext,
        language: discord.Option(
            str, 
            description="转写语言", 
            choices=["zh", "en"], 
            default="zh"
        ),
    ) -> None:
        """Start real-time voice chat with transcription."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.respond("你需要先进入一个语音频道。", ephemeral=True)
            return
            
        guild = ctx.guild
        if guild is None:
            await ctx.respond("该指令只能在服务器内使用。", ephemeral=True)
            return
            
        if guild.id in self.active_sessions:
            await ctx.respond("当前服务器已经启动了语音聊天模式。", ephemeral=True)
            return
            
        voice_channel = ctx.author.voice.channel
        voice_client = guild.voice_client
        
        if voice_client and voice_client.channel != voice_channel:
            await voice_client.move_to(voice_channel)
        elif not voice_client:
            voice_client = await voice_channel.connect()
            
        # Connect to OpenClaw
        await self.bridge.connect()
        
        # Initialize session
        self.active_sessions[guild.id] = {
            "voice_client": voice_client,
            "text_channel": ctx.channel,
            "language": language,
            "start_time": time.time(),
            "guild_id": guild.id
        }
        
        # Start audio processing loop
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._audio_processing_loop())
        
        # Set up recording with per-user callback
        sink = discord.sinks.WaveSink()
        
        def on_recording_finished(sink, *args):
            asyncio.create_task(self._handle_recording_data(sink, guild.id))
            
        voice_client.start_recording(
            sink, 
            on_recording_finished,
            ctx.channel
        )
        
        await ctx.respond(
            f"🎙️ 已启动实时语音聊天模式！\n"
            f"语音频道: {voice_channel.mention}\n"
            f"转写语言: `{language}`\n"
            f"使用 `/stop_voice_chat` 停止",
            ephemeral=False
        )
        
    @commands.slash_command(
        name="stop_voice_chat",
        description="停止实时语音聊天模式"
    )
    async def stop_voice_chat(self, ctx: discord.ApplicationContext) -> None:
        """Stop real-time voice chat."""
        guild = ctx.guild
        if guild is None:
            await ctx.respond("该指令只能在服务器内使用。", ephemeral=True)
            return
            
        if guild.id not in self.active_sessions:
            await ctx.respond("当前没有进行中的语音聊天。", ephemeral=True)
            return
            
        session = self.active_sessions.pop(guild.id)
        voice_client = session["voice_client"]
        
        if voice_client and voice_client.is_recording():
            voice_client.stop_recording()
            
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect(force=True)
            
        await self.bridge.disconnect()
        
        await ctx.respond("🛑 已停止实时语音聊天模式。", ephemeral=False)
        
    @commands.slash_command(
        name="register_voice",
        description="注册你的声纹，让机器人能识别你的声音"
    )
    async def register_voice(
        self,
        ctx: discord.ApplicationContext,
        seconds: discord.Option(
            int,
            description="录音时长（秒）",
            min_value=5,
            max_value=30,
            default=10
        )
    ) -> None:
        """Register user's voice for speaker identification."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.respond("你需要先进入一个语音频道。", ephemeral=True)
            return
            
        guild = ctx.guild
        if guild is None:
            await ctx.respond("该指令只能在服务器内使用。", ephemeral=True)
            return
            
        await ctx.respond(
            f"🎤 请对着麦克风说话 {seconds} 秒，我会记录你的声纹...",
            ephemeral=True
        )
        
        # Quick voice recording for registration
        voice_channel = ctx.author.voice.channel
        voice_client = guild.voice_client
        
        if not voice_client:
            voice_client = await voice_channel.connect()
            
        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"register_{ctx.author.id}.wav"
        
        try:
            # Record short sample
            sink = discord.sinks.WaveSink()
            
            def on_finish(sink, *args):
                audio_data = getattr(sink, "audio_data", {})
                if ctx.author.id in audio_data:
                    data = audio_data[ctx.author.id]
                    file_obj = getattr(data, "file", None)
                    if file_obj:
                        file_obj.seek(0)
                        temp_wav.write_bytes(file_obj.read())
                        
            voice_client.start_recording(sink, on_finish, ctx.channel)
            await asyncio.sleep(seconds)
            voice_client.stop_recording()
            
            # Wait a bit for processing
            await asyncio.sleep(1)
            
            if temp_wav.exists() and temp_wav.stat().st_size > 0:
                success = self.speaker_recognizer.register_speaker(
                    ctx.author.id,
                    ctx.author.display_name,
                    temp_wav
                )
                
                if success:
                    await ctx.followup.send(
                        f"✅ 声纹注册成功！现在我能识别 `{ctx.author.display_name}` 的声音了。",
                        ephemeral=True
                    )
                else:
                    await ctx.followup.send(
                        "❌ 声纹提取失败，请再试一次（确保环境安静且清晰说话）。",
                        ephemeral=True
                    )
            else:
                await ctx.followup.send(
                    "❌ 没有采集到音频，请检查麦克风权限。",
                    ephemeral=True
                )
                
        except Exception as e:
            LOGGER.exception("Voice registration failed")
            await ctx.followup.send(
                f"❌ 注册失败: {e}",
                ephemeral=True
            )
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    @commands.slash_command(
        name="set_openclaw_channel",
        description="设置 OpenClaw 回复消息的目标文字频道"
    )
    async def set_openclaw_channel(
        self,
        ctx: discord.ApplicationContext,
        channel: discord.Option(
            discord.TextChannel,
            description="目标文字频道",
            required=True
        )
    ) -> None:
        """Set the target channel for OpenClaw responses."""
        guild = ctx.guild
        if guild is None:
            await ctx.respond("该指令只能在服务器内使用。", ephemeral=True)
            return
            
        if guild.id not in self.active_sessions:
            await ctx.respond("请先启动语音聊天模式 (/start_voice_chat)。", ephemeral=True)
            return
            
        self.active_sessions[guild.id]["text_channel"] = channel
        await ctx.respond(
            f"✅ OpenClaw 回复将发送到 {channel.mention}",
            ephemeral=True
        )
        
    async def _handle_recording_data(self, sink: discord.sinks.Sink, guild_id: int):
        """Handle incoming recording data from Discord."""
        audio_data = getattr(sink, "audio_data", {})
        
        for user_id, data in audio_data.items():
            file_obj = getattr(data, "file", None)
            if file_obj is None:
                continue
                
            file_obj.seek(0)
            raw_audio = file_obj.read()
            
            if not raw_audio:
                continue
                
            # Accumulate audio in buffer
            if user_id not in self.audio_buffers:
                self.audio_buffers[user_id] = AudioBuffer(user_id=user_id)
                
            buffer = self.audio_buffers[user_id]
            buffer.data += raw_audio
            buffer.last_activity = time.time()
            
    async def _audio_processing_loop(self):
        """Main audio processing loop."""
        while True:
            try:
                await asyncio.sleep(0.5)  # Process every 500ms
                
                current_time = time.time()
                
                # Process buffers that have enough data
                for user_id, buffer in list(self.audio_buffers.items()):
                    # Check if we have enough audio (2+ seconds)
                    # Discord audio is 48kHz stereo, 16-bit = 192KB per second
                    min_bytes = int(self.chunk_duration * SAMPLE_RATE * 2 * 2)  # duration * rate * channels * bytes
                    
                    if len(buffer.data) >= min_bytes:
                        await self._process_audio_chunk(user_id, buffer)
                        
                    # Clean up old inactive buffers
                    elif current_time - buffer.last_activity > 10:
                        del self.audio_buffers[user_id]
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.exception("Audio processing error")
                
    async def _process_audio_chunk(self, user_id: int, buffer: AudioBuffer):
        """Process a chunk of audio for transcription."""
        # Take chunk from buffer
        chunk_size = int(self.chunk_duration * SAMPLE_RATE * 2 * 2)
        audio_chunk = buffer.data[:chunk_size]
        buffer.data = buffer.data[chunk_size:]  # Keep remaining
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Convert stereo to mono
        audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        # Resample from 48kHz to 16kHz
        audio_array = self._resample(audio_array, SAMPLE_RATE, TARGET_SAMPLE_RATE)
        
        # Normalize to float32 [-1, 1]
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Check for speech using VAD
        speech_timestamps = self.vad.get_speech_timestamps(audio_float, VAD_THRESHOLD)
        
        if not speech_timestamps:
            return  # No speech detected
            
        # Save to temp file for transcription
        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"chunk_{user_id}_{int(time.time())}.wav"
        
        try:
            import soundfile as sf
            sf.write(str(temp_wav), audio_float, TARGET_SAMPLE_RATE)
            
            # Identify speaker
            speaker_id, confidence = self.speaker_recognizer.identify_speaker(temp_wav)
            
            # Get speaker name
            if speaker_id and speaker_id in self.speaker_recognizer.speaker_profiles:
                speaker_name = self.speaker_recognizer.speaker_profiles[speaker_id].username
            else:
                # Try to get from Discord
                speaker_name = f"User_{user_id}"
                for guild_id, session in self.active_sessions.items():
                    guild = self.bot.get_guild(guild_id)
                    if guild:
                        member = guild.get_member(user_id)
                        if member:
                            speaker_name = member.display_name
                            break
            
            # Transcribe
            for guild_id, session in self.active_sessions.items():
                language = session.get("language", "zh")
                
                result = mlx_whisper.transcribe(
                    str(temp_wav),
                    path_or_hf_repo="mlx-community/whisper-small-mlx",
                    language=language,
                    word_timestamps=False
                )
                
                text = result.get("text", "").strip()
                
                if text:
                    LOGGER.info(f"[{speaker_name}] {text}")
                    
                    # Send to Discord text channel
                    text_channel = session.get("text_channel")
                    if text_channel:
                        await text_channel.send(
                            f"🎙️ **[{speaker_name}]**: {text}"
                        )
                    
                    # Send to OpenClaw and get response
                    try:
                        response = await self.bridge.send_transcription(
                            text=text,
                            speaker_name=speaker_name,
                            user_id=user_id,
                            channel_id=text_channel.id if text_channel else None
                        )
                        
                        if response and text_channel:
                            await text_channel.send(f"🤖 **OpenClaw**: {response}")
                            
                    except Exception as e:
                        LOGGER.error(f"Failed to send to OpenClaw: {e}")
                        
                break  # Only process for first active session
                
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
            
        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        
        indices = np.linspace(0, len(audio) - 1, new_length)
        indices_floor = np.floor(indices).astype(np.int64)
        indices_ceil = np.minimum(indices_floor + 1, len(audio) - 1)
        fractions = indices - indices_floor
        
        return audio[indices_floor] * (1 - fractions) + audio[indices_ceil] * fractions


def setup(bot: commands.Bot) -> None:
    """Add the cog to the bot."""
    bot.add_cog(RealtimeVoiceCog(bot))
