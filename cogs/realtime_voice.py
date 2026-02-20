import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import discord
import mlx_whisper
import numpy as np
import torch
import torchaudio.transforms as T
from discord.ext import commands
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

LOGGER = logging.getLogger(__name__)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

# Audio settings
SAMPLE_RATE = 48000       # Discord voice sample rate
TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 2.0      # Process audio in 2-second chunks
VAD_THRESHOLD = 0.5       # Voice activity detection threshold
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to process

# Persistence paths
PROFILES_DIR = Path("data/speaker_profiles")
PROFILES_JSON = PROFILES_DIR / "profiles.json"
EMBEDDINGS_DIR = PROFILES_DIR / "embeddings"


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


class RealTimeAudioSink(discord.sinks.Sink):
    """Custom sink that pushes raw PCM data directly into memory buffers."""

    def __init__(self, audio_buffers: dict):
        super().__init__()
        self.audio_buffers = audio_buffers

    def write(self, data: bytes, user_id: int) -> None:
        if user_id not in self.audio_buffers:
            self.audio_buffers[user_id] = AudioBuffer(user_id=user_id)
        buf = self.audio_buffers[user_id]
        buf.data += data
        buf.last_activity = time.time()

    def cleanup(self) -> None:
        pass  # Nothing to clean up — no files written


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

    def has_speech(self, audio: np.ndarray, threshold: float = VAD_THRESHOLD) -> bool:
        """Return True if the audio contains speech above threshold."""
        wav = torch.from_numpy(audio).float()
        window_size_samples = 512

        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i:i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
            with torch.no_grad():
                prob = self.model(chunk, TARGET_SAMPLE_RATE).item()
                if prob >= threshold:
                    return True
        return False


class SpeakerRecognizer:
    """Speaker recognition using pyannote/speechbrain embeddings with disk persistence."""

    def __init__(self):
        self.embedding_model = None
        self.speaker_profiles: dict[int, SpeakerProfile] = {}
        self.similarity_threshold = 0.7
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load speaker profiles from disk on startup."""
        if not PROFILES_JSON.exists():
            return
        try:
            data = json.loads(PROFILES_JSON.read_text())
            for uid_str, meta in data.items():
                user_id = int(uid_str)
                emb_path = PROFILES_DIR / meta["embedding_file"]
                embedding = np.load(str(emb_path)) if emb_path.exists() else None
                self.speaker_profiles[user_id] = SpeakerProfile(
                    user_id=user_id,
                    username=meta["username"],
                    embedding=embedding,
                    registered_at=meta.get("registered_at", time.time()),
                )
            LOGGER.info("Loaded %d speaker profile(s) from disk", len(self.speaker_profiles))
        except Exception:
            LOGGER.exception("Failed to load speaker profiles")

    def _save_profiles(self) -> None:
        """Persist speaker profiles to disk."""
        try:
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            index: dict[str, dict] = {}
            for user_id, profile in self.speaker_profiles.items():
                emb_filename = f"embeddings/{user_id}.npy"
                if profile.embedding is not None:
                    np.save(str(PROFILES_DIR / emb_filename), profile.embedding)
                index[str(user_id)] = {
                    "user_id": user_id,
                    "username": profile.username,
                    "registered_at": profile.registered_at,
                    "embedding_file": emb_filename,
                }
            PROFILES_JSON.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        except Exception:
            LOGGER.exception("Failed to save speaker profiles")

    def load_model(self) -> None:
        """Lazy-load the speaker embedding model."""
        if self.embedding_model is None and HF_TOKEN:
            self.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/ecapa-tdnn",
                device=torch.device("cpu")
            )
            LOGGER.info("Speaker embedding model loaded")

    def extract_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        """Extract speaker embedding from an audio file."""
        if self.embedding_model is None:
            return None
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            with torch.no_grad():
                embedding = self.embedding_model(waveform)
                return embedding.numpy()
        except Exception:
            LOGGER.exception("Failed to extract embedding")
            return None

    def register_speaker(self, user_id: int, username: str, audio_path: Path) -> bool:
        """Register a new speaker profile and persist it."""
        self.load_model()
        embedding = self.extract_embedding(audio_path)
        if embedding is not None:
            self.speaker_profiles[user_id] = SpeakerProfile(
                user_id=user_id,
                username=username,
                embedding=embedding,
            )
            self._save_profiles()
            LOGGER.info("Registered speaker: %s (ID: %s)", username, user_id)
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
            if profile.embedding is None:
                continue
            similarity = np.dot(embedding.flatten(), profile.embedding.flatten()) / (
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
        self.session = None
        self.connected = False
        self.reconnect_delay = 5
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_callbacks: dict[str, asyncio.Future] = {}

    async def connect(self) -> None:
        import aiohttp
        try:
            self.session = aiohttp.ClientSession()
            self.ws = await self.session.ws_connect(
                self.gateway_url,
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None
            )
            self.connected = True
            LOGGER.info("Connected to OpenClaw Gateway")
            asyncio.create_task(self._listen())
            asyncio.create_task(self._process_queue())
        except Exception:
            LOGGER.exception("Failed to connect to OpenClaw")
            self.connected = False

    async def disconnect(self) -> None:
        self.connected = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        LOGGER.info("Disconnected from OpenClaw Gateway")

    async def _listen(self) -> None:
        import aiohttp
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    import json as _json
                    data = _json.loads(msg.data)
                    await self._handle_message(data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception:
            LOGGER.exception("WebSocket error")
        finally:
            self.connected = False
            await asyncio.sleep(self.reconnect_delay)
            if self.gateway_url:
                await self.connect()

    async def _handle_message(self, data: dict) -> None:
        if data.get("type") == "response":
            request_id = data.get("request_id")
            if request_id in self.response_callbacks:
                self.response_callbacks[request_id].set_result(data.get("content", ""))
                del self.response_callbacks[request_id]

    async def _process_queue(self) -> None:
        import json as _json
        while self.connected:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                if self.ws and self.connected:
                    await self.ws.send_str(_json.dumps(message))
            except asyncio.TimeoutError:
                continue
            except Exception:
                LOGGER.exception("Error processing OpenClaw queue")

    async def send_transcription(
        self,
        text: str,
        speaker_name: str,
        user_id: Optional[int] = None,
        channel_id: Optional[int] = None,
    ) -> str:
        request_id = f"{int(time.time() * 1000)}_{user_id or 'unknown'}"
        message = {
            "type": "voice_transcription",
            "request_id": request_id,
            "content": text,
            "speaker": speaker_name,
            "user_id": user_id,
            "channel_id": channel_id,
            "timestamp": time.time(),
        }
        response_future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.response_callbacks[request_id] = response_future
        await self.message_queue.put(message)
        try:
            return await asyncio.wait_for(response_future, timeout=30.0)
        except asyncio.TimeoutError:
            LOGGER.warning("Timeout waiting for OpenClaw response: %s", request_id)
            self.response_callbacks.pop(request_id, None)
            return ""


class RealtimeVoiceCog(commands.Cog):
    """Real-time voice transcription cog (Phase 1)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.active_sessions: dict[int, dict] = {}  # guild_id -> session
        self.audio_buffers: dict[int, AudioBuffer] = {}  # user_id -> buffer
        self.speaker_recognizer = SpeakerRecognizer()
        self.vad = SileroVAD()
        self.bridge = OpenClawBridge(OPENCLAW_GATEWAY_URL, OPENCLAW_GATEWAY_TOKEN)
        self.chunk_duration = CHUNK_DURATION
        self.processing_task: Optional[asyncio.Task] = None

    def cog_unload(self) -> None:
        if self.processing_task:
            self.processing_task.cancel()

    # ------------------------------------------------------------------ #
    #  Slash commands                                                      #
    # ------------------------------------------------------------------ #

    @commands.slash_command(
        name="start_voice_chat",
        description="启动实时语音转录（语音 → Discord 文字频道）"
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

        # Only connect to OpenClaw when fully configured
        if OPENCLAW_GATEWAY_URL and OPENCLAW_GATEWAY_TOKEN:
            await self.bridge.connect()

        self.active_sessions[guild.id] = {
            "voice_client": voice_client,
            "text_channel": ctx.channel,
            "language": language,
            "start_time": time.time(),
            "guild_id": guild.id,
            "transcript": [],  # For post-meeting summary
        }

        # Start the background processing loop (one loop for all guilds)
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._audio_processing_loop())

        # Start recording with our custom real-time sink
        sink = RealTimeAudioSink(self.audio_buffers)
        voice_client.start_recording(sink, self._on_recording_finished, ctx.channel)

        await ctx.respond(
            f"🎙️ 已启动实时语音转录！\n"
            f"语音频道: {voice_channel.mention}\n"
            f"转写语言: `{language}`\n"
            f"使用 `/stop_voice_chat` 停止",
            ephemeral=False
        )

    @commands.slash_command(
        name="stop_voice_chat",
        description="停止实时语音转录，输出会议纪要"
    )
    async def stop_voice_chat(self, ctx: discord.ApplicationContext) -> None:
        guild = ctx.guild
        if guild is None:
            await ctx.respond("该指令只能在服务器内使用。", ephemeral=True)
            return

        if guild.id not in self.active_sessions:
            await ctx.respond("当前没有进行中的语音转录。", ephemeral=True)
            return

        session = self.active_sessions.pop(guild.id)
        voice_client = session["voice_client"]

        if voice_client and voice_client.is_recording():
            voice_client.stop_recording()
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect(force=True)

        if self.bridge.connected:
            await self.bridge.disconnect()

        # Post meeting summary
        transcript = session.get("transcript", [])
        text_channel = session.get("text_channel")
        if transcript and text_channel:
            participants = list(dict.fromkeys(entry["speaker"] for entry in transcript))
            duration_secs = int(time.time() - session["start_time"])
            lines = "\n".join(f"[{e['speaker']}]: {e['text']}" for e in transcript)
            summary = (
                f"--- 会议纪要 ---\n"
                f"时长: {duration_secs // 60} 分钟 {duration_secs % 60} 秒\n"
                f"参与者: {', '.join(participants)}\n\n"
                f"{lines}"
            )
            # Discord message limit: 2000 chars
            for i in range(0, len(summary), 1900):
                await text_channel.send(summary[i:i + 1900])

        await ctx.respond("🛑 已停止实时语音转录。", ephemeral=False)

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

        voice_channel = ctx.author.voice.channel
        voice_client = guild.voice_client
        connected_here = False

        if not voice_client:
            voice_client = await voice_channel.connect()
            connected_here = True

        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"register_{ctx.author.id}.wav"

        try:
            # Use WaveSink for registration (one-shot recording is fine here)
            reg_sink = discord.sinks.WaveSink()

            def on_finish(sink, *args):
                audio_data = getattr(sink, "audio_data", {})
                if ctx.author.id in audio_data:
                    data = audio_data[ctx.author.id]
                    file_obj = getattr(data, "file", None)
                    if file_obj:
                        file_obj.seek(0)
                        temp_wav.write_bytes(file_obj.read())

            voice_client.start_recording(reg_sink, on_finish, ctx.channel)
            await asyncio.sleep(seconds)
            voice_client.stop_recording()
            await asyncio.sleep(1)  # Give on_finish time to run

            if temp_wav.exists() and temp_wav.stat().st_size > 0:
                success = await asyncio.to_thread(
                    self.speaker_recognizer.register_speaker,
                    ctx.author.id,
                    ctx.author.display_name,
                    temp_wav,
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
        except Exception:
            LOGGER.exception("Voice registration failed")
            await ctx.followup.send("❌ 注册失败，请查看日志。", ephemeral=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            # Disconnect if we connected solely for registration
            if connected_here and voice_client.is_connected():
                await voice_client.disconnect(force=True)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _on_recording_finished(self, sink: discord.sinks.Sink, *args) -> None:
        """Callback when stop_recording() is called — not used for real-time flow."""
        pass  # RealTimeAudioSink already wrote data to buffers continuously

    async def _audio_processing_loop(self) -> None:
        """Background loop: drain audio buffers every 500ms and transcribe."""
        while True:
            try:
                await asyncio.sleep(0.5)
                current_time = time.time()
                # min bytes for one chunk: duration * 48kHz * 2ch * 2 bytes
                min_bytes = int(CHUNK_DURATION * SAMPLE_RATE * 2 * 2)

                for user_id, buffer in list(self.audio_buffers.items()):
                    if len(buffer.data) >= min_bytes:
                        await self._process_audio_chunk(user_id, buffer)
                    elif current_time - buffer.last_activity > 10:
                        del self.audio_buffers[user_id]

            except asyncio.CancelledError:
                break
            except Exception:
                LOGGER.exception("Audio processing loop error")

    async def _process_audio_chunk(self, user_id: int, buffer: AudioBuffer) -> None:
        """Convert one chunk of raw PCM → text and post to Discord."""
        chunk_size = int(CHUNK_DURATION * SAMPLE_RATE * 2 * 2)
        audio_chunk = buffer.data[:chunk_size]
        buffer.data = buffer.data[chunk_size:]

        # PCM int16 → numpy
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

        # Stereo → mono
        audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # 48kHz → 16kHz (high-quality resample via torchaudio)
        audio_float = self._resample_tensor(audio_array, SAMPLE_RATE, TARGET_SAMPLE_RATE)

        # VAD: skip silent chunks
        if not self.vad.has_speech(audio_float):
            return

        # Save temp WAV for mlx-whisper and speaker ID
        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"chunk_{user_id}_{int(time.time())}.wav"

        try:
            import soundfile as sf
            sf.write(str(temp_wav), audio_float, TARGET_SAMPLE_RATE)

            # Resolve speaker name
            speaker_name = await self._resolve_speaker(user_id, temp_wav)

            # Transcribe (blocking → thread)
            for guild_id, session in self.active_sessions.items():
                language = session.get("language", "zh")

                result = await asyncio.to_thread(
                    mlx_whisper.transcribe,
                    str(temp_wav),
                    path_or_hf_repo="mlx-community/whisper-small-mlx",
                    language=language,
                    word_timestamps=False,
                )

                text: str = result.get("text", "").strip()
                if not text:
                    break

                LOGGER.info("[%s] %s", speaker_name, text)

                # Post to Discord
                text_channel = session.get("text_channel")
                if text_channel:
                    await text_channel.send(f"🎙️ **[{speaker_name}]**: {text}")

                # Append to session transcript
                session["transcript"].append({"speaker": speaker_name, "text": text})

                # Forward to OpenClaw (optional)
                if self.bridge.connected and text_channel:
                    try:
                        response = await self.bridge.send_transcription(
                            text=text,
                            speaker_name=speaker_name,
                            user_id=user_id,
                            channel_id=text_channel.id,
                        )
                        if response:
                            await text_channel.send(f"🤖 **OpenClaw**: {response}")
                    except Exception:
                        LOGGER.exception("Failed to send to OpenClaw")

                break  # Only process for first active session found
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _resolve_speaker(self, user_id: int, audio_path: Path) -> str:
        """Return display name for user_id, using voice embedding if available."""
        # Try speaker embedding recognition (blocking → thread)
        speaker_id, _ = await asyncio.to_thread(
            self.speaker_recognizer.identify_speaker, audio_path
        )
        if speaker_id and speaker_id in self.speaker_recognizer.speaker_profiles:
            return self.speaker_recognizer.speaker_profiles[speaker_id].username

        # Fall back to Discord display name
        for guild_id in self.active_sessions:
            guild = self.bot.get_guild(guild_id)
            if guild:
                member = guild.get_member(user_id)
                if member:
                    return member.display_name

        return f"User_{user_id}"

    def _resample_tensor(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High-quality resampling via torchaudio.transforms.Resample."""
        if orig_sr == target_sr:
            return audio.astype(np.float32) / 32768.0
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(wav).squeeze(0)
        # Normalize int16 range to float [-1, 1]
        return (resampled / 32768.0).numpy()


def setup(bot: commands.Bot) -> None:
    """Add the cog to the bot."""
    bot.add_cog(RealtimeVoiceCog(bot))
