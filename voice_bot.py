import asyncio
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import discord
import mlx_whisper
import numpy as np
import requests
from discord.ext import commands

LOGGER = logging.getLogger(__name__)

DISCORD_TOKEN = os.getenv("VOICE_DISCORD_TOKEN", "")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

SAMPLE_RATE = 48000
TARGET_RATE = 16000
CHUNK_DURATION = 3.0  # seconds per transcription chunk


@dataclass
class AudioBuffer:
    user_id: int
    username: str
    data: bytes = field(default_factory=bytes)
    last_activity: float = field(default_factory=time.time)


class SimpleVAD:
    def has_speech(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        audio = audio.astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > threshold




class VoiceBot(commands.Bot):

    def __init__(self):
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.messages = True
        super().__init__(intents=intents)

        self.active_sessions: dict[int, dict] = {}
        self.vad = SimpleVAD()
        self.processing_task = None

    async def on_ready(self):
        LOGGER.info(f"Voice Bot logged in as {self.user}")

        if self.processing_task is None:
            await self.sync_commands()
            LOGGER.info("Slash commands synced.")
            self.processing_task = asyncio.create_task(self._audio_processing_loop())

    async def _audio_processing_loop(self):
        while True:
            try:
                await asyncio.sleep(1.0)
                min_bytes = int(CHUNK_DURATION * SAMPLE_RATE * 2 * 2)  # stereo int16

                for session in list(self.active_sessions.values()):
                    sink = session.get("sink")
                    guild = session.get("guild")
                    if not sink or not guild:
                        continue

                    for user_id, audio_data in list(sink.audio_data.items()):
                        audio_data.file.seek(0)
                        raw = audio_data.file.read()
                        audio_data.file.seek(0, 2)  # seek to end for continued writing

                        if len(raw) < min_bytes:
                            continue

                        username = f"User_{user_id}"
                        member = guild.get_member(user_id)
                        if member:
                            username = member.display_name

                        # consume min_bytes, leave the rest
                        chunk = raw[:min_bytes]
                        remaining = raw[min_bytes:]
                        audio_data.file.seek(0)
                        audio_data.file.truncate()
                        audio_data.file.write(remaining)
                        audio_data.file.seek(0, 2)

                        buf = AudioBuffer(user_id=user_id, username=username, data=chunk)
                        await self._process_chunk(user_id, buf)

            except asyncio.CancelledError:
                break
            except Exception:
                LOGGER.exception("Audio processing loop error")

    async def _process_chunk(self, user_id: int, buf: AudioBuffer):
        chunk_size = int(CHUNK_DURATION * SAMPLE_RATE * 2 * 2)
        audio_chunk = buf.data[:chunk_size]
        buf.data = buf.data[chunk_size:]

        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

        # stereo → mono
        if audio_array.size % 2 == 0:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # 48kHz → 16kHz
        audio_array = self._resample(audio_array, SAMPLE_RATE, TARGET_RATE)

        if not self.vad.has_speech(audio_array):
            return

        # find language from any active session
        language = "zh"
        for session in self.active_sessions.values():
            language = session.get("language", "zh")
            break

        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"chunk_{user_id}.wav"
        try:
            import soundfile as sf
            audio_float = audio_array.astype(np.float32) / 32768.0
            sf.write(str(temp_wav), audio_float, TARGET_RATE)

            result = await asyncio.to_thread(
                mlx_whisper.transcribe,
                str(temp_wav),
                path_or_hf_repo="mlx-community/whisper-small-mlx",
                language=language,
                word_timestamps=False,
            )

            text = result.get("text", "").strip()
            if text and len(text) > 2:
                LOGGER.info(f"[{buf.username}] {text}")
                for session in self.active_sessions.values():
                    ch = session.get("text_channel")
                    if ch:
                        await ch.send(f"🎙️ **{buf.username}**: {text}")
                await self._send_to_openclaw(text, buf.username, user_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _send_to_openclaw(self, text: str, username: str, user_id: int):
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
                "channel": "discord",
            }
            await asyncio.to_thread(
                lambda: requests.post(
                    f"{OPENCLAW_GATEWAY_URL}/api/v1/events",
                    json=payload,
                    headers=headers,
                    timeout=10,
                )
            )
        except Exception:
            LOGGER.debug("OpenClaw send failed (not critical)")

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        new_length = int(len(audio) * target_sr / orig_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        idx_floor = np.floor(indices).astype(np.int64)
        idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
        frac = indices - idx_floor
        return (audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac).astype(np.int16)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not discord.opus.is_loaded():
        discord.opus.load_opus("/opt/homebrew/lib/libopus.dylib")

    if not DISCORD_TOKEN:
        LOGGER.error("Missing VOICE_DISCORD_TOKEN environment variable!")
        return

    bot = VoiceBot()

    @bot.slash_command(name="voice_on", description="开启实时语音转文字")
    async def voice_on(ctx: discord.ApplicationContext, language: str = "zh"):
        member = ctx.guild.get_member(ctx.author.id) if ctx.guild else None
        if not member or not member.voice or not member.voice.channel:
            await ctx.respond("❌ 你需要先进入语音频道！", ephemeral=True)
            return

        guild_id = ctx.guild_id
        if guild_id in bot.active_sessions:
            await ctx.respond("⚠️ 已经在转录中了！", ephemeral=True)
            return

        await ctx.defer()

        voice_channel = member.voice.channel
        voice_client = await voice_channel.connect()

        sink = discord.sinks.WaveSink()

        def on_stop(sink, *args):
            pass  # no-op: audio consumed by processing loop

        voice_client.start_recording(sink, on_stop)

        bot.active_sessions[guild_id] = {
            "voice_client": voice_client,
            "sink": sink,
            "guild": ctx.guild,
            "text_channel": ctx.channel,
            "language": language,
        }

        await ctx.followup.send(
            f"🎙️ **实时语音转录已开启！**\n"
            f"频道: {voice_channel.mention}\n"
            f"语言: `{language}`\n"
            f"使用 `/voice_off` 停止"
        )

    @bot.slash_command(name="voice_off", description="关闭实时语音转文字")
    async def voice_off(ctx: discord.ApplicationContext):
        guild_id = ctx.guild_id

        if guild_id not in bot.active_sessions:
            await ctx.respond("⚠️ 没有在运行的转录。", ephemeral=True)
            return

        await ctx.defer()

        session = bot.active_sessions.pop(guild_id)
        voice_client = session["voice_client"]

        try:
            voice_client.stop_recording()
        except Exception:
            pass

        if voice_client.is_connected():
            await voice_client.disconnect(force=True)

        await ctx.followup.send("🛑 语音转录已停止。")

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
