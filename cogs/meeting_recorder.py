import asyncio
import datetime
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import discord
import mlx_whisper
import torch
from discord.ext import commands
from pyannote.audio import Pipeline


# Place your HuggingFace token in env `HF_TOKEN`.
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HUGGINGFACE_TOKEN")

LOGGER = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str


@dataclass
class MeetingSession:
    temp_dir: Path
    language: str


class MeetingRecorder(commands.Cog):
    def __init__(self, bot: discord.Bot):
        self.bot = bot
        self._sessions: dict[int, MeetingSession] = {}

    @commands.slash_command(name="join_meeting", description="加入语音频道并开始会议录音")
    async def join_meeting(
        self,
        ctx: discord.ApplicationContext,
        language: discord.Option(str, description="转写语言", choices=["zh", "en"], default="zh"),
    ) -> None:
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.respond("你需要先进入一个语音频道。", ephemeral=True)
            return

        guild = ctx.guild
        if guild is None:
            await ctx.respond("该指令只能在服务器内使用。", ephemeral=True)
            return

        if guild.id in self._sessions:
            await ctx.respond("当前服务器已经在录音中，请先执行 /leave_meeting。", ephemeral=True)
            return

        voice_channel = ctx.author.voice.channel
        voice_client = guild.voice_client

        if voice_client and voice_client.channel != voice_channel:
            await voice_client.move_to(voice_channel)
        elif not voice_client:
            voice_client = await voice_channel.connect()

        temp_dir = Path(tempfile.mkdtemp(prefix=f"meeting_{guild.id}_"))
        self._sessions[guild.id] = MeetingSession(temp_dir=temp_dir, language=language)

        sink = discord.sinks.WaveSink()
        try:
            voice_client.start_recording(sink, self._on_recording_stopped, ctx.channel)
        except Exception:
            self._sessions.pop(guild.id, None)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        await ctx.respond(f"已开始录音：{voice_channel.mention}，转写语言 `{language}`。结束时执行 /leave_meeting。")

    @commands.slash_command(name="leave_meeting", description="停止录音并生成会议转写")
    async def leave_meeting(self, ctx: discord.ApplicationContext) -> None:
        guild = ctx.guild
        if guild is None:
            await ctx.respond("该指令只能在服务器内使用。", ephemeral=True)
            return

        voice_client = guild.voice_client
        if guild.id not in self._sessions or not voice_client:
            await ctx.respond("当前没有进行中的会议录音。", ephemeral=True)
            return

        voice_client.stop_recording()
        await ctx.respond("已停止录音，正在后台执行声纹识别与转录，请稍候。")

    def _on_recording_stopped(self, sink: discord.sinks.Sink, text_channel: discord.abc.Messageable, *_) -> None:
        self.bot.loop.create_task(self._handle_recording_stopped(sink, text_channel))

    async def _handle_recording_stopped(
        self,
        sink: discord.sinks.Sink,
        text_channel: discord.abc.Messageable,
    ) -> None:
        guild = sink.vc.guild if sink.vc else None
        if guild is None:
            return

        session = self._sessions.pop(guild.id, None)
        if session is None:
            return

        meeting_wav = session.temp_dir / "meeting.wav"

        try:
            audio_files = self._extract_audio_files(sink, session.temp_dir)
            if not audio_files:
                await text_channel.send("未采集到有效音频数据，请检查机器人语音权限与频道输入源。")
                return

            # 单麦克风场景：取最长轨道作为会议主录音。
            primary = max(audio_files, key=lambda p: p.stat().st_size)
            shutil.copy2(primary, meeting_wav)

            transcript = await asyncio.to_thread(self._run_pipeline, meeting_wav, session.language)
            if not transcript.strip():
                transcript = "未识别到有效语音内容。"

            await self._send_transcript(text_channel, transcript)
        except Exception as exc:
            LOGGER.exception("Meeting pipeline failed")
            await text_channel.send(f"会议处理失败: {exc}")
        finally:
            if sink.vc and sink.vc.is_connected():
                await sink.vc.disconnect(force=True)
            shutil.rmtree(session.temp_dir, ignore_errors=True)

    def _extract_audio_files(self, sink: discord.sinks.Sink, out_dir: Path) -> list[Path]:
        files: list[Path] = []
        audio_data = getattr(sink, "audio_data", {})

        for user_id, data in audio_data.items():
            file_obj = getattr(data, "file", None)
            if file_obj is None:
                continue

            try:
                file_obj.seek(0)
                raw = file_obj.read()
            except Exception:
                continue

            if not raw:
                continue

            file_path = out_dir / f"user_{user_id}.wav"
            file_path.write_bytes(raw)
            files.append(file_path)

        return files

    def _run_pipeline(self, meeting_wav: Path, language: str) -> str:
        diarization_segments = self._run_diarization(meeting_wav)
        whisper_segments = self._run_transcription(meeting_wav, language)
        merged = self._merge_segments(whisper_segments, diarization_segments)
        return "\n".join(merged)

    def _run_diarization(self, meeting_wav: Path) -> list[DiarizationSegment]:
        if HF_TOKEN == "YOUR_HUGGINGFACE_TOKEN":
            raise RuntimeError("请先设置 HF_TOKEN 环境变量后再执行声纹识别。")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        pipeline.to(torch.device("cpu"))

        diarization = pipeline(str(meeting_wav))

        segments: list[DiarizationSegment] = []
        for turn, _track, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))

        segments.sort(key=lambda s: s.start)
        return segments

    def _run_transcription(self, meeting_wav: Path, language: str) -> list[WhisperSegment]:
        result: dict[str, Any] = mlx_whisper.transcribe(
            audio=str(meeting_wav),
            path_or_hf_repo="mlx-community/whisper-small-mlx",
            language=language,
            word_timestamps=True,
        )

        segments: list[WhisperSegment] = []
        for seg in result.get("segments", []):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if end < start:
                end = start

            segments.append(WhisperSegment(start=start, end=end, text=text))

        return segments

    def _merge_segments(
        self,
        whisper_segments: list[WhisperSegment],
        diarization_segments: list[DiarizationSegment],
    ) -> list[str]:
        speaker_alias: dict[str, str] = {}

        def alias(raw_speaker: str) -> str:
            if raw_speaker not in speaker_alias:
                speaker_alias[raw_speaker] = f"Speaker {len(speaker_alias) + 1}"
            return speaker_alias[raw_speaker]

        lines: list[str] = []
        for ws in whisper_segments:
            speaker = self._find_speaker_for_segment(ws, diarization_segments)
            speaker_name = alias(speaker) if speaker != "Unknown" else "Unknown"
            lines.append(
                f"[{speaker_name}] ({self._fmt_time(ws.start)}-{self._fmt_time(ws.end)}): {ws.text}"
            )

        return lines

    def _find_speaker_for_segment(
        self,
        whisper_segment: WhisperSegment,
        diarization_segments: list[DiarizationSegment],
    ) -> str:
        best_speaker = "Unknown"
        best_overlap = 0.0

        for ds in diarization_segments:
            overlap_start = max(whisper_segment.start, ds.start)
            overlap_end = min(whisper_segment.end, ds.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds.speaker

        if best_overlap > 0:
            return best_speaker

        midpoint = (whisper_segment.start + whisper_segment.end) / 2
        nearest = None
        nearest_distance = float("inf")
        for ds in diarization_segments:
            center = (ds.start + ds.end) / 2
            distance = abs(midpoint - center)
            if distance < nearest_distance:
                nearest = ds
                nearest_distance = distance

        return nearest.speaker if nearest else "Unknown"

    def _fmt_time(self, seconds: float) -> str:
        total = int(max(0, round(seconds)))
        mm, ss = divmod(total, 60)
        hh, mm = divmod(mm, 60)
        if hh:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        return f"{mm:02d}:{ss:02d}"

    def _save_transcript(self, transcript: str) -> Path:
        """Save transcript to data/transcripts/<datetime>.txt and return the path."""
        out_dir = Path("data/transcripts")
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        out_path = out_dir / filename
        out_path.write_text(transcript, encoding="utf-8")
        LOGGER.info("Transcript saved to %s", out_path)
        return out_path

    async def _send_transcript(self, text_channel: discord.abc.Messageable, transcript: str) -> None:
        out_path = self._save_transcript(transcript)
        header = f"会议纪要如下：（已保存至 `{out_path}`）"
        max_text = 1800
        if len(transcript) <= max_text:
            await text_channel.send(f"{header}\n```\n{transcript}\n```")
            return

        await text_channel.send(header)
        start = 0
        while start < len(transcript):
            chunk = transcript[start : start + max_text]
            await text_channel.send(f"```\n{chunk}\n```")
            start += max_text


def setup(bot: discord.Bot) -> None:
    bot.add_cog(MeetingRecorder(bot))
