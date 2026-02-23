import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import wave

from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import discord
import mlx_whisper
import numpy as np
import requests
import torch
import torchaudio.transforms as T
from discord.ext import commands
from speechbrain.inference.classifiers import EncoderClassifier

LOGGER = logging.getLogger(__name__)

DISCORD_TOKEN = os.getenv("VOICE_DISCORD_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

SAMPLE_RATE = 48000
TARGET_RATE = 16000
VAD_THRESHOLD = 0.5
VAD_FRAME_DURATION = 0.2   # seconds per VAD check window
SILENCE_TIMEOUT = 0.8      # seconds of silence before flushing a sentence
MAX_SPEECH_DURATION = 15.0  # seconds max before forced flush

PROFILES_DIR = Path("data/speaker_profiles")
PROFILES_JSON = PROFILES_DIR / "profiles.json"
EMBEDDINGS_DIR = PROFILES_DIR / "embeddings"

# Whisper initial_prompt：引导输出简体中文，避免输出繁体
WHISPER_INITIAL_PROMPT = {
    "zh": "以下是普通话的转录，使用简体中文。",
    "en": "",
}

MIN_SPEECH_DURATION = 1.0  # 低于此秒数的片段不送 Whisper


def _is_hallucination(text: str) -> bool:
    """检测 Whisper 幻觉：单个字符重复超过 10 次，或最长重复子串占比超过 50%。"""
    if not text:
        return False
    # 单字符重复检测
    for ch in set(text):
        if text.count(ch) > max(10, len(text) * 0.5):
            return True
    # 短语重复检测：取前 10 个字符作为模式，看重复次数
    if len(text) >= 20:
        pattern = text[:5]
        if text.count(pattern) > len(text) / (len(pattern) * 2):
            return True
    return False


@dataclass
class SpeakerProfile:
    user_id: int
    username: str
    embedding: Optional[np.ndarray] = None
    registered_at: float = field(default_factory=time.time)


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
        wav = torch.from_numpy(audio).float()
        window_size_samples = 512
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i:i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
            with torch.no_grad():
                prob = self.model(chunk, TARGET_RATE).item()
                if prob >= threshold:
                    return True
        return False


class SpeakerRecognizer:
    """Speaker recognition using speechbrain/ecapa-tdnn embeddings with disk persistence."""

    def __init__(self):
        self.embedding_model = None
        self.speaker_profiles: dict[int, SpeakerProfile] = {}
        self.similarity_threshold = 0.55
        self._load_profiles()

    def _load_profiles(self) -> None:
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
            for uid, p in self.speaker_profiles.items():
                emb_shape = p.embedding.shape if p.embedding is not None else None
                LOGGER.info("  Profile: uid=%s name=%s embedding=%s", uid, p.username, emb_shape)
            LOGGER.info("Loaded %d speaker profile(s) from disk", len(self.speaker_profiles))
        except Exception:
            LOGGER.exception("Failed to load speaker profiles")

    def _save_profiles(self) -> None:
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
        if self.embedding_model is None:
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
            LOGGER.info("Speaker embedding model loaded")

    def extract_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        if self.embedding_model is None:
            return None
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            if waveform.numel() == 0:
                LOGGER.warning("Empty waveform loaded from %s", audio_path)
                return None
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(waveform)
                return embedding.squeeze().cpu().numpy()
        except Exception:
            LOGGER.exception("Failed to extract embedding")
            return None

    def register_speaker(self, user_id: int, username: str, audio_path: Path) -> bool:
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


class VoiceBot(commands.Bot):

    def __init__(self):
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.messages = True
        super().__init__(intents=intents)

        self.active_sessions: dict[int, dict] = {}
        self.vad = SileroVAD()
        self.speaker_recognizer = SpeakerRecognizer()
        self.processing_task = None
        self._diarization_pipeline = None  # 懒加载，避免每次重新下载
        # Per-user speech state: {user_id: {"buffer": bytes, "last_speech": float, "speech_start": float}}
        self.speech_state: dict[int, dict] = {}
        # Per-user read position into sink BytesIO (avoids seek/truncate race with sink writer)
        self.sink_read_pos: dict[int, int] = {}
        # Per-user full PCM accumulator for post-meeting transcription: {user_id: bytes}
        self.full_audio: dict[int, bytes] = {}
        # Per-user last successful speaker identification cache: {user_id: speaker_name}
        self.speaker_cache: dict[int, str] = {}

    async def on_ready(self):
        LOGGER.info(f"Voice Bot logged in as {self.user}")

        if self.processing_task is None:
            await self.sync_commands()
            LOGGER.info("Slash commands synced.")
            self.processing_task = asyncio.create_task(self._audio_processing_loop())

    async def _audio_processing_loop(self):
        # VAD frame size in raw stereo int16 bytes
        frame_bytes = int(VAD_FRAME_DURATION * SAMPLE_RATE * 2 * 2)

        while True:
            try:
                await asyncio.sleep(VAD_FRAME_DURATION)
                now = time.time()

                for session in list(self.active_sessions.values()):
                    sink = session.get("sink")
                    guild = session.get("guild")
                    if not sink or not guild:
                        continue

                    for user_id, audio_data in list(sink.audio_data.items()):
                        # ── 修复竞争：用读指针只取新增数据，不 truncate ──
                        pos = self.sink_read_pos.get(user_id, 0)
                        audio_data.file.seek(pos)
                        new_data = audio_data.file.read()
                        if not new_data:
                            continue
                        self.sink_read_pos[user_id] = pos + len(new_data)

                        # 按 VAD 帧处理
                        pending = self.speech_state.get(user_id, {}).get("_pending", b"") + new_data
                        state = self.speech_state.setdefault(user_id, {
                            "buffer": b"",
                            "last_speech": 0.0,
                            "speech_start": 0.0,
                            "_pending": b"",
                        })

                        while len(pending) >= frame_bytes:
                            frame = pending[:frame_bytes]
                            pending = pending[frame_bytes:]

                            arr = np.frombuffer(frame, dtype=np.int16)
                            if arr.size >= 2 and arr.size % 2 == 0:
                                arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
                            audio_float = self._resample_tensor(arr, SAMPLE_RATE, TARGET_RATE)
                            has_voice = self.vad.has_speech(audio_float)

                            if has_voice:
                                if not state["buffer"]:
                                    state["speech_start"] = now
                                state["buffer"] += frame
                                state["last_speech"] = now
                                # 只累积有语音的帧到完整录音，避免送大段静音给 Whisper 产生幻觉
                                self.full_audio[user_id] = self.full_audio.get(user_id, b"") + frame
                            else:
                                if state["buffer"]:
                                    silence_secs = now - state["last_speech"]
                                    speech_secs = now - state["speech_start"]
                                    if silence_secs >= SILENCE_TIMEOUT or speech_secs >= MAX_SPEECH_DURATION:
                                        await self._flush_speech(user_id, state, guild)

                        state["_pending"] = pending

            except asyncio.CancelledError:
                break
            except Exception:
                LOGGER.exception("Audio processing loop error")

    def _get_display_name(self, user_id: int) -> str:
        """获取 Discord 显示名，找不到则返回 User_{id}。"""
        for session in self.active_sessions.values():
            guild = session.get("guild")
            if guild:
                member = guild.get_member(user_id)
                if member:
                    return member.display_name
        return f"User_{user_id}"

    async def _flush_speech(self, user_id: int, state: dict, guild) -> None:
        """Transcribe the accumulated speech buffer and post to Discord."""
        audio_data = state["buffer"]
        state["buffer"] = b""
        state["last_speech"] = 0.0
        state["speech_start"] = 0.0

        arr = np.frombuffer(audio_data, dtype=np.int16)
        if arr.size >= 2 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
        audio_float = self._resample_tensor(arr, SAMPLE_RATE, TARGET_RATE)

        # 音频太短不送 Whisper，避免幻觉
        duration = len(audio_float) / TARGET_RATE
        if duration < MIN_SPEECH_DURATION:
            LOGGER.debug("Skipping short segment %.2fs for user %d", duration, user_id)
            return

        language = "zh"
        for session in self.active_sessions.values():
            language = session.get("language", "zh")
            break

        # 实时字幕直接用 Discord 显示名，声纹识别放到会议纪要阶段
        speaker_name = self._get_display_name(user_id)

        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"speech_{user_id}.wav"
        try:
            import soundfile as sf
            sf.write(str(temp_wav), audio_float, TARGET_RATE)

            result = await asyncio.to_thread(
                mlx_whisper.transcribe,
                str(temp_wav),
                path_or_hf_repo="mlx-community/whisper-small-mlx",
                language=language,
                word_timestamps=False,
                initial_prompt=WHISPER_INITIAL_PROMPT.get(language, ""),
            )

            text = result.get("text", "").strip()
            if text and len(text) > 2 and not _is_hallucination(text):
                LOGGER.info("[%s] %s", speaker_name, text)
                for session in self.active_sessions.values():
                    ch = session.get("text_channel")
                    if ch:
                        try:
                            await ch.send(f"🎙️ **[{speaker_name}]**: {text}")
                        except Exception:
                            LOGGER.warning("Failed to send transcription to Discord (network error), skipping")
                    session.setdefault("transcript", []).append({
                        "speaker": speaker_name,
                        "text": text,
                        "ts": time.time(),
                    })
                await self._send_to_openclaw(text, speaker_name, user_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_diarization(self, wav_path: Path) -> list[tuple[float, float, str]]:
        """
        用 pyannote speaker-diarization-3.1 对混音音频做说话人分离。
        返回 [(start_sec, end_sec, speaker_label), ...]，按时间排序。
        需要 HF_TOKEN 且已在 HuggingFace 接受 pyannote 模型使用条款。
        """
        if self._diarization_pipeline is None:
            from pyannote.audio import Pipeline
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN,
            )
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            LOGGER.info("Diarization pipeline loaded, using device: %s", device)
            self._diarization_pipeline.to(device)
        diarization = self._diarization_pipeline(str(wav_path))
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        segments.sort(key=lambda x: x[0])
        return segments

    def _match_diarization_to_profiles(
        self,
        wav_path: Path,
        segments: list[tuple[float, float, str]],
        audio_float: np.ndarray,
    ) -> dict[str, str]:
        """
        对 diarization 分出的每个 speaker label，取其所有片段的音频，
        用 ecapa-tdnn 提取 embedding，与注册声纹库对比，返回映射表：
        {speaker_label: display_name}
        """
        import soundfile as sf

        # 按 speaker label 归并所有片段的音频
        label_audio: dict[str, list[np.ndarray]] = {}
        for start, end, label in segments:
            s = int(start * TARGET_RATE)
            e = int(end * TARGET_RATE)
            chunk = audio_float[s:e]
            if len(chunk) > 0:
                label_audio.setdefault(label, []).append(chunk)

        self.speaker_recognizer.load_model()

        # 第一步：提取每个 label 的 embedding
        label_embedding: dict[str, Optional[np.ndarray]] = {}
        for label, chunks in label_audio.items():
            combined = np.concatenate(chunks)
            if len(combined) < TARGET_RATE * 0.5:
                label_embedding[label] = None
                continue
            temp_dir = Path(tempfile.mkdtemp())
            try:
                temp_wav = temp_dir / f"spk_{label}.wav"
                sf.write(str(temp_wav), combined, TARGET_RATE)
                self.speaker_recognizer.load_model()
                emb = self.speaker_recognizer.extract_embedding(temp_wav)
                label_embedding[label] = emb
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # 第二步：构建 label × profile 相似度矩阵，做最优一对一分配
        # 每个注册人只能匹配给相似度最高的那个 label，避免所有 label 都匹配到同一人
        profiles = {
            uid: p for uid, p in self.speaker_recognizer.speaker_profiles.items()
            if p.embedding is not None
        }
        label_to_name: dict[str, str] = {}

        if profiles:
            # 计算所有 (label, profile) 对的相似度
            scores: list[tuple[float, str, int]] = []  # (score, label, user_id)
            for label, emb in label_embedding.items():
                if emb is None:
                    continue
                for uid, profile in profiles.items():
                    sim = float(np.dot(emb.flatten(), profile.embedding.flatten()) / (
                        np.linalg.norm(emb) * np.linalg.norm(profile.embedding) + 1e-9
                    ))
                    LOGGER.info("Similarity %s vs %s: %.4f (threshold=%.2f)",
                                label, profile.username, sim,
                                self.speaker_recognizer.similarity_threshold)
                    scores.append((sim, label, uid))

            # 贪心最优分配：按相似度从高到低，每个 label 和每个 profile 只用一次
            scores.sort(reverse=True)
            used_labels: set[str] = set()
            used_profiles: set[int] = set()
            for sim, label, uid in scores:
                if label in used_labels or uid in used_profiles:
                    continue
                if sim >= self.speaker_recognizer.similarity_threshold:
                    name = profiles[uid].username
                    LOGGER.info("Diarization %s → %s (score=%.3f)", label, name, sim)
                    label_to_name[label] = name
                    used_labels.add(label)
                    used_profiles.add(uid)

        # 未匹配到的 label 标记为匿名
        for label in label_embedding:
            if label not in label_to_name:
                LOGGER.info("Diarization %s → 未匹配，标记匿名", label)
                label_to_name[label] = f"{label}(匿名)"

        return label_to_name

    async def _generate_final_minutes(
        self,
        session: dict,
        user_pcm: dict[int, bytes],
        language: str,
        text_channel,
    ) -> None:
        """
        会议纪要生成：
        1. pyannote diarization 分离说话人片段
        2. ecapa-tdnn 匹配注册声纹，给每个 speaker label 打名字
        3. Whisper 对每个片段转录
        4. 按时间顺序拼装纪要
        """
        await text_channel.send("⏳ **正在生成会议纪要（diarization + 转录），请稍候...**")

        import soundfile as sf

        entries: list[dict] = []  # {"speaker": str, "start": float, "text": str}

        for user_id, pcm_bytes in user_pcm.items():
            if not pcm_bytes:
                continue

            display_name = self._get_display_name(user_id)
            temp_dir = Path(tempfile.mkdtemp())
            try:
                # ── 1. 解码为 mono float32 @ 16kHz ──
                arr = np.frombuffer(pcm_bytes, dtype=np.int16)
                if arr.size >= 2 and arr.size % 2 == 0:
                    arr_mono = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
                else:
                    arr_mono = arr
                audio_float = self._resample_tensor(arr_mono, SAMPLE_RATE, TARGET_RATE)

                full_wav = temp_dir / f"full_{user_id}.wav"
                sf.write(str(full_wav), audio_float, TARGET_RATE)

                # ── 2. Diarization ──
                LOGGER.info("Running diarization for user %s (%s)...", user_id, display_name)
                try:
                    segments = await asyncio.to_thread(self._run_diarization, full_wav)
                except Exception:
                    LOGGER.exception("Diarization failed for user %s, falling back to single-speaker", user_id)
                    segments = [(0.0, len(audio_float) / TARGET_RATE, "SPEAKER_0")]

                if not segments:
                    continue

                LOGGER.info("Diarization found %d segments for user %s", len(segments), user_id)

                # ── 3. 声纹匹配：给每个 speaker label 打名字 ──
                if self.speaker_recognizer.speaker_profiles:
                    label_to_name = await asyncio.to_thread(
                        self._match_diarization_to_profiles, full_wav, segments, audio_float
                    )
                else:
                    # 没有注册声纹，用 diarization label 直接显示
                    unique_labels = list(dict.fromkeys(s[2] for s in segments))
                    label_to_name = {lbl: f"{display_name}_{lbl}" for lbl in unique_labels}

                # ── 4. 按片段转录 ──
                for start, end, label in segments:
                    duration = end - start
                    if duration < 0.5:  # 太短跳过
                        continue

                    s_idx = int(start * TARGET_RATE)
                    e_idx = int(end * TARGET_RATE)
                    chunk = audio_float[s_idx:e_idx]
                    if len(chunk) == 0:
                        continue

                    seg_wav = temp_dir / f"seg_{label}_{int(start*1000)}.wav"
                    sf.write(str(seg_wav), chunk, TARGET_RATE)

                    try:
                        result = await asyncio.to_thread(
                            mlx_whisper.transcribe,
                            str(seg_wav),
                            path_or_hf_repo="mlx-community/whisper-small-mlx",
                            language=language,
                            word_timestamps=False,
                            initial_prompt=WHISPER_INITIAL_PROMPT.get(language, ""),
                        )
                        text = result.get("text", "").strip()
                        if text and len(text) > 1 and not _is_hallucination(text):
                            speaker_name = label_to_name.get(label, label)
                            entries.append({"speaker": speaker_name, "start": start, "text": text})
                    except Exception:
                        LOGGER.exception("Segment transcription failed: %s %.1f-%.1f", label, start, end)

            except Exception:
                LOGGER.exception("Final minutes failed for user %s", user_id)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if not entries:
            await text_channel.send("⚠️ 没有录到有效音频，无法生成会议纪要。")
            return

        # ── 5. 按时间排序，合并连续同一说话人的片段 ──
        entries.sort(key=lambda x: x["start"])
        merged: list[dict] = []
        for e in entries:
            if merged and merged[-1]["speaker"] == e["speaker"]:
                merged[-1]["text"] += "　" + e["text"]
            else:
                merged.append({"speaker": e["speaker"], "text": e["text"]})

        duration_secs = int(time.time() - session.get("start_time", time.time()))
        participants = list(dict.fromkeys(e["speaker"] for e in merged))
        lines = "\n\n".join(f"【{e['speaker']}】\n{e['text']}" for e in merged)
        summary = (
            f"--- 会议纪要（diarization 版）---\n"
            f"时长: {duration_secs // 60} 分 {duration_secs % 60} 秒\n"
            f"参与者: {', '.join(participants)}\n\n"
            f"{lines}"
        )
        for i in range(0, len(summary), 1900):
            await text_channel.send(f"```\n{summary[i:i+1900]}\n```")

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

    def _resample_tensor(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High-quality resampling via torchaudio, returns float32 [-1, 1]."""
        if orig_sr == target_sr:
            return audio.astype(np.float32) / 32768.0
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(wav).squeeze(0)
        return (resampled / 32768.0).numpy()

    async def _resolve_speaker(self, user_id: int, audio_path: Path) -> str:
        """声纹匹配优先；短片段失败时复用缓存结果；最终回退到 Discord 显示名+(匿名)。"""
        speaker_id, score = await asyncio.to_thread(
            self.speaker_recognizer.identify_speaker, audio_path
        )
        if speaker_id and speaker_id in self.speaker_recognizer.speaker_profiles:
            name = self.speaker_recognizer.speaker_profiles[speaker_id].username
            self.speaker_cache[user_id] = name  # 更新缓存
            LOGGER.debug("Speaker identified: %s (score=%.3f)", name, score)
            return name

        # 未达阈值：先看缓存（同一用户连续发言声纹稳定）
        LOGGER.debug("Speaker not matched (score=%.3f), checking cache for user %d", score, user_id)
        if user_id in self.speaker_cache:
            return self.speaker_cache[user_id]

        # 彻底未知 → Discord 显示名 + (匿名)
        for session in self.active_sessions.values():
            guild = session.get("guild")
            if guild:
                member = guild.get_member(user_id)
                if member:
                    return f"{member.display_name}(匿名)"
        return f"User_{user_id}(匿名)"


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
        # defer 必须在 3 秒内调用，放在最前面避免后续操作超时
        if not ctx.response.is_done():
            await ctx.defer()

        member = ctx.guild.get_member(ctx.author.id) if ctx.guild else None
        if not member or not member.voice or not member.voice.channel:
            await ctx.followup.send("❌ 你需要先进入语音频道！")
            return

        guild_id = ctx.guild_id
        if guild_id in bot.active_sessions:
            await ctx.followup.send("⚠️ 已经在转录中了！")
            return

        voice_channel = member.voice.channel
        voice_client = await voice_channel.connect()

        sink = discord.sinks.WaveSink()

        async def on_stop(sink, *args):
            pass  # no-op: audio consumed by processing loop

        voice_client.start_recording(sink, on_stop)

        # 清空上次会议的状态，避免跨会议污染
        bot.full_audio.clear()
        bot.sink_read_pos.clear()
        bot.speech_state.clear()
        bot.speaker_cache.clear()

        bot.active_sessions[guild_id] = {
            "voice_client": voice_client,
            "sink": sink,
            "guild": ctx.guild,
            "text_channel": ctx.channel,
            "language": language,
            "start_time": time.time(),
            "transcript": [],
        }

        await ctx.followup.send(
            f"🎙️ **实时语音转录已开启！**\n"
            f"频道: {voice_channel.mention}\n"
            f"语言: `{language}`\n"
            f"使用 `/voice_off` 停止"
        )

    @bot.slash_command(name="register_voice", description="注册声纹。name 可选，不填则注册自己，填则为指定人注册")
    async def register_voice(ctx: discord.ApplicationContext, name: str = "", seconds: int = 10):
        member = ctx.guild.get_member(ctx.author.id) if ctx.guild else None
        if not member or not member.voice or not member.voice.channel:
            await ctx.respond("❌ 你需要先进入语音频道！", ephemeral=True)
            return

        if seconds < 5:
            seconds = 5
        if seconds > 30:
            seconds = 30

        await ctx.respond(
            f"🎤 请对着麦克风说话 {seconds} 秒，我会记录你的声纹...",
            ephemeral=True
        )

        voice_channel = member.voice.channel
        guild_id = ctx.guild_id

        existing_session = bot.active_sessions.get(guild_id)
        temp_dir = Path(tempfile.mkdtemp())
        temp_wav = temp_dir / f"register_{ctx.author.id}.wav"
        connected_here = False

        try:
            if existing_session:
                # /voice_on 已在运行：等待 seconds 秒，从 full_audio 取该用户已积累的声音
                # 清除旧的积累，重新开始采集
                bot.full_audio.pop(ctx.author.id, None)
                await asyncio.sleep(seconds)
                pcm = bot.full_audio.get(ctx.author.id, b"")
                if pcm:
                    arr = np.frombuffer(pcm, dtype=np.int16)
                    if arr.size >= 2 and arr.size % 2 == 0:
                        arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    audio_float = bot._resample_tensor(arr, SAMPLE_RATE, TARGET_RATE)
                    import soundfile as sf
                    sf.write(str(temp_wav), audio_float, TARGET_RATE)
            else:
                # 没有 /voice_on：单独连接录音
                voice_client = await voice_channel.connect()
                connected_here = True
                reg_sink = discord.sinks.WaveSink()

                async def on_finish(sink, *args):
                    pass  # 数据在下面手动读取

                voice_client.start_recording(reg_sink, on_finish, ctx.channel)
                await asyncio.sleep(seconds)
                voice_client.stop_recording()
                await asyncio.sleep(0.5)

                # 直接从 sink 读取该用户的音频数据
                audio_data = reg_sink.audio_data.get(ctx.author.id)
                if audio_data and audio_data.file:
                    audio_data.file.seek(0)
                    raw = audio_data.file.read()
                    if raw:
                        # 转成 16kHz mono float WAV
                        arr = np.frombuffer(raw, dtype=np.int16)
                        if arr.size >= 2 and arr.size % 2 == 0:
                            arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
                        audio_float = bot._resample_tensor(arr, SAMPLE_RATE, TARGET_RATE)
                        import soundfile as sf
                        sf.write(str(temp_wav), audio_float, TARGET_RATE)

            if temp_wav.exists() and temp_wav.stat().st_size > 0:
                # 传了 name 则用名字的 hash 作为 ID，避免覆盖真实 Discord 用户
                if name:
                    reg_id = abs(hash(name)) % (10 ** 15)
                    reg_name = name
                else:
                    reg_id = ctx.author.id
                    reg_name = ctx.author.display_name
                success = await asyncio.to_thread(
                    bot.speaker_recognizer.register_speaker,
                    reg_id,
                    reg_name,
                    temp_wav,
                )
                if success:
                    await ctx.followup.send(
                        f"✅ 声纹注册成功！现在我能识别 **{reg_name}** 的声音了。",
                        ephemeral=True
                    )
                else:
                    await ctx.followup.send(
                        "❌ 声纹提取失败，请再试一次（确保环境安静且清晰说话）。",
                        ephemeral=True
                    )
            else:
                await ctx.followup.send("❌ 没有采集到音频，请检查麦克风权限。", ephemeral=True)
        except Exception:
            LOGGER.exception("Voice registration failed")
            await ctx.followup.send("❌ 注册失败，请查看日志。", ephemeral=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if connected_here and voice_client.is_connected():
                await voice_client.disconnect(force=True)

    @bot.slash_command(name="voice_off", description="关闭实时语音转文字")
    async def voice_off(ctx: discord.ApplicationContext):
        guild_id = ctx.guild_id

        if guild_id not in bot.active_sessions:
            await ctx.respond("⚠️ 没有在运行的转录。", ephemeral=True)
            return

        if not ctx.response.is_done():
            await ctx.defer()

        session = bot.active_sessions.pop(guild_id)
        voice_client = session["voice_client"]
        language = session.get("language", "zh")
        text_channel = session.get("text_channel")

        try:
            voice_client.stop_recording()
        except Exception:
            pass

        if voice_client.is_connected():
            await voice_client.disconnect(force=True)

        await ctx.followup.send("🛑 语音转录已停止。")

        # full_audio 是处理循环逐帧积累的完整 PCM，直接用它做会后纪要
        user_pcm: dict[int, bytes] = {}
        for user_id in list(bot.full_audio.keys()):
            data = bot.full_audio.pop(user_id)
            if data:
                user_pcm[user_id] = data
                LOGGER.info("Collected full audio for user %s: %d bytes", user_id, len(data))

        # 清理该 guild 的读指针和 speech state
        for user_id in list(bot.sink_read_pos.keys()):
            bot.sink_read_pos.pop(user_id, None)
        for user_id in list(bot.speech_state.keys()):
            bot.speech_state.pop(user_id, None)

        # 用完整录音生成会后纪要
        if text_channel and user_pcm:
            await bot._generate_final_minutes(session, user_pcm, language, text_channel)
        elif text_channel:
            await text_channel.send("⚠️ 没有录到音频，无法生成会议纪要。")

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
