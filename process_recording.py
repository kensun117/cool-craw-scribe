#!/usr/bin/env python3
"""
离线重跑会议纪要：对已保存的 WAV 文件执行 diarization + 声纹匹配 + 转录。

用法：
    python process_recording.py data/recordings/20260223_110000_user123.wav
    python process_recording.py data/recordings/20260223_110000_user123.wav --language zh
"""
import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

# 从 voice_bot 导入所有共享的处理函数和常量
from voice_bot import (
    run_diarization_on_file,
    match_labels_to_speakers,
    transcribe_segments,
    format_minutes,
    HF_TOKEN,
    SIMILARITY_THRESHOLD,
    PROFILES_DIR,
    PROFILES_JSON,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_speaker_profiles() -> dict:
    import json
    if not PROFILES_JSON.exists():
        return {}
    data = json.loads(PROFILES_JSON.read_text())
    profiles = {}
    for uid_str, meta in data.items():
        emb_path = PROFILES_DIR / meta["embedding_file"]
        if emb_path.exists():
            profiles[int(uid_str)] = {
                "username": meta["username"],
                "embedding": np.load(str(emb_path)),
            }
    LOGGER.info("Loaded %d speaker profile(s)", len(profiles))
    return profiles


def load_diarization_pipeline():
    from pyannote.audio import Pipeline
    LOGGER.info("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    LOGGER.info("Diarization using device: %s", device)
    pipeline.to(device)
    pipeline._segmentation.batch_size = 64
    pipeline._embedding.batch_size = 64
    return pipeline


def load_embedding_model():
    from speechbrain.inference.classifiers import EncoderClassifier
    LOGGER.info("Loading speaker embedding model...")
    savedir = Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": "cpu"},
    )


def extract_embedding(wav_path: Path, model) -> np.ndarray | None:
    import torchaudio
    import torchaudio.transforms as T
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.numel() == 0:
            return None
        if sr != 16000:
            waveform = T.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = model.encode_batch(waveform)
            return emb.squeeze().cpu().numpy()
    except Exception:
        LOGGER.exception("Failed to extract embedding from %s", wav_path)
        return None


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def process(wav_path: Path, language: str = "zh", output_path: Path | None = None, num_speakers: int | None = None) -> None:
    t_total = time.time()

    LOGGER.info("Processing: %s", wav_path)
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    LOGGER.info("Audio: %.1f sec @ %d Hz", len(audio) / sr, sr)

    t0 = time.time()
    profiles = load_speaker_profiles()
    LOGGER.info("[TIMER] load_speaker_profiles: %s", _elapsed(t0))

    t0 = time.time()
    pipeline = load_diarization_pipeline()
    LOGGER.info("[TIMER] load_diarization_pipeline: %s", _elapsed(t0))

    t0 = time.time()
    LOGGER.info("Running diarization on %s... (num_speakers=%s)", wav_path, num_speakers)
    segments = run_diarization_on_file(wav_path, pipeline, num_speakers=num_speakers)
    LOGGER.info("[TIMER] diarization: %s", _elapsed(t0))

    # 释放 pipeline，避免 MPS context 和后续 mlx-whisper 的 Metal 使用冲突
    del pipeline
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if not segments:
        LOGGER.warning("No segments found")
        return

    LOGGER.info("Found %d segments, %d unique speakers",
                len(segments), len(set(s[2] for s in segments)))

    if profiles:
        t0 = time.time()
        emb_model = load_embedding_model()
        LOGGER.info("[TIMER] load_embedding_model: %s", _elapsed(t0))

        t0 = time.time()
        label_to_name = match_labels_to_speakers(
            segments, audio, profiles,
            lambda p: extract_embedding(p, emb_model),
            SIMILARITY_THRESHOLD,
        )
        LOGGER.info("[TIMER] match_labels_to_speakers: %s", _elapsed(t0))
    else:
        unique_labels = list(dict.fromkeys(s[2] for s in segments))
        label_to_name = {lbl: f"{lbl}(匿名)" for lbl in unique_labels}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        t0 = time.time()
        entries = transcribe_segments(segments, audio, label_to_name, language, tmp_dir)
        LOGGER.info("[TIMER] transcribe_segments: %s", _elapsed(t0))
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not entries:
        LOGGER.warning("No transcription output")
        return

    duration_secs = int(len(audio) / sr)
    summary = format_minutes(entries, duration_secs)

    header = "=" * 60 + f"\n文件: {wav_path.name}\n" + "=" * 60
    output = f"\n{header}\n{summary}\n{'=' * 60}\n"

    print(output)

    # 默认输出到 WAV 同目录同名 .txt，--output 可覆盖
    save_path = output_path or wav_path.with_suffix(".txt")
    save_path.write_text(output, encoding="utf-8")
    LOGGER.info("Transcript saved to: %s", save_path)
    LOGGER.info("[TIMER] total: %s", _elapsed(t_total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线重跑会议纪要")
    parser.add_argument("wav", help="WAV 文件路径")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="语言（默认 zh）")
    parser.add_argument("--output", default=None, help="输出文本文件路径（默认与 WAV 同目录同名 .txt）")
    parser.add_argument("--num-speakers", type=int, default=None, help="已知说话人数量（指定后跳过自动估计，提高准确性）")
    args = parser.parse_args()
    process(Path(args.wav), args.language, Path(args.output) if args.output else None, num_speakers=args.num_speakers)
