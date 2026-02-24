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
    return pipeline


def load_embedding_model():
    from speechbrain.inference.classifiers import EncoderClassifier
    LOGGER.info("Loading speaker embedding model...")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
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


def process(wav_path: Path, language: str = "zh") -> None:
    LOGGER.info("Processing: %s", wav_path)
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    LOGGER.info("Audio: %.1f sec @ %d Hz", len(audio) / sr, sr)

    profiles = load_speaker_profiles()
    pipeline = load_diarization_pipeline()

    LOGGER.info("Running diarization on %s...", wav_path)
    segments = run_diarization_on_file(wav_path, pipeline)

    if not segments:
        LOGGER.warning("No segments found")
        return

    LOGGER.info("Found %d segments, %d unique speakers",
                len(segments), len(set(s[2] for s in segments)))

    if profiles:
        emb_model = load_embedding_model()
        label_to_name = match_labels_to_speakers(
            segments, audio, profiles,
            lambda p: extract_embedding(p, emb_model),
            SIMILARITY_THRESHOLD,
        )
    else:
        unique_labels = list(dict.fromkeys(s[2] for s in segments))
        label_to_name = {lbl: f"{lbl}(匿名)" for lbl in unique_labels}

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        entries = transcribe_segments(segments, audio, label_to_name, language, tmp_dir)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not entries:
        LOGGER.warning("No transcription output")
        return

    duration_secs = int(len(audio) / sr)
    summary = format_minutes(entries, duration_secs)

    print("\n" + "=" * 60)
    print(f"文件: {wav_path.name}")
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线重跑会议纪要")
    parser.add_argument("wav", help="WAV 文件路径")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="语言（默认 zh）")
    args = parser.parse_args()
    process(Path(args.wav), args.language)
