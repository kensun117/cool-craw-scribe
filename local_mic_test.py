import argparse
import tempfile
from pathlib import Path

import mlx_whisper
import sounddevice as sd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local microphone recording test with MLX Whisper")
    parser.add_argument("--seconds", type=int, default=12, help="record duration in seconds")
    parser.add_argument("--samplerate", type=int, default=16000, help="record sample rate")
    parser.add_argument("--channels", type=int, default=1, help="number of input channels")
    parser.add_argument("--stream", action="store_true", help="stream mode: print text chunk by chunk")
    parser.add_argument("--chunk-seconds", type=int, default=4, help="chunk length for stream mode")
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="transcription language: zh or en",
    )
    return parser.parse_args()


def record_to_wav(out_wav: Path, seconds: int, samplerate: int, channels: int) -> None:
    print("[INFO] Recorder started successfully.")
    print(f"[INFO] Input config: samplerate={samplerate}, channels={channels}")
    print("[INFO] Prepare to speak in 3...")
    print("[INFO] Prepare to speak in 2...")
    print("[INFO] Prepare to speak in 1...")
    print(f"[1/3] Recording {seconds}s... start speaking now")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
    sd.wait()
    print("[INFO] Recording finished.")
    sf.write(str(out_wav), audio, samplerate)
    print(f"Saved audio: {out_wav}")


def transcribe(wav_path: Path, language: str) -> str:
    print(f"[2/3] Transcribing with mlx-whisper (language={language})...")
    result = mlx_whisper.transcribe(
        audio=str(wav_path),
        path_or_hf_repo="mlx-community/whisper-small-mlx",
        language=language,
        word_timestamps=True,
    )
    segments = result.get("segments", [])
    if not segments:
        return ""

    lines = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        if text:
            lines.append(f"[{start:6.2f}-{end:6.2f}] {text}")
    return "\n".join(lines)


def transcribe_compact(wav_path: Path, language: str) -> str:
    result = mlx_whisper.transcribe(
        audio=str(wav_path),
        path_or_hf_repo="mlx-community/whisper-small-mlx",
        language=language,
        word_timestamps=True,
    )
    texts = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if text:
            texts.append(text)
    return " ".join(texts).strip()


def stream_transcribe(total_seconds: int, chunk_seconds: int, samplerate: int, channels: int, language: str) -> None:
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be > 0")
    print("[INFO] Stream mode started successfully.")
    print(f"[INFO] Input config: samplerate={samplerate}, channels={channels}, language={language}")
    print("[INFO] Prepare to speak in 3...")
    print("[INFO] Prepare to speak in 2...")
    print("[INFO] Prepare to speak in 1...")
    print(f"[1/2] Stream recording started: total={total_seconds}s, chunk={chunk_seconds}s")

    elapsed = 0
    idx = 1
    with tempfile.TemporaryDirectory(prefix="local_mic_stream_") as td:
        while elapsed < total_seconds:
            sec = min(chunk_seconds, total_seconds - elapsed)
            print(f"[INFO] Recording chunk {idx} ({sec}s)...")
            audio = sd.rec(int(sec * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
            sd.wait()
            wav = Path(td) / f"chunk_{idx:03d}.wav"
            sf.write(str(wav), audio, samplerate)
            print(f"[INFO] Transcribing chunk {idx}...")
            text = transcribe_compact(wav, language)
            start_ts = elapsed
            end_ts = elapsed + sec
            if text:
                print(f"[{start_ts:>3}s-{end_ts:>3}s] {text}")
            else:
                print(f"[{start_ts:>3}s-{end_ts:>3}s] (silence)")
            elapsed += sec
            idx += 1

    print("[2/2] Stream test done.")


def main() -> None:
    args = parse_args()
    if args.stream:
        stream_transcribe(
            total_seconds=args.seconds,
            chunk_seconds=args.chunk_seconds,
            samplerate=args.samplerate,
            channels=args.channels,
            language=args.language,
        )
        return

    with tempfile.TemporaryDirectory(prefix="local_mic_test_") as td:
        wav = Path(td) / "mic_test.wav"
        record_to_wav(wav, args.seconds, args.samplerate, args.channels)
        text = transcribe(wav, args.language)

    print("[3/3] Result:")
    if text.strip():
        print(text)
    else:
        print("No speech recognized.")


if __name__ == "__main__":
    main()
