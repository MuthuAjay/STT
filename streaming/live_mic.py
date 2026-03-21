"""
Live Microphone Streaming Demo
================================
Real-time speech-to-text from the microphone using the whisper_streaming
local-agreement algorithm (ufal/whisper_streaming).

The audio is captured via sounddevice in small chunks (default 0.5s),
fed into OnlineASRProcessor, and printed as stable partial transcripts.

Usage:
    python streaming/live_mic.py
    python streaming/live_mic.py --model base --chunk-sec 1.0
    python streaming/live_mic.py --vad --model small
"""
import argparse
import logging
import os
import queue
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd

# ── make whisper_online importable from this folder ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)

SAMPLE_RATE = 16000


def build_asr(model_size: str, language: str, use_vad: bool):
    """Initialise FasterWhisperASR with CPU float16 fallback."""
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    print(f"[mic] Loading faster-whisper '{model_size}' …", flush=True)

    # Patch load_model to use CPU when CUDA is unavailable
    import faster_whisper

    class CPUFasterWhisperASR(FasterWhisperASR):
        def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
            size = model_dir or modelsize
            try:
                return faster_whisper.WhisperModel(
                    size, device="cuda", compute_type="float16"
                )
            except Exception:
                return faster_whisper.WhisperModel(
                    size, device="cpu", compute_type="float32"
                )

    asr = CPUFasterWhisperASR(language, modelsize=model_size)
    if use_vad:
        asr.use_vad()

    processor = OnlineASRProcessor(
        asr,
        tokenizer=None,
        buffer_trimming=("segment", 15),
        logfile=open(os.devnull, "w"),
    )
    print(f"[mic] Model ready. VAD={'on' if use_vad else 'off'}", flush=True)
    return processor


def run(model_size: str, language: str, chunk_sec: float,
        use_vad: bool, max_sec: int):

    processor = build_asr(model_size, language, use_vad)

    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(chunk_sec * SAMPLE_RATE)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    print()
    print("═" * 60)
    print(f"  LIVE TRANSCRIPTION  │  model={model_size}  lang={language}")
    print(f"  chunk={chunk_sec}s  │  Ctrl+C to stop")
    print("═" * 60)
    print()

    processor.init()
    committed_text = ""
    session_start = time.perf_counter()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        callback=callback,
    ):
        try:
            while True:
                if max_sec and (time.perf_counter() - session_start) > max_sec:
                    print("\n[mic] Max duration reached.")
                    break

                try:
                    chunk = audio_q.get(timeout=2.0)
                except queue.Empty:
                    continue

                processor.insert_audio_chunk(chunk)
                beg, end, text = processor.process_iter()

                if text and text.strip():
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\r[{ts}] {text.strip():<70}", end="", flush=True)
                    if text != committed_text:
                        committed_text = text
                        if text.endswith((".", "!", "?")):
                            print()  # newline on sentence end

        except KeyboardInterrupt:
            print("\n\n[mic] Stopping …")
        finally:
            beg, end, text = processor.finish()
            if text and text.strip():
                print(f"\n[mic] Final: {text.strip()}")
            elapsed = time.perf_counter() - session_start
            print(f"[mic] Session: {elapsed:.1f}s")


def main():
    p = argparse.ArgumentParser(description="Live mic STT with whisper_streaming")
    p.add_argument("--model",     default="base",
                   choices=["tiny", "tiny.en", "base", "base.en",
                            "small", "small.en", "medium", "large-v2", "large-v3"])
    p.add_argument("--language",  default="en")
    p.add_argument("--chunk-sec", type=float, default=0.5,
                   help="Audio chunk size in seconds (default: 0.5)")
    p.add_argument("--vad",       action="store_true",
                   help="Enable Silero VAD filter inside Whisper")
    p.add_argument("--max-sec",   type=int, default=0,
                   help="Auto-stop after N seconds (0 = unlimited)")
    args = p.parse_args()

    run(args.model, args.language, args.chunk_sec, args.vad, args.max_sec)


if __name__ == "__main__":
    main()
