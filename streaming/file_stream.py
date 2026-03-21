"""
File Streaming Demo
====================
Simulates real-time streaming by replaying an audio file in fixed-size chunks,
feeding each chunk into OnlineASRProcessor exactly as a live mic would.

Shows:
  • Per-chunk latency (RTF = real-time factor)
  • Partial (unstable) vs committed (stable) transcript
  • Final full transcript with WER if a reference is provided

Usage:
    python streaming/file_stream.py audio.wav
    python streaming/file_stream.py audio.wav --ref "ground truth text"
    python streaming/file_stream.py audio.wav --chunk-sec 2.0 --model small
    python streaming/file_stream.py audio.wav --offline   # batch mode for comparison
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING)

SAMPLE_RATE = 16000


def load_wav(path: str) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data


def build_asr(model_size: str, language: str):
    from whisper_online import FasterWhisperASR, OnlineASRProcessor
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

    print(f"[stream] Loading faster-whisper '{model_size}' …")
    asr = CPUFasterWhisperASR(language, modelsize=model_size)
    asr.use_vad()

    processor = OnlineASRProcessor(
        asr,
        tokenizer=None,
        buffer_trimming=("segment", 15),
        logfile=open(os.devnull, "w"),
    )
    return processor


def run_streaming(audio: np.ndarray, processor, chunk_sec: float, filename: str):
    """Simulate live streaming: feed chunks and print partial transcript."""
    chunk_sz   = int(chunk_sec * SAMPLE_RATE)
    n_chunks   = max(1, len(audio) // chunk_sz)
    duration   = len(audio) / SAMPLE_RATE

    print()
    print("═" * 65)
    print(f"  FILE STREAMING  │  {os.path.basename(filename)}")
    print(f"  Duration: {duration:.1f}s  │  Chunk: {chunk_sec}s  │  Chunks: {n_chunks}")
    print("═" * 65)

    processor.init()
    all_parts  = []
    latencies  = []
    t_wall_start = time.perf_counter()

    for i in range(n_chunks + 1):
        chunk = audio[i * chunk_sz:(i + 1) * chunk_sz]
        if len(chunk) < SAMPLE_RATE * 0.1:
            break

        audio_time_start = i * chunk_sec
        audio_time_end   = audio_time_start + len(chunk) / SAMPLE_RATE

        t0 = time.perf_counter()
        processor.insert_audio_chunk(chunk)
        beg, end, text = processor.process_iter()
        latency = time.perf_counter() - t0
        latencies.append(latency)
        rtf = latency / (len(chunk) / SAMPLE_RATE)

        status = "▶" if text and text.strip() else "·"
        print(f"  {status} chunk {i+1:>3}/{n_chunks}  "
              f"[{audio_time_start:.1f}s→{audio_time_end:.1f}s]  "
              f"lat={latency:.2f}s  RTF={rtf:.3f}", end="")
        if text and text.strip():
            print(f"\n    └─ {text.strip()}")
            all_parts.append(text.strip())
        else:
            print()

    # Flush remaining
    beg, end, text = processor.finish()
    if text and text.strip():
        print(f"\n  [final flush] {text.strip()}")
        all_parts.append(text.strip())

    full_transcript = " ".join(all_parts)
    total_wall = time.perf_counter() - t_wall_start

    print()
    print("─" * 65)
    print(f"  Full transcript:")
    print(f"  {full_transcript}")
    print("─" * 65)
    print(f"  Total wall time  : {total_wall:.2f}s  (audio: {duration:.1f}s)")
    print(f"  Avg chunk latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"  Overall RTF      : {total_wall/duration:.3f}x")

    return full_transcript


def run_offline(audio: np.ndarray, processor, filename: str):
    """Batch mode: feed entire audio at once for best WER."""
    print()
    print("═" * 65)
    print(f"  OFFLINE (BATCH) MODE  │  {os.path.basename(filename)}")
    print("═" * 65)

    processor.init()
    t0 = time.perf_counter()
    processor.insert_audio_chunk(audio)
    beg, end, text = processor.finish()
    elapsed = time.perf_counter() - t0

    print(f"  Transcript : {text.strip() if text else '(empty)'}")
    print(f"  Time       : {elapsed:.2f}s  "
          f"(RTF={elapsed/(len(audio)/SAMPLE_RATE):.3f}x)")
    return text.strip() if text else ""


def compute_wer(ref: str, hyp: str) -> float:
    from jiwer import wer
    from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
    T = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    return wer(T(ref), T(hyp)) * 100


def main():
    p = argparse.ArgumentParser(description="File-based streaming demo")
    p.add_argument("audio",       help="Path to WAV audio file")
    p.add_argument("--model",     default="base",
                   choices=["tiny", "tiny.en", "base", "base.en",
                            "small", "small.en", "medium", "large-v2", "large-v3"])
    p.add_argument("--language",  default="en")
    p.add_argument("--chunk-sec", type=float, default=1.0,
                   help="Chunk size in seconds (default: 1.0)")
    p.add_argument("--offline",   action="store_true",
                   help="Batch mode: feed all audio at once (best WER)")
    p.add_argument("--ref",       default=None,
                   help="Reference transcript for WER calculation")
    args = p.parse_args()

    if not os.path.exists(args.audio):
        print(f"[error] File not found: {args.audio}")
        sys.exit(1)

    print(f"[stream] Loading audio: {args.audio}")
    audio = load_wav(args.audio)
    print(f"[stream] Duration: {len(audio)/SAMPLE_RATE:.2f}s  |  Samples: {len(audio)}")

    processor = build_asr(args.model, args.language)

    if args.offline:
        hyp = run_offline(audio, processor, args.audio)
    else:
        hyp = run_streaming(audio, processor, args.chunk_sec, args.audio)

    if args.ref and hyp:
        try:
            w = compute_wer(args.ref, hyp)
            print(f"\n  WER vs reference: {w:.2f}%")
            print(f"  REF: {args.ref}")
            print(f"  HYP: {hyp}")
        except Exception as e:
            print(f"  [WER error] {e}")


if __name__ == "__main__":
    main()
