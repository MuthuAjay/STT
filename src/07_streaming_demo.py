"""
Step 7 – Streaming / Live STT Demo
=====================================
Real-time speech transcription using the microphone.

Two modes:
  --mode chunk   (default) Record fixed-length chunks, transcribe each chunk.
  --mode vad     Voice-activity-detection: transcribe on silence detection.

Run:
    python src/07_streaming_demo.py
    python src/07_streaming_demo.py --mode vad --chunk-sec 3
    python src/07_streaming_demo.py --file path/to/audio.wav   # file streaming
"""
import argparse
import os
import sys
import time
import queue
import threading
import tempfile
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from faster_whisper import WhisperModel

from config import WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE

# ── Model ─────────────────────────────────────────────────────────────────

def load_model():
    print(f"[streaming] Loading whisper-{WHISPER_MODEL_SIZE} …")
    return WhisperModel(WHISPER_MODEL_SIZE,
                        compute_type=WHISPER_COMPUTE_TYPE,
                        num_workers=1)


# ── Transcription helpers ─────────────────────────────────────────────────

def transcribe_chunk(model: WhisperModel, audio: np.ndarray,
                     samplerate: int = 16000) -> str:
    """Transcribe a numpy float32 audio array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    # Write to temp wav
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

    segments, _ = model.transcribe(
        tmp_path,
        language="en",
        beam_size=3,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    os.unlink(tmp_path)
    return text


# ── Chunk mode ────────────────────────────────────────────────────────────

def run_chunk_mode(model: WhisperModel,
                   chunk_sec: float = 4.0,
                   samplerate: int = 16000,
                   max_chunks: int = 0):
    """Record fixed-length chunks from the mic and transcribe each."""
    print(f"\n[streaming] CHUNK mode  |  chunk={chunk_sec}s  |  Ctrl+C to stop")
    print("─" * 60)
    chunk_samples = int(chunk_sec * samplerate)
    chunk_count = 0

    try:
        while True:
            if max_chunks and chunk_count >= max_chunks:
                break
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening …", end=" ", flush=True)
            audio = sd.rec(chunk_samples, samplerate=samplerate,
                           channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()
            chunk_count += 1

            t0 = time.perf_counter()
            text = transcribe_chunk(model, audio, samplerate)
            latency = time.perf_counter() - t0

            if text:
                print(f"\033[32m{text}\033[0m  \033[90m({latency:.2f}s)\033[0m")
            else:
                print("\033[90m[silence]\033[0m")

    except KeyboardInterrupt:
        print("\n[streaming] Stopped by user.")


# ── VAD mode ──────────────────────────────────────────────────────────────

def run_vad_mode(model: WhisperModel,
                 chunk_sec: float = 3.0,
                 samplerate: int = 16000,
                 silence_threshold: float = 0.01,
                 silence_sec: float = 1.0):
    """Buffer audio; transcribe when silence is detected."""
    print(f"\n[streaming] VAD mode  |  silence_threshold={silence_threshold}  |  Ctrl+C to stop")
    print("─" * 60)

    audio_queue: queue.Queue = queue.Queue()
    buffer = []
    silence_chunks = 0
    frames_per_check = int(samplerate * 0.1)   # 100 ms windows
    silence_limit = int(silence_sec / 0.1)

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1,
                        dtype="float32", blocksize=frames_per_check,
                        callback=callback):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening … (Ctrl+C to stop)")
        try:
            while True:
                chunk = audio_queue.get(timeout=5).flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > silence_threshold:
                    buffer.append(chunk)
                    silence_chunks = 0
                else:
                    if buffer:
                        silence_chunks += 1
                        buffer.append(chunk)
                        if silence_chunks >= silence_limit:
                            audio = np.concatenate(buffer)
                            buffer.clear()
                            silence_chunks = 0

                            t0 = time.perf_counter()
                            text = transcribe_chunk(model, audio, samplerate)
                            latency = time.perf_counter() - t0
                            ts = datetime.now().strftime("%H:%M:%S")
                            if text:
                                print(f"[{ts}] \033[32m{text}\033[0m  \033[90m({latency:.2f}s)\033[0m")

        except KeyboardInterrupt:
            print("\n[streaming] Stopped by user.")
        except queue.Empty:
            print("[streaming] No audio received. Check microphone.")


# ── File streaming mode ───────────────────────────────────────────────────

def run_file_streaming(model: WhisperModel, file_path: str, chunk_sec: float = 4.0):
    """Simulate streaming by processing an audio file in chunks."""
    print(f"\n[streaming] FILE mode  |  {file_path}  |  chunk={chunk_sec}s")
    print("─" * 60)

    samplerate, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    chunk_samples = int(chunk_sec * samplerate)
    total_chunks = len(data) // chunk_samples
    transcript_parts = []

    for i in range(total_chunks + 1):
        chunk = data[i * chunk_samples:(i + 1) * chunk_samples]
        if len(chunk) < samplerate * 0.5:
            break
        t0 = time.perf_counter()
        text = transcribe_chunk(model, chunk, samplerate)
        latency = time.perf_counter() - t0
        ts = datetime.now().strftime("%H:%M:%S")
        if text:
            print(f"[{ts}] chunk {i+1}/{total_chunks}  \033[32m{text}\033[0m  \033[90m({latency:.2f}s)\033[0m")
            transcript_parts.append(text)

    full_transcript = " ".join(transcript_parts)
    print(f"\n--- Full transcript ---\n{full_transcript}")
    return full_transcript


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time STT streaming demo")
    parser.add_argument("--mode", choices=["chunk", "vad", "file"],
                        default="chunk", help="Streaming mode")
    parser.add_argument("--chunk-sec", type=float, default=4.0,
                        help="Chunk length in seconds (chunk/file mode)")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--silence-threshold", type=float, default=0.01,
                        help="RMS threshold for silence detection (VAD mode)")
    parser.add_argument("--file", type=str, default=None,
                        help="Audio file path (file mode)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Stop after N chunks (0 = unlimited)")
    args = parser.parse_args()

    model = load_model()

    if args.mode == "file" or args.file:
        path = args.file
        if not path:
            parser.error("--file required for file mode")
        if not os.path.exists(path):
            parser.error(f"File not found: {path}")
        run_file_streaming(model, path, args.chunk_sec)
    elif args.mode == "vad":
        run_vad_mode(model, args.chunk_sec, args.samplerate, args.silence_threshold)
    else:
        run_chunk_mode(model, args.chunk_sec, args.samplerate, args.max_chunks)


if __name__ == "__main__":
    main()
