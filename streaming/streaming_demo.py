"""
Streaming Demo – Unified Runner
================================
Runs both file streaming and live mic modes with rich terminal output,
logging, and a session summary.

Usage:
    # Live mic  (default, 60s session)
    python streaming/streaming_demo.py --mode mic

    # File streaming
    python streaming/streaming_demo.py --mode file --audio path/to/file.wav

    # File + compare streaming vs offline WER
    python streaming/streaming_demo.py --mode compare --audio file.wav

    # Batch test on the Common Voice samples
    python streaming/streaming_demo.py --mode batch --samples-dir ../common_au_en_op/samples --n 10
"""
import argparse
import glob
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"streaming_{datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("streaming_demo")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

SAMPLE_RATE = 16000


import librosa

def load_wav(path):
    # librosa automatically handles format conversion and resamples to 16kHz
    data, sr = librosa.load(path, sr=16000, mono=True)
    return sr, data

# ── ASR factory ───────────────────────────────────────────────────────────

def make_processor(model_size: str, language: str):
    from whisper_online import FasterWhisperASR, OnlineASRProcessor
    import faster_whisper

    class CPUFasterWhisperASR(FasterWhisperASR):
        def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
            size = model_dir or modelsize
            try:
                return faster_whisper.WhisperModel(
                    size, device="cuda", compute_type="float16")
            except Exception:
                log.info(f"  CUDA unavailable — using CPU (float32)")
                return faster_whisper.WhisperModel(
                    size, device="cpu", compute_type="float32")

    log.info(f"  Loading faster-whisper '{model_size}' (lang={language}) …")
    asr = CPUFasterWhisperASR(language, modelsize=model_size)
    asr.use_vad()
    proc = OnlineASRProcessor(
        asr, tokenizer=None, buffer_trimming=("segment", 15),
        logfile=open(os.devnull, "w"),
    )
    log.info("  Model ready.")
    return proc


# ── Audio loader ──────────────────────────────────────────────────────────

# def load_wav(path: str) -> np.ndarray:
#     from scipy.io import wavfile
#     sr, data = wavfile.read(path)
#     if data.ndim > 1:
#         data = data.mean(axis=1)
#     if data.dtype != np.float32:
#         data = data.astype(np.float32) / np.iinfo(data.dtype).max
#     if sr != SAMPLE_RATE:
#         from scipy.signal import resample_poly
#         from math import gcd
#         g = gcd(sr, SAMPLE_RATE)
#         data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
#     return data


def compute_wer(ref: str, hyp: str) -> float:
    from jiwer import wer
    from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
    T = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    return round(wer(T(ref), T(hyp)) * 100, 2)


# ── Modes ──────────────────────────────────────────────────────────────────

def mode_mic(proc, chunk_sec: float, max_sec: int):
    """Live microphone transcription."""
    import queue
    import sounddevice as sd

    log.info("=" * 60)
    log.info("  LIVE MIC MODE")
    log.info(f"  chunk={chunk_sec}s  |  max={max_sec}s  |  Ctrl+C to stop")
    log.info("=" * 60)

    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(chunk_sec * SAMPLE_RATE)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    proc.init()
    t_start = time.perf_counter()
    n_chunks = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=chunk_samples, callback=callback):
        log.info("  🎤 Listening … (speak now)")
        try:
            while True:
                if max_sec and (time.perf_counter() - t_start) > max_sec:
                    log.info("  Max duration reached.")
                    break
                try:
                    chunk = audio_q.get(timeout=2.0)
                except queue.Empty:
                    continue

                proc.insert_audio_chunk(chunk)
                beg, end, text = proc.process_iter()
                n_chunks += 1

                if text and text.strip():
                    ts = datetime.now().strftime("%H:%M:%S")
                    log.info(f"  [{ts}] {text.strip()}")

        except KeyboardInterrupt:
            log.info("  Stopped by user.")
        finally:
            _, _, text = proc.finish()
            if text and text.strip():
                log.info(f"  [final] {text.strip()}")

    elapsed = time.perf_counter() - t_start
    log.info(f"\n  Session: {elapsed:.1f}s  |  Chunks processed: {n_chunks}")
    log.info(f"  Log saved → {LOG_FILE}")


def mode_file(proc, audio_path: str, chunk_sec: float, ref: str = None):
    """Simulate streaming from a file."""
    from tqdm import tqdm

    audio    = load_wav(audio_path)
    duration = len(audio) / SAMPLE_RATE
    chunk_sz = int(chunk_sec * SAMPLE_RATE)
    n_chunks = max(1, len(audio) // chunk_sz)

    log.info("=" * 60)
    log.info(f"  FILE STREAMING  │  {os.path.basename(audio_path)}")
    log.info(f"  Duration: {duration:.1f}s  │  Chunk: {chunk_sec}s  │  Chunks: {n_chunks}")
    log.info("=" * 60)

    proc.init()
    all_parts, latencies = [], []
    t_wall = time.perf_counter()

    with tqdm(total=n_chunks, desc="  Streaming", unit="chunk",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for i in range(n_chunks + 1):
            chunk = audio[i * chunk_sz:(i + 1) * chunk_sz]
            if len(chunk) < SAMPLE_RATE * 0.1:
                break

            t0 = time.perf_counter()
            proc.insert_audio_chunk(chunk)
            beg, end, text = proc.process_iter()
            lat = time.perf_counter() - t0
            latencies.append(lat)
            rtf = lat / (len(chunk) / SAMPLE_RATE)

            pbar.update(1)
            pbar.set_postfix({"lat": f"{lat:.2f}s", "RTF": f"{rtf:.3f}"})

            if text and text.strip():
                log.info(f"  [{i*chunk_sec:.1f}s] {text.strip()}")
                all_parts.append(text.strip())

    _, _, text = proc.finish()
    if text and text.strip():
        log.info(f"  [flush] {text.strip()}")
        all_parts.append(text.strip())

    hyp = " ".join(all_parts)
    total_wall = time.perf_counter() - t_wall

    log.info("\n  ── Full Transcript ──")
    log.info(f"  {hyp}")
    log.info(f"\n  Wall time : {total_wall:.2f}s  (audio: {duration:.1f}s)")
    log.info(f"  Avg lat   : {sum(latencies)/len(latencies):.3f}s/chunk")
    log.info(f"  RTF       : {total_wall/duration:.3f}x")

    if ref:
        w = compute_wer(ref, hyp)
        log.info(f"\n  WER vs ref: {w:.2f}%")
        log.info(f"  REF: {ref}")
        log.info(f"  HYP: {hyp}")

    return hyp, total_wall, duration


def mode_compare(proc, audio_path: str, chunk_sec: float, ref: str = None):
    """Compare streaming vs offline on same file."""
    from whisper_online import OnlineASRProcessor
    import faster_whisper

    log.info("=" * 60)
    log.info(f"  COMPARE MODE  │  streaming vs offline")
    log.info("=" * 60)

    audio    = load_wav(audio_path)
    duration = len(audio) / SAMPLE_RATE

    # ── Streaming ──
    log.info("\n  [1/2] Streaming …")
    hyp_stream, t_stream, _ = mode_file(proc, audio_path, chunk_sec, ref=None)

    # ── Offline ── (re-init same processor)
    log.info("\n  [2/2] Offline (batch) …")
    proc.init()
    t0 = time.perf_counter()
    proc.insert_audio_chunk(audio)
    _, _, hyp_offline = proc.finish()
    t_offline = time.perf_counter() - t0
    hyp_offline = hyp_offline.strip() if hyp_offline else ""
    log.info(f"  Offline transcript: {hyp_offline}")
    log.info(f"  Offline time: {t_offline:.2f}s  RTF={t_offline/duration:.3f}x")

    log.info("\n  ── Comparison ──")
    log.info(f"  {'Mode':<12} {'Time':>7}  {'RTF':>6}  {'WER':>7}")
    log.info(f"  {'─'*12} {'─'*7}  {'─'*6}  {'─'*7}")

    for mode_name, hyp, t in [("Streaming", hyp_stream, t_stream),
                               ("Offline",   hyp_offline, t_offline)]:
        rtf = t / duration
        wer_val = compute_wer(ref, hyp) if ref else "n/a"
        log.info(f"  {mode_name:<12} {t:>6.2f}s  {rtf:>6.3f}  {str(wer_val):>6}%")

    if ref:
        log.info(f"\n  REF: {ref}")


def mode_batch(proc, samples_dir: str, manifest_csv: str, n: int, chunk_sec: float):
    """Run file streaming on N samples, compute WER, log summary table."""
    from tqdm import tqdm

    log.info("=" * 60)
    log.info(f"  BATCH STREAMING MODE  │  n={n}  chunk={chunk_sec}s")
    log.info(f"  Samples dir: {samples_dir}")
    log.info("=" * 60)

    # Load manifest if available
    ref_map = {}
    if manifest_csv and os.path.exists(manifest_csv):
        df = pd.read_csv(manifest_csv)
        ref_map = dict(zip(df["audio_path"].str.replace(".wav", "").str.split("/").str[-1],
                           df["transcript"]))

    wav_files = sorted(glob.glob(os.path.join(samples_dir, "*.wav")))[:n]
    if not wav_files:
        log.error(f"No WAV files found in {samples_dir}")
        return

    log.info(f"  Files found: {len(wav_files)}")

    results = []
    with tqdm(total=len(wav_files), desc="  Processing", unit="file",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for wav_path in wav_files:
            fid  = os.path.splitext(os.path.basename(wav_path))[0]
            ref  = ref_map.get(fid, None)
            audio = load_wav(wav_path)
            duration = len(audio) / SAMPLE_RATE

            chunk_sz = int(chunk_sec * SAMPLE_RATE)
            proc.init()
            all_parts, lats = [], []

            for i in range(max(1, len(audio) // chunk_sz) + 1):
                chunk = audio[i * chunk_sz:(i + 1) * chunk_sz]
                if len(chunk) < SAMPLE_RATE * 0.1:
                    break
                t0 = time.perf_counter()
                proc.insert_audio_chunk(chunk)
                beg, end, text = proc.process_iter()
                lats.append(time.perf_counter() - t0)
                if text and text.strip():
                    all_parts.append(text.strip())

            _, _, text = proc.finish()
            if text and text.strip():
                all_parts.append(text.strip())

            hyp = " ".join(all_parts)
            avg_lat = sum(lats) / len(lats) if lats else 0
            wer_val = compute_wer(ref, hyp) if ref else None

            results.append({
                "file": fid[:30], "dur_s": round(duration, 1),
                "avg_lat": round(avg_lat, 3), "wer": wer_val,
                "hyp": hyp[:60],
            })
            pbar.update(1)

    # Summary table
    df_res = pd.DataFrame(results)
    log.info("\n  ── Batch Results ──")
    log.info(f"  {'File':<30} {'Dur':>5} {'AvgLat':>8} {'WER':>7}")
    log.info(f"  {'─'*30} {'─'*5} {'─'*8} {'─'*7}")
    for _, r in df_res.iterrows():
        wer_s = f"{r['wer']:.1f}%" if r["wer"] is not None else "  n/a"
        log.info(f"  {r['file']:<30} {r['dur_s']:>4.1f}s {r['avg_lat']:>7.3f}s {wer_s:>7}")

    if df_res["wer"].notna().any():
        mean_wer = df_res["wer"].dropna().mean()
        log.info(f"\n  Mean streaming WER: {mean_wer:.2f}%")

    log.info(f"\n  Log saved → {LOG_FILE}")
    return df_res


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Whisper Streaming Demo")
    p.add_argument("--mode", choices=["mic", "file", "compare", "batch"],
                   default="mic")
    p.add_argument("--audio",       default=None,  help="WAV file (file/compare mode)")
    p.add_argument("--ref",         default=None,  help="Reference text for WER")
    p.add_argument("--model",       default="base")
    p.add_argument("--language",    default="en")
    p.add_argument("--chunk-sec",   type=float, default=1.0)
    p.add_argument("--max-sec",     type=int,   default=60,
                   help="Max mic recording seconds (mic mode, default: 60)")
    p.add_argument("--samples-dir", default=None,
                   help="Directory with WAV files (batch mode)")
    p.add_argument("--manifest",    default=None,
                   help="manifest.csv with transcripts (batch mode)")
    p.add_argument("--n",           type=int, default=10,
                   help="Number of files for batch mode (default: 10)")
    args = p.parse_args()

    log.info(f"Streaming Demo  |  mode={args.mode}  model={args.model}  "
             f"lang={args.language}")
    log.info(f"Log → {LOG_FILE}")

    proc = make_processor(args.model, args.language)

    if args.mode == "mic":
        mode_mic(proc, args.chunk_sec, args.max_sec)

    elif args.mode == "file":
        if not args.audio:
            p.error("--audio required for file mode")
        mode_file(proc, args.audio, args.chunk_sec, args.ref)

    elif args.mode == "compare":
        if not args.audio:
            p.error("--audio required for compare mode")
        mode_compare(proc, args.audio, args.chunk_sec, args.ref)

    elif args.mode == "batch":
        samples_dir = args.samples_dir or os.path.join(
            os.path.dirname(__file__), "..", "common_au_en_op", "samples"
        )
        manifest = args.manifest or os.path.join(
            os.path.dirname(__file__), "..", "common_au_en_op", "manifest.csv"
        )
        mode_batch(proc, samples_dir, manifest, args.n, args.chunk_sec)


if __name__ == "__main__":
    main()
