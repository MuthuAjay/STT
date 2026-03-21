"""
Enhanced Streaming Demo — Pipeline-Improved
============================================
Extends streaming_demo.py with three improvements from the offline pipeline:

  1. domain initial_prompt  — biases Whisper decoder toward technical vocabulary
                              (DeepSeekMoE, GShard, ZeRO, MoE, ...)
  2. StreamingPostProcessor — applies text normalisation + vocab bias to every
                              committed chunk before display / scoring
  3. Richer WER reporting   — WER / CER / MER / WIL via pipeline Evaluator
                              (batch/compare modes)

All four modes are supported:
    mic      — live microphone
    file     — simulate streaming from a WAV file
    compare  — streaming vs offline on the same file (shows WER delta)
    batch    — process N samples from a manifest, full metrics table

Usage:
    # Live mic with domain prompt + post-processing
    python streaming/enhanced_demo.py --mode mic --model large-v3

    # File mode with reference WER
    python streaming/enhanced_demo.py --mode file \\
        --audio path/to/file.wav --ref "ground truth" --model large-v3

    # Batch — DeepSeekMoE synthetic data
    python streaming/enhanced_demo.py --mode batch \\
        --manifest synthetic_data_generation/deepseekMOE_v2/manifest.csv \\
        --samples-dir synthetic_data_generation/deepseekMOE_v2/audio \\
        --error-json deepseekMOE_v3/error_analysis.json \\
        --model large-v3 --n 20

Env var overrides (same as pipeline/):
    STT_PROMPT="YourTerms"   override default initial_prompt
    STT_EXPERIMENT=name      pick up error_analysis.json from that experiment folder
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT    = Path(__file__).resolve().parent.parent
_STREAM_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(_STREAM_DIR))   # whisper_online, post_processor
sys.path.insert(0, str(_REPO_ROOT))    # pipeline/

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_DIR = _STREAM_DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"enhanced_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers = [
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("enhanced_demo")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

SAMPLE_RATE = 16000

# ── Default domain initial_prompt ────────────────────────────────────────────
_DEFAULT_PROMPT = os.environ.get(
    "STT_PROMPT",
    (
        # ── Model names & sizes (highest mis-recognition rate — keep first) ──
        "DeepSeekMoE, DeepSeekMoE-2B, DeepSeekMoE-16B, DeepSeekMoE-145B, "
        "DeepSeek-MoE, GShard, GShard-137B, ZeRO, ZeRO-Offload, DeepSpeed, "

        # ── MoE architecture terms ──────────────────────────────────────────
        "MoE, mixture-of-experts, routed experts, shared experts, "
        "fine-grained expert segmentation, shared expert isolation, "
        "expert specialization, activated parameters, expert parameters, "
        "top-K routing, load balancing, knowledge redundancy, knowledge hybridity, "

        # ── Model architecture & training ───────────────────────────────────
        "Transformer, feed-forward network, FFN, hidden dimension, "
        "warmup scheduler, learning rate scheduler, cosine annealing, "

        # ── Benchmarks (proper nouns — Whisper often mishears these) ────────
        "Pile loss, HellaSwag, MMLU, TriviaQA, ARC, "

        # ── Common paper terms ──────────────────────────────────────────────
        "large language model, LLM, pre-training, fine-tuning, parameter efficiency, PyTorch."
    ),
)

# Token budget note: Whisper hard-truncates initial_prompt at 224 tokens.
# The expanded prompt above is ~130 tokens — safely within budget.
# Terms are ordered by mis-recognition priority (most impactful first).


# ── Audio loader ──────────────────────────────────────────────────────────────

def load_wav(path: str) -> np.ndarray:
    import librosa
    data, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return data


# ── ASR factory ───────────────────────────────────────────────────────────────

def make_processor(model_size: str, language: str, initial_prompt: str | None,
                   silence_ms: int = 2000):
    """Build OnlineASRProcessor with initial_prompt and configurable VAD silence.

    Args:
        silence_ms: Milliseconds of silence before a speech segment is
                    committed.  faster-whisper default is 2000 ms (2 s).
                    Lower values (e.g. 500) give snappier commits but may
                    cut mid-sentence.  Higher values wait longer before
                    flushing, useful for slower or pausy speech.
    """
    from whisper_online import FasterWhisperASR, OnlineASRProcessor
    import faster_whisper

    _prompt     = initial_prompt   # capture in closure
    _silence_ms = silence_ms       # capture in closure

    class EnhancedFasterWhisperASR(FasterWhisperASR):
        """Subclass that injects initial_prompt and VAD silence into every transcribe call."""

        def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
            size = model_dir or modelsize
            try:
                return faster_whisper.WhisperModel(
                    size, device="cuda", compute_type="float16"
                )
            except Exception:
                log.info("  CUDA unavailable — falling back to CPU (float32)")
                return faster_whisper.WhisperModel(
                    size, device="cpu", compute_type="float32"
                )

        def transcribe(self, audio, init_prompt=""):
            merged = " ".join(filter(None, [_prompt, init_prompt])).strip()
            # Inject vad_parameters so silence threshold is honoured
            self.transcribe_kargs["vad_parameters"] = {
                "min_silence_duration_ms": _silence_ms,
            }
            return super().transcribe(audio, init_prompt=merged)

    log.info("  Loading faster-whisper '%s' (lang=%s) …", model_size, language)
    log.info("  VAD silence threshold : %d ms", silence_ms)
    if initial_prompt:
        log.info("  initial_prompt: %s…", initial_prompt[:80])

    asr  = EnhancedFasterWhisperASR(language, modelsize=model_size)
    asr.use_vad()
    proc = OnlineASRProcessor(
        asr, tokenizer=None, buffer_trimming=("segment", 15),
        logfile=open(os.devnull, "w"),
    )
    log.info("  Model ready.")
    return proc


# ── WER helper ────────────────────────────────────────────────────────────────

def _score(ref: str, hyp: str) -> dict:
    """Return WER/CER/MER/WIL using the pipeline Evaluator."""
    try:
        from pipeline.core import Evaluator
        ev      = Evaluator()
        metrics = ev.score([ref], [hyp])
        return {"wer": metrics.wer, "cer": metrics.cer,
                "mer": metrics.mer, "wil": metrics.wil}
    except ImportError:
        # Fallback to basic WER if pipeline not on path
        from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
        T = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
        return {"wer": round(wer(T(ref), T(hyp)) * 100, 2)}


# ── Modes ─────────────────────────────────────────────────────────────────────

def mode_mic(proc, pp, chunk_sec: float, max_sec: int):
    import queue
    import sounddevice as sd

    log.info("=" * 60)
    log.info("  LIVE MIC MODE  (enhanced)")
    log.info("  chunk=%.1fs  |  max=%ds  |  Ctrl+C to stop", chunk_sec, max_sec)
    log.info("=" * 60)

    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(chunk_sec * SAMPLE_RATE)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    proc.init()
    t_start  = time.perf_counter()
    n_chunks = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=chunk_samples, callback=callback):
        log.info("  Listening … (speak now)")
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
                _, _, text = proc.process_iter()
                n_chunks += 1

                if text and text.strip():
                    cleaned = pp.process(text.strip())
                    ts = datetime.now().strftime("%H:%M:%S")
                    log.info("  [%s]  RAW: %s", ts, text.strip())
                    if cleaned != text.strip():
                        log.info("  [%s]  ENH: %s", ts, cleaned)

        except KeyboardInterrupt:
            log.info("  Stopped by user.")
        finally:
            _, _, text = proc.finish()
            if text and text.strip():
                cleaned = pp.process(text.strip())
                log.info("  [final] %s", cleaned)

    elapsed = time.perf_counter() - t_start
    log.info("\n  Session: %.1fs  |  Chunks: %d  |  Log: %s",
             elapsed, n_chunks, _LOG_FILE)


def mode_file(proc, pp, audio_path: str, chunk_sec: float,
              ref: str | None = None) -> tuple[str, str, float, float]:
    """Returns (raw_hyp, enhanced_hyp, wall_time, duration)."""
    from tqdm import tqdm

    audio    = load_wav(audio_path)
    duration = len(audio) / SAMPLE_RATE
    chunk_sz = int(chunk_sec * SAMPLE_RATE)
    n_chunks = max(1, len(audio) // chunk_sz)

    log.info("=" * 60)
    log.info("  FILE STREAMING  (enhanced)  │  %s", os.path.basename(audio_path))
    log.info("  Duration: %.1fs  │  Chunk: %.1fs  │  Chunks: %d",
             duration, chunk_sec, n_chunks)
    log.info("=" * 60)

    proc.init()
    raw_parts, enh_parts, latencies = [], [], []
    t_wall = time.perf_counter()

    with tqdm(total=n_chunks, desc="  Streaming", unit="chunk",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for i in range(n_chunks + 1):
            chunk = audio[i * chunk_sz:(i + 1) * chunk_sz]
            if len(chunk) < SAMPLE_RATE * 0.1:
                break

            t0 = time.perf_counter()
            proc.insert_audio_chunk(chunk)
            _, _, text = proc.process_iter()
            lat = time.perf_counter() - t0
            latencies.append(lat)

            pbar.update(1)
            pbar.set_postfix({"lat": f"{lat:.2f}s",
                              "RTF": f"{lat/(len(chunk)/SAMPLE_RATE):.3f}"})

            if text and text.strip():
                raw   = text.strip()
                clean = pp.process(raw)
                log.info("  [%.1fs]  ENH: %s", i * chunk_sec, clean)
                raw_parts.append(raw)
                enh_parts.append(clean)

    _, _, text = proc.finish()
    if text and text.strip():
        raw   = text.strip()
        clean = pp.process(raw)
        log.info("  [flush]  ENH: %s", clean)
        raw_parts.append(raw)
        enh_parts.append(clean)

    raw_hyp = " ".join(raw_parts)
    enh_hyp = " ".join(enh_parts)
    total_wall = time.perf_counter() - t_wall

    log.info("\n  ── Transcripts ──")
    log.info("  RAW: %s", raw_hyp)
    log.info("  ENH: %s", enh_hyp)
    log.info("\n  Wall: %.2fs  (audio: %.1fs)  Avg lat: %.3fs  RTF: %.3fx",
             total_wall, duration,
             sum(latencies) / len(latencies),
             total_wall / duration)

    if ref:
        raw_s = _score(ref, raw_hyp)
        enh_s = _score(ref, enh_hyp)
        log.info("\n  ── WER Comparison ──")
        log.info("  %-10s  %s", "Raw",     _fmt_scores(raw_s))
        log.info("  %-10s  %s", "Enhanced", _fmt_scores(enh_s))
        log.info("  REF: %s", ref)

    return raw_hyp, enh_hyp, total_wall, duration


def mode_compare(proc, pp, audio_path: str, chunk_sec: float, ref: str | None):
    log.info("=" * 60)
    log.info("  COMPARE MODE  │  streaming vs offline")
    log.info("=" * 60)

    audio    = load_wav(audio_path)
    duration = len(audio) / SAMPLE_RATE

    log.info("\n  [1/2] Streaming …")
    raw_stream, enh_stream, t_stream, _ = mode_file(
        proc, pp, audio_path, chunk_sec, ref=None
    )

    log.info("\n  [2/2] Offline (batch) …")
    proc.init()
    t0 = time.perf_counter()
    proc.insert_audio_chunk(audio)
    _, _, text = proc.finish()
    t_offline  = time.perf_counter() - t0
    raw_offline = text.strip() if text else ""
    enh_offline = pp.process(raw_offline)
    log.info("  RAW: %s", raw_offline)
    log.info("  ENH: %s", enh_offline)
    log.info("  Time: %.2fs  RTF=%.3fx", t_offline, t_offline / duration)

    log.info("\n  ── Comparison ──")
    header = f"  {'Mode':<18} {'Time':>7}  {'RTF':>6}  {'WER (raw)':>10}  {'WER (enh)':>10}"
    log.info(header)
    log.info("  " + "─" * (len(header) - 2))
    for name, raw_h, enh_h, t in [
        ("Streaming", raw_stream, enh_stream, t_stream),
        ("Offline",   raw_offline, enh_offline, t_offline),
    ]:
        rtf   = t / duration
        raw_w = f"{_score(ref, raw_h)['wer']:.2f}%" if ref else "n/a"
        enh_w = f"{_score(ref, enh_h)['wer']:.2f}%" if ref else "n/a"
        log.info("  %-18s %6.2fs  %6.3f  %10s  %10s",
                 name, t, rtf, raw_w, enh_w)

    if ref:
        log.info("\n  REF: %s", ref)


def mode_batch(proc, pp, samples_dir: str, manifest_csv: str,
               n: int, chunk_sec: float):
    from tqdm import tqdm

    log.info("=" * 60)
    log.info("  BATCH STREAMING MODE  (enhanced)  │  n=%d  chunk=%.1fs", n, chunk_sec)
    log.info("=" * 60)

    ref_map: dict[str, str] = {}
    if manifest_csv and os.path.exists(manifest_csv):
        df = pd.read_csv(manifest_csv)
        ref_map = dict(zip(
            df["audio_path"].apply(lambda p: Path(p).stem),
            df["transcript"],
        ))

    wav_files = sorted(glob.glob(os.path.join(samples_dir, "*.wav")))[:n]
    if not wav_files:
        log.error("No WAV files found in %s", samples_dir)
        return

    log.info("  Files found: %d", len(wav_files))
    results = []

    with tqdm(total=len(wav_files), desc="  Processing", unit="file",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for wav_path in wav_files:
            fid      = Path(wav_path).stem
            ref      = ref_map.get(fid)
            audio    = load_wav(wav_path)
            duration = len(audio) / SAMPLE_RATE
            chunk_sz = int(chunk_sec * SAMPLE_RATE)

            proc.init()
            raw_parts, enh_parts, lats = [], [], []

            for i in range(max(1, len(audio) // chunk_sz) + 1):
                chunk = audio[i * chunk_sz:(i + 1) * chunk_sz]
                if len(chunk) < SAMPLE_RATE * 0.1:
                    break
                t0 = time.perf_counter()
                proc.insert_audio_chunk(chunk)
                _, _, text = proc.process_iter()
                lats.append(time.perf_counter() - t0)
                if text and text.strip():
                    raw_parts.append(text.strip())
                    enh_parts.append(pp.process(text.strip()))

            _, _, text = proc.finish()
            if text and text.strip():
                raw_parts.append(text.strip())
                enh_parts.append(pp.process(text.strip()))

            raw_hyp = " ".join(raw_parts)
            enh_hyp = " ".join(enh_parts)
            avg_lat = sum(lats) / len(lats) if lats else 0.0

            raw_wer = _score(ref, raw_hyp)["wer"] if ref else None
            enh_wer = _score(ref, enh_hyp)["wer"] if ref else None

            results.append({
                "file":    fid[:28],
                "dur_s":   round(duration, 1),
                "avg_lat": round(avg_lat, 3),
                "raw_wer": raw_wer,
                "enh_wer": enh_wer,
                "delta":   round(raw_wer - enh_wer, 2) if (raw_wer and enh_wer) else None,
            })
            pbar.update(1)

    df_res = pd.DataFrame(results)
    log.info("\n  ── Batch Results ──")
    log.info("  %-28s %5s %8s %9s %9s %7s",
             "File", "Dur", "AvgLat", "WER-raw", "WER-enh", "Delta")
    log.info("  " + "─" * 75)
    for _, r in df_res.iterrows():
        rw = f"{r['raw_wer']:.1f}%" if r["raw_wer"] is not None else "  n/a"
        ew = f"{r['enh_wer']:.1f}%" if r["enh_wer"] is not None else "  n/a"
        dw = f"▼{r['delta']:.1f}%"  if r["delta"]   is not None else "  n/a"
        log.info("  %-28s %4.1fs %7.3fs %9s %9s %7s",
                 r["file"], r["dur_s"], r["avg_lat"], rw, ew, dw)

    notna = df_res["raw_wer"].notna()
    if notna.any():
        mean_raw = df_res.loc[notna, "raw_wer"].mean()
        mean_enh = df_res.loc[notna, "enh_wer"].mean()
        log.info("\n  Mean WER   raw: %.2f%%   enhanced: %.2f%%   Δ %.2f%%",
                 mean_raw, mean_enh, mean_raw - mean_enh)

    log.info("  Log → %s", _LOG_FILE)
    return df_res


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_scores(scores: dict) -> str:
    parts = [f"WER={scores['wer']:.2f}%"]
    if "cer" in scores:
        parts += [f"CER={scores['cer']:.2f}%",
                  f"MER={scores['mer']:.2f}%",
                  f"WIL={scores['wil']:.2f}%"]
    return "  ".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enhanced Streaming Demo — pipeline-improved",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode",        choices=["mic", "file", "compare", "batch"],
                   default="mic")
    p.add_argument("--audio",       default=None,  help="WAV file (file/compare mode)")
    p.add_argument("--ref",         default=None,  help="Reference text for WER")
    p.add_argument("--model",       default="large-v3")
    p.add_argument("--language",    default="en")
    p.add_argument("--chunk-sec",   type=float, default=1.0)
    p.add_argument("--max-sec",     type=int,   default=60)
    p.add_argument("--samples-dir", default=None)
    p.add_argument("--manifest",    default=None,
                   help="manifest.csv (batch mode) — used for reference transcripts")
    p.add_argument("--n",           type=int, default=10)
    p.add_argument("--error-json",  default=None,
                   help="Path to error_analysis.json for auto-corrections")
    p.add_argument("--no-prompt",   action="store_true",
                   help="Disable domain initial_prompt")
    p.add_argument("--silence-ms",  type=int, default=2000,
                   help="Milliseconds of silence before a speech segment is "
                        "committed by the VAD (default: 2000). Lower = faster "
                        "commits, higher = waits longer before flushing.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    initial_prompt = None if args.no_prompt else _DEFAULT_PROMPT

    # Resolve error_analysis.json — explicit flag > STT_EXPERIMENT env folder
    error_json: Path | None = None
    if args.error_json:
        error_json = Path(args.error_json)
    else:
        exp = os.environ.get("STT_EXPERIMENT")
        if exp:
            candidate = _REPO_ROOT / exp / "error_analysis.json"
            if candidate.exists():
                error_json = candidate

    log.info("Enhanced Streaming Demo  |  mode=%s  model=%s  lang=%s",
             args.mode, args.model, args.language)
    log.info("Log → %s", _LOG_FILE)

    from post_processor import StreamingPostProcessor
    pp   = StreamingPostProcessor(error_json_path=error_json)
    proc = make_processor(args.model, args.language, initial_prompt,
                          silence_ms=args.silence_ms)

    if args.mode == "mic":
        mode_mic(proc, pp, args.chunk_sec, args.max_sec)

    elif args.mode == "file":
        if not args.audio:
            sys.exit("--audio required for file mode")
        mode_file(proc, pp, args.audio, args.chunk_sec, args.ref)

    elif args.mode == "compare":
        if not args.audio:
            sys.exit("--audio required for compare mode")
        mode_compare(proc, pp, args.audio, args.chunk_sec, args.ref)

    elif args.mode == "batch":
        samples_dir = args.samples_dir or str(
            _REPO_ROOT / "common_au_en_op" / "samples"
        )
        manifest = args.manifest or str(
            _REPO_ROOT / "common_au_en_op" / "manifest.csv"
        )
        mode_batch(proc, pp, samples_dir, manifest, args.n, args.chunk_sec)


if __name__ == "__main__":
    main()
