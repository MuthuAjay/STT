# Real-Time Streaming STT — `streaming/`

Built on top of [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming), which implements a
**local-agreement policy** over faster-whisper to produce low-latency, stable partial transcripts.

---

## How it works

```
Audio chunks (every 1 s)
       │
       ▼
 OnlineASRProcessor          ← local-agreement algorithm
  ├── insert_audio_chunk()   ← buffer audio
  ├── process_iter()         ← run Whisper, compare new/old hypothesis
  │     returns (beg, end, committed_text)
  └── finish()               ← flush remaining
```

The processor only emits text once two consecutive Whisper passes **agree** on the same tokens,
avoiding the hallucination "flicker" of naive streaming.
VAD (Silero) is enabled to skip silence, reducing unnecessary inference.

---

## Files

| File | Description |
|------|-------------|
| `whisper_online.py` | Core streaming engine from ufal/whisper_streaming |
| `silero_vad_iterator.py` | Silero VAD helper (used by VAC mode) |
| `live_mic.py` | Live microphone demo |
| `file_stream.py` | File-based streaming simulation |
| `streaming_demo.py` | **Unified runner** (mic / file / compare / batch) |
| `logs/` | Auto-created per-session log files |

---

## Quick start

```bash
# From repo root (activate venv first)
source .venv/bin/activate
pip install sounddevice scipy tqdm  # if not already installed

# 1. Live mic — transcribe in real time (60 s, Ctrl+C to stop early)
python streaming/streaming_demo.py --mode mic --model base

# 2. File streaming — simulate real-time on a WAV file
python streaming/streaming_demo.py --mode file \
    --audio common_au_en_op/samples/common_voice_en_111370.wav

# 3. Compare — streaming vs offline batch on same file
python streaming/streaming_demo.py --mode compare \
    --audio common_au_en_op/samples/common_voice_en_111370.wav \
    --ref "The apples are in the basket near the door on your way out."

# 4. Batch — run on N samples and compute mean WER
python streaming/streaming_demo.py --mode batch \
    --samples-dir common_au_en_op/samples \
    --manifest   common_au_en_op/manifest.csv \
    --n 50
```

Or use the individual scripts directly:

```bash
# Live mic (simple)
python streaming/live_mic.py --model base --chunk-sec 0.5 --vad

# File streaming (verbose per-chunk output)
python streaming/file_stream.py common_au_en_op/samples/common_voice_en_111370.wav \
    --ref "The apples are in the basket near the door on your way out."

# Offline batch comparison
python streaming/file_stream.py audio.wav --offline
```

---

## Arguments (`streaming_demo.py`)

| Arg | Default | Description |
|-----|---------|-------------|
| `--mode` | `mic` | `mic` / `file` / `compare` / `batch` |
| `--audio` | — | WAV file path (file/compare modes) |
| `--ref` | — | Reference text for WER |
| `--model` | `base` | Whisper model size |
| `--language` | `en` | Language code |
| `--chunk-sec` | `1.0` | Chunk size in seconds |
| `--max-sec` | `60` | Max mic recording seconds |
| `--samples-dir` | — | Directory of WAVs (batch mode) |
| `--manifest` | — | manifest.csv with transcripts (batch mode) |
| `--n` | `10` | Number of files for batch mode |

---

## Sample results (Common Voice AU-EN, base model, 10 files)

```
File                             Dur   AvgLat     WER
────────────────────────────── ───── ──────── ───────
common_voice_en_111370          3.6s   0.063s    0.0%
common_voice_en_111396          3.6s   0.034s    0.0%
common_voice_en_14233970        2.5s   0.030s    0.0%
common_voice_en_14705319        1.5s   0.023s    0.0%
common_voice_en_14767           3.7s   0.026s    0.0%
common_voice_en_149445          4.9s   0.033s    6.2%
common_voice_en_149579          4.2s   0.050s    0.0%
common_voice_en_15733296        4.3s   0.048s    0.0%
common_voice_en_15734613        4.1s   0.030s    0.0%
common_voice_en_15735116        4.8s   0.081s    0.0%

Mean streaming WER : 0.62%
Overall RTF        : ~0.07x  (14× faster than real time)
Avg chunk latency  : ~0.04 s
```

---

## Dependencies

```
faster-whisper
sounddevice       # mic input
scipy             # WAV loading / resampling
tqdm              # progress bars
jiwer             # WER computation (batch mode)
pandas            # batch results table
```

CUDA is auto-detected. Falls back to CPU float32 if unavailable.
