# Speech-to-Text (STT) Transcription Quality Improvement

An end-to-end pipeline to analyse and improve STT transcription quality using **open-source models only**.

---

## System Design

```
Audio Input ──► faster-whisper ──► Baseline Hypothesis
                                          │
                           ┌──────────────▼────────────────┐
                           │       Post-Processing Stack    │
                           │  1. Text Normalisation (regex) │
                           │  2. Vocabulary Biasing (dict)  │
                           │  3. LM Post-Correction (LLM)   │
                           └──────────────┬────────────────┘
                                          │
                           ┌──────────────▼────────────────┐
                           │  Evaluation: WER · CER · MER  │
                           └───────────────────────────────┘
```

| Component | Technology |
|-----------|-----------|
| ASR Engine | `faster-whisper` (CTranslate2) — tiny → large-v3 |
| Evaluation | `jiwer` (WER / CER / MER / WIL) |
| LM Correction | Ollama `qwen3.5:4b` (runs locally) |
| Audio I/O | `sounddevice` + `scipy` |
| Datasets | LibriSpeech, CommonVoice, LJSpeech |

---

## Project Structure

```
STT/
├── pipeline/                        # Main package (python -m pipeline)
│   ├── __init__.py
│   ├── __main__.py                  # Entry point
│   ├── config.py                    # PathConfig & PipelineConfig dataclasses
│   ├── cli.py                       # Argument parser + logging setup
│   ├── base.py                      # Step abstract base class
│   ├── runner.py                    # PipelineRunner (orchestrates steps)
│   ├── core/
│   │   ├── transcriber.py           # faster-whisper wrapper (lazy loading)
│   │   ├── evaluator.py             # WER / CER / MER / WIL scoring
│   │   ├── error_analyzer.py        # Substitution / deletion / insertion analysis
│   │   ├── improver.py              # 5-strategy post-processing
│   │   └── comparator.py           # Baseline vs improved comparison
│   └── steps/
│       ├── transcription.py         # Step 1: Baseline transcription
│       ├── evaluation.py            # Step 2: Baseline evaluation
│       ├── error_analysis.py        # Step 3: Error analysis
│       ├── improvement.py           # Step 4: Apply improvements
│       └── reevaluation.py          # Step 5: Compare before/after
├── src/                             # Legacy standalone scripts
│   ├── config.py
│   ├── 02_baseline_transcription.py
│   ├── 03_evaluation.py
│   ├── 04_error_analysis.py
│   ├── 05_improvement.py
│   ├── 06_reevaluation.py
│   ├── 07_streaming_demo.py
│   ├── run_pipeline.py
│   ├── download_librispeech.py
│   └── download_commonvoice.py
├── streaming/                       # Real-time transcription demos
│   ├── live_mic.py
│   ├── file_stream.py
│   └── whisper_online.py
├── notebooks/
│   └── STT_Pipeline.ipynb
├── data/
│   ├── manifest.csv                 # Audio file list + references
│   └── samples/                     # Audio clips
├── outputs/                         # Per-experiment results
└── requirements.txt
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/MuthuAjay/STT.git
cd STT
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama + pull model

```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen3.5:4b
```

### 3. Prepare a dataset

```bash
# Download LibriSpeech test-other (300 utterances, ~34 min audio) from HuggingFace
python src/download_librispeech.py --num-samples 300
# Audio saved to data/samples/, manifest written to data/manifest.csv
```

---

## Running the Pipeline

### Primary — `python -m pipeline`

```bash
# Full pipeline (all 5 steps)
python -m pipeline --manifest data/manifest.csv

# Skip LLM correction (faster)
python -m pipeline --manifest data/manifest.csv --no-lm

# Skip re-transcription with initial prompt
python -m pipeline --manifest data/manifest.csv --no-retranscribe

# Force re-run even if cached outputs exist
python -m pipeline --manifest data/manifest.csv --force

# Run only specific steps
python -m pipeline --manifest data/manifest.csv --steps 1 2 3

# Use a named experiment folder
python -m pipeline --manifest data/manifest.csv --experiment my_experiment
```

### Environment variable overrides

All settings can be controlled via env vars — no code changes needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_EXPERIMENT` | `common_au_en_op` | Experiment folder name under `outputs/` |
| `STT_MANIFEST` | `{outputs_dir}/manifest.csv` | Path to audio manifest CSV |
| `STT_MODEL` | `large-v3` | Whisper model size (`tiny` / `base` / `small` / `medium` / `large` / `large-v2` / `large-v3`) |
| `STT_COMPUTE` | `float16` | CTranslate2 compute type (`float16` / `float32` / `int8` / `int8_float16`) |
| `STT_SAMPLES` | `300` | Number of audio samples to process |
| `STT_OLLAMA` | `qwen3.5:4b` | Ollama model identifier |
| `STT_PROMPT` | *(domain terms)* | Initial prompt for Whisper vocabulary biasing |

```bash
STT_MODEL=large-v3 STT_COMPUTE=int8 STT_EXPERIMENT=run_01 \
  python -m pipeline --manifest data/manifest.csv
```

### Alternative — Jupyter Notebook

```bash
jupyter notebook notebooks/STT_Pipeline.ipynb
```

### Alternative — Legacy scripts

```bash
cd src
python run_pipeline.py               # Full pipeline
python run_pipeline.py --no-lm      # Skip LLM
python run_pipeline.py --force      # Force re-run
```

### Streaming / Live Demo

```bash
# Live microphone (4 s chunks, default)
python src/07_streaming_demo.py

# Voice-activity detection mode
python src/07_streaming_demo.py --mode vad

# Simulate streaming from a file
python src/07_streaming_demo.py --mode file --file path/to/audio.wav

# Custom chunk length
python src/07_streaming_demo.py --chunk-sec 3
```

---

## Pipeline Steps

| Step | Name | Description | Output |
|------|------|-------------|--------|
| 1 | Transcription | Transcribe all audio with faster-whisper | `baseline_results.csv` |
| 2 | Evaluation | Compute WER / CER / MER / WIL; show best & worst utterances | *(display only)* |
| 3 | Error Analysis | Analyse substitution / deletion / insertion patterns | `error_analysis.json`, `wer_distribution.png`, `top_substitutions.png` |
| 4 | Improvement | Apply 5-strategy post-processing | `improved_results.csv` |
| 5 | Re-evaluation | Stage-by-stage comparison; top improved utterances | `improvement_comparison.png` |

All outputs go to `outputs/{STT_EXPERIMENT}/`.

---

## Improvement Strategies

| # | Strategy | What it does |
|---|----------|-------------|
| 0 | **Re-transcription** | Re-runs Whisper with an `initial_prompt` to bias towards domain vocabulary (optional) |
| 1 | **Text Normalisation** | Removes trailing ellipsis / dashes, collapses spaces, fixes comma-conjunction artifacts |
| 2 | **Vocabulary Biasing** | Applies hard-coded corrections + auto-corrections derived from `error_analysis.json` (min count ≥ 5) |
| 3 | **LM Post-correction** | Sends hypotheses to a local Ollama LLM for grammar / context correction (optional) |
| 4 | **Lowercase** | Final normalisation to match reference casing convention |

---

## Results (LibriSpeech test-other, 300 utterances, whisper-base)

| Stage | WER (%) | CER (%) |
|-------|---------|---------|
| Baseline | 12.51 | 5.50 |
| After Text Normalisation | 12.51 | 5.50 |
| After Vocabulary Biasing | 12.44 | 5.41 |
| After LM Correction | 12.44 | 5.41 |

Best individual utterances show 14–17% WER reduction via vocabulary biasing
(e.g. number-word normalisation, proper-noun corrections).

---

## Dependencies

```
faster-whisper    # ASR engine (CTranslate2)
jiwer             # WER / CER / MER / WIL metrics
sounddevice       # Audio capture
scipy             # Audio processing
numpy
pandas
matplotlib
seaborn
tqdm
ollama            # Local LLM client
jupyter
ipykernel
```

All open-source. No external API keys required.
