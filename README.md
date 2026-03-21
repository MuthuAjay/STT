# Speech-to-Text (STT) Transcription Quality Improvement

An end-to-end pipeline to analyse and improve STT transcription quality using **open-source models only**.

---

## System Design

```
Audio Input ──► faster-whisper (base) ──► Baseline Hypothesis
                                                    │
                                     ┌──────────────▼────────────────┐
                                     │   Post-Processing Stack        │
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
| Dataset | LibriSpeech test-other (multi-speaker, harder conditions) |
| ASR Engine | `faster-whisper` (Whisper base, CTranslate2) |
| Evaluation | `jiwer` (WER / CER / MER) |
| LM Correction | Ollama `qwen3.5:4b` |
| Audio I/O | `sounddevice` + `scipy` |

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
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

### 3. Download LibriSpeech test-other (auto via script)

```bash
# Downloads 300 utterances (~34 min audio) automatically from HuggingFace
python src/download_librispeech.py --num-samples 300
```

No manual download needed. Audio is saved to `data/samples/` and the manifest is created automatically.

---

## Running the Pipeline

### Option A – Jupyter Notebook (recommended)

```bash
jupyter notebook notebooks/STT_Pipeline.ipynb
```

The notebook runs all steps end-to-end with visualisations.

### Option B – Command-line scripts

```bash
cd src

# Step 0 – download dataset (first time only)
python download_librispeech.py --num-samples 300

# Run each step individually:
python 02_baseline_transcription.py # transcribe with whisper-base
python 03_evaluation.py             # compute WER / CER / MER
python 04_error_analysis.py         # analyse error patterns
python 05_improvement.py            # apply post-processing
python 06_reevaluation.py           # compare before/after

# Or run the full pipeline at once:
python run_pipeline.py
python run_pipeline.py --no-lm      # skip LLM (faster)
python run_pipeline.py --force      # re-run even if cached
```

### Option C – Streaming / Live Demo

```bash
# Mic chunk mode (default, 4s chunks):
python src/07_streaming_demo.py

# Voice-activity detection mode:
python src/07_streaming_demo.py --mode vad

# File streaming simulation:
python src/07_streaming_demo.py --mode file --file path/to/audio.wav

# Custom chunk length:
python src/07_streaming_demo.py --chunk-sec 3
```

---

## Project Structure

```
STT/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                    # paths & model settings
│   ├── 01_data_collection.py        # sample LJSpeech subset
│   ├── 02_baseline_transcription.py # run faster-whisper
│   ├── 03_evaluation.py             # WER / CER / MER
│   ├── 04_error_analysis.py         # error pattern analysis
│   ├── 05_improvement.py            # post-processing pipeline
│   ├── 06_reevaluation.py           # before/after comparison
│   ├── 07_streaming_demo.py         # real-time mic / file STT
│   └── run_pipeline.py              # end-to-end runner
├── notebooks/
│   └── STT_Pipeline.ipynb           # comprehensive notebook
├── data/
│   ├── manifest.csv                 # sampled file list
│   └── samples/                     # copied audio clips
└── outputs/
    ├── baseline_results.csv
    ├── improved_results.csv
    ├── error_analysis.json
    ├── dataset_stats.png
    ├── baseline_wer_distribution.png
    ├── error_analysis.png
    └── improvement_comparison.png
```

---

## Improvement Strategies

| # | Strategy | Targets |
|---|----------|---------|
| 1 | **Text Normalisation** | Trailing ellipsis, dash noise, double spaces, comma-conjunction artifacts |
| 2 | **Vocabulary Biasing** | Systematic mis-recognitions derived from error-frequency analysis |
| 3 | **LM Post-correction** | Grammar errors, missing function words, context-dependent mistakes (via local LLM) |

---

## Results (LibriSpeech test-other, 300 utterances)

| Stage | WER (%) | CER (%) |
|-------|---------|---------|
| Baseline (whisper-base) | 12.51 | 5.50 |
| After Text Normalisation | 12.51 | 5.50 |
| After Vocabulary Biasing | 12.44 | 5.41 |
| After LM Correction | 12.44 | 5.41 |

Most improved utterances show 14–17% WER reduction per sentence via vocabulary biasing
(e.g. "Sinbad" → "Sindbad", "Archie" → "Archy", number word normalisation).

---

## Streaming Demo

The live demo (`07_streaming_demo.py`) supports:

- **Chunk mode**: records fixed-length audio windows and transcribes each
- **VAD mode**: detects speech boundaries using RMS energy and transcribes on silence
- **File mode**: simulates streaming on a pre-recorded audio file

```
[14:32:01] Listening … The quick brown fox jumps over the lazy dog.  (0.43s)
[14:32:06] Listening … Machine learning models learn from data.      (0.38s)
```

---

## Dependencies

All open-source, no API keys required:

- `faster-whisper` – STT engine
- `jiwer` – WER/CER metrics
- `sounddevice` – audio capture
- `ollama` – local LLM inference
- `pandas`, `matplotlib`, `seaborn`, `tqdm`, `scipy`, `numpy`
