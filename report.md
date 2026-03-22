# STT Pipeline — Technical Report

**End-to-end Speech-to-Text quality improvement system**
*From synthetic data creation to real-time streaming*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dependencies](#2-dependencies)
3. [Phase 1 — Synthetic Data Generation](#3-phase-1--synthetic-data-generation)
4. [Phase 2 — Offline Pipeline](#4-phase-2--offline-pipeline)
   - 4.1 Configuration
   - 4.2 Step 1: Baseline Transcription
   - 4.3 Step 2: Evaluation
   - 4.4 Step 3: Error Analysis
   - 4.5 Step 4: Improvement Strategies
   - 4.6 Step 5: Re-evaluation & Visualisation
5. [Phase 3 — Real-Time Streaming](#5-phase-3--real-time-streaming)
   - 5.1 ASR Backends
   - 5.2 Local-Agreement Buffering
   - 5.3 Voice Activity Detection
   - 5.4 Streaming Post-Processor
   - 5.5 Enhanced Streaming Demo
6. [Data Flow Diagrams](#6-data-flow-diagrams)
7. [Results](#7-results)
8. [Key Design Decisions](#8-key-design-decisions)

---

## 1. System Overview

This system addresses a fundamental limitation of general-purpose ASR models like Whisper: they are trained on broad internet audio (podcasts, YouTube, conversational speech) and struggle with domain-specific technical language found in research papers — compound proper nouns (`DeepSeekMoE`, `GShard`), technical abbreviations (`MoE`, `FFN`, `KV-cache`), and mixed alphanumeric tokens (`top-K routing`, `DeepSeekMoE-16B`).

The solution is a three-phase pipeline:

| Phase | What it does |
|---|---|
| **Synthetic Data Generation** | Extracts sentences from ML research paper PDFs and synthesises audio with a local TTS model, producing perfectly aligned audio-transcript pairs |
| **Offline Pipeline** | Transcribes, evaluates, analyses errors, applies improvement strategies, and re-evaluates |
| **Real-Time Streaming** | Runs lightweight post-processing improvements in real time with sub-second latency |

All components are **fully open-source** — no API keys required.

---

## 2. Dependencies

| Package | Version | Role |
|---|---|---|
| `faster-whisper` | ≥ 1.0.0 | ASR engine — CTranslate2-optimised Whisper |
| `jiwer` | ≥ 3.0.0 | WER / CER / MER / WIL computation |
| `sounddevice` | ≥ 0.4.6 | Real-time audio capture (microphone) |
| `scipy` | ≥ 1.10.0 | WAV file I/O and resampling |
| `numpy` | ≥ 1.24.0 | Audio array manipulation |
| `pandas` | ≥ 2.0.0 | Tabular results (manifest, baseline, improved CSVs) |
| `matplotlib` | ≥ 3.7.0 | Plotting (WER distributions, comparison charts) |
| `seaborn` | ≥ 0.12.0 | Statistical chart styling |
| `tqdm` | ≥ 4.65.0 | Progress bars for batch transcription |
| `jupyter` | ≥ 1.0.0 | Interactive notebook exploration |
| `ipykernel` | ≥ 6.0.0 | Jupyter kernel |
| `PyMuPDF` (fitz) | — | PDF text extraction for synthetic data |
| `fish-speech` | — | TTS engine for audio synthesis (local, GPU) |
| `Silero VAD` | — | Voice activity detection in streaming |

---

## 3. Phase 1 — Synthetic Data Generation

**Entry point**: `synthetic_data_generation/pdf_to_synthetic_data.py`

### Why Synthetic Data?

Standard audio datasets (LibriSpeech, CommonVoice) contain none of the technical vocabulary that causes errors with domain-specific content. Rather than collecting real recordings, synthetic audio is generated directly from research paper text, producing:
- Perfectly aligned audio-transcript pairs
- Exactly the vocabulary the model needs to handle
- Acoustic variety through 5 prosody styles

### 3.1 Stage 1 — PDF Text Extraction

The pipeline begins by loading a research paper PDF using PyMuPDF (`fitz`). Raw extraction introduces two classes of artifact that must be repaired before any further processing.

**Ligature Repair (P0)**

PDF renderers drop font ligature glyph mappings, producing broken character sequences:

| Broken (PDF artifact) | Corrected |
|---|---|
| `efciency` | `efficiency` |
| `signicantly` | `significantly` |
| `dened` | `defined` |
| `rstly` | `firstly` |
| `oating` | `floating` |
| `exibility` | `flexibility` |
| `ofexperts` | `of experts` |

Over **100 regex patterns** cover `fi`, `fl`, `ffi`, `ffl`, and compound ligature sequences. This step runs before any sentence splitting — a single corrupted word in a reference transcript would poison both the TTS audio and the ground truth.

**Typo Repair (P0b)**

Common OCR-style character transpositions are fixed: `wtih → with`, `teh → the`, etc.

### 3.2 Stage 2 — Sentence Cleaning and Filtering

The repaired raw text is split into candidate sentences and subjected to a multi-pass filter.

**Split and Clean**

- Split on sentence boundaries (`.`, `!`, `?`)
- Normalise whitespace
- Remove non-ASCII characters
- Remove brackets and their contents
- Expand intra-word hyphens to spaces

**Skip Filter (`_RE_SKIP` regex)**

Each candidate sentence is rejected if it matches any of the following:

| Pattern | Reason |
|---|---|
| Blank / whitespace only | Empty |
| Standalone page numbers | Not speech |
| URLs (`http://`, `www.`) | Not speech |
| Citation fragments (`et al.`, `[1,2]`) | Not speech |
| Figure/table/algorithm captions | Noise |
| arXiv metadata | Not speech |
| Math equations (`=`, `<`, `>` dominance) | TTS-unfriendly |
| Table rows (consecutive ALL-CAPS tokens) | Not speech |
| Parenthetical citations `(Author, 2024)` | Noise |
| Merged number ranges `1.07 1.29` | Noise |
| Model name lists `MoE 4, MoE 32` | Noise |

**Symbol Density Filter**

Sentences where more than 20% of tokens are non-alphabetic are discarded. This removes equation-heavy lines that slip past the regex filter.

**Domain Keyword Filter**

Only sentences containing at least one term from the ML-domain keyword list are kept. Keywords include: `model`, `layer`, `attention`, `transformer`, `training`, `expert`, `routing`, `parameter`, `architecture`, etc. This ensures the synthesised dataset focuses on technically relevant vocabulary.

**Word Count Bounds**

- Minimum: **6 words** — avoids degenerate single-phrase utterances
- Maximum: **60 words** — avoids overly long utterances that cause TTS quality degradation

**Deduplication**

Two-column PDF layouts repeat text across columns. A normalised-text hash deduplicates before synthesis.

### 3.3 Stage 3 — TTS Synthesis (Fish Audio S2 Pro)

**Model selection**

| Alternative | Reason rejected |
|---|---|
| Festival / eSpeak | Robotic quality, poor prosody |
| Google Cloud TTS | Requires API key |
| Coqui TTS | Limited prosody control |
| **Fish Audio S2 Pro** | Natural speech, inline prosody tags, fully local, no API key |

Fish Audio S2 Pro runs locally on GPU (CUDA float16) or CPU (float32) with a `MAX_SEQ_LEN=4096` cap (limits KV cache to ~2.3 GB VRAM).

**Inference parameters**

| Parameter | Value | Effect |
|---|---|---|
| `temperature` | 0.8 | Controlled naturalness |
| `top_p` | 0.8 | Nucleus sampling |
| `repetition_penalty` | 1.1 | Prevents stuck loops |
| `max_new_tokens` | 1024 | Caps generation length |

**Prosody Variety**

To prevent the model from over-fitting to a single speaking style, 5 prosody tags are cycled across utterances:

| Index | Tag | Character |
|---|---|---|
| 0 | *(none)* | Neutral / default |
| 1 | `[emphasis]` | Slightly stressed |
| 2 | `[low voice]` | Quieter, intimate |
| 3 | `[excited tone]` | Energetic |
| 4 | `[professional broadcast tone]` | Clean, broadcast-style |

**Audio format**

Fish Audio S2 Pro outputs 44.1 kHz. Audio is downsampled to **16 kHz mono** for Whisper compatibility before saving.

### 3.4 Stage 4 — Manifest

Each synthesised utterance is written as a row in `manifest.csv`:

| Column | Content |
|---|---|
| `audio_path` | Absolute path to WAV file |
| `transcript` | Ground-truth reference text |
| `style` | Prosody style tag used |
| `word_count` | Word count of the sentence |
| `char_len` | Character length |

### 3.5 Dataset Statistics

| Run | Utterances | Duration |
|---|---|---|
| Research paper — run 1 | 200 | 32.7 min |
| Research paper — run 2 | 200 | 31.1 min |
| Research paper — run 3 | 175 | 28.8 min |
| Research paper — run 4 | 182 | 26.1 min |
| **Total** | **757** | **118.7 min (1.98 hrs)** |

---

## 4. Phase 2 — Offline Pipeline

**Entry point**: `python -m pipeline`

The pipeline is built as an OOP chain of five `Step` subclasses orchestrated by `PipelineRunner`. Each step follows a skip-or-execute contract and is independently re-runnable.

### 4.1 Configuration

Two frozen dataclasses handle all configuration:

**`PathConfig`** (environment variables)

| Variable | Default | Description |
|---|---|---|
| `STT_EXPERIMENT` | `common_au_en_op` | Output folder name |
| `STT_MANIFEST` | `{outputs_dir}/manifest.csv` | Path to manifest CSV |
| `STT_MODEL` | `large-v3` | Whisper model size |
| `STT_COMPUTE` | `float16` | CTranslate2 compute type |
| `STT_SAMPLES` | `300` | Max utterances to process |
| `STT_PROMPT` | *(79 domain terms)* | `initial_prompt` for Whisper |

The default `initial_prompt` contains 79 domain-specific terms covering MoE architectures, DeepSeek model naming conventions, and common abbreviations (`FFN`, `KV-cache`, `top-K`, `FLOP`, etc.), ordered by mis-recognition priority (model names first, abbreviations last).

**`PipelineConfig`** (CLI arguments)

| Flag | Default | Description |
|---|---|---|
| `--force` | False | Re-run even if output exists |
| `--no-retranscribe` | — | Skip Strategy 0 (re-transcription) |
| `--steps N [N ...]` | 1 2 3 4 5 | Run only these steps |

### 4.2 Step 1 — Baseline Transcription

**Class**: `TranscriptionStep` (`pipeline/steps/transcription.py`)
**Output**: `{experiment}/baseline_results.csv`

The `Transcriber` class lazy-loads `faster-whisper`'s `WhisperModel` and transcribes each audio file in the manifest:

```
model.transcribe(
    audio_path,
    language="en",
    beam_size=5,
    vad_filter=True,
    initial_prompt=None   ← no prompt at baseline
)
```

Output columns added: `hypothesis` (raw Whisper output), `runtime_s` (inference time per file).

**Skip condition**: `baseline_results.csv` already exists and `--force` not set.

### 4.3 Step 2 — Evaluation

**Class**: `EvaluationStep` (`pipeline/steps/evaluation.py`)
**Output**: Console only

`Evaluator` normalises both reference and hypothesis text before scoring:

1. Lowercase
2. Remove punctuation
3. Collapse spaces
4. Expand intra-word hyphens to spaces (e.g. `top-K` → `top K`)

jiwer's `process_words()` aligns reference and hypothesis word sequences and counts substitutions, deletions, and insertions. Four metrics are reported:

| Metric | Formula | Measures |
|---|---|---|
| **WER** | (S + D + I) / N | Word-level error rate |
| **CER** | Character-level equivalent | Fine-grained accuracy |
| **MER** | S / (S + D + I + correct) | Match error rate |
| **WIL** | 1 − (correct²) / (N × M) | Word information loss |

Top-5 best and worst utterances by per-utterance WER are printed to console.

### 4.4 Step 3 — Error Analysis

**Class**: `ErrorAnalysisStep` (`pipeline/steps/error_analysis.py`)
**Output**: `error_analysis.json`, `wer_distribution.png`, `top_substitutions.png`

`ErrorAnalyzer` iterates every utterance and builds frequency tables of:

- **Substitutions**: (ref_word, hyp_word) pairs — top 30 saved
- **Deletions**: ref_word that was dropped — top 20 saved
- **Insertions**: hyp_word that was hallucinated — top 20 saved

Substitution pairs are categorised into four groups:

| Category | Criterion |
|---|---|
| `number_mismatch` | Either token is a digit or number word |
| `proper_noun` | Reference token starts with uppercase |
| `short_function_word` | Reference token ≤ 3 characters |
| `content_word` | All other substitutions |

**Plots**

- `wer_distribution.png`: histogram of per-utterance WER values + bucketed bar chart (0%, 1–5%, 6–10%, 11–30%, >30%)
- `top_substitutions.png`: horizontal bar chart of top-15 (ref→hyp) substitution pairs

![WER Distribution](deepseekMOE_v4/wer_distribution.png)

![Top Substitutions](deepseekMOE_v4/top_substitutions.png)

### 4.5 Step 4 — Improvement Strategies

**Class**: `ImprovementStep` (`pipeline/steps/improvement.py`)
**Output**: `{experiment}/improved_results.csv`

`Improver` applies a chain of strategies in order, with each strategy building on the previous output:

---

**Strategy 0 — Re-transcription with `initial_prompt`**

Whisper's decoder is biased by prepending domain terms as a "fake" prior transcript. The 79-term prompt is ordered by mis-recognition priority (model names → compound nouns → abbreviations). Whisper hard-truncates at 224 tokens, so prompt ordering matters.

This step re-runs Whisper on every audio file with `initial_prompt` set. Because Whisper conditions its decoder on the prompt tokens, it strongly biases toward those surface forms, resolving most OOV errors.

Output column: `hypothesis_retranscribed`

---

**Strategy 1 — Text Normalisation**

A set of deterministic regex rules cleans surface artifacts:

| Rule | Pattern | Example |
|---|---|---|
| Trailing ellipsis | `\s*\.\.\.\s*$` | `"results..."` → `"results"` |
| Trailing dash | `\s*-\s*$` | `"layer-"` → `"layer"` |
| Double space | `\s{2,}` | `"two  spaces"` → `"two spaces"` |
| Comma conjunction | `, and` / `, but` | `"fast, and efficient"` → `"fast and efficient"` |

Output column: `hypothesis_norm`

---

**Strategy 2 — Vocabulary Biasing**

Three layers of correction are applied:

**Layer A — Hand-crafted phrase corrections**

Domain-specific multi-word substitutions known from error patterns:
- `"deep seek"` → `"DeepSeek"`
- Common model name fragments reassembled

**Layer B — Regex corrections**

| Pattern | Replacement | Example |
|---|---|---|
| `(\w+) th\b` | `\1th` | Ordinal restoration |
| `one hundred` | `100` | Number normalisation |

**Layer C — Auto-corrections from `error_analysis.json`**

The top-30 substitution pairs from Step 3 are turned into word-boundary replacement rules, but only if they pass **six guards**:

| Guard | Purpose |
|---|---|
| `count >= 5` | Reject rare corrections that may not generalise |
| HYP not a function word | Avoid replacing `"the"`, `"a"`, `"and"` |
| HYP does not start with REF | Avoid compound-split artifacts (`"and"` → `"andstep"`) |
| HYP is not a substring of REF | Avoid substring over-corrections |
| REF has at least one vowel | Avoid ligature artifacts in reference |
| Neither token is a digit or number word | Handled by normalisation, not biasing |

**Layer D — Exponent Normalisation** (`pipeline/core/math_normalizer.py`)

Converts spoken exponent expressions to symbolic notation:

| Spoken | Symbolic |
|---|---|
| `ten to the power of negative four` | `10^-4` |
| `two to the power of thirty two` | `2^32` |
| `10 to the power of minus 6` | `10^-6` |

The regex matches `<base> <bridge> [±]<exponent>` where:
- Base/exponent: word (`zero`–`ninety-nine`) or digit string
- Bridge: `"to the power of"`, `"to the power"`, `"power of"`, `"power"`
- Sign words: `negative`, `minus`, `positive`, `plus`

Output column: `hypothesis_vocab`

---

### 4.6 Step 5 — Re-evaluation and Visualisation

**Class**: `ReevaluationStep` (`pipeline/steps/reevaluation.py`)
**Output**: `improvement_comparison.png`, `before_after.png`

`Comparator` scores each strategy stage independently and produces:

**Console output**

- Baseline metrics (WER / CER / MER / WIL)
- Final improved metrics with absolute and relative deltas
- Stage-by-stage WER/CER table
- Top N utterances with largest WER improvement

**Plots**

- `improvement_comparison.png`: stage-by-stage WER and CER as a line+bar chart — shows where improvement is concentrated
- `before_after.png`: grouped bar chart comparing baseline vs final across all four metrics (WER, CER, MER, WIL)

![Stage-by-Stage Improvement](deepseekMOE_v4/improvement_comparison.png)

![Before vs After Post-Processing](deepseekMOE_v4/before_after.png)

---

## 5. Phase 3 — Real-Time Streaming

**Directory**: `streaming/`

The streaming subsystem provides real-time transcription with the same text quality improvements as the offline pipeline, minus the LLM correction (which would break real-time factor).

### 5.1 ASR Backends

`whisper_online.py` abstracts four ASR backends behind a common `ASRBase` interface:

| Backend | Class | Use case |
|---|---|---|
| `faster-whisper` | `FasterWhisperASR` | Primary — CUDA float16, fastest |
| `whisper_timestamped` | `WhisperTimestampedASR` | Older, slower alternative |
| `mlx-whisper` | `MLXWhisper` | Apple Silicon (M1/M2/M3) |
| OpenAI API | `OpenaiApiASR` | Cloud fallback |

`FasterWhisperASR` configuration:
```
model.transcribe(
    audio,
    language="en",
    beam_size=5,
    vad_filter=True,
    word_timestamps=True,
    condition_on_previous_text=True,
    initial_prompt=<domain_prompt>
)
```

`word_timestamps=True` is critical — it returns the start/end time for every word, enabling precise buffer management by the `HypothesisBuffer`.

Words with `no_speech_prob > 0.9` are discarded to suppress hallucination on silence.

### 5.2 Local-Agreement Buffering

**Classes**: `HypothesisBuffer` and `OnlineASRProcessor` in `whisper_online.py`

The central challenge in streaming ASR is that Whisper is not designed for streaming — it re-transcribes the entire buffer from scratch on every chunk. The local-agreement algorithm stabilises the output by only committing words that are agreed upon across multiple consecutive transcriptions.

**Algorithm**

At each iteration, `OnlineASRProcessor.process_iter()`:

1. Transcribes the current audio buffer with a `prompt` derived from the last 200 characters of committed text plus the 79-term domain vocabulary
2. Extracts timestamped words from the result
3. Calls `HypothesisBuffer.insert(new_words, time_offset)` — compares new hypothesis with the uncommitted buffer using n-gram overlap (max length 5). Words in the longest common prefix across two consecutive transcriptions are "agreed upon" and moved to the committed list
4. `flush()` returns and clears the newly committed words
5. Committed words are passed to `StreamingPostProcessor`

**Buffer Trimming**

To prevent the audio buffer from growing unboundedly (which would slow Whisper down), the processor trims at natural boundaries:

- **Segment-level** (default): if audio buffer exceeds 15 seconds, trim at the last segment boundary
- **Sentence-level**: if audio buffer exceeds threshold, trim at the last sentence-ending punctuation

After trimming, older audio is discarded and the hypothesis buffer's committed words are adjusted by the trim offset.

**`OnlineASRProcessor` key parameters**

| Parameter | Default | Description |
|---|---|---|
| `buffer_trimming` | `("segment", 15)` | Trim mode and threshold (seconds) |
| `logfile` | stderr | Logging target |

### 5.3 Voice Activity Detection

**Class**: `FixedVADIterator` in `silero_vad_iterator.py`

Silero VAD is a lightweight neural model that classifies 512-sample chunks (32 ms at 16 kHz) as speech or silence. The `FixedVADIterator` wrapper allows variable-length inputs by internally buffering and processing 512-sample windows.

**Parameters**

| Parameter | Default | Effect |
|---|---|---|
| `threshold` | 0.5 | Speech probability above which segment is triggered |
| `min_silence_duration_ms` | 500 | Minimum silence to end a speech segment |
| `speech_pad_ms` | 100 | Padding added to both sides of detected speech |

**Output**

- `{'start': sample_idx}` — speech onset detected
- `{'end': sample_idx}` — silence detected (speech segment ended)

When VAD is enabled in `OnlineASRProcessor`, audio is only fed to Whisper during detected speech segments, suppressing hallucinations on silence and reducing compute.

### 5.4 Streaming Post-Processor

**Class**: `StreamingPostProcessor` in `streaming/post_processor.py`

This is a lightweight, low-latency subset of the offline `Improver`. It applies every improvement that can run in under 1 ms:

| Strategy | Included | Reason |
|---|---|---|
| Text Normalisation | Yes | Pure regex, < 1 ms |
| Vocabulary Biasing | Yes | Dict lookup + regex, < 1 ms |
| Exponent Normalisation | Yes | Regex, < 1 ms |

The same six guards used in the offline pipeline govern which auto-corrections are loaded from `error_analysis.json`. This means the streaming system automatically benefits from corrections discovered during the offline error analysis phase, without any manual tuning.

**Processing pipeline per committed chunk**

```
committed_text
  → _normalise()           # trailing ..., double spaces, comma conjunctions
  → _vocab_correct()       # hand-crafted + regex + auto-corrections
  → normalise_exponents()  # spoken exponents → symbolic
  → enhanced_text
```

### 5.5 Enhanced Streaming Demo

**File**: `streaming/enhanced_demo.py`

The enhanced demo integrates all of the above and supports four operating modes:

| Mode | Description |
|---|---|
| `mic` | Live microphone — prints raw vs enhanced transcript in real time |
| `file` | Simulates streaming from a WAV file chunk by chunk, measures RTF |
| `compare` | Runs the same file in both streaming and offline modes, compares WER |
| `batch` | Processes N utterances from a manifest, produces a full metrics table |

**Domain initial_prompt**

At startup, the 79-term domain prompt is passed to `OnlineASRProcessor`, which prepends it to every Whisper call as a fake prior transcript. This biases the decoder toward technical vocabulary on every chunk.

**Dual output**

In `file` and `mic` modes, each committed chunk is printed twice:

```
[RAW]  deep seek moe uses fine grained expert segmentation
[ENH]  DeepSeekMoE uses fine-grained expert segmentation
```

**Batch mode output table**

| Column | Description |
|---|---|
| `WER_raw` | WER of raw Whisper output |
| `WER_enh` | WER after post-processing |
| `delta` | WER_raw − WER_enh (positive = improvement) |
| `CER_raw`, `CER_enh` | Character-level equivalents |

**RTF measurement**

For each chunk: `RTF = processing_latency / chunk_audio_duration`

Values consistently below 1.0 indicate real-time capability. The system targets RTF < 0.5 with `faster-whisper` on GPU.

---

## 6. Data Flow Diagrams

### Offline Pipeline

```
Research paper PDF
        │
        ▼
  [PDF Text Extraction] (PyMuPDF / fitz)
        │  Ligature repair (100+ patterns: fi/fl/ffi/ffl glyphs)
        │  Typo repair (OCR character transpositions)
        ▼
  [Sentence Filtering]
        │  Skip regex (URLs, citations, equations, tables, captions)
        │  Symbol density filter (> 20% non-alpha → reject)
        │  Domain keyword filter (ML terms only)
        │  Word count bounds (6 – 60 words)
        │  Deduplication (two-column PDF artifact)
        ▼
  [TTS Synthesis] (Fish Audio S2 Pro)
        │  5 prosody styles cycled per utterance
        │  temperature=0.8, top_p=0.8, repetition_penalty=1.1
        │  44.1 kHz output → 16 kHz downsample (Whisper compatibility)
        ▼
  manifest.csv
  (audio_path | transcript | style | word_count | char_len)
        │
        ▼
  [Step 1: Baseline Transcription]
        │  faster-whisper large-v3
        │  beam_size=5, vad_filter=True
        │  No initial_prompt at baseline
        ▼
  baseline_results.csv (+ hypothesis | runtime_s columns)
        │
        ├──► [Step 2: Evaluation] (jiwer)
        │         Normalise → lowercase, strip punctuation, expand hyphens
        │         process_words() → S / D / I counts
        │         WER / CER / MER / WIL
        │         Console report + top-5 best/worst utterances
        │
        ├──► [Step 3: Error Analysis]
        │         Substitution pairs (top 30)
        │         Deletions (top 20) / Insertions (top 20)
        │         Categorise: number_mismatch / proper_noun /
        │                     short_function_word / content_word
        │         → error_analysis.json
        │         → wer_distribution.png
        │         → top_substitutions.png
        │
        ▼
  [Step 4: Improvement]
        │
        │  S0. Re-transcription
        │      faster-whisper + initial_prompt (79 domain terms)
        │      ordered: model names → compound nouns → abbreviations
        │
        │  S1. Text Normalisation (regex)
        │      trailing ... / - → strip
        │      double spaces → single
        │      ", and" / ", but" → remove comma
        │
        │  S2. Vocabulary Biasing
        │      hand-crafted phrase corrections
        │      regex corrections (ordinals, numbers)
        │      auto-corrections from error_analysis.json
        │        (6 guards: count≥5, no function words,
        │         no compound-split, no substring, vowel check,
        │         no digit swaps)
        │      exponent normalisation (spoken → symbolic)
        │
        ▼
  improved_results.csv
  (hypothesis_norm | hypothesis_vocab | hypothesis)
        │
        ▼
  [Step 5: Re-evaluation & Visualisation]
        │  Stage-by-stage WER/CER table
        │  Absolute + relative deltas vs baseline
        │  Top N improved utterances
        │  → improvement_comparison.png (stage line/bar chart)
        │  → before_after.png (grouped bar: 4 metrics)
        ▼
  Final metrics (WER / CER / MER / WIL before and after)
```

### Streaming Pipeline

```
Audio Input (microphone / WAV file)
        │
        ▼
  [FixedVADIterator] (Silero VAD)
        │  Buffers input → processes 512-sample (32ms) windows
        │  threshold=0.5
        │  silence end: min_silence_duration_ms=500
        │  padding: speech_pad_ms=100
        │  Suppresses: silence-only segments
        ▼
  [OnlineASRProcessor]
        │
        ├── insert_audio_chunk(chunk)   ← called every 0.5s
        │
        ├── process_iter()
        │     │
        │     ├── prompt()
        │     │     last 200 chars of committed text
        │     │     + 79-term domain initial_prompt
        │     │
        │     ├── FasterWhisperASR.transcribe(audio_buffer, prompt)
        │     │     word_timestamps=True
        │     │     condition_on_previous_text=True
        │     │     no_speech_prob > 0.9 → skip word
        │     │
        │     ├── HypothesisBuffer.insert(timestamped_words, offset)
        │     │     n-gram overlap (max length 5)
        │     │     longest common prefix → committed
        │     │
        │     ├── flush() → newly committed words
        │     │
        │     └── Buffer trimming
        │           segment-level: > 15s → trim at segment boundary
        │           sentence-level: > threshold → trim at punctuation
        │
        ▼
  [StreamingPostProcessor]
        │  _normalise()            regex, < 1ms
        │  _vocab_correct()        dict + regex, < 1ms
        │  normalise_exponents()   regex, < 1ms
        ▼
  Enhanced committed text
        │
        ├── [mic / file mode]  print [RAW] and [ENH] side by side
        ├── [compare mode]     streaming WER vs offline WER
        └── [batch mode]       WER_raw / WER_enh / delta table
```

---

## 7. Results

### DeepSeekMoE Synthetic Data — 200 utterances, whisper large-v3

#### v4 — Expanded domain prompt (184 tokens)

| Stage | WER | CER | MER | WIL |
|---|---|---|---|---|
| Baseline | 8.21% | 1.89% | 8.00% | 12.86% |
| After Re-transcription + Norm | 2.85% | 0.75% | — | — |
| After Vocabulary Biasing | **2.83%** | **0.74%** | **2.81%** | **4.64%** |

**65.5% relative WER reduction · 60.8% relative CER reduction**

#### Effect of expanding the initial_prompt (v3 vs v4)

| | Baseline | v3 (40-token prompt) | v4 (184-token prompt) |
|---|---|---|---|
| WER | 8.21% | 3.54% | **2.83%** |
| CER | 1.89% | 0.86% | **0.74%** |
| Zero-WER utterances | — | 132 / 200 | **139 / 200** |

Expanding the prompt within the 224-token Whisper limit adds **0.71% absolute WER improvement** at zero additional compute cost.

#### Per-utterance WER distribution (v4 final)

| Bucket | Count |
|---|---|
| 0% (perfect) | 139 |
| 1–5% | 15 |
| 6–10% | 24 |
| 11–30% | 21 |
| >30% | 1 |

#### Visualisations

![Before vs After Post-Processing](deepseekMOE_v4/before_after.png)

![Stage-by-Stage Improvement](deepseekMOE_v4/improvement_comparison.png)

![WER Distribution](deepseekMOE_v4/wer_distribution.png)

![Top Substitution Errors](deepseekMOE_v4/top_substitutions.png)

---

## 8. Key Design Decisions

**Why `initial_prompt` before re-transcription?**

Whisper's decoder is biased by prepending domain terms as a "fake" prior transcript. Terms must be ≤ 224 tokens (Whisper hard-truncates beyond this). The prompt is ordered by mis-recognition priority — model names first (`DeepSeekMoE`, `GShard`), then compound nouns, then abbreviations — because the decoder pays more attention to tokens near the start of the prompt.

**Why are auto-corrections guarded with six rules?**

Naive substitution rules derived from error analysis caused WER regressions in early experiments. The pair `"step" → "andstep"` arose from hyphen-stripped references where `"and-step"` normalised to `"andstep"`. Six guards prevent compound-split artifacts, substring over-corrections, and digit/number-word swaps from being applied.

**Why is the streaming post-processor limited to deterministic strategies?**

Any non-deterministic operation (e.g. LLM call) adds 1–3 seconds per utterance. This pushes RTF above 1.0, breaking real-time capability. The streaming post-processor therefore applies only normalisation, vocabulary biasing, and exponent normalisation — all under 1 ms — leaving the transcript latency dominated entirely by the Whisper inference time.

**Why Fish Audio S2 Pro for TTS?**

The key requirement was inline prosody control to produce acoustic variety without collecting real recordings. Fish Audio S2 Pro is the only fully local, open-source TTS model that supports inline prosody style tags, runs on consumer GPU hardware, and produces naturalness comparable to cloud TTS services.

**Why the local-agreement algorithm instead of a fixed time window?**

Fixed-window approaches commit words after a fixed delay regardless of whether subsequent transcriptions agree. Local-agreement waits until a word appears in the same position across two consecutive transcriptions before committing it, dramatically reducing mid-word correction flicker at the cost of a small additional latency (~0.5–1.0 s).

**Why two separate packages (`pipeline/` and `src/`)?**

`src/` contains the original numbered scripts kept intact for reproducibility — they can be run independently to verify individual steps without the OOP wrapper. `pipeline/` is a clean rewrite that imports nothing from `src/`, implementing the same logic with explicit skip conditions, structured configuration, and timing instrumentation.
