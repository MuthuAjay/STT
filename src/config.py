"""
Central configuration for the STT pipeline.

All values can be overridden via environment variables — no source edits needed.

  STT_EXPERIMENT=synthetic_ml python src/run_pipeline.py
  STT_MANIFEST=/path/to/manifest.csv python src/run_pipeline.py --steps 1 2
  STT_MODEL=large-v3 STT_EXPERIMENT=common_au_en_op python src/run_pipeline.py
"""
import os
from pathlib import Path

# ── Repo root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


# ── Experiment ─────────────────────────────────────────────────────────────
# Override: STT_EXPERIMENT=librispeech_op python src/run_pipeline.py
EXPERIMENT = os.environ.get("STT_EXPERIMENT", "common_au_en_op")

OUTPUTS_DIR  = BASE_DIR / EXPERIMENT
SAMPLES_DIR  = OUTPUTS_DIR / "samples"
MANIFEST_CSV = str(os.environ.get("STT_MANIFEST", OUTPUTS_DIR / "manifest.csv"))

# Output files — all live inside OUTPUTS_DIR
BASELINE_CSV        = str(OUTPUTS_DIR / "baseline_results.csv")
IMPROVED_CSV        = str(OUTPUTS_DIR / "improved_results.csv")
ERROR_ANALYSIS_JSON = str(OUTPUTS_DIR / "error_analysis.json")

OUTPUTS_DIR  = str(OUTPUTS_DIR)
SAMPLES_DIR  = str(SAMPLES_DIR)


# ── Dataset sources ────────────────────────────────────────────────────────
# Override these via env vars so no file edits are needed on a different machine.
CV_LOCAL_DIR = os.environ.get(
    "STT_CV_DIR",
    "/home/muthuajay/Documents/Datasets/commonvoice-v24_en-AU",
)
LJSPEECH_DIR = os.environ.get(
    "STT_LJSPEECH_DIR",
    "/home/muthuajay/Documents/Datasets/LJSpeech-1.1",
)


# ── Model settings ─────────────────────────────────────────────────────────
# Override: STT_MODEL=large-v3 STT_COMPUTE=int8 python src/run_pipeline.py
WHISPER_MODEL_SIZE   = os.environ.get("STT_MODEL",   "large-v3")
WHISPER_COMPUTE_TYPE = os.environ.get("STT_COMPUTE", "float16")
OLLAMA_MODEL         = os.environ.get("STT_OLLAMA",  "qwen3.5:4b")


# ── Evaluation settings ────────────────────────────────────────────────────
NUM_SAMPLES = int(os.environ.get("STT_SAMPLES", "300"))

# ── Whisper initial prompt ─────────────────────────────────────────────────
# Primes the Whisper decoder with domain vocabulary to reduce OOV errors.
# Override: STT_PROMPT="YourTerms here" python src/run_pipeline.py
WHISPER_INITIAL_PROMPT = os.environ.get(
    "STT_PROMPT",
    (
        "DeepSeekMoE, GShard, ZeRO, DeepSpeed, MoE, mixture-of-experts, "
        "fine-grained expert segmentation, shared expert isolation, "
        "expert specialization, Transformer, hidden dimension, "
        "warmup scheduler, activated parameters, language model."
    ),
)


# ── Validation ─────────────────────────────────────────────────────────────
_VALID_MODELS   = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
_VALID_COMPUTES = {"float16", "float32", "int8", "int8_float16"}

def validate():
    """Call once at pipeline startup to catch bad config early."""
    errors = []
    if not EXPERIMENT:
        errors.append("STT_EXPERIMENT must not be empty")
    if WHISPER_MODEL_SIZE not in _VALID_MODELS:
        errors.append(f"STT_MODEL '{WHISPER_MODEL_SIZE}' not in {_VALID_MODELS}")
    if WHISPER_COMPUTE_TYPE not in _VALID_COMPUTES:
        errors.append(f"STT_COMPUTE '{WHISPER_COMPUTE_TYPE}' not in {_VALID_COMPUTES}")
    if NUM_SAMPLES < 1:
        errors.append(f"STT_SAMPLES must be >= 1, got {NUM_SAMPLES}")
    if errors:
        raise ValueError("Config errors:\n  " + "\n  ".join(errors))


def makedirs():
    """Explicitly create output directories. Call only when actually writing outputs."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)


def summary() -> str:
    """Human-readable config summary for logging."""
    return (
        f"  experiment : {EXPERIMENT}\n"
        f"  outputs    : {OUTPUTS_DIR}\n"
        f"  manifest   : {MANIFEST_CSV}\n"
        f"  samples    : {SAMPLES_DIR}\n"
        f"  model      : faster-whisper {WHISPER_MODEL_SIZE} ({WHISPER_COMPUTE_TYPE})\n"
        f"  ollama     : {OLLAMA_MODEL}\n"
        f"  n_samples  : {NUM_SAMPLES}\n"
        f"  cv_dir     : {CV_LOCAL_DIR}"
    )
