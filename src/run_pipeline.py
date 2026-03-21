"""
STT Quality Improvement Pipeline – End-to-End Runner
=====================================================
Thin orchestrator: imports and delegates to each step's standalone script.
No logic is duplicated here — all evaluation, transcription, and improvement
code lives in the numbered scripts.

Usage:
    python src/run_pipeline.py                        # full pipeline
    python src/run_pipeline.py --no-lm                # skip LLM correction
    python src/run_pipeline.py --no-retranscribe      # skip initial_prompt re-transcription
    python src/run_pipeline.py --force                # re-run even if cached
    python src/run_pipeline.py --steps 1 2 3          # specific steps only
    python src/run_pipeline.py --manifest path/to/manifest.csv

Env var overrides (no file edits needed):
    STT_EXPERIMENT=librispeech_op python src/run_pipeline.py
    STT_MODEL=large-v3 STT_SAMPLES=500 python src/run_pipeline.py
    STT_PROMPT="YourTerms" python src/run_pipeline.py
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

# ── Path setup ────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import config

# ── Logger ────────────────────────────────────────────────────────────────
log = logging.getLogger("pipeline")


def _setup_logging() -> Path:
    config.makedirs()
    log_file = Path(config.OUTPUTS_DIR) / "pipeline.log"
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.FileHandler(log_file, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log_file


def _load(module_name: str):
    """Import a numbered module (e.g. '01_data_collection') via importlib."""
    return importlib.import_module(module_name)


# ── UI helpers ────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 60) -> None:
    log.info("=" * width)
    log.info(f"  {title}")
    log.info("=" * width)


def _step_header(n: int, name: str) -> None:
    log.info("")
    log.info(f"┌{'─'*58}┐")
    log.info(f"│  STEP {n}  {name:<51}│")
    log.info(f"└{'─'*58}┘")


@contextmanager
def _timed(name: str):
    """Context manager that logs step completion with elapsed time."""
    t0 = time.perf_counter()
    yield
    log.info(f"  ✓  {name} completed in {time.perf_counter() - t0:.1f}s")


# ── Step definition ───────────────────────────────────────────────────────

@dataclass
class Step:
    n:       int
    name:    str
    run:     Callable
    skip_if: Callable[[], tuple[bool, str]]
    kwargs:  dict = field(default_factory=dict)

    def execute(self) -> None:
        _step_header(self.n, self.name)
        should_skip, reason = self.skip_if()
        if should_skip:
            log.info(f"  ⏭  {self.name} skipped — {reason}")
            return
        with _timed(self.name):
            self.run(**self.kwargs)


# ── Step registry ─────────────────────────────────────────────────────────

def _build_steps(args: argparse.Namespace) -> list[Step]:
    """Build the ordered step list, closing over parsed args and config."""

    manifest  = Path(config.MANIFEST_CSV)
    baseline  = Path(config.BASELINE_CSV)
    improved  = Path(config.IMPROVED_CSV)
    error_json = Path(config.ERROR_ANALYSIS_JSON)

    def skip_if_exists(path: Path, label: str) -> tuple[bool, str]:
        if not args.force and path.exists():
            return True, f"{label} exists"
        return False, ""

    def requires(path: Path, label: str) -> tuple[bool, str]:
        if not path.exists():
            return True, f"{label} missing — run the preceding step first"
        return False, ""

    return [
        Step(
            n=0, name="Data Preparation",
            run=lambda: _load("01_data_collection").main(),
            skip_if=lambda: (
                (True, f"manifest exists at {manifest}")
                if args.manifest or (not args.force and manifest.exists())
                else (False, "")
            ),
        ),
        Step(
            n=1, name="Baseline Transcription",
            run=lambda: _load("02_baseline_transcription").main(),
            skip_if=lambda: skip_if_exists(baseline, "baseline_results.csv"),
        ),
        Step(
            n=2, name="Evaluation (Baseline)",
            run=lambda: _load("03_evaluation").evaluate(
                config.BASELINE_CSV,
                label=f"Baseline (whisper-{config.WHISPER_MODEL_SIZE})",
                n_examples=5,
            ),
            skip_if=lambda: requires(baseline, "baseline_results.csv"),
        ),
        Step(
            n=3, name="Error Analysis",
            run=lambda: _load("04_error_analysis").main(),
            skip_if=lambda: (
                requires(baseline, "baseline_results.csv")
                if not baseline.exists()
                else skip_if_exists(error_json, "error_analysis.json")
            ),
        ),
        Step(
            n=4, name="Improvement",
            run=lambda **kw: _load("05_improvement").main(**kw),
            skip_if=lambda: (
                requires(baseline, "baseline_results.csv")
                if not baseline.exists()
                else skip_if_exists(improved, "improved_results.csv")
            ),
            kwargs={"use_lm": not args.no_lm, "retranscribe": not args.no_retranscribe},
        ),
        Step(
            n=5, name="Re-evaluation",
            run=lambda: _load("06_reevaluation").main(),
            skip_if=lambda: requires(improved, "improved_results.csv"),
        ),
    ]


# ── Argument parsing ──────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STT Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
env var overrides (no file edits needed):
  STT_EXPERIMENT=librispeech_op   switch experiment folder
  STT_MANIFEST=path/to/manifest   use an existing manifest
  STT_MODEL=large-v3              whisper model size
  STT_SAMPLES=500                 number of samples
  STT_COMPUTE=int8                compute type
  STT_PROMPT="term1, term2"       initial_prompt for Whisper
        """,
    )
    parser.add_argument("--no-lm",           action="store_true", help="Skip LLM correction (step 4)")
    parser.add_argument("--no-retranscribe", action="store_true", help="Skip re-transcription with initial_prompt (step 4)")
    parser.add_argument("--force",           action="store_true", help="Re-run even if cached outputs exist")
    parser.add_argument("--steps",           nargs="+", type=int, metavar="N",
                        help="Run only these step numbers (e.g. --steps 1 2 3)")
    parser.add_argument("--manifest",        type=str, metavar="PATH",
                        help="Path to an existing manifest.csv — skips step 0")
    parser.add_argument("--experiment",      type=str, metavar="NAME",
                        help="Experiment name (overrides STT_EXPERIMENT env var)")
    return parser.parse_args()


# ── Config overrides ──────────────────────────────────────────────────────

def _apply_overrides(args: argparse.Namespace) -> None:
    import os
    if args.experiment:
        os.environ["STT_EXPERIMENT"] = args.experiment
        importlib.reload(config)
        config.makedirs()
    if args.manifest:
        os.environ["STT_MANIFEST"] = args.manifest
        importlib.reload(config)

    try:
        config.validate()
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    if args.manifest and not Path(config.MANIFEST_CSV).exists():
        sys.exit(f"[ERROR] --manifest path does not exist: {config.MANIFEST_CSV}")


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _apply_overrides(args)

    log_file = _setup_logging()
    run_steps = set(args.steps) if args.steps else set(range(6))
    steps = _build_steps(args)

    _banner(f"STT Pipeline  ·  experiment: {config.EXPERIMENT}")
    log.info(config.summary())
    log.info(f"  lm            : {config.OLLAMA_MODEL if not args.no_lm else 'disabled'}")
    log.info(f"  retranscribe  : {'disabled' if args.no_retranscribe else 'enabled'}")
    log.info(f"  log           : {log_file}")
    log.info(f"  started       : {datetime.now():%Y-%m-%d %H:%M:%S}")

    t_start = time.perf_counter()

    # Guard: manifest must exist before steps 1–5 can run
    if any(n in run_steps for n in range(1, 6)):
        if not Path(config.MANIFEST_CSV).exists():
            log.error(
                f"Manifest not found: {config.MANIFEST_CSV}\n"
                f"  Run step 0 first, or pass --manifest <path>."
            )
            sys.exit(1)

    for step in steps:
        if step.n in run_steps:
            step.execute()

    _banner("Pipeline Complete")
    log.info(f"  total time : {time.perf_counter() - t_start:.1f}s")
    log.info(f"  outputs    : {config.OUTPUTS_DIR}")
    log.info(f"  log        : {log_file}")
    log.info(f"  finished   : {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
