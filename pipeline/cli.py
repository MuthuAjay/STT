"""Command-line entry point.

Usage (run from repo root):
    python -m pipeline
    python -m pipeline --no-lm
    python -m pipeline --no-retranscribe
    python -m pipeline --force
    python -m pipeline --steps 1 2 3
    python -m pipeline --manifest path/to/manifest.csv
    python -m pipeline --experiment Deepseek_moe_3

Env var overrides:
    STT_EXPERIMENT=Deepseek_moe_3  python -m pipeline
    STT_MODEL=large-v3             python -m pipeline
    STT_SAMPLES=500                python -m pipeline
    STT_PROMPT="YourTerms"        python -m pipeline
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PathConfig, PipelineConfig
from .runner import PipelineRunner
from .steps  import build_steps

log = logging.getLogger("pipeline")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="STT Quality Improvement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
env var overrides (no file edits needed):
  STT_EXPERIMENT=name     switch experiment / output folder
  STT_MANIFEST=path       use an existing manifest.csv
  STT_MODEL=large-v3      whisper model size
  STT_SAMPLES=500         number of samples
  STT_COMPUTE=int8        compute type
  STT_PROMPT="terms"      initial_prompt for Whisper
        """,
    )
    p.add_argument("--no-lm",           action="store_true",
                   help="Skip LLM correction (step 4)")
    p.add_argument("--no-retranscribe", action="store_true",
                   help="Skip re-transcription with initial_prompt (step 4)")
    p.add_argument("--force",           action="store_true",
                   help="Re-run even if cached outputs exist")
    p.add_argument("--steps",           nargs="+", type=int, metavar="N",
                   help="Run only these step numbers (1–5)")
    p.add_argument("--manifest",        type=str, metavar="PATH",
                   help="Path to an existing manifest.csv")
    p.add_argument("--experiment",      type=str, metavar="NAME",
                   help="Experiment name (overrides STT_EXPERIMENT)")
    return p.parse_args()


def _setup_logging(outputs_dir: Path) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_file = outputs_dir / "pipeline.log"
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    )
    for handler in (
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log_file


def main() -> None:
    args    = _parse_args()
    run_cfg = PipelineConfig.from_args(args)
    run_cfg.apply_env()                    # push experiment/manifest into os.environ

    path_cfg = PathConfig.from_env()
    path_cfg.validate()
    path_cfg.makedirs()

    log_file = _setup_logging(path_cfg.outputs_dir)
    log.info("  log : %s", log_file)

    steps  = build_steps(path_cfg, run_cfg)
    runner = PipelineRunner(steps=steps, path_cfg=path_cfg, run_cfg=run_cfg)
    runner.run()
