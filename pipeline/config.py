"""Pipeline configuration.

Two concerns kept separate:
  - PathConfig  : paths and model settings, driven by env vars
  - PipelineConfig : runtime flags from the CLI (force, use_lm, etc.)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ── Path / model config ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathConfig:
    """All filesystem paths and model knobs for one experiment run.

    Constructed from env vars; never mutated after creation.
    """

    experiment:      str
    outputs_dir:     Path
    manifest_csv:    Path
    baseline_csv:    Path
    improved_csv:    Path
    error_json:      Path
    samples_dir:     Path

    model_size:      str
    compute_type:    str
    num_samples:     int
    ollama_model:    str
    initial_prompt:  str

    # Dataset source dirs (needed only for data-prep, kept for completeness)
    cv_local_dir:    str
    ljspeech_dir:    str

    _VALID_MODELS   = frozenset({"tiny", "base", "small", "medium",
                                  "large", "large-v2", "large-v3"})
    _VALID_COMPUTES = frozenset({"float16", "float32", "int8", "int8_float16"})

    @classmethod
    def from_env(cls) -> "PathConfig":
        """Build from environment variables using the same defaults as src/config.py."""
        base_dir    = Path(__file__).resolve().parent.parent
        experiment  = os.environ.get("STT_EXPERIMENT", "common_au_en_op")
        outputs_dir = base_dir / experiment

        return cls(
            experiment      = experiment,
            outputs_dir     = outputs_dir,
            manifest_csv    = Path(os.environ.get("STT_MANIFEST",
                                                   str(outputs_dir / "manifest.csv"))),
            baseline_csv    = outputs_dir / "baseline_results.csv",
            improved_csv    = outputs_dir / "improved_results.csv",
            error_json      = outputs_dir / "error_analysis.json",
            samples_dir     = outputs_dir / "samples",
            model_size      = os.environ.get("STT_MODEL",   "large-v3"),
            compute_type    = os.environ.get("STT_COMPUTE", "float16"),
            num_samples     = int(os.environ.get("STT_SAMPLES", "300")),
            ollama_model    = os.environ.get("STT_OLLAMA",  "qwen3.5:4b"),
            initial_prompt  = os.environ.get(
                "STT_PROMPT",
                (
                    "DeepSeekMoE, DeepSeekMoE-2B, DeepSeekMoE-16B, DeepSeekMoE-145B, "
                    "DeepSeek-MoE, GShard, GShard-137B, ZeRO, ZeRO-Offload, DeepSpeed, "
                    "MoE, mixture-of-experts, routed experts, shared experts, "
                    "fine-grained expert segmentation, shared expert isolation, "
                    "expert specialization, activated parameters, expert parameters, "
                    "top-K routing, load balancing, knowledge redundancy, knowledge hybridity, "
                    "Transformer, feed-forward network, FFN, hidden dimension, "
                    "warmup scheduler, learning rate scheduler, cosine annealing, "
                    "Pile loss, HellaSwag, MMLU, TriviaQA, ARC, "
                    "large language model, LLM, pre-training, fine-tuning, parameter efficiency, PyTorch."
                ),
            ),
            cv_local_dir    = os.environ.get(
                "STT_CV_DIR",
                "/home/muthuajay/Documents/Datasets/commonvoice-v24_en-AU",
            ),
            ljspeech_dir    = os.environ.get(
                "STT_LJSPEECH_DIR",
                "/home/muthuajay/Documents/Datasets/LJSpeech-1.1",
            ),
        )

    def validate(self) -> None:
        errors = []
        if not self.experiment:
            errors.append("STT_EXPERIMENT must not be empty")
        if self.model_size not in self._VALID_MODELS:
            errors.append(f"STT_MODEL '{self.model_size}' not in {self._VALID_MODELS}")
        if self.compute_type not in self._VALID_COMPUTES:
            errors.append(f"STT_COMPUTE '{self.compute_type}' not in {self._VALID_COMPUTES}")
        if self.num_samples < 1:
            errors.append(f"STT_SAMPLES must be >= 1, got {self.num_samples}")
        if errors:
            raise ValueError("Config errors:\n  " + "\n  ".join(errors))

    def makedirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        return (
            f"  experiment : {self.experiment}\n"
            f"  outputs    : {self.outputs_dir}\n"
            f"  manifest   : {self.manifest_csv}\n"
            f"  model      : faster-whisper {self.model_size} ({self.compute_type})\n"
            f"  ollama     : {self.ollama_model}\n"
            f"  n_samples  : {self.num_samples}"
        )


# ── Runtime / CLI config ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Immutable runtime flags parsed from the CLI."""

    force:         bool            = False
    use_lm:        bool            = True
    retranscribe:  bool            = True
    steps:         frozenset       = field(default_factory=lambda: frozenset(range(1, 6)))
    experiment:    str | None      = None
    manifest_path: Path | None     = None

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        steps = frozenset(args.steps) if args.steps else frozenset(range(1, 6))
        return cls(
            force         = args.force,
            use_lm        = not args.no_lm,
            retranscribe  = not args.no_retranscribe,
            steps         = steps,
            experiment    = args.experiment or None,
            manifest_path = Path(args.manifest) if args.manifest else None,
        )

    def apply_env(self) -> None:
        """Push CLI experiment / manifest overrides into the environment."""
        if self.experiment:
            os.environ["STT_EXPERIMENT"] = self.experiment
        if self.manifest_path:
            os.environ["STT_MANIFEST"] = str(self.manifest_path)

    def should_run(self, step_number: int) -> bool:
        return step_number in self.steps
