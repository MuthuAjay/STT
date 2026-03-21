"""STT pipeline — OOP modular package.

Public surface:
    PathConfig      — filesystem paths and model settings (env-driven)
    PipelineConfig  — CLI runtime flags (force, use_lm, retranscribe, steps)
    PipelineRunner  — executes an ordered sequence of Steps
    Step            — abstract base class for all pipeline steps
    build_steps     — factory that wires Steps together

Core classes (pipeline.core):
    Transcriber     — faster-whisper wrapper with lazy model load
    Evaluator       — WER / CER / MER / WIL scoring
    ErrorAnalyzer   — substitution / deletion / insertion analysis
    Improver        — 5-strategy post-processing chain
    Comparator      — baseline vs stage comparison + chart

Entry point:
    python -m pipeline          (from repo root)
"""
from .config  import PathConfig, PipelineConfig
from .runner  import PipelineRunner
from .base    import Step, SkipResult
from .steps   import build_steps

__all__ = [
    "PathConfig",
    "PipelineConfig",
    "PipelineRunner",
    "Step",
    "SkipResult",
    "build_steps",
]
