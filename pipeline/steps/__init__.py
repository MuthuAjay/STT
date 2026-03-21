"""Step factory."""
from __future__ import annotations

from typing import Sequence

from ..base import Step
from ..config import PathConfig, PipelineConfig
from .transcription  import TranscriptionStep
from .evaluation     import EvaluationStep
from .error_analysis import ErrorAnalysisStep
from .improvement    import ImprovementStep
from .reevaluation   import ReevaluationStep


def build_steps(path_cfg: PathConfig, run_cfg: PipelineConfig) -> Sequence[Step]:
    return [
        TranscriptionStep(path_cfg, force=run_cfg.force),
        EvaluationStep(path_cfg),
        ErrorAnalysisStep(path_cfg, force=run_cfg.force),
        ImprovementStep(
            path_cfg,
            force        = run_cfg.force,
            use_lm       = run_cfg.use_lm,
            retranscribe = run_cfg.retranscribe,
        ),
        ReevaluationStep(path_cfg),
    ]
