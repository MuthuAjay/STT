"""Step 2 – Evaluation (Baseline)."""
from __future__ import annotations

from ..base import Step, SkipResult
from ..config import PathConfig
from ..core import Evaluator


class EvaluationStep(Step):

    def __init__(self, cfg: PathConfig) -> None:
        self._cfg = cfg

    @property
    def number(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "Evaluation (Baseline)"

    def should_skip(self) -> SkipResult:
        if not self._cfg.baseline_csv.exists():
            return SkipResult.yes("baseline_results.csv missing — run step 1 first")
        return SkipResult.no()

    def run(self) -> None:
        import pandas as pd

        df      = pd.read_csv(self._cfg.baseline_csv)
        ev      = Evaluator()
        metrics = ev.score(df["transcript"].tolist(), df["hypothesis"].tolist())
        ev.print_report(
            f"Baseline (whisper-{self._cfg.model_size})", metrics
        )
        df = ev.per_utterance(df)
        ev.show_examples(df, n=5, worst=False)
        ev.show_examples(df, n=5, worst=True)
