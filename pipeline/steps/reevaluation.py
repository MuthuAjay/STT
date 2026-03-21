"""Step 5 – Re-evaluation & Comparison."""
from __future__ import annotations

from ..base import Step, SkipResult
from ..config import PathConfig
from ..core import Comparator


class ReevaluationStep(Step):

    def __init__(self, cfg: PathConfig) -> None:
        self._cfg = cfg

    @property
    def number(self) -> int:
        return 5

    @property
    def name(self) -> str:
        return "Re-evaluation"

    def should_skip(self) -> SkipResult:
        if not self._cfg.improved_csv.exists():
            return SkipResult.yes("improved_results.csv missing — run step 4 first")
        return SkipResult.no()

    def run(self) -> None:
        import pandas as pd

        baseline_df = pd.read_csv(self._cfg.baseline_csv)
        improved_df = pd.read_csv(self._cfg.improved_csv)

        comparator = Comparator()
        report     = comparator.compare(baseline_df, improved_df)

        comparator.print_report(report)
        comparator.print_top_improved(improved_df)
        comparator.plot(report, self._cfg.outputs_dir / "improvement_comparison.png")
