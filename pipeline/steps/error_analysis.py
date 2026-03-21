"""Step 3 – Error Analysis."""
from __future__ import annotations

from ..base import Step, SkipResult
from ..config import PathConfig
from ..core import ErrorAnalyzer


class ErrorAnalysisStep(Step):

    def __init__(self, cfg: PathConfig, force: bool) -> None:
        self._cfg   = cfg
        self._force = force

    @property
    def number(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "Error Analysis"

    def should_skip(self) -> SkipResult:
        if not self._cfg.baseline_csv.exists():
            return SkipResult.yes("baseline_results.csv missing — run step 1 first")
        if not self._force and self._cfg.error_json.exists():
            return SkipResult.yes("error_analysis.json exists")
        return SkipResult.no()

    def run(self) -> None:
        import pandas as pd

        df       = pd.read_csv(self._cfg.baseline_csv)
        analyzer = ErrorAnalyzer()

        print(f"[error_analysis] Analysing {len(df)} utterances …")
        report = analyzer.analyze(df)

        analyzer.save(report, self._cfg.error_json)
        analyzer.print_report(report)
        analyzer.plot_distribution(
            df, self._cfg.outputs_dir / "wer_distribution.png"
        )
        analyzer.plot_substitutions(
            report, self._cfg.outputs_dir / "top_substitutions.png"
        )
