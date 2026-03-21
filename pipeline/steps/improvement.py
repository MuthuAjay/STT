"""Step 4 – Improvement."""
from __future__ import annotations

from ..base import Step, SkipResult
from ..config import PathConfig
from ..core import Transcriber, Improver


class ImprovementStep(Step):

    def __init__(self, cfg: PathConfig, force: bool,
                 use_lm: bool, retranscribe: bool) -> None:
        self._cfg          = cfg
        self._force        = force
        self._use_lm       = use_lm
        self._retranscribe = retranscribe

    @property
    def number(self) -> int:
        return 4

    @property
    def name(self) -> str:
        return "Improvement"

    def should_skip(self) -> SkipResult:
        if not self._cfg.baseline_csv.exists():
            return SkipResult.yes("baseline_results.csv missing — run step 1 first")
        if not self._force and self._cfg.improved_csv.exists():
            return SkipResult.yes("improved_results.csv exists")
        return SkipResult.no()

    def run(self) -> None:
        import pandas as pd

        df = pd.read_csv(self._cfg.baseline_csv)
        print(f"[improvement] Processing {len(df)} utterances …")

        transcriber = (
            Transcriber(self._cfg.model_size, self._cfg.compute_type)
            if self._retranscribe else None
        )

        improver = Improver(
            use_lm       = self._use_lm,
            retranscribe = self._retranscribe,
            ollama_model = self._cfg.ollama_model,
            transcriber  = transcriber,
        )

        result_df = improver.run(
            df,
            initial_prompt  = self._cfg.initial_prompt,
            error_json_path = self._cfg.error_json,
        )
        result_df.to_csv(self._cfg.improved_csv, index=False)
        print(f"[improvement] Saved → {self._cfg.improved_csv}")

        print("\n=== Sample Improvements ===")
        for _, row in result_df.sample(5, random_state=42).iterrows():
            print(f"\n  REF   : {row['transcript']}")
            print(f"  NORM  : {row.get('hypothesis_norm', '')}")
            if "hypothesis_lm" in result_df.columns:
                print(f"  LM    : {row.get('hypothesis_lm', '')}")
            print(f"  FINAL : {row['hypothesis']}")
