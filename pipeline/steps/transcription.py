"""Step 1 – Baseline Transcription."""
from __future__ import annotations

from pathlib import Path

from ..base import Step, SkipResult
from ..config import PathConfig
from ..core import Transcriber


class TranscriptionStep(Step):

    def __init__(self, cfg: PathConfig, force: bool) -> None:
        self._cfg   = cfg
        self._force = force

    @property
    def number(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "Baseline Transcription"

    def should_skip(self) -> SkipResult:
        if not self._force and self._cfg.baseline_csv.exists():
            return SkipResult.yes("baseline_results.csv exists")
        return SkipResult.no()

    def run(self) -> None:
        import pandas as pd

        df = pd.read_csv(self._cfg.manifest_csv)
        print(f"[transcription] Transcribing {len(df)} files …")

        transcriber        = Transcriber(self._cfg.model_size, self._cfg.compute_type)
        hypotheses, runtimes = transcriber.transcribe_batch(df)

        df["hypothesis"] = hypotheses
        df["runtime_s"]  = runtimes
        df.to_csv(self._cfg.baseline_csv, index=False)
        print(f"[transcription] Saved → {self._cfg.baseline_csv}")

        print("\n=== Sample Transcriptions ===")
        for _, row in df.head(5).iterrows():
            print(f"\n  REF : {row['transcript']}")
            print(f"  HYP : {row['hypothesis']}")
            print(f"  Time: {row['runtime_s']}s")
        print(f"\n[transcription] Avg inference time: {df['runtime_s'].mean():.3f}s")
