"""Evaluator — WER / CER / MER / WIL scoring with jiwer."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from jiwer import (
    wer, cer, mer, wil,
    process_words,
    Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip,
)


# ── Metrics value object ──────────────────────────────────────────────────────

@dataclass
class Metrics:
    wer: float
    cer: float
    mer: float
    wil: float
    substitutions: int
    deletions:     int
    insertions:    int
    total:         int

    def __str__(self) -> str:
        lines = [
            f"  WER  : {self.wer:>6.2f}%",
            f"  CER  : {self.cer:>6.2f}%",
            f"  MER  : {self.mer:>6.2f}%",
            f"  WIL  : {self.wil:>6.2f}%",
            f"  --- word counts ---",
            f"  SUB={self.substitutions}  DEL={self.deletions}  "
            f"INS={self.insertions}  TOT={self.total}",
        ]
        return "\n".join(lines)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """Scores ASR hypotheses against reference transcripts.

    Text normalisation applied before scoring:
      lowercase → remove punctuation → collapse spaces
      + intra-word hyphen expansion (``fine-grained`` → ``fine grained``)
    """

    _JIWER_PIPELINE = Compose([
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip(),
    ])

    def normalize(self, text: str) -> str:
        # Expand intra-word hyphens before jiwer (avoids inflate from
        # "fine-grained" vs "fine grained" treating as single token)
        text = re.sub(r"(?<=\w)-(?=\w)", " ", str(text))
        return self._JIWER_PIPELINE(text)

    # ── Scoring ───────────────────────────────────────────────────────────

    def score(self, references: list[str], hypotheses: list[str]) -> Metrics:
        refs_n = [self.normalize(r) for r in references]
        hyps_n = [self.normalize(h) for h in hypotheses]
        pw     = process_words(refs_n, hyps_n)
        total  = pw.hits + pw.substitutions + pw.deletions
        return Metrics(
            wer           = round(wer(refs_n, hyps_n) * 100, 2),
            cer           = round(cer(refs_n, hyps_n) * 100, 2),
            mer           = round(mer(refs_n, hyps_n) * 100, 2),
            wil           = round(wil(refs_n, hyps_n) * 100, 2),
            substitutions = pw.substitutions,
            deletions     = pw.deletions,
            insertions    = pw.insertions,
            total         = total,
        )

    def per_utterance(self, df: pd.DataFrame,
                      ref_col: str = "transcript",
                      hyp_col: str = "hypothesis") -> pd.DataFrame:
        """Add a ``utt_wer`` column with per-utterance WER (%)."""
        df = df.copy()
        df["utt_wer"] = df.apply(
            lambda row: round(
                wer(self.normalize(str(row[ref_col])),
                    self.normalize(str(row[hyp_col]))) * 100,
                2,
            ),
            axis=1,
        )
        return df

    # ── Reporting ─────────────────────────────────────────────────────────

    def print_report(self, label: str, metrics: Metrics) -> None:
        print(f"\n{'=' * 40}")
        print(f"  {label}")
        print("=" * 40)
        print(metrics)

    def show_examples(self, df: pd.DataFrame, n: int = 5,
                      worst: bool = False) -> None:
        label  = "Worst" if worst else "Best"
        subset = df.nlargest(n, "utt_wer") if worst else df.nsmallest(n, "utt_wer")
        print(f"\n--- {label} {n} utterances ---")
        for _, row in subset.iterrows():
            print(f"  WER={row['utt_wer']:5.1f}%")
            print(f"    REF: {row['transcript']}")
            print(f"    HYP: {row['hypothesis']}")
