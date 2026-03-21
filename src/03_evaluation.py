"""
Step 3 – Evaluation (WER / CER / MER / WIL)
============================================
Computes Word Error Rate, Character Error Rate, Match Error Rate,
and Word Information Lost for baseline and (optionally) improved transcriptions.

Run:
    python src/03_evaluation.py                       # baseline only
    python src/03_evaluation.py --improved            # both
    python src/03_evaluation.py --output results.csv  # save per-utterance results
    python src/03_evaluation.py --examples 10         # show 10 best/worst utterances
"""
import os
import re
import argparse
import pandas as pd
from jiwer import (
    wer, cer, mer, wil,
    process_words,
    Compose, ToLowerCase, RemovePunctuation,
    RemoveMultipleSpaces, Strip,
)
from config import BASELINE_CSV, IMPROVED_CSV


def _normalize(text: str) -> str:
    """Pre-processing before jiwer TRANSFORM: collapse intra-word hyphens."""
    # "fine-grained" → "fine grained", "mixture-of-experts" → "mixture of experts"
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    return text


# ── Text normalisation applied before scoring ──────────────────────────────
_JIWER_TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def TRANSFORM(text: str) -> str:
    return _JIWER_TRANSFORM(_normalize(text))


def score(references: list[str], hypotheses: list[str]) -> dict:
    refs_norm = [TRANSFORM(r) for r in references]
    hyps_norm = [TRANSFORM(h) for h in hypotheses]
    # Single pass for word-level stats
    pw = process_words(refs_norm, hyps_norm)
    total = pw.hits + pw.substitutions + pw.deletions
    return {
        "WER": round(wer(refs_norm, hyps_norm) * 100, 2),
        "CER": round(cer(refs_norm, hyps_norm) * 100, 2),
        "MER": round(mer(refs_norm, hyps_norm) * 100, 2),
        "WIL": round(wil(refs_norm, hyps_norm) * 100, 2),
        "SUB": pw.substitutions,
        "DEL": pw.deletions,
        "INS": pw.insertions,
        "TOT": total,
    }


def per_utterance_wer(df: pd.DataFrame,
                      ref_col: str = "transcript",
                      hyp_col: str = "hypothesis") -> pd.DataFrame:
    df = df.copy()
    df["utt_wer"] = df.apply(
        lambda row: round(
            wer(TRANSFORM(str(row[ref_col])), TRANSFORM(str(row[hyp_col]))) * 100, 2
        ),
        axis=1,
    )
    return df


def print_metrics(label: str, metrics: dict):
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    for k in ("WER", "CER", "MER", "WIL"):
        print(f"  {k:<5}: {metrics[k]:>6.2f}%")
    print(f"  --- word counts ---")
    print(f"  SUB={metrics['SUB']}  DEL={metrics['DEL']}  INS={metrics['INS']}  TOT={metrics['TOT']}")


def show_examples(df: pd.DataFrame, n: int = 5, worst: bool = False):
    """Show best or worst utterances by per-utterance WER."""
    label = "Worst" if worst else "Best"
    subset = df.nlargest(n, "utt_wer") if worst else df.nsmallest(n, "utt_wer")
    print(f"\n--- {label} {n} utterances ---")
    for _, row in subset.iterrows():
        print(f"  WER={row['utt_wer']:5.1f}%")
        print(f"    REF: {row['transcript']}")
        print(f"    HYP: {row['hypothesis']}")


def evaluate(csv_path: str, label: str = "Baseline",
             n_examples: int = 5) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    metrics = score(df["transcript"].tolist(), df["hypothesis"].tolist())
    print_metrics(label, metrics)
    df = per_utterance_wer(df)
    show_examples(df, n=n_examples, worst=False)
    show_examples(df, n=n_examples, worst=True)
    return df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--improved", action="store_true",
                        help="Also evaluate improved results")
    parser.add_argument("--output", type=str, default=None,
                        help="Save per-utterance results to this CSV path")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of best/worst examples to show (default: 5)")
    args = parser.parse_args()

    baseline_df, baseline_metrics = evaluate(
        BASELINE_CSV, "Baseline (whisper-large-v3)", n_examples=args.examples
    )

    if args.improved:
        if not os.path.exists(IMPROVED_CSV):
            print(f"[eval] {IMPROVED_CSV} not found. Run 05_improvement.py first.")
        else:
            improved_df, improved_metrics = evaluate(
                IMPROVED_CSV, "Improved", n_examples=args.examples
            )
            print("\n=== Delta ===")
            for k in ("WER", "CER", "MER", "WIL"):
                delta = improved_metrics[k] - baseline_metrics[k]
                arrow = "▼" if delta < 0 else "▲"
                print(f"  {k}: {baseline_metrics[k]:.2f}% → {improved_metrics[k]:.2f}%  "
                      f"({arrow} {abs(delta):.2f}%)")

    if args.output:
        baseline_df.to_csv(args.output, index=False)
        print(f"\n[eval] Per-utterance results saved to {args.output}")

    return baseline_df, baseline_metrics


if __name__ == "__main__":
    main()
