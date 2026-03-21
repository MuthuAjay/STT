"""
Step 6 – Re-evaluation & Comparison
=====================================
Loads baseline and improved CSVs, computes metrics for every stage, and
produces a comparison chart.

Run:
    python src/06_reevaluation.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

from config import BASELINE_CSV, IMPROVED_CSV, OUTPUTS_DIR

TRANSFORM = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])


def score(refs: list[str], hyps: list[str]) -> dict:
    refs_n = [TRANSFORM(r) for r in refs]
    hyps_n = [TRANSFORM(h) for h in hyps]
    return {
        "WER (%)": round(wer(refs_n, hyps_n) * 100, 2),
        "CER (%)": round(cer(refs_n, hyps_n) * 100, 2),
    }


def compare_stages(improved_df: pd.DataFrame) -> pd.DataFrame:
    refs = improved_df["transcript"].tolist()

    stages = {
        "After Norm.":      improved_df.get("hypothesis_norm",  pd.Series()).tolist(),
        "After Vocab Bias": improved_df.get("hypothesis_vocab", pd.Series()).tolist(),
        "After LM Corr.":   improved_df.get("hypothesis_lm",   pd.Series()).tolist(),
        "After Lowercase":  improved_df["hypothesis"].tolist(),
    }

    rows = []
    for label, hyps in stages.items():
        if hyps and any(str(h) != "nan" for h in hyps):
            m = score(refs, [str(h) for h in hyps])
            rows.append({"Stage": label, **m})

    return pd.DataFrame(rows)


def plot_comparison(comparison_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, metric in zip(axes, ["WER (%)", "CER (%)"]):
        bars = ax.bar(comparison_df["Stage"], comparison_df[metric],
                      color=colors[:len(comparison_df)], edgecolor="white", width=0.5)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Stage")
        ax.set_ylim(0, comparison_df[metric].max() * 1.2)
        ax.bar_label(bars, fmt="%.2f%%", padding=3)
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("STT Improvement: Baseline vs Post-Processing Stages", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[reevaluation] Comparison chart → {output_path}")


def main():
    if not os.path.exists(IMPROVED_CSV):
        print(f"[reevaluation] {IMPROVED_CSV} not found. Run 05_improvement.py first.")
        return

    # Load baseline for original hypothesis column
    base_df = pd.read_csv(BASELINE_CSV)
    improved_df = pd.read_csv(IMPROVED_CSV)

    # Baseline WER uses the original un-normalised hypothesis
    base_metrics = score(base_df["transcript"].tolist(),
                         base_df["hypothesis"].tolist())
    print("\n=== Baseline Metrics ===")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.2f}%")

    improved_metrics = score(improved_df["transcript"].tolist(),
                              improved_df["hypothesis"].tolist())
    print("\n=== Improved Metrics (after all stages) ===")
    for k, v in improved_metrics.items():
        print(f"  {k}: {v:.2f}%")

    print("\n=== Improvement Delta ===")
    for k in base_metrics:
        delta = improved_metrics[k] - base_metrics[k]
        arrow = "▼" if delta < 0 else "▲"
        print(f"  {k}: {base_metrics[k]:.2f}% → {improved_metrics[k]:.2f}%  "
              f"({arrow} {abs(delta):.2f}% {'reduction' if delta < 0 else 'increase'})")

    comparison_df = compare_stages(improved_df)
    print("\n=== Stage-by-Stage Comparison ===")
    print(comparison_df.to_string(index=False))

    plot_comparison(comparison_df, os.path.join(OUTPUTS_DIR, "improvement_comparison.png"))

    # Show top improved utterances (baseline = after norm, final = after lowercase)
    if "hypothesis_norm" in improved_df.columns:
        from jiwer import wer as jwer
        improved_df["base_wer"] = [
            jwer(TRANSFORM(r), TRANSFORM(h)) * 100
            for r, h in zip(improved_df["transcript"], improved_df["hypothesis_norm"])
        ]
        improved_df["final_wer"] = [
            jwer(TRANSFORM(r), TRANSFORM(h)) * 100
            for r, h in zip(improved_df["transcript"], improved_df["hypothesis"])
        ]
        improved_df["wer_delta"] = improved_df["base_wer"] - improved_df["final_wer"]

        print("\n=== Top 5 Most Improved Utterances ===")
        for _, row in improved_df.nlargest(5, "wer_delta").iterrows():
            print(f"\n  REF    : {row['transcript'][:80]}")
            print(f"  NORM   : {row['hypothesis_norm'][:80]}")
            print(f"  LOWER  : {row['hypothesis'][:80]}")
            print(f"  WER    : {row['base_wer']:.1f}% → {row['final_wer']:.1f}%  (Δ {row['wer_delta']:.1f}%)")

    return comparison_df


if __name__ == "__main__":
    main()
