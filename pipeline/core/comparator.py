"""Comparator — baseline vs improvement stage comparison + chart."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jiwer import wer as jwer, cer as jcer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

from .evaluator import Evaluator, Metrics


# ── Value object ──────────────────────────────────────────────────────────────

@dataclass
class StageMetrics:
    label: str
    wer:   float
    cer:   float


@dataclass
class ComparisonReport:
    baseline_metrics: Metrics
    improved_metrics: Metrics
    stages:           list[StageMetrics] = field(default_factory=list)


# ── Comparator ────────────────────────────────────────────────────────────────

class Comparator:
    """Compares baseline and every improvement stage."""

    def __init__(self) -> None:
        self._evaluator = Evaluator()

    # ── Public API ────────────────────────────────────────────────────────

    def compare(self, baseline_df: pd.DataFrame,
                improved_df: pd.DataFrame) -> ComparisonReport:
        refs_base = baseline_df["transcript"].tolist()
        refs_imp  = improved_df["transcript"].tolist()

        baseline_metrics = self._evaluator.score(
            refs_base, baseline_df["hypothesis"].tolist()
        )
        improved_metrics = self._evaluator.score(
            refs_imp, improved_df["hypothesis"].tolist()
        )

        stages = self._score_stages(improved_df)

        return ComparisonReport(
            baseline_metrics = baseline_metrics,
            improved_metrics = improved_metrics,
            stages           = stages,
        )

    def print_report(self, report: ComparisonReport) -> None:
        self._evaluator.print_report("Baseline", report.baseline_metrics)
        self._evaluator.print_report("Improved (all stages)", report.improved_metrics)

        print("\n=== Improvement Delta ===")
        for attr, label in [("wer", "WER"), ("cer", "CER"),
                             ("mer", "MER"), ("wil", "WIL")]:
            b = getattr(report.baseline_metrics, attr)
            i = getattr(report.improved_metrics, attr)
            delta = i - b
            arrow = "▼" if delta < 0 else "▲"
            tag   = "reduction" if delta < 0 else "increase"
            print(f"  {label}: {b:.2f}% → {i:.2f}%  ({arrow} {abs(delta):.2f}% {tag})")

        print("\n=== Stage-by-Stage ===")
        rows = [(s.label, f"{s.wer:.2f}%", f"{s.cer:.2f}%") for s in report.stages]
        col_w = max(len(r[0]) for r in rows) + 2
        print(f"  {'Stage':<{col_w}}  {'WER':>8}  {'CER':>8}")
        for label, w, c in rows:
            print(f"  {label:<{col_w}}  {w:>8}  {c:>8}")

    def print_top_improved(self, improved_df: pd.DataFrame, n: int = 5) -> None:
        if "hypothesis_norm" not in improved_df.columns:
            return
        ev = self._evaluator
        improved_df = improved_df.copy()
        improved_df["base_wer"] = [
            jwer(ev.normalize(r), ev.normalize(h)) * 100
            for r, h in zip(improved_df["transcript"], improved_df["hypothesis_norm"])
        ]
        improved_df["final_wer"] = [
            jwer(ev.normalize(r), ev.normalize(h)) * 100
            for r, h in zip(improved_df["transcript"], improved_df["hypothesis"])
        ]
        improved_df["wer_delta"] = improved_df["base_wer"] - improved_df["final_wer"]

        print(f"\n=== Top {n} Most Improved Utterances ===")
        for _, row in improved_df.nlargest(n, "wer_delta").iterrows():
            print(f"\n  REF  : {row['transcript'][:80]}")
            print(f"  NORM : {row['hypothesis_norm'][:80]}")
            print(f"  FINAL: {row['hypothesis'][:80]}")
            print(f"  WER  : {row['base_wer']:.1f}% → {row['final_wer']:.1f}%  "
                  f"(Δ {row['wer_delta']:.1f}%)")

    def plot_before_after(self, report: ComparisonReport, output_path: Path) -> None:
        """Grouped bar chart: Baseline vs Improved across all four metrics."""
        metrics = ["WER", "CER", "MER", "WIL"]
        before  = [report.baseline_metrics.wer, report.baseline_metrics.cer,
                   report.baseline_metrics.mer, report.baseline_metrics.wil]
        after   = [report.improved_metrics.wer, report.improved_metrics.cer,
                   report.improved_metrics.mer, report.improved_metrics.wil]

        import numpy as np
        x     = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        bars_b = ax.bar(x - width / 2, before, width, label="Baseline",
                        color="#4C72B0", edgecolor="white")
        bars_a = ax.bar(x + width / 2, after,  width, label="Improved",
                        color="#55A868", edgecolor="white")

        ax.bar_label(bars_b, fmt="%.2f%%", padding=3, fontsize=9)
        ax.bar_label(bars_a, fmt="%.2f%%", padding=3, fontsize=9)

        # Annotate reduction arrows
        for i, (b, a) in enumerate(zip(before, after)):
            delta = b - a
            ax.annotate(
                f"▼{delta:.2f}%",
                xy=(x[i], max(b, a) + 0.5),
                ha="center", fontsize=8, color="#C44E52", fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, max(before) * 1.35)
        ax.set_title("Before vs After Post-Processing", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[comparator] Before/after chart → {output_path}")

    def plot(self, report: ComparisonReport, output_path: Path) -> None:
        if not report.stages:
            return
        df = pd.DataFrame([{"Stage": s.label, "WER (%)": s.wer, "CER (%)": s.cer}
                           for s in report.stages])
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ["WER (%)", "CER (%)"]):
            bars = ax.bar(df["Stage"], df[metric],
                          color=colors[:len(df)], edgecolor="white", width=0.5)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by Stage")
            ax.set_ylim(0, df[metric].max() * 1.2)
            ax.bar_label(bars, fmt="%.2f%%", padding=3)
            ax.tick_params(axis="x", rotation=20)

        plt.suptitle("STT Improvement: Baseline vs Post-Processing Stages", fontsize=13)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[comparator] Chart → {output_path}")

    # ── Private helpers ───────────────────────────────────────────────────

    def _score_stages(self, improved_df: pd.DataFrame) -> list[StageMetrics]:
        refs   = improved_df["transcript"].tolist()
        ev     = self._evaluator
        stages = [
            ("After Norm.",      "hypothesis_norm"),
            ("After Vocab Bias", "hypothesis_vocab"),
            ("After LM Corr.",   "hypothesis_lm"),
        ]
        results = []
        for label, col in stages:
            if col not in improved_df.columns:
                continue
            hyps = [str(h) for h in improved_df[col].tolist()]
            if not any(h != "nan" for h in hyps):
                continue
            refs_n = [ev.normalize(r) for r in refs]
            hyps_n = [ev.normalize(h) for h in hyps]
            results.append(StageMetrics(
                label = label,
                wer   = round(jwer(refs_n, hyps_n) * 100, 2),
                cer   = round(jcer(refs_n, hyps_n) * 100, 2),
            ))
        return results
