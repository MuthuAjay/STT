"""ErrorAnalyzer — substitution / deletion / insertion analysis."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jiwer import wer, process_words, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip


# ── Value objects ─────────────────────────────────────────────────────────────

@dataclass
class ErrorReport:
    totals:            dict
    top_substitutions: list[dict]
    top_deletions:     list[dict]
    top_insertions:    list[dict]
    categories:        dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "totals":            self.totals,
            "top_substitutions": self.top_substitutions,
            "top_deletions":     self.top_deletions,
            "top_insertions":    self.top_insertions,
            "categories":        self.categories,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorReport":
        return cls(
            totals            = d["totals"],
            top_substitutions = d["top_substitutions"],
            top_deletions     = d["top_deletions"],
            top_insertions    = d["top_insertions"],
            categories        = d.get("categories", {}),
        )


# ── Analyzer ──────────────────────────────────────────────────────────────────

class ErrorAnalyzer:
    """Analyses ASR edit operations and groups errors into categories."""

    _TRANSFORM = Compose([
        ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()
    ])

    def _normalize(self, text: str) -> str:
        return self._TRANSFORM(str(text))

    # ── Core analysis ─────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> ErrorReport:
        """Run full error analysis over a baseline results DataFrame."""
        substitutions: Counter = Counter()
        deletions:     Counter = Counter()
        insertions:    Counter = Counter()
        total_s = total_i = total_d = total_h = 0

        for _, row in df.iterrows():
            ref    = self._normalize(row["transcript"])
            hyp    = self._normalize(row["hypothesis"])
            result = process_words(ref, hyp)

            for chunk in result.alignments[0]:
                op = chunk.type
                if op == "substitute":
                    ref_w = result.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
                    hyp_w = result.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]
                    for r, h in zip(ref_w, hyp_w):
                        substitutions[(r, h)] += 1
                    total_s += len(ref_w)
                elif op == "delete":
                    ref_w = result.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
                    for w in ref_w:
                        deletions[w] += 1
                    total_d += len(ref_w)
                elif op == "insert":
                    hyp_w = result.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]
                    for w in hyp_w:
                        insertions[w] += 1
                    total_i += len(hyp_w)

            total_h += len(result.references[0])

        report = ErrorReport(
            totals={
                "substitutions":  total_s,
                "deletions":      total_d,
                "insertions":     total_i,
                "total_ref_words": total_h,
            },
            top_substitutions=[
                {"ref": r, "hyp": h, "count": c}
                for (r, h), c in substitutions.most_common(30)
            ],
            top_deletions=[
                {"word": w, "count": c} for w, c in deletions.most_common(20)
            ],
            top_insertions=[
                {"word": w, "count": c} for w, c in insertions.most_common(20)
            ],
        )
        report.categories = self._categorise(report)
        return report

    def _categorise(self, report: ErrorReport) -> dict:
        categories: dict = defaultdict(list)
        for sub in report.top_substitutions:
            r, h = sub["ref"], sub["hyp"]
            if re.fullmatch(r"\d+(\.\d+)?", r) or re.fullmatch(r"\d+(\.\d+)?", h):
                categories["number_mismatch"].append(sub)
            elif r[0].isupper() or h[0].isupper():
                categories["proper_noun"].append(sub)
            elif len(r) <= 3 or len(h) <= 3:
                categories["short_function_word"].append(sub)
            else:
                categories["content_word"].append(sub)
        return dict(categories)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, report: ErrorReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[error_analyzer] Saved → {path}")

    def load(self, path: Path) -> ErrorReport:
        with open(path) as f:
            return ErrorReport.from_dict(json.load(f))

    # ── Plots ─────────────────────────────────────────────────────────────

    def plot_distribution(self, df: pd.DataFrame, output_path: Path) -> None:
        utt_wers = [
            wer(self._normalize(row["transcript"]),
                self._normalize(row["hypothesis"])) * 100
            for _, row in df.iterrows()
        ]
        mean_wer = sum(utt_wers) / len(utt_wers)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(utt_wers, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[0].axvline(mean_wer, color="red", linestyle="--",
                        label=f"Mean WER = {mean_wer:.1f}%")
        axes[0].set_xlabel("Utterance WER (%)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("WER Distribution per Utterance")
        axes[0].legend()

        buckets = {"0%": 0, "1-10%": 0, "11-30%": 0, "31-60%": 0, ">60%": 0}
        for w in utt_wers:
            if w == 0:       buckets["0%"] += 1
            elif w <= 10:    buckets["1-10%"] += 1
            elif w <= 30:    buckets["11-30%"] += 1
            elif w <= 60:    buckets["31-60%"] += 1
            else:            buckets[">60%"] += 1

        axes[1].bar(buckets.keys(), buckets.values(),
                    color=sns.color_palette("Blues_d", 5))
        axes[1].set_xlabel("WER Bucket")
        axes[1].set_ylabel("Number of Utterances")
        axes[1].set_title("WER Bucket Distribution")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[error_analyzer] WER distribution → {output_path}")

    def plot_substitutions(self, report: ErrorReport, output_path: Path) -> None:
        top  = report.top_substitutions[:15]
        labels = [f"'{s['ref']}' → '{s['hyp']}'" for s in top]
        counts = [s["count"] for s in top]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels[::-1], counts[::-1], color="#DD8452")
        ax.set_xlabel("Count")
        ax.set_title("Top 15 Substitution Pairs")
        ax.bar_label(bars, padding=3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[error_analyzer] Substitution chart → {output_path}")

    # ── Console report ────────────────────────────────────────────────────

    def print_report(self, report: ErrorReport) -> None:
        t = report.totals
        total_errors = t["substitutions"] + t["deletions"] + t["insertions"]
        print("\n=== Error Operation Breakdown ===")
        print(f"  Substitutions : {t['substitutions']:>5}  "
              f"({t['substitutions']/total_errors*100:.1f}%)")
        print(f"  Deletions     : {t['deletions']:>5}  "
              f"({t['deletions']/total_errors*100:.1f}%)")
        print(f"  Insertions    : {t['insertions']:>5}  "
              f"({t['insertions']/total_errors*100:.1f}%)")
        print(f"  Total errors  : {total_errors:>5}")
        print(f"  Total ref wds : {t['total_ref_words']:>5}")

        print("\n=== Top 10 Substitution Pairs ===")
        for sub in report.top_substitutions[:10]:
            print(f"  '{sub['ref']}' → '{sub['hyp']}'  (×{sub['count']})")

        print("\n=== Top 10 Deleted Words ===")
        for d in report.top_deletions[:10]:
            print(f"  '{d['word']}'  (×{d['count']})")

        print("\n=== Top 10 Inserted Words ===")
        for ins in report.top_insertions[:10]:
            print(f"  '{ins['word']}'  (×{ins['count']})")

        print("\n=== Error Categories ===")
        for cat, items in report.categories.items():
            print(f"  {cat}: {len(items)} pairs")
