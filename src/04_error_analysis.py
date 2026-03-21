"""
Step 4 – Error Analysis
========================
Analyses the baseline transcription errors and identifies:
  1. Overall edit-operation breakdown (S / I / D)
  2. Most common substitution pairs
  3. Most frequent deleted / inserted words
  4. Error categories (numbers, proper nouns, punctuation artifacts, etc.)
  5. Utterance-level WER distribution

Results are saved to outputs/error_analysis.json and printed to stdout.

Run:
    python src/04_error_analysis.py
"""
import json
import re
from collections import Counter, defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from jiwer import process_words, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from config import BASELINE_CSV, OUTPUTS_DIR

TRANSFORM = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])


# ── Helpers ───────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return TRANSFORM(str(text))


def classify_word(word: str) -> str:
    """Heuristic category for a word."""
    if re.fullmatch(r"\d+(\.\d+)?", word):
        return "number"
    if word[0].isupper() and len(word) > 1:
        return "proper_noun_candidate"
    if len(word) <= 2:
        return "short_word"
    return "common_word"


def analyse_errors(df: pd.DataFrame) -> dict:
    substitutions: Counter = Counter()
    deletions: Counter = Counter()
    insertions: Counter = Counter()
    total_s = total_i = total_d = total_h = 0

    for _, row in df.iterrows():
        ref = normalize(row["transcript"])
        hyp = normalize(row["hypothesis"])
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

    return {
        "totals": {
            "substitutions": total_s,
            "deletions": total_d,
            "insertions": total_i,
            "total_ref_words": total_h,
        },
        "top_substitutions": [
            {"ref": r, "hyp": h, "count": c}
            for (r, h), c in substitutions.most_common(30)
        ],
        "top_deletions": [
            {"word": w, "count": c} for w, c in deletions.most_common(20)
        ],
        "top_insertions": [
            {"word": w, "count": c} for w, c in insertions.most_common(20)
        ],
    }


def categorise_substitutions(analysis: dict) -> dict:
    """Group substitution pairs into high-level categories."""
    categories = defaultdict(list)
    for sub in analysis["top_substitutions"]:
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


def plot_error_distribution(df: pd.DataFrame, output_path: str):
    from jiwer import wer
    utt_wers = []
    for _, row in df.iterrows():
        ref = normalize(row["transcript"])
        hyp = normalize(row["hypothesis"])
        utt_wers.append(wer(ref, hyp) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # WER distribution
    axes[0].hist(utt_wers, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(sum(utt_wers) / len(utt_wers), color="red",
                    linestyle="--", label=f"Mean WER = {sum(utt_wers)/len(utt_wers):.1f}%")
    axes[0].set_xlabel("Utterance WER (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("WER Distribution per Utterance")
    axes[0].legend()

    # WER buckets
    buckets = {"0%": 0, "1-10%": 0, "11-30%": 0, "31-60%": 0, ">60%": 0}
    for w in utt_wers:
        if w == 0:
            buckets["0%"] += 1
        elif w <= 10:
            buckets["1-10%"] += 1
        elif w <= 30:
            buckets["11-30%"] += 1
        elif w <= 60:
            buckets["31-60%"] += 1
        else:
            buckets[">60%"] += 1

    axes[1].bar(buckets.keys(), buckets.values(), color=sns.color_palette("Blues_d", 5))
    axes[1].set_xlabel("WER Bucket")
    axes[1].set_ylabel("Number of Utterances")
    axes[1].set_title("WER Bucket Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[error_analysis] WER distribution plot → {output_path}")


def plot_top_substitutions(analysis: dict, output_path: str):
    top_subs = analysis["top_substitutions"][:15]
    labels = [f"'{s['ref']}' → '{s['hyp']}'" for s in top_subs]
    counts = [s["count"] for s in top_subs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], counts[::-1], color="#DD8452")
    ax.set_xlabel("Count")
    ax.set_title("Top 15 Substitution Pairs")
    ax.bar_label(bars, padding=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[error_analysis] Substitution chart → {output_path}")


def main():
    import os
    df = pd.read_csv(BASELINE_CSV)
    print(f"[error_analysis] Analysing {len(df)} utterances …")

    analysis = analyse_errors(df)
    analysis["categories"] = categorise_substitutions(analysis)

    # Save JSON
    json_path = os.path.join(OUTPUTS_DIR, "error_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[error_analysis] Saved → {json_path}")

    # Print summary
    t = analysis["totals"]
    total_errors = t["substitutions"] + t["deletions"] + t["insertions"]
    print("\n=== Error Operation Breakdown ===")
    print(f"  Substitutions : {t['substitutions']:>5}  ({t['substitutions']/total_errors*100:.1f}%)")
    print(f"  Deletions     : {t['deletions']:>5}  ({t['deletions']/total_errors*100:.1f}%)")
    print(f"  Insertions    : {t['insertions']:>5}  ({t['insertions']/total_errors*100:.1f}%)")
    print(f"  Total errors  : {total_errors:>5}")
    print(f"  Total ref wds : {t['total_ref_words']:>5}")

    print("\n=== Top 10 Substitution Pairs ===")
    for sub in analysis["top_substitutions"][:10]:
        print(f"  '{sub['ref']}' → '{sub['hyp']}'  (×{sub['count']})")

    print("\n=== Top 10 Deleted Words ===")
    for d in analysis["top_deletions"][:10]:
        print(f"  '{d['word']}'  (×{d['count']})")

    print("\n=== Top 10 Inserted Words ===")
    for ins in analysis["top_insertions"][:10]:
        print(f"  '{ins['word']}'  (×{ins['count']})")

    print("\n=== Error Categories ===")
    for cat, items in analysis["categories"].items():
        print(f"  {cat}: {len(items)} pairs")

    # Plots
    plot_error_distribution(df, os.path.join(OUTPUTS_DIR, "wer_distribution.png"))
    plot_top_substitutions(analysis, os.path.join(OUTPUTS_DIR, "top_substitutions.png"))

    return analysis


if __name__ == "__main__":
    main()
