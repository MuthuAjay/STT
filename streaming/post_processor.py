"""StreamingPostProcessor — applies pipeline improvement strategies to streaming output.

Strategies applied per committed chunk (zero-copy, sub-millisecond):
  1. Text normalisation   — trailing artifacts, double spaces, comma conjunctions
  2. Vocabulary biasing   — hand-crafted + auto-corrections from error_analysis.json

Strategy 3 (LLM) is intentionally excluded — it takes 1–3s per call and would
destroy real-time factor.  Strategy 4 (lowercase) is skipped because the
Evaluator already lowercases before scoring.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from math_normalizer import normalise_exponents


# ── Normalisation rules (same as pipeline/core/improver.py) ──────────────────

_NORMALISATION_RULES = [
    (r"\s*\.\.\.\s*$",         ""),
    (r"\s*-\s*$",              ""),
    (r"  +",                   " "),
    (r"^\s+|\s+$",             ""),
    (r",\s+(and|but|or|so)\b", r" \1"),
]

# ── Hand-crafted vocab corrections (domain-specific) ─────────────────────────

_VOCAB_CORRECTIONS: dict[str, str] = {
    "fourteen fifty-five": "1455",
    "fourteen fifty five": "1455",
    " an the ":            " and the ",
    "rooted experts":      "routed experts",
    "rooted expert":       "routed expert",
}

_REGEX_CORRECTIONS = [
    (r"\b(\w+) th\b",    r"\1th"),
    (r"\bone hundred\b", "100"),
]

# Words that should never be the HYP source of an auto-correction
_SKIP_HYP_WORDS = {
    'a','an','the','is','in','on','at','to','of','and','or','but','as',
    'it','its','he','she','we','they','new','old','big','one','two','all',
    'no','not','so','buy','by','be','do','go','my','up','now','had',
}
_NUMBER_WORDS = {
    'zero','one','two','three','four','five','six','seven','eight','nine',
    'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
    'seventeen','eighteen','nineteen','twenty','hundred','thousand','million',
}


class StreamingPostProcessor:
    """Applies text normalisation and vocabulary correction to streaming chunks.

    Usage::

        pp = StreamingPostProcessor(error_json_path=Path("exp/error_analysis.json"))

        # Inside your streaming loop:
        beg, end, text = proc.process_iter()
        if text:
            text = pp.process(text)
            print(text)
    """

    def __init__(self, error_json_path: Path | None = None,
                 min_count: int = 5) -> None:
        self._auto_corrections = self._load_auto_corrections(
            error_json_path, min_count
        ) if error_json_path and error_json_path.exists() else {}

        n = len(self._auto_corrections)
        src = str(error_json_path) if error_json_path else "none"
        print(f"[post_processor] Auto-corrections: {n} rules  (source: {src})")

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """Normalise and vocab-correct a committed chunk.  Returns cleaned text."""
        text = self._normalise(text)
        text = self._vocab_correct(text)
        text = normalise_exponents(text)
        return text

    # ── Strategy 1: text normalisation ───────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        for pattern, repl in _NORMALISATION_RULES:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text.strip()

    # ── Strategy 2: vocabulary correction ────────────────────────────────

    def _vocab_correct(self, text: str) -> str:
        # Phrase-level replacements
        for wrong, right in _VOCAB_CORRECTIONS.items():
            text = text.replace(wrong, right)
        # Regex corrections
        for pattern, repl in _REGEX_CORRECTIONS:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        # Auto-corrections (word boundary)
        for wrong, right in self._auto_corrections.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', right,
                          text, flags=re.IGNORECASE)
        return text.strip()

    # ── Auto-correction loader ────────────────────────────────────────────

    @staticmethod
    def _load_auto_corrections(path: Path, min_count: int) -> dict[str, str]:
        """Build word-level substitution map from error_analysis.json.

        Same guards as pipeline/core/improver.py to avoid over-correction.
        """
        with open(path) as f:
            analysis = json.load(f)

        corrections: dict[str, str] = {}
        for sub in analysis.get("top_substitutions", []):
            if sub["count"] < min_count:
                continue
            ref_w = sub["ref"].strip().lower()
            hyp_w = sub["hyp"].strip().lower()
            if ' ' in ref_w or ' ' in hyp_w:
                continue
            if hyp_w in _SKIP_HYP_WORDS:
                continue
            if hyp_w.startswith(ref_w) and len(hyp_w) > len(ref_w):
                continue
            if hyp_w in ref_w and len(ref_w) > len(hyp_w):
                continue
            if not re.search(r'[aeiou]', ref_w):
                continue
            if ref_w in _NUMBER_WORDS or hyp_w in _NUMBER_WORDS:
                continue
            if re.fullmatch(r'\d+', ref_w) or re.fullmatch(r'\d+', hyp_w):
                continue
            corrections[hyp_w] = sub["ref"].strip()

        return corrections
