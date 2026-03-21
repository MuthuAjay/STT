"""Improver — five-strategy post-processing pipeline."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── Value object ──────────────────────────────────────────────────────────────

@dataclass
class ImprovementResult:
    after_norm:  str
    after_vocab: str
    after_lm:    str
    after_lower: str


# ── Strategy constants ────────────────────────────────────────────────────────

_NORMALISATION_RULES = [
    (r"\s*\.\.\.\s*$",          ""),
    (r"\s*-\s*$",               ""),
    (r"  +",                    " "),
    (r"^\s+|\s+$",              ""),
    (r",\s+(and|but|or|so)\b",  r" \1"),
]

_VOCAB_CORRECTIONS: dict[str, str] = {
    "fourteen fifty-five": "1455",
    "fourteen fifty five": "1455",
    " an the ":            " and the ",
    "i.e.":                "i.e.",
    "viz.":                "viz.",
}

_REGEX_CORRECTIONS = [
    (r"\b(\w+) th\b",    r"\1th"),
    (r"\bone hundred\b", "100"),
]

_LM_SYSTEM = """\
You are a minimal speech-to-text post-correction system.
Your task: fix ONLY clear transcription errors. When in doubt, leave the text unchanged.

Correct:
- Obvious phonetic mis-recognitions (e.g. "Sina Cherub" → "Sennacherib")
- Number words that should be digits or vice versa when context is clear
- Split compound proper nouns (e.g. "New gate" → "Newgate")

Do NOT:
- Add or remove words unless clearly wrong
- Change punctuation style
- Rewrite or paraphrase sentences
- Add explanations or commentary

Return ONLY the corrected text, nothing else. If nothing needs fixing, return the text as-is.\
"""

_LM_USER = "Correct any transcription errors (return text as-is if correct):\n\n{text}"

_NUMBER_WORDS = {
    'zero','one','two','three','four','five','six','seven','eight','nine',
    'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
    'seventeen','eighteen','nineteen','twenty','hundred','thousand','million',
}

_SKIP_HYP_WORDS = {
    'a','an','the','is','in','on','at','to','of','and','or','but','as',
    'it','its','he','she','we','they','new','old','big','one','two','all',
    'no','not','so','buy','by','be','do','go','my','up','now','had',
}


# ── Improver ──────────────────────────────────────────────────────────────────

class Improver:
    """Applies five sequential strategies to reduce WER.

    Strategy 0 — Re-transcription with domain initial_prompt (optional)
    Strategy 1 — Text normalisation
    Strategy 2 — Vocabulary / domain biasing
    Strategy 3 — LLM post-correction via Ollama (optional)
    Strategy 4 — Lowercase
    """

    def __init__(
        self,
        use_lm:       bool = True,
        retranscribe: bool = True,
        ollama_model: str  = "qwen3.5:4b",
        transcriber   = None,          # Transcriber | None
    ) -> None:
        self._use_lm       = use_lm
        self._retranscribe = retranscribe
        self._ollama_model = ollama_model
        self._transcriber  = transcriber

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame,
            initial_prompt: str | None,
            error_json_path: Path) -> pd.DataFrame:
        """Apply all strategies to every row. Returns augmented DataFrame."""
        df = df.copy()

        if self._retranscribe:
            if self._transcriber is None:
                raise ValueError("retranscribe=True but no Transcriber provided")
            print(f"[improver] Re-transcribing {len(df)} files with initial_prompt …")
            hyps, _ = self._transcriber.transcribe_batch(
                df, initial_prompt=initial_prompt, desc="Re-transcribing"
            )
            df["hypothesis"] = hyps
        else:
            print("[improver] Skipping re-transcription.")

        auto_corrections = self._build_auto_corrections(error_json_path)
        print(f"[improver] Auto-corrections loaded: {len(auto_corrections)} rules")

        after_norm, after_vocab, after_lm, after_lower = [], [], [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Improving"):
            result = self._apply_chain(row["hypothesis"], auto_corrections)
            after_norm .append(result.after_norm)
            after_vocab.append(result.after_vocab)
            after_lm   .append(result.after_lm)
            after_lower.append(result.after_lower)

        df["hypothesis_norm"]  = after_norm
        df["hypothesis_vocab"] = after_vocab
        df["hypothesis_lm"]    = after_lm
        df["hypothesis"]       = after_lower

        return df

    # ── Strategy chain ────────────────────────────────────────────────────

    def _apply_chain(self, hypothesis: str,
                     auto_corrections: dict) -> ImprovementResult:
        s1 = self._normalise(hypothesis)
        s2 = self._vocab_correct(s1, auto_corrections)
        s3 = self._lm_correct(s2) if self._use_lm else s2
        s4 = s3.lower()
        return ImprovementResult(
            after_norm=s1, after_vocab=s2, after_lm=s3, after_lower=s4
        )

    # ── Strategy 1: text normalisation ───────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        for pattern, repl in _NORMALISATION_RULES:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text.strip()

    # ── Strategy 2: vocabulary biasing ───────────────────────────────────

    @staticmethod
    def _vocab_correct(text: str, auto_corrections: dict) -> str:
        for wrong, right in _VOCAB_CORRECTIONS.items():
            text = text.replace(wrong, right)
        for pattern, repl in _REGEX_CORRECTIONS:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        for wrong, right in auto_corrections.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', right,
                          text, flags=re.IGNORECASE)
        return text.strip()

    # ── Strategy 3: LLM correction ───────────────────────────────────────

    def _lm_correct(self, text: str) -> str:
        try:
            from ollama import chat
            response = chat(
                model    = self._ollama_model,
                messages = [
                    {"role": "system", "content": _LM_SYSTEM},
                    {"role": "user",   "content": _LM_USER.format(text=text)},
                ],
                options={"temperature": 0.0, "num_predict": 256},
            )
            result = response["message"]["content"].strip()
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return result if result else text
        except Exception as exc:
            print(f"[improver] LM error: {exc}")
            return text

    # ── Auto-correction builder ───────────────────────────────────────────

    @staticmethod
    def _build_auto_corrections(error_json_path: Path,
                                min_count: int = 5) -> dict:
        """Derive substitution corrections from error_analysis.json.

        Five guards prevent over-correction:
          1. min_count >= 5
          2. Skip common function words as HYP source
          3. Skip if HYP starts with REF (compound-split artifact)
          4. Skip if HYP is a substring of REF (model-name suffix artifact)
          5. Skip digit / number-word swaps (TRANSFORM handles these)
          6. Skip REF tokens with no vowels (ligature artifact)
        """
        if not error_json_path.exists():
            return {}
        with open(error_json_path) as f:
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
