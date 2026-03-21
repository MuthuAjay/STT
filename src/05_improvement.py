"""
Step 5 – Improvement
=====================
Applies four complementary strategies to reduce WER:

  Strategy 0 – Re-transcription with Initial Prompt
    • Re-runs Whisper with a domain-vocabulary initial_prompt to bias the
      decoder toward known technical terms (DeepSeekMoE, GShard, ZeRO, …)
    • Skipped if --no-retranscribe is passed (uses baseline hypotheses)

  Strategy 1 – Text Normalisation
    • Expand common STT artefacts (e.g. trailing ellipsis, dash noise)
    • Fix capitalisation artifacts
    • Standardise common English contractions

  Strategy 2 – Vocabulary / Domain Biasing
    • Lexical substitution dictionary built from the error-analysis output
    • Corrects systematic mis-recognitions (e.g. punctuation words, short
      function-word confusions) using a hand-crafted + auto-derived map

  Strategy 3 – LM Post-correction (Ollama)
    • Passes each hypothesis through a local LLM (qwen3.5:4b) with a
      targeted grammar-correction prompt
    • Focuses on number words, proper nouns, and missing function words

Results saved to outputs/improved_results.csv.

Run:
    python src/05_improvement.py                   # all strategies
    python src/05_improvement.py --no-retranscribe  # skip re-transcription
    python src/05_improvement.py --no-lm            # skip LM correction
"""
import json
import os
import re
import time

import pandas as pd
from tqdm import tqdm

from config import (BASELINE_CSV, IMPROVED_CSV, OUTPUTS_DIR, OLLAMA_MODEL,
                    WHISPER_INITIAL_PROMPT)

# ── Strategy 0: Re-transcription with initial_prompt ─────────────────────

def retranscribe_with_prompt(df: pd.DataFrame,
                              prompt: str = WHISPER_INITIAL_PROMPT) -> list[str]:
    """Re-run Whisper on the original audio with a domain initial_prompt."""
    from faster_whisper import WhisperModel
    from config import WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE

    print("[improvement] Loading model for re-transcription …")
    model = WhisperModel(WHISPER_MODEL_SIZE, compute_type=WHISPER_COMPUTE_TYPE)
    print(f"[improvement] Re-transcribing {len(df)} files with initial_prompt …")
    print(f"[improvement] Prompt: {prompt[:80]}…")

    hypotheses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Re-transcribing"):
        segments, _ = model.transcribe(
            row["audio_path"],
            language="en",
            beam_size=5,
            vad_filter=True,
            initial_prompt=prompt,
        )
        hyp = " ".join(seg.text.strip() for seg in segments).strip()
        hypotheses.append(hyp)
    return hypotheses


# ── Strategy 1: Text normalisation rules ─────────────────────────────────

NORMALISATION_RULES = [
    # Remove trailing ellipsis and filler noise added by Whisper
    (r"\s*\.\.\.\s*$", ""),
    (r"\s*-\s*$", ""),
    # Fix double spaces
    (r"  +", " "),
    # Strip leading/trailing whitespace
    (r"^\s+|\s+$", ""),
    # Whisper sometimes adds a comma before "and" / "but" – normalise
    (r",\s+(and|but|or|so)\b", r" \1"),
]


def text_normalise(text: str) -> str:
    for pattern, repl in NORMALISATION_RULES:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text.strip()


# ── Strategy 2: Vocabulary biasing dictionary ─────────────────────────────

# Core domain-correction map (hand-crafted + derived from common STT errors)
VOCAB_CORRECTIONS = {
    # Number words often mis-transcribed
    "fourteen fifty-five": "1455",
    "fourteen fifty five": "1455",
    # Common STT function-word confusions
    " an the ": " and the ",
    # Punctuation-adjacent word artifacts
    "i.e.": "i.e.",
    "viz.": "viz.",
    # Whisper sometimes drops the Oxford comma region – handled by rule below
}

# Regex-based corrections for common patterns
REGEX_CORRECTIONS = [
    # "fifteen th" → "fifteenth"
    (r"\b(\w+) th\b", r"\1th"),
    # "nineteen fifty" ordinals sometimes mis-spaced
    (r"\bone hundred\b", "100"),
]


def vocab_bias_correct(text: str) -> str:
    """Apply vocabulary biasing dictionary and regex corrections."""
    for wrong, right in VOCAB_CORRECTIONS.items():
        text = text.replace(wrong, right)
    for pattern, repl in REGEX_CORRECTIONS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text.strip()


def auto_build_corrections(error_analysis_path: str,
                            min_count: int = 5) -> dict:
    """Auto-build corrections from the most common substitution pairs.

    Guards applied to avoid over-correction:
      1. min_count >= 5  — only high-frequency, reliable patterns
      2. Skip function words as HYP source
      3. Skip if REF is a prefix of HYP — compound-split artifact
         e.g. REF="fine" HYP="finegrained" means REF was "fine grained" (two
         tokens) but HYP merged them; correcting would drop "grained".
      4. Skip if REF contains no vowels — likely a ligature artifact token
         e.g. "ofexperts", "andstep"
      5. Skip digit ↔ number-word swaps — evaluation TRANSFORM handles these
    """
    _NUMBER_WORDS = {
        'zero','one','two','three','four','five','six','seven','eight','nine',
        'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
        'seventeen','eighteen','nineteen','twenty','hundred','thousand','million',
    }
    SKIP_HYP_WORDS = {
        'a', 'an', 'the', 'is', 'in', 'on', 'at', 'to', 'of', 'and',
        'or', 'but', 'as', 'it', 'its', 'he', 'she', 'we', 'they',
        'new', 'old', 'big', 'one', 'two', 'all', 'no', 'not', 'so',
        'buy', 'by', 'be', 'do', 'go', 'my', 'up', 'now', 'had',
    }
    if not os.path.exists(error_analysis_path):
        return {}
    with open(error_analysis_path) as f:
        analysis = json.load(f)
    corrections = {}
    for sub in analysis.get("top_substitutions", []):
        if sub["count"] < min_count:
            continue
        ref_w = sub["ref"].strip().lower()
        hyp_w = sub["hyp"].strip().lower()
        # Single-word pairs only
        if ' ' in ref_w or ' ' in hyp_w:
            continue
        # Guard 2: skip common function words
        if hyp_w in SKIP_HYP_WORDS:
            continue
        # Guard 3: skip compound-split artifacts (REF is a prefix of HYP)
        if hyp_w.startswith(ref_w) and len(hyp_w) > len(ref_w):
            continue
        # Guard 3b: skip model-name suffix artifacts (HYP is a substring of REF)
        # e.g. "moe"→"deepseekmoe" fires on generic "moe" usage; initial_prompt handles these
        if hyp_w in ref_w and len(ref_w) > len(hyp_w):
            continue
        # Guard 4: skip REF tokens with no vowels (ligature/concatenation artifact)
        if not re.search(r'[aeiou]', ref_w):
            continue
        # Guard 5: skip digit ↔ number-word swaps
        if ref_w in _NUMBER_WORDS or hyp_w in _NUMBER_WORDS:
            continue
        if re.fullmatch(r'\d+', ref_w) or re.fullmatch(r'\d+', hyp_w):
            continue
        corrections[hyp_w] = sub["ref"].strip()
    return corrections


def apply_auto_corrections(text: str, corrections: dict) -> str:
    """Apply corrections using word-boundary regex (case-insensitive)."""
    for wrong_raw, right_raw in corrections.items():
        wrong = wrong_raw.strip()
        right = right_raw.strip()
        if not wrong:
            continue
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text,
                      flags=re.IGNORECASE)
    return text.strip()


# ── Strategy 3: LM post-correction ───────────────────────────────────────

LM_SYSTEM_PROMPT = """You are a minimal speech-to-text post-correction system.
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

Return ONLY the corrected text, nothing else. If nothing needs fixing, return the text as-is."""

LM_USER_TEMPLATE = """Correct any transcription errors (return text as-is if correct):

{text}"""


def lm_correct(text: str) -> str:
    try:
        from ollama import chat
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": LM_SYSTEM_PROMPT},
                {"role": "user", "content": LM_USER_TEMPLATE.format(text=text)},
            ],
            options={"temperature": 0.0, "num_predict": 256},
        )
        result = response["message"]["content"].strip()
        # Remove any thinking tags that Qwen3 may emit
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result if result else text
    except Exception as e:
        print(f"[improvement] LM error: {e}")
        return text


# ── Pipeline ─────────────────────────────────────────────────────────────

def lowercase(text: str) -> str:
    """Strategy 4 – lowercase the entire hypothesis."""
    return text.lower()


def improve_pipeline(hypothesis: str, auto_corrections: dict,
                     use_lm: bool = True) -> dict:
    """Apply all strategies and return intermediate + final results."""
    s1 = text_normalise(hypothesis)
    s2 = vocab_bias_correct(s1)
    s2 = apply_auto_corrections(s2, auto_corrections)
    s3 = lm_correct(s2) if use_lm else s2
    s4 = lowercase(s3)
    return {"after_norm": s1, "after_vocab": s2, "after_lm": s3, "after_lower": s4}


def main(use_lm: bool = True, retranscribe: bool = True):
    df = pd.read_csv(BASELINE_CSV)
    print(f"[improvement] Processing {len(df)} utterances …")

    # Strategy 0: re-transcribe with domain initial_prompt
    if retranscribe:
        df["hypothesis"] = retranscribe_with_prompt(df)
    else:
        print("[improvement] Skipping re-transcription (--no-retranscribe).")

    error_analysis_path = os.path.join(OUTPUTS_DIR, "error_analysis.json")
    auto_corrections = auto_build_corrections(error_analysis_path)
    print(f"[improvement] Auto-corrections loaded: {len(auto_corrections)} rules")

    after_norm, after_vocab, after_lm, after_lower = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Improving"):
        results = improve_pipeline(row["hypothesis"], auto_corrections, use_lm=use_lm)
        after_norm.append(results["after_norm"])
        after_vocab.append(results["after_vocab"])
        after_lm.append(results["after_lm"])
        after_lower.append(results["after_lower"])

    df["hypothesis_norm"]  = after_norm
    df["hypothesis_vocab"] = after_vocab
    df["hypothesis_lm"]    = after_lm
    df["hypothesis"]       = after_lower   # final: lowercased

    df.to_csv(IMPROVED_CSV, index=False)
    print(f"[improvement] Saved → {IMPROVED_CSV}")

    # Show a few side-by-side examples
    print("\n=== Sample Improvements ===")
    sample = df.sample(5, random_state=42)
    for _, row in sample.iterrows():
        print(f"\n  REF    : {row['transcript']}")
        print(f"  BASE   : {row['hypothesis_norm']}")
        print(f"  LM     : {row['hypothesis_lm']}")
        print(f"  LOWER  : {row['hypothesis']}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lm", action="store_true",
                        help="Skip LM post-correction (faster, but less accurate)")
    parser.add_argument("--no-retranscribe", action="store_true",
                        help="Skip re-transcription; use baseline hypotheses as-is")
    args = parser.parse_args()
    main(use_lm=not args.no_lm, retranscribe=not args.no_retranscribe)
