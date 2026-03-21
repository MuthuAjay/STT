"""Standalone math/exponent normalizer.

Converts spoken exponent expressions to symbolic notation.

Examples
--------
  "ten to the power of negative four"  → "10^-4"
  "ten power minus four"               → "10^-4"
  "two to the power of thirty two"     → "2^32"
  "10 to the power of minus 6"         → "10^-6"

Usage (CLI)
-----------
  python math_normalizer.py "the learning rate is ten to the power of negative four"
  python math_normalizer.py --file transcript.txt
  python math_normalizer.py --file transcript.txt --out fixed.txt

Usage (import)
--------------
  from math_normalizer import normalise_exponents
  text = normalise_exponents("learning rate ten to the power of negative four")
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# ── Number-word → digit mapping ───────────────────────────────────────────────

_ONES: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# Combined single-token word → integer (covers 0–99)
_WORD_TO_INT: dict[str, int] = {**_ONES, **_TENS}

# Two-word tens+ones combos: "thirty two" → 32  (built programmatically)
_COMPOUND_TO_INT: dict[str, int] = {
    f"{t} {o}": tv + ov
    for t, tv in _TENS.items()
    for o, ov in _ONES.items()
    if 1 <= ov <= 9
}

# All patterns sorted longest-first so compound matches win over single tokens
_ALL_WORDS: list[tuple[str, int]] = sorted(
    list(_WORD_TO_INT.items()) + list(_COMPOUND_TO_INT.items()),
    key=lambda kv: -len(kv[0]),
)

# Pre-build the OR alternation (longest first → correct greedy matching)
_WORD_PATTERN = "|".join(re.escape(w) for w, _ in _ALL_WORDS)
_WORD_TO_INT_MERGED: dict[str, int] = {**_COMPOUND_TO_INT, **_WORD_TO_INT}


def _parse_number(token: str) -> str:
    """Convert a number token (word or digit string) to its integer string."""
    token = token.strip().lower()
    if token.isdigit():
        return token
    if token in _WORD_TO_INT_MERGED:
        return str(_WORD_TO_INT_MERGED[token])
    return token  # fallback: return as-is


# ── Core regex ────────────────────────────────────────────────────────────────

# Bridge phrases between base and exponent
_BRIDGE = (
    r"(?:"
    r"to\s+the\s+power\s+of"
    r"|to\s+the\s+power"
    r"|to\s+the"
    r"|power\s+of"
    r"|power"
    r")"
)

# Optional sign word
_SIGN = r"(?P<sign>negative|minus|positive|plus)?\s*"

# Base token: digit string OR number-word (compound before single)
_BASE_TOK = rf"(?P<base>{_WORD_PATTERN}|\d+)"

# Exponent token: same
_EXP_TOK = rf"(?P<exp>{_WORD_PATTERN}|\d+)"

# Full pattern (case-insensitive)
_EXPONENT_RE = re.compile(
    rf"\b{_BASE_TOK}\s+{_BRIDGE}\s+{_SIGN}{_EXP_TOK}\b",
    re.IGNORECASE,
)


def _replace_match(m: re.Match) -> str:
    base_str = _parse_number(m.group("base"))
    exp_str  = _parse_number(m.group("exp"))
    sign     = m.group("sign") or ""
    sign_ch  = "-" if sign.lower() in ("negative", "minus") else ""
    return f"{base_str}^{sign_ch}{exp_str}"


# ── Public API ────────────────────────────────────────────────────────────────

def normalise_exponents(text: str) -> str:
    """Replace spoken exponent expressions with symbolic notation.

    Parameters
    ----------
    text : str
        Raw transcription text.

    Returns
    -------
    str
        Text with exponent expressions replaced, e.g. ``10^-4``.
    """
    return _EXPONENT_RE.sub(_replace_match, text)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert spoken exponent expressions to symbolic notation."
    )
    parser.add_argument(
        "text", nargs="?", default=None,
        help="Inline text to normalise (wrap in quotes).",
    )
    parser.add_argument(
        "--file", "-f", type=Path, default=None,
        help="Input text file (one transcript per line).",
    )
    parser.add_argument(
        "--out", "-o", type=Path, default=None,
        help="Output file (default: print to stdout).",
    )
    args = parser.parse_args()

    if args.text:
        lines = [args.text]
    elif args.file:
        lines = args.file.read_text().splitlines()
    else:
        parser.print_help()
        sys.exit(1)

    results = [normalise_exponents(line) for line in lines]

    if args.out:
        args.out.write_text("\n".join(results) + "\n")
        print(f"Written to {args.out}")
    else:
        for orig, fixed in zip(lines, results):
            if orig != fixed:
                print(f"  BEFORE: {orig}")
                print(f"  AFTER : {fixed}")
                print()
            else:
                print(f"  (unchanged) {orig}")


# ── Self-test ─────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    cases = [
        ("ten to the power of negative four",        "10^-4"),
        ("ten to the power of minus four",            "10^-4"),
        ("ten power negative four",                   "10^-4"),
        ("ten power minus four",                      "10^-4"),
        ("two to the power of thirty two",            "2^32"),
        ("10 to the power of minus 6",                "10^-6"),
        ("three to the power of positive two",        "3^2"),
        ("five power of negative twenty",             "5^-20"),
        ("learning rate is ten to the power of negative four please", "learning rate is 10^-4 please"),
        ("no match here at all",                      "no match here at all"),
    ]

    passed = failed = 0
    for inp, expected in cases:
        got = normalise_exponents(inp)
        ok  = got == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {inp!r}")
        if not ok:
            print(f"         expected: {expected!r}")
            print(f"         got     : {got!r}")

    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        _run_tests()
    else:
        _cli()
