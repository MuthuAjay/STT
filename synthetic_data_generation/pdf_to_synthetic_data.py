"""
PDF → Synthetic STT Dataset
============================
1. Extract text from a research-paper PDF (PyMuPDF)
2. Clean & segment into sentence-level chunks (ML domain)
3. Synthesise each chunk with Fish Audio S2 Pro (multiple prosody styles)
4. Write manifest.csv  (audio_path, transcript, style, char_len, word_count)

Usage:
    python TTS/pdf_to_synthetic_data.py --pdf /path/to/paper.pdf
    python TTS/pdf_to_synthetic_data.py --pdf /path/to/paper.pdf --max-sentences 200 --out-dir TTS/synthetic_ml

Requirements:
    pip install git+https://github.com/fishaudio/fish-speech
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import fitz  # PyMuPDF

MODEL_PATH = Path(os.environ.get(
    "FISH_SPEECH_MODEL",
    Path.home() / ".cache/huggingface/hub/models--fishaudio--s2-pro/snapshots/1de9996b6be38b745688de084d87a5633f714e4e"
))

try:
    import queue
    import threading
    import traceback

    import hydra
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.text2semantic.inference import (
        GenerateRequest,
        WrappedGenerateResponse,
        generate_long,
        init_model,
    )
    from fish_speech.utils.schema import ServeTTSRequest
    from omegaconf import OmegaConf
except ImportError:
    sys.exit(
        "fish-speech not installed.\n"
        "Run: pip install git+https://github.com/fishaudio/fish-speech"
    )


# Max context — generate_long reserves 2048 tokens for audio output, so
# max_seq_len must be >= prompt_len + 2048.  4096 covers any sentence prompt
# (~100 tokens) + full audio output, while cutting KV cache from ~18 GB
# (at default 32768) to ~2.3 GB.  Total VRAM stays well under 24 GB.
MAX_SEQ_LEN = 4096


def _launch_queue(checkpoint_path, device, precision):
    """Like launch_thread_safe_queue but caps max_seq_len to MAX_SEQ_LEN."""
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(checkpoint_path, device, precision, compile=False)
        model.config.max_seq_len = MAX_SEQ_LEN  # override before cache allocation
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=MAX_SEQ_LEN,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()
        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break
            kwargs = item.request
            response_queue = item.response_queue
            try:
                for chunk in generate_long(model=model, decode_one_token=decode_one_token, **kwargs):
                    response_queue.put(WrappedGenerateResponse(status="success", response=chunk))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                from loguru import logger
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()
    return input_queue

_DAC_CONFIG = Path(__file__).parent.parent / ".venv/lib64/python3.11/site-packages/fish_speech/configs/modded_dac_vq.yaml"
if not _DAC_CONFIG.exists():
    # fallback: search in site-packages
    import fish_speech
    _DAC_CONFIG = Path(fish_speech.__file__).parent / "configs/modded_dac_vq.yaml"


def _load_dac(checkpoint_path: Path, device: str):
    """Load the DAC vocoder directly via hydra instantiation (avoids pyrootutils)."""
    cfg = OmegaConf.load(_DAC_CONFIG)
    model = hydra.utils.instantiate(cfg)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# S2 Pro supports free-form inline prosody tags — cycle through styles
# instead of voices to add acoustic variety to the dataset.
STYLES = [
    "",                            # neutral / default
    "[emphasis]",                  # slightly stressed
    "[low voice]",                 # quieter, intimate
    "[excited tone]",              # energetic
    "[professional broadcast tone]",  # clean, broadcast
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MIN_WORDS = 6
MAX_WORDS = 60

# ---------------------------------------------------------------------------
# P0 — Ligature repair
# PDF ligature glyphs (fi, fl, ff, ffi, ffl) are often dropped entirely by
# PyMuPDF, producing broken words like "efciency", "signicantly", "dene".
# We repair them before any other processing so reference text is correct.
# ---------------------------------------------------------------------------

# (broken_pattern, correct_replacement)  — order matters (longer first)
_LIGATURE_REPAIRS = [
    # ffi ligature
    (re.compile(r"\befciency\b",      re.I), "efficiency"),
    (re.compile(r"\befciencies\b",    re.I), "efficiencies"),
    (re.compile(r"\befcient\b",       re.I), "efficient"),
    (re.compile(r"\befciently\b",     re.I), "efficiently"),
    (re.compile(r"\binefcient\b",     re.I), "inefficient"),
    (re.compile(r"\bcoefcient\b",     re.I), "coefficient"),
    (re.compile(r"\bcoefcients\b",    re.I), "coefficients"),
    (re.compile(r"\binsufcient\b",    re.I), "insufficient"),
    (re.compile(r"\binsufciently\b",  re.I), "insufficiently"),
    (re.compile(r"\bsufciently\b",    re.I), "sufficiently"),
    (re.compile(r"\bsufcient\b",      re.I), "sufficient"),
    (re.compile(r"\bofcial\b",        re.I), "official"),
    (re.compile(r"\bofcially\b",      re.I), "officially"),
    (re.compile(r"\bofce\b",          re.I), "office"),
    # fi ligature
    (re.compile(r"\bsignicantly\b",   re.I), "significantly"),
    (re.compile(r"\bsignicance\b",    re.I), "significance"),
    (re.compile(r"\bsignicant\b",     re.I), "significant"),
    (re.compile(r"\bdene\b",          re.I), "define"),
    (re.compile(r"\bdened\b",         re.I), "defined"),
    (re.compile(r"\bdenes\b",         re.I), "defines"),
    (re.compile(r"\bdenition\b",      re.I), "definition"),
    (re.compile(r"\bdenitions\b",     re.I), "definitions"),
    (re.compile(r"\bundened\b",       re.I), "undefined"),
    (re.compile(r"\bbenecial\b",      re.I), "beneficial"),
    (re.compile(r"\bbenets\b",        re.I), "benefits"),
    (re.compile(r"\bbenet\b",         re.I), "benefit"),
    (re.compile(r"\bspecc\b",         re.I), "specific"),
    (re.compile(r"\bspeccally\b",     re.I), "specifically"),
    (re.compile(r"\bspeccation\b",    re.I), "specification"),
    (re.compile(r"\bclassier\b",      re.I), "classifier"),
    (re.compile(r"\bclassiers\b",     re.I), "classifiers"),
    (re.compile(r"\bclassication\b",  re.I), "classification"),
    (re.compile(r"\bmodied\b",        re.I), "modified"),
    (re.compile(r"\bmodication\b",    re.I), "modification"),
    (re.compile(r"\bverication\b",    re.I), "verification"),
    (re.compile(r"\bnotication\b",    re.I), "notification"),
    (re.compile(r"\bconguration\b",   re.I), "configuration"),
    (re.compile(r"\bsimplied\b",      re.I), "simplified"),
    (re.compile(r"\bunied\b",         re.I), "unified"),
    (re.compile(r"\bamplied\b",       re.I), "amplified"),
    (re.compile(r"\bramied\b",        re.I), "ramified"),
    (re.compile(r"\bcalcied\b",       re.I), "calcified"),
    (re.compile(r"\bsatised\b",       re.I), "satisfied"),
    (re.compile(r"\bjustied\b",       re.I), "justified"),
    (re.compile(r"\bidentied\b",      re.I), "identified"),
    (re.compile(r"\bndings\b",        re.I), "findings"),
    (re.compile(r"\bnding\b",         re.I), "finding"),
    (re.compile(r"\bnds\b",           re.I), "finds"),
    (re.compile(r"\bnd\b",            re.I), "find"),
    (re.compile(r"\bnally\b",         re.I), "finally"),
    (re.compile(r"\bnalize\b",        re.I), "finalize"),
    (re.compile(r"\bnalized\b",       re.I), "finalized"),
    (re.compile(r"\bnalizes\b",       re.I), "finalizes"),
    (re.compile(r"\bnalizing\b",      re.I), "finalizing"),
    (re.compile(r"\bnite\b",          re.I), "finite"),
    (re.compile(r"\bnitely\b",        re.I), "finitely"),
    # fi ligature — "first" family (must come before bare "rst")
    (re.compile(r"\brstly\b",         re.I), "firstly"),
    (re.compile(r"\brst\b",           re.I), "first"),
    # fl ligature
    (re.compile(r"\boating\b",        re.I), "floating"),
    (re.compile(r"\boat\b",           re.I), "float"),
    (re.compile(r"\boats\b",          re.I), "floats"),
    (re.compile(r"\bows\b",           re.I), "flows"),
    (re.compile(r"\bow\b",            re.I), "flow"),
    (re.compile(r"\bexible\b",        re.I), "flexible"),
    (re.compile(r"\bexibility\b",     re.I), "flexibility"),
    (re.compile(r"\buctuate\b",       re.I), "fluctuate"),
    (re.compile(r"\buctuates\b",      re.I), "fluctuates"),
    (re.compile(r"\buctuation\b",     re.I), "fluctuation"),
    (re.compile(r"\buctuations\b",    re.I), "fluctuations"),
    (re.compile(r"\battening\b",      re.I), "flattening"),
    (re.compile(r"\battened\b",       re.I), "flattened"),
    # Compound tokens (space + ligature lost together)
    (re.compile(r"\bofexperts\b",     re.I), "of experts"),
    (re.compile(r"\bwload\b",         re.I), "workload"),
    (re.compile(r"\btflopsgpu\b",     re.I), "TFLOPs per GPU"),
    (re.compile(r"\bhandtuned\b",     re.I), "hand-tuned"),
    (re.compile(r"\btimestep\b",      re.I), "time step"),
]


# ---------------------------------------------------------------------------
# P0b — OCR typo repairs (character transpositions, not ligature drops)
# ---------------------------------------------------------------------------
_TYPO_REPAIRS = [
    (re.compile(r"\bwtih\b"),   "with"),
    (re.compile(r"\bwiht\b"),   "with"),
    (re.compile(r"\bteh\b"),    "the"),
    (re.compile(r"\badn\b"),    "and"),
    (re.compile(r"\bnad\b"),    "and"),
    (re.compile(r"\bfo\b"),     "of"),
    (re.compile(r"\bthsi\b"),   "this"),
    (re.compile(r"\bparmeter\b",  re.I), "parameter"),
    (re.compile(r"\bparmaeter\b", re.I), "parameter"),
    (re.compile(r"\bmodle\b",     re.I), "model"),
]


def repair_ligatures(text: str) -> str:
    """Fix PDF ligature artifacts in raw extracted text."""
    for pattern, replacement in _LIGATURE_REPAIRS:
        text = pattern.sub(replacement, text)
    return text


def repair_typos(text: str) -> str:
    """Fix common PDF OCR character-transposition errors."""
    for pattern, replacement in _TYPO_REPAIRS:
        text = pattern.sub(replacement, text)
    return text


def _is_ligature_artifact(ref: str, hyp: str) -> bool:
    """
    Returns True if ref looks like hyp with ONE ligature sequence (fi/fl/ff/ffi)
    stripped. Tests each ligature independently on the original string so that
    'efficiency' → try remove 'fi' → 'efciency' ✓ (not chained, which would
    consume 'ffi' first and give the wrong result 'eciency').
    Used in P2 vocab rule validation.
    """
    if len(hyp) <= len(ref):
        return False
    hyp_l = hyp.lower()
    ref_l = ref.lower()
    for lig in ("ffi", "ffl", "fi", "fl", "ff"):
        if hyp_l.replace(lig, "", 1) == ref_l:
            return True
    return False


# ---------------------------------------------------------------------------
# P1 — Improved sentence filters
# ---------------------------------------------------------------------------

# Regex to match clearly broken / non-sentence lines
_RE_SKIP = re.compile(
    r"^\s*$"                                    # blank
    r"|^\s*\d+\s*$"                             # page numbers
    r"|^\s*[A-Z\s]{2,}\s*$"                    # all-caps headings
    r"|http\S+"                                 # URLs
    r"|@"                                       # emails / citations
    r"|\[\d[\d,\s]*\]"                         # reference tags [1], [1,2]
    r"|Figure\s+\d|Table\s+\d"                 # figure/table captions
    r"|Algorithm\s+\d|Equation\s+\d"
    r"|arXiv|doi:|preprint"
    r"|[Uu]nder\s+review"                      # paper submission metadata
    r"|[Cc]onference\s+paper"
    r"|[Ww]orkshop\s+paper"
    r"|\bICLR\b|\bICML\b|\bNeurIPS\b|\bACL\b|\bEMNLP\b"  # venue names
    r"|\bet\s+al\b"                             # citation fragments "et al."
    r"|[=<>]{1}\s*\w"                           # math equations containing = < >
    r"|\b[A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,}"   # table rows (e.g. "GNMT Mono GNMT Multi")
    r"|\(\s*[A-Z][a-z]+.*?\d{4}\s*\)"          # parenthetical citations (Author, 2024)
    r"|\d+\.\d+\.\d+"                           # merged numbers e.g. "0.740.90"
    r"|\d+\.\d+\s+\d+\.\d+"                    # space-separated number ranges e.g. "1.07 1.29"
    r"|(?:MoE|model)\s+\d+.*?(?:MoE|model)\s+\d+.*?(?:MoE|model)\s+\d+"  # model name lists "MoE 4, MoE 32, MoE 256"
)

# Characters that break TTS / cause hallucinations
_RE_CLEAN = [
    (re.compile(r"\s+"), " "),                           # normalise whitespace
    (re.compile(r"[^\x00-\x7F]"), ""),                   # drop non-ASCII
    (re.compile(r"(?<=[a-zA-Z])\d[\d,]*(?=\s|,|$)"), ""),  # strip affiliation superscripts: "Dai1,2" → "Dai"
    (re.compile(r"[\(\)\[\]\{\}]"), ""),                 # remove brackets
    (re.compile(r"\b(\w+)\s*[-–]\s*(\w+)\b"), r"\1 \2"),  # hyphen → space
    (re.compile(r"(?<=[a-z])\.(?=[A-Z])"), ". "),        # fix missing space after period
    (re.compile(r"\d+\.\d{3,}"), ""),                    # remove merged decimals like "0.740.90"
    (re.compile(r"\s{2,}"), " "),                        # collapse spaces again
]

# Reject sentences with high symbol density (math notation leaking through)
def _has_symbol_noise(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    symbol_tokens = sum(1 for w in words if re.fullmatch(r"[^a-zA-Z]+", w))
    # Reject if >20% of tokens are pure non-alphabetic
    return symbol_tokens / len(words) > 0.20

# ML-domain keyword filter — only keep sentences that mention ML concepts
_ML_KEYWORDS = re.compile(
    r"\b(model|layer|loss|gradient|attention|transformer|neural|training|"
    r"inference|embedding|token|weight|parameter|batch|epoch|learning rate|"
    r"encoder|decoder|head|softmax|relu|dropout|dataset|accuracy|precision|"
    r"recall|benchmark|fine.?tun|pre.?train|backprop|optimiz|LLM|GPT|BERT|"
    r"mixture.of.expert|MoE|router|expert|gating|sparse|dense|feed.forward|"
    r"self.attention|cross.attention|residual|normali[sz]|tokeniz|vocabulary|"
    r"sequence|context|window|perplexity|FLOP|compute|activation|convolution)\b",
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    raw = "\n".join(pages)
    raw = repair_ligatures(raw)   # P0:  fix missing glyph ligature artifacts
    raw = repair_typos(raw)       # P0b: fix character-transposition OCR errors
    return raw


def split_into_sentences(raw_text: str) -> list[str]:
    """Split on sentence boundaries; keep only clean ML sentences."""
    # Join hyphenated line-breaks (common in two-column PDFs)
    text = re.sub(r"-\n(\w)", r"\1", raw_text)
    text = re.sub(r"\n", " ", text)

    # Rough sentence split on '. ', '! ', '? '
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text)

    cleaned = []
    for part in parts:
        part = part.strip()

        # P1: skip broken / noisy lines
        if _RE_SKIP.search(part):
            continue

        # Apply cleaning transforms
        for pattern, repl in _RE_CLEAN:
            part = pattern.sub(repl, part)
        part = part.strip()

        # P1: reject high symbol-density (math notation)
        if _has_symbol_noise(part):
            continue

        words = part.split()
        if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
            continue

        # Must be ML-domain
        if not _ML_KEYWORDS.search(part):
            continue

        # Must end with a real sentence terminator
        if not part[-1] in ".!?":
            part += "."

        cleaned.append(part)

    # Deduplicate (PDF columns can duplicate text)
    seen = set()
    unique = []
    for s in cleaned:
        key = re.sub(r"\s+", " ", s.lower())
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


# ---------------------------------------------------------------------------
# TTS synthesis — Fish Audio S2 Pro
# ---------------------------------------------------------------------------

def load_tts_engine() -> TTSInferenceEngine:
    """Load the S2 Pro model and return a ready-to-use inference engine."""
    print(f"  Loading model from {MODEL_PATH} on {DEVICE} …")
    llama_queue = _launch_queue(MODEL_PATH, DEVICE, DTYPE)
    decoder_model = _load_dac(MODEL_PATH / "codec.pth", DEVICE)
    return TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=DTYPE,
        compile=False,
    )


def synthesise_one(engine: TTSInferenceEngine, text: str, style: str, out_path: Path) -> None:
    """Synthesise a single sentence and save as WAV."""
    # Prepend style tag if non-empty
    tagged_text = f"{style} {text}".strip() if style else text
    request = ServeTTSRequest(
        text=tagged_text,
        format="wav",
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    audio_data = None
    sample_rate = 44100
    for result in engine.inference(request):
        if result.code == "final":
            sample_rate, audio_data = result.audio
            break
        elif result.code == "error":
            raise RuntimeError(f"TTS inference error: {result.error}")
    if audio_data is None:
        raise RuntimeError(f"No audio produced for: {text[:60]}")
    sf.write(str(out_path), audio_data, sample_rate)


def synthesise_all(
    engine: TTSInferenceEngine,
    sentences: list[str],
    out_dir: Path,
    styles: list[str],
) -> list[dict]:
    """Synthesise each sentence, cycling through styles. Returns manifest rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(sentences)
    for idx, sentence in enumerate(sentences):
        style = styles[idx % len(styles)]
        filename = f"ml_synth_{idx:04d}.wav"
        audio_path = out_dir / filename
        synthesise_one(engine, sentence, style, audio_path)
        records.append({
            "audio_path": str(audio_path),
            "transcript": sentence,
            "style": style or "neutral",
            "word_count": len(sentence.split()),
            "char_len": len(sentence),
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1}/{total}] synthesised", flush=True)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_manifest(records: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "transcript", "style", "word_count", "char_len"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nManifest written → {manifest_path}  ({len(records)} entries)")


def main(pdf_path: str, out_dir: str, max_sentences: int) -> None:
    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.csv"

    print(f"=== PDF → Synthetic ML STT Dataset (Fish Audio S2 Pro) ===")
    print(f"PDF        : {pdf_path.name}")
    print(f"Output dir : {out_dir}")
    print(f"Model      : {MODEL_PATH.name}")
    print(f"Device     : {DEVICE}")
    print(f"Styles     : {len(STYLES)} prosody variants")

    # Step 1 — Extract
    print("\n[1/4] Extracting text from PDF …")
    raw = extract_text_from_pdf(str(pdf_path))
    print(f"  Extracted {len(raw):,} characters")

    # Step 2 — Clean & segment
    print("\n[2/4] Cleaning & segmenting …")
    sentences = split_into_sentences(raw)
    print(f"  Found {len(sentences)} ML sentences")

    if not sentences:
        print("  No suitable sentences found. Check the PDF or relax filters.")
        return

    if max_sentences and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
        print(f"  Capped to {max_sentences} sentences")

    # Preview
    print("\n  Sample sentences:")
    for s in sentences[:5]:
        print(f"    · {s[:100]}")

    # Step 3 — Load model
    print("\n[3/4] Loading Fish Audio S2 Pro …")
    engine = load_tts_engine()
    print("  Model ready.")

    # Step 4 — Synthesise
    print(f"\n[4/4] Synthesising {len(sentences)} audio clips …")
    records = synthesise_all(engine, sentences, audio_dir, STYLES)

    # Manifest
    write_manifest(records, manifest_path)

    # Stats
    total_words = sum(r["word_count"] for r in records)
    print(f"\n  Total words  : {total_words:,}")
    print(f"  Audio files  : {len(records)} WAVs in {audio_dir}")
    print(f"  Styles used  : {len(STYLES)}")
    print("\nDone. Use this manifest as input to the STT pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → Synthetic ML STT Dataset (Fish Audio S2 Pro)")
    parser.add_argument("--pdf",           required=True,                         help="Path to research paper PDF")
    parser.add_argument("--out",           default="synthetic_data_generation/output", help="Output directory")
    parser.add_argument("--max-sentences", type=int, default=200,                 help="Cap number of sentences (0=all)")
    parser.add_argument("--model",         default=None,                          help="Override model path")
    args = parser.parse_args()

    if args.model:
        MODEL_PATH = Path(args.model)

    main(args.pdf, args.out, args.max_sentences)
