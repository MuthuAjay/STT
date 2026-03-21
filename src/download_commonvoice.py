"""
Download Mozilla Common Voice (English) via HuggingFace datasets.

Pre-requisites (one-time):
  1. Create a free account at https://huggingface.co
  2. Accept the dataset licence at:
       https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
  3. Generate a read token at:
       https://huggingface.co/settings/tokens
  4. Either:
       a) Pass it via --hf-token <TOKEN>
       b) Set env var:  export HF_TOKEN=<TOKEN>
       c) Run once:     huggingface-cli login

Run:
    python src/download_commonvoice.py --num-samples 300
    python src/download_commonvoice.py --num-samples 300 --hf-token hf_xxxx
    python src/download_commonvoice.py --num-samples 300 --split validated
"""
import argparse
import csv
import io
import os
import sys
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, SAMPLES_DIR

os.makedirs(SAMPLES_DIR, exist_ok=True)

CV_DATASET   = "mozilla-foundation/common_voice_17_0"
TARGET_SR    = 16000


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    g = gcd(orig_sr, target_sr)
    return resample_poly(data, target_sr // g, orig_sr // g).astype(np.float32)


def save_wav(path: str, data: np.ndarray, sr: int = TARGET_SR):
    data_i16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(path, sr, data_i16)


def download(num_samples: int, split: str, hf_token: str | None):
    try:
        import soundfile as sf
        from datasets import load_dataset, Audio
    except ImportError:
        print("[error] Run: pip install datasets soundfile")
        sys.exit(1)

    # Authenticate
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print(f"[cv] Logged in with provided token.")
    elif os.environ.get("HF_TOKEN"):
        print("[cv] Using HF_TOKEN from environment.")
    else:
        print("[cv] No token provided — trying anonymous access (may fail).")
        print("     If it fails, re-run with:  --hf-token hf_xxxx")

    print(f"[cv] Streaming Common Voice 17 · en · {split} (first {num_samples}) …")

    try:
        ds = load_dataset(
            CV_DATASET, "en",
            split=split,
            streaming=True,
            token=hf_token or os.environ.get("HF_TOKEN") or True,
        )
        ds = ds.cast_column("audio", Audio(decode=False, sampling_rate=None))
    except Exception as e:
        print(f"\n[error] Could not load dataset: {e}")
        print("\nMake sure you have:")
        print("  1. Accepted the licence at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        print("  2. Generated a token at https://huggingface.co/settings/tokens")
        print("  3. Passed it with --hf-token  or  export HF_TOKEN=...")
        sys.exit(1)

    rows   = []
    saved  = 0
    skipped = 0

    for sample in ds:
        if saved >= num_samples:
            break

        sentence  = sample.get("sentence", "").strip()
        utt_id    = sample.get("path", f"cv_{saved:06d}")
        utt_id    = os.path.splitext(os.path.basename(utt_id))[0]
        audio_d   = sample.get("audio", {})
        age       = sample.get("age", "")
        gender    = sample.get("gender", "")
        accent    = sample.get("accent", "")

        if not sentence or not audio_d.get("bytes"):
            skipped += 1
            continue

        # Decode
        try:
            data, sr = sf.read(io.BytesIO(audio_d["bytes"]))
        except Exception:
            skipped += 1
            continue

        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        data = resample(data, sr, TARGET_SR)

        out_path = os.path.join(SAMPLES_DIR, f"{utt_id}.wav")
        save_wav(out_path, data, TARGET_SR)

        rows.append({
            "file_id":    utt_id,
            "transcript": sentence,
            "audio_path": out_path,
            "duration_s": round(len(data) / TARGET_SR, 2),
            "age":        age,
            "gender":     gender,
            "accent":     accent,
        })
        saved += 1

        if saved % 50 == 0:
            print(f"  saved {saved}/{num_samples} …")

    # Write manifest
    manifest_path = os.path.join(DATA_DIR, "manifest.csv")
    fieldnames = ["file_id", "transcript", "audio_path", "duration_s", "age", "gender", "accent"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_dur  = sum(r["duration_s"] for r in rows)
    accents    = set(r["accent"] for r in rows if r["accent"])
    genders    = set(r["gender"] for r in rows if r["gender"])

    print(f"\n[cv] Done.")
    print(f"  Saved       : {len(rows)} utterances  ({skipped} skipped)")
    print(f"  Total audio : {total_dur/60:.1f} min")
    print(f"  Accents     : {', '.join(sorted(accents)) or 'n/a'}")
    print(f"  Genders     : {', '.join(sorted(genders)) or 'n/a'}")
    print(f"  Manifest    : {manifest_path}")

    return rows


def print_samples(rows: list):
    import pandas as pd
    df = pd.DataFrame(rows)
    wc = df["transcript"].str.split().str.len()
    print(f"\n  Avg words/utt : {wc.mean():.1f}")
    print(f"  Avg duration  : {df['duration_s'].mean():.1f}s")
    print("\nSample entries:")
    print(df[["file_id", "transcript", "duration_s", "accent"]].head(8).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Mozilla Common Voice (English)")
    parser.add_argument("--num-samples", type=int, default=300,
                        help="Number of utterances to download (default: 300)")
    parser.add_argument("--split",       default="test",
                        choices=["test", "dev", "validated", "train"],
                        help="Dataset split (default: test)")
    parser.add_argument("--hf-token",    default=None,
                        help="HuggingFace read token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    rows = download(args.num_samples, args.split, args.hf_token)
    if rows:
        print_samples(rows)
