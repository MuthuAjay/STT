"""
Step 1 – Data Collection
========================
Samples a reproducible subset from LJSpeech-1.1 and saves a manifest CSV
that all downstream scripts can use.

LJSpeech-1.1 contains 13,100 single-speaker English utterances (~24 h total).
We draw 200 samples (~≈ 8–10 min) for fast iteration and another 1-hour split
for full reporting.

Run:
    python src/01_data_collection.py
"""
import os
import random
import shutil
import pandas as pd
from pathlib import Path
from config import (LJSPEECH_METADATA, LJSPEECH_WAVS,
                    SAMPLES_DIR, MANIFEST_CSV, NUM_SAMPLES)

RANDOM_SEED = 42


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(
        LJSPEECH_METADATA,
        sep="|",
        header=None,
        names=["file_id", "transcript", "normalized_transcript"],
    )
    print(f"[data_collection] Loaded {len(df)} entries from LJSpeech-1.1")
    return df


def sample_dataset(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    sampled = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    print(f"[data_collection] Sampled {len(sampled)} utterances (seed={seed})")
    return sampled


def copy_audio(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Copy sampled wavs to data/samples/ and add audio_path column."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    missing = 0
    for _, row in df.iterrows():
        src = os.path.join(LJSPEECH_WAVS, f"{row['file_id']}.wav")
        dst = os.path.join(out_dir, f"{row['file_id']}.wav")
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            paths.append(dst)
        else:
            paths.append(None)
            missing += 1
    df = df.copy()
    df["audio_path"] = paths
    if missing:
        print(f"[data_collection] WARNING: {missing} audio files not found")
    df = df.dropna(subset=["audio_path"]).reset_index(drop=True)
    return df


def main():
    df = load_metadata()
    sampled = sample_dataset(df, NUM_SAMPLES)
    sampled = copy_audio(sampled, SAMPLES_DIR)

    sampled.to_csv(MANIFEST_CSV, index=False)
    print(f"[data_collection] Manifest saved → {MANIFEST_CSV}")

    # Dataset statistics
    print("\n=== Dataset Statistics ===")
    lengths = sampled["transcript"].str.split().str.len()
    print(f"  Total utterances : {len(sampled)}")
    print(f"  Avg words/utt    : {lengths.mean():.1f}")
    print(f"  Min / Max words  : {lengths.min()} / {lengths.max()}")
    char_counts = sampled["transcript"].str.len()
    print(f"  Avg chars/utt    : {char_counts.mean():.1f}")
    print("\nSample entries:")
    print(sampled[["file_id", "transcript"]].head(5).to_string(index=False))
    return sampled


if __name__ == "__main__":
    main()
