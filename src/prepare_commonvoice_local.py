"""
Prepare Mozilla Common Voice (Australian English) from a local download.

Reads the CSV, filters for quality clips, converts MP3 → WAV at 16 kHz,
and writes data/manifest.csv for the rest of the pipeline.

Run:
    python src/prepare_commonvoice_local.py
    python src/prepare_commonvoice_local.py --num-samples 300 --min-votes 1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

CV_DIR      = config.CV_LOCAL_DIR
CV_CSV      = os.path.join(CV_DIR, "commonvoice-v24_en-AU.csv")
AUDIO_DIR   = os.path.join(CV_DIR, "audio_files")
TARGET_SR   = 16000
RANDOM_SEED = 42


def resample_audio(data: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return data.astype(np.float32)
    g = gcd(orig_sr, TARGET_SR)
    return resample_poly(data, TARGET_SR // g, orig_sr // g).astype(np.float32)


def save_wav(path: str, data: np.ndarray):
    data_i16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(path, TARGET_SR, data_i16)


def load_metadata(min_votes: int) -> pd.DataFrame:
    df = pd.read_csv(CV_CSV, index_col=0)
    print(f"[cv_local] Total entries in CSV : {len(df):,}")

    # Quality filter
    df = df[df["up_votes"] >= min_votes].copy()
    df = df[df["up_votes"] > df["down_votes"]].copy()
    print(f"[cv_local] After quality filter  : {len(df):,}")

    # Drop empty sentences
    df = df.dropna(subset=["sentence"]).copy()
    df["sentence"] = df["sentence"].str.strip()
    df = df[df["sentence"] != ""]

    # Check audio exists
    df["audio_exists"] = df["path"].apply(
        lambda p: os.path.exists(os.path.join(AUDIO_DIR, p))
    )
    df = df[df["audio_exists"]].drop(columns=["audio_exists"])
    print(f"[cv_local] With existing audio   : {len(df):,}")

    return df.reset_index(drop=True)


def convert_and_save(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    sampled = df.sample(n=min(num_samples, len(df)),
                        random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[cv_local] Sampled {len(sampled)} utterances → converting to 16 kHz WAV …")

    rows = []
    failed = 0
    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Converting"):
        src_path = os.path.join(AUDIO_DIR, row["path"])
        utt_id   = os.path.splitext(row["path"])[0]
        dst_path = os.path.join(config.SAMPLES_DIR, f"{utt_id}.wav")

        try:
            data, sr = sf.read(src_path, always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = resample_audio(data, sr)
            save_wav(dst_path, data)
        except Exception as e:
            failed += 1
            continue

        rows.append({
            "file_id":    utt_id,
            "transcript": row["sentence"],
            "audio_path": dst_path,
            "duration_s": round(len(data) / TARGET_SR, 2),
            "age":        row.get("age", ""),
            "gender":     row.get("gender", ""),
            "accent":     row.get("accents", "Australian English"),
            "up_votes":   int(row.get("up_votes", 0)),
        })

    if failed:
        print(f"[cv_local] WARNING: {failed} files failed to convert")
    return pd.DataFrame(rows)


def main(num_samples: int = config.NUM_SAMPLES, min_votes: int = 1):
    config.makedirs()
    df = load_metadata(min_votes)

    result_df = convert_and_save(df, num_samples)

    result_df.to_csv(config.MANIFEST_CSV, index=False)
    print(f"[cv_local] Manifest saved → {config.MANIFEST_CSV}")

    # Stats
    print("\n=== Dataset Statistics ===")
    print(f"  Utterances   : {len(result_df)}")
    print(f"  Total audio  : {result_df['duration_s'].sum()/60:.1f} min")
    print(f"  Avg duration : {result_df['duration_s'].mean():.1f}s")
    wc = result_df["transcript"].str.split().str.len()
    print(f"  Avg words    : {wc.mean():.1f}  (min={wc.min()}, max={wc.max()})")
    ages    = result_df["age"].value_counts().to_dict()
    genders = result_df["gender"].value_counts().to_dict()
    print(f"  Ages         : {ages}")
    print(f"  Genders      : {genders}")
    print("\nSample entries:")
    print(result_df[["file_id","transcript","duration_s","age","gender"]].head(8).to_string(index=False))

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=config.NUM_SAMPLES)
    parser.add_argument("--min-votes",   type=int, default=1,
                        help="Minimum up_votes required (default: 1)")
    args = parser.parse_args()
    main(args.num_samples, args.min_votes)
