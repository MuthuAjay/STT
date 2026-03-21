"""
Download LibriSpeech test-other via HuggingFace datasets and build manifest.
Replaces the LJSpeech manifest with a harder, multi-speaker dataset.

Run:
    python src/download_librispeech.py --num-samples 300
"""
import argparse
import io
import os
import sys
import csv

import numpy as np
import soundfile as sf
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MANIFEST_CSV, SAMPLES_DIR

os.makedirs(SAMPLES_DIR, exist_ok=True)


def download_and_save(num_samples: int = 300):
    from datasets import load_dataset, Audio

    print(f"[download] Streaming LibriSpeech test-other (first {num_samples} samples) …")
    ds = load_dataset(
        "openslr/librispeech_asr", "other",
        split="test",
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False, sampling_rate=None))

    rows = []
    saved = 0

    for i, sample in enumerate(ds):
        if saved >= num_samples:
            break

        utt_id   = sample["id"]
        text     = sample["text"].strip()
        audio_d  = sample["audio"]          # {"bytes": ..., "path": ...}

        if not text:
            continue

        # Decode audio bytes → numpy float32 at 16 kHz
        audio_bytes = audio_d.get("bytes")
        if audio_bytes is None:
            continue

        try:
            data, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            print(f"  [skip] {utt_id}: {e}")
            continue

        # Resample to 16 kHz if needed
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16000)
            data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
            sr = 16000

        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # Save wav
        out_path = os.path.join(SAMPLES_DIR, f"{utt_id}.wav")
        data_i16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(out_path, sr, data_i16)

        rows.append({
            "file_id":    utt_id,
            "transcript": text,
            "audio_path": out_path,
            "speaker_id": sample.get("speaker_id", ""),
            "duration_s": round(len(data) / sr, 2),
        })
        saved += 1

        if saved % 50 == 0:
            print(f"  saved {saved}/{num_samples} …")

    # Write manifest
    fieldnames = ["file_id", "transcript", "audio_path", "speaker_id", "duration_s"]
    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_dur = sum(r["duration_s"] for r in rows)
    n_speakers = len(set(r["speaker_id"] for r in rows))
    print(f"\n[download] Done.")
    print(f"  Utterances : {len(rows)}")
    print(f"  Speakers   : {n_speakers}")
    print(f"  Total audio: {total_dur/60:.1f} min")
    print(f"  Manifest   : {MANIFEST_CSV}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=300)
    args = parser.parse_args()

    rows = download_and_save(args.num_samples)

    # Print sample
    import pandas as pd
    df = pd.DataFrame(rows)
    word_counts = df["transcript"].str.split().str.len()
    print(f"\n  Avg words/utt: {word_counts.mean():.1f}")
    print(f"  Avg duration : {df['duration_s'].mean():.1f}s")
    print("\nSample entries:")
    print(df[["file_id", "transcript", "duration_s"]].head(5).to_string(index=False))
