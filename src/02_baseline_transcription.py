"""
Step 2 – Baseline Transcription
================================
Runs faster-whisper (base) on every audio file in the manifest and saves
raw transcriptions to outputs/baseline_results.csv.

Run:
    python src/02_baseline_transcription.py
"""
import os
import time
import pandas as pd
from tqdm import tqdm
from faster_whisper import WhisperModel
from config import (MANIFEST_CSV, BASELINE_CSV,
                    WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE,
                    WHISPER_INITIAL_PROMPT)


def load_model(model_size: str = WHISPER_MODEL_SIZE,
               compute_type: str = WHISPER_COMPUTE_TYPE) -> WhisperModel:
    print(f"[baseline] Loading faster-whisper '{model_size}' ({compute_type}) …")
    model = WhisperModel(model_size, compute_type=compute_type)
    print("[baseline] Model loaded.")
    return model


def transcribe_file(model: WhisperModel, audio_path: str,
                    initial_prompt: str | None = None) -> str:
    segments, _ = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        initial_prompt=initial_prompt,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def run_baseline(manifest_path: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    print(f"[baseline] Transcribing {len(df)} files …")

    model = load_model()

    hypotheses, runtimes = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        t0 = time.perf_counter()
        hyp = transcribe_file(model, row["audio_path"])
        elapsed = time.perf_counter() - t0
        hypotheses.append(hyp)
        runtimes.append(round(elapsed, 3))

    df["hypothesis"] = hypotheses
    df["runtime_s"] = runtimes
    df.to_csv(output_csv, index=False)
    print(f"[baseline] Results saved → {output_csv}")
    return df


def main():
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(
            f"manifest.csv not found at {MANIFEST_CSV}. Run the data prep script first.")

    df = run_baseline(MANIFEST_CSV, BASELINE_CSV)

    print("\n=== Sample Transcriptions ===")
    for _, row in df.head(5).iterrows():
        print(f"\n  REF : {row['transcript']}")
        print(f"  HYP : {row['hypothesis']}")
        print(f"  Time: {row['runtime_s']}s")

    avg_rt = df["runtime_s"].mean()
    print(f"\n[baseline] Avg inference time per file: {avg_rt:.3f}s")
    return df


if __name__ == "__main__":
    main()
