"""Transcriber — wraps faster-whisper with lazy model loading."""
from __future__ import annotations

import time

import pandas as pd
from tqdm import tqdm


class Transcriber:
    """Thin, stateful wrapper around a faster-whisper WhisperModel.

    The model is loaded lazily on first use via ``load()``.  Callers that
    need explicit control over when GPU memory is allocated can call
    ``load()`` before the batch loop.
    """

    def __init__(
        self,
        model_size:   str = "large-v3",
        compute_type: str = "float16",
        beam_size:    int = 5,
        language:     str = "en",
        vad_filter:   bool = True,
    ) -> None:
        self._model_size   = model_size
        self._compute_type = compute_type
        self._beam_size    = beam_size
        self._language     = language
        self._vad_filter   = vad_filter
        self._model        = None

    # ── Model lifecycle ───────────────────────────────────────────────────

    def load(self) -> "Transcriber":
        """Explicitly load the model.  Returns self for chaining."""
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"[transcriber] Loading faster-whisper '{self._model_size}' "
                  f"({self._compute_type}) …")
            self._model = WhisperModel(self._model_size,
                                        compute_type=self._compute_type)
            print("[transcriber] Model loaded.")
        return self

    # ── Transcription API ─────────────────────────────────────────────────

    def transcribe(self, audio_path: str,
                   initial_prompt: str | None = None) -> str:
        """Transcribe a single audio file and return the hypothesis string."""
        self.load()
        segments, _ = self._model.transcribe(
            audio_path,
            language       = self._language,
            beam_size      = self._beam_size,
            vad_filter     = self._vad_filter,
            initial_prompt = initial_prompt,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_batch(
        self,
        df: pd.DataFrame,
        initial_prompt: str | None = None,
        desc: str = "Transcribing",
    ) -> tuple[list[str], list[float]]:
        """Transcribe every row in *df* (must have an ``audio_path`` column).

        Returns (hypotheses, runtimes_seconds).
        """
        self.load()
        hypotheses: list[str]   = []
        runtimes:   list[float] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            t0  = time.perf_counter()
            hyp = self.transcribe(row["audio_path"], initial_prompt=initial_prompt)
            hypotheses.append(hyp)
            runtimes.append(round(time.perf_counter() - t0, 3))

        return hypotheses, runtimes
