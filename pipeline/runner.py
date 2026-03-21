"""PipelineRunner — iterates steps, guards the manifest, logs timing."""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from typing import Sequence

from .base import Step
from .config import PathConfig, PipelineConfig

log = logging.getLogger("pipeline")


class PipelineRunner:
    """Runs an ordered list of Steps.

    Contains zero business logic — all of that lives in core/ and steps/.
    """

    def __init__(self, steps: Sequence[Step],
                 path_cfg: PathConfig, run_cfg: PipelineConfig) -> None:
        self._steps    = steps
        self._path_cfg = path_cfg
        self._run_cfg  = run_cfg

    def run(self) -> None:
        self._banner(f"STT Pipeline  ·  experiment: {self._path_cfg.experiment}")
        log.info(self._path_cfg.summary())
        log.info("  lm           : %s",
                 self._path_cfg.ollama_model if self._run_cfg.use_lm else "disabled")
        log.info("  retranscribe : %s",
                 "enabled" if self._run_cfg.retranscribe else "disabled")
        log.info("  started      : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self._guard_manifest()

        t0 = time.perf_counter()
        for step in self._steps:
            if self._run_cfg.should_run(step.number):
                step.execute()

        self._banner("Pipeline Complete")
        log.info("  total time : %.1fs", time.perf_counter() - t0)
        log.info("  outputs    : %s", self._path_cfg.outputs_dir)
        log.info("  finished   : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _guard_manifest(self) -> None:
        needs = any(self._run_cfg.should_run(n) for n in range(1, 6))
        if needs and not self._path_cfg.manifest_csv.exists():
            log.error(
                "Manifest not found: %s\n"
                "  Provide one via --manifest <path> or STT_MANIFEST env var.",
                self._path_cfg.manifest_csv,
            )
            sys.exit(1)

    @staticmethod
    def _banner(title: str, width: int = 60) -> None:
        log.info("=" * width)
        log.info("  %s", title)
        log.info("=" * width)
