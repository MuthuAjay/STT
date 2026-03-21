"""Step abstract base class."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

log = logging.getLogger("pipeline")


@dataclass
class SkipResult:
    skip:   bool
    reason: str = ""

    @classmethod
    def yes(cls, reason: str) -> "SkipResult":
        return cls(skip=True, reason=reason)

    @classmethod
    def no(cls) -> "SkipResult":
        return cls(skip=False)


class Step(ABC):
    """Contract every pipeline step must satisfy.

    Subclasses implement ``should_skip()`` and ``run()``.
    ``execute()`` is the template method — never override it.
    """

    @property
    @abstractmethod
    def number(self) -> int:
        """Step index (1-based)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label."""

    @abstractmethod
    def should_skip(self) -> SkipResult:
        """Decide whether to skip before calling ``run()``."""

    @abstractmethod
    def run(self) -> None:
        """Execute the step's core logic."""

    # ── Template method ───────────────────────────────────────────────────

    def execute(self) -> None:
        self._print_header()
        result = self.should_skip()
        if result.skip:
            log.info("  ⏭  %s skipped — %s", self.name, result.reason)
            return
        with self._timed():
            self.run()

    # ── Private helpers ───────────────────────────────────────────────────

    def _print_header(self) -> None:
        log.info("")
        log.info("┌%s┐", "─" * 58)
        log.info("│  STEP %-2d  %-51s│", self.number, self.name)
        log.info("└%s┘", "─" * 58)

    @contextmanager
    def _timed(self) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        yield
        log.info("  ✓  %s completed in %.1fs", self.name, time.perf_counter() - t0)
