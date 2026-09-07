from __future__ import annotations

import re
import time
from dataclasses import dataclass


def normalize_plate(text: str) -> str:
    """Keep only uppercase Latin letters and digits returned by OCR."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


@dataclass(frozen=True)
class Recognition:
    plate: str
    confidence: float
    timestamp: float
    image_path: str | None = None


class CooldownFilter:
    """Suppress duplicate readings of the same plate for a short period."""

    def __init__(self, seconds: float = 10.0) -> None:
        self.seconds = seconds
        self._last_seen: dict[str, float] = {}

    def allow(self, plate: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        previous = self._last_seen.get(plate)
        if previous is not None and now - previous < self.seconds:
            return False
        self._last_seen[plate] = now
        return True

