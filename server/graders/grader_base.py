from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, final_state: Dict[str, Any]) -> float:
        """Return a score strictly between 0.0 and 1.0 (exclusive)."""

    def _clamp(self, value: float) -> float:
        return max(1e-6, min(1.0 - 1e-6, float(value)))
