from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, final_state: Dict[str, Any]) -> float:
        """Return a score in [0.0, 1.0]."""

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
