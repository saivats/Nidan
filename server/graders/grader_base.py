from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, final_state: Dict[str, Any]) -> float:
        """Return a score strictly within (0.0, 1.0)."""

    def _clamp(self, value: float) -> float:
        eps = 1e-6
        return max(eps, min(1.0 - eps, float(value)))
