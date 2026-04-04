from __future__ import annotations

from typing import Any, Dict

from server.graders.grader_base import BaseGrader

TASK2_SCORE_DENOMINATOR = 0.80


class Task2Grader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        final_macro_auc = float(final_state.get("current_auc", 0.0))
        score = min(final_macro_auc / TASK2_SCORE_DENOMINATOR, 1.0)
        return self._clamp(score)
