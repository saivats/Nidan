from __future__ import annotations

from typing import Any, Dict

from server.graders.grader_base import BaseGrader

TASK1_SUCCESS_THRESHOLD = 0.72
TASK1_BUDGET = 40
TASK1_EFFICIENCY_BONUS_FACTOR = 0.2


class Task1Grader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        final_auc = float(final_state.get("current_auc", 0.0))
        budget_used = int(final_state.get("budget_used", TASK1_BUDGET))

        base_score = min(final_auc / TASK1_SUCCESS_THRESHOLD, 1.0)

        if final_auc >= TASK1_SUCCESS_THRESHOLD:
            efficiency_bonus = 1.0 + TASK1_EFFICIENCY_BONUS_FACTOR * (
                1.0 - budget_used / TASK1_BUDGET
            )
            score = base_score * efficiency_bonus
        else:
            score = base_score

        return self._clamp(score)
