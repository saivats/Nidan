from __future__ import annotations

from typing import Any, Dict

from server.graders.grader_base import BaseGrader

TASK2_SUCCESS_THRESHOLD = 0.75
TASK2_BUDGET = 30
TASK2_EFFICIENCY_BONUS_FACTOR = 0.15


class Task2Grader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        final_macro_auc = float(final_state.get("current_auc", 0.0))
        budget_used = int(final_state.get("budget_used", TASK2_BUDGET))

        base_score = min(final_macro_auc / TASK2_SUCCESS_THRESHOLD, 1.0)

        if final_macro_auc >= TASK2_SUCCESS_THRESHOLD:
            efficiency_bonus = 1.0 + TASK2_EFFICIENCY_BONUS_FACTOR * (
                1.0 - budget_used / TASK2_BUDGET
            )
            score = base_score * efficiency_bonus
        else:
            score = base_score

        return self._clamp(score)

