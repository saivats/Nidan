from __future__ import annotations

from typing import Any, Dict

from server.graders.grader_base import BaseGrader

RARE_POSITIVES_REQUIRED = 3
TASK3_AUC_DENOMINATOR = 0.70
POSITIVE_CASE_WEIGHT = 0.5
AUC_WEIGHT = 0.5


class Task3Grader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        final_auc = float(final_state.get("current_auc", 0.0))
        rare_positives_found = int(final_state.get("rare_positives_found", 0))

        positive_case_score = (
            min(rare_positives_found / RARE_POSITIVES_REQUIRED, 1.0) * POSITIVE_CASE_WEIGHT
        )
        auc_score = min(final_auc / TASK3_AUC_DENOMINATOR, 1.0) * AUC_WEIGHT
        score = positive_case_score + auc_score
        return self._clamp(score)
