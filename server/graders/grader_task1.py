from __future__ import annotations

from typing import Any, Dict

from server.graders.grader_base import BaseGrader

TASK1_SUCCESS_THRESHOLD = 0.72
TASK1_BUDGET = 40
AUC_WEIGHT = 0.6
EFFICIENCY_WEIGHT = 0.2
CLASS_BALANCE_WEIGHT = 0.2


class Task1Grader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        final_auc = float(final_state.get("current_auc", 0.0))
        budget_used = int(final_state.get("budget_used", TASK1_BUDGET))

        auc_component = min(final_auc / TASK1_SUCCESS_THRESHOLD, 1.0) * AUC_WEIGHT

        efficiency_ratio = 1.0 - (budget_used / TASK1_BUDGET)
        efficiency_component = max(0.0, efficiency_ratio) * EFFICIENCY_WEIGHT

        label_dist = final_state.get("label_distribution", {})
        if label_dist and sum(label_dist.values()) > 0:
            total_labels = sum(label_dist.values())
            proportions = [count / total_labels for count in label_dist.values()]
            n_classes = len(proportions)
            if n_classes > 1:
                ideal = 1.0 / n_classes
                imbalance = sum(abs(p - ideal) for p in proportions) / (2.0 * (1.0 - ideal))
                balance_score = 1.0 - min(imbalance, 1.0)
            else:
                balance_score = 0.5
        else:
            balance_score = 0.5

        class_balance_component = balance_score * CLASS_BALANCE_WEIGHT

        score = auc_component + efficiency_component + class_balance_component
        return self._clamp(score)
