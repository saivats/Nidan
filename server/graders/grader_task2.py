from __future__ import annotations

from typing import Any, Dict

from server.graders.grader_base import BaseGrader

TASK2_SUCCESS_THRESHOLD = 0.65
TASK2_BUDGET = 30
AUC_WEIGHT = 0.5
EFFICIENCY_WEIGHT = 0.25
CLASS_COVERAGE_WEIGHT = 0.25


class Task2Grader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        final_macro_auc = float(final_state.get("current_auc", 0.0))
        budget_used = int(final_state.get("budget_used", TASK2_BUDGET))

        auc_component = min(final_macro_auc / TASK2_SUCCESS_THRESHOLD, 1.0) * AUC_WEIGHT

        efficiency_ratio = 1.0 - (budget_used / TASK2_BUDGET)
        efficiency_component = max(0.0, efficiency_ratio) * EFFICIENCY_WEIGHT

        class_coverage = float(final_state.get("class_coverage_ratio", 0.0))
        if class_coverage == 0.0:
            label_dist = final_state.get("label_distribution", {})
            if label_dist:
                classes_with_labels = sum(1 for v in label_dist.values() if v > 0)
                total_classes = len(label_dist)
                class_coverage = classes_with_labels / total_classes if total_classes > 0 else 0.0

        coverage_component = class_coverage * CLASS_COVERAGE_WEIGHT

        score = auc_component + efficiency_component + coverage_component
        return self._clamp(score)
