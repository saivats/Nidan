from __future__ import annotations

from typing import Any, Dict, List

from server.graders.grader_base import BaseGrader

RARE_POSITIVES_REQUIRED = 3
TASK3_AUC_DENOMINATOR = 0.70
RARE_DISCOVERY_WEIGHT = 0.4
AUC_WEIGHT = 0.3
DISCOVERY_SPEED_WEIGHT = 0.3


class Task3Grader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        final_auc = float(final_state.get("current_auc", 0.0))
        rare_positives_found = int(final_state.get("rare_positives_found", 0))

        rare_discovery_component = (
            min(rare_positives_found / RARE_POSITIVES_REQUIRED, 1.0) * RARE_DISCOVERY_WEIGHT
        )

        auc_component = min(final_auc / TASK3_AUC_DENOMINATOR, 1.0) * AUC_WEIGHT

        discovery_speed = self._compute_discovery_speed(final_state)
        speed_component = discovery_speed * DISCOVERY_SPEED_WEIGHT

        score = rare_discovery_component + auc_component + speed_component
        return self._clamp(score)

    def _compute_discovery_speed(self, final_state: Dict[str, Any]) -> float:
        episode_history: List[Dict[str, Any]] = final_state.get("episode_history", [])
        if not episode_history:
            return 0.0

        total_steps = len(episode_history)
        if total_steps == 0:
            return 0.0

        rare_step_indices: List[int] = []
        seen_classes = set()

        for idx, step_data in enumerate(episode_history):
            label = step_data.get("revealed_label", "")
            if label and label != "normal" and label not in seen_classes:
                rare_step_indices.append(idx)
                seen_classes.add(label)

        if not rare_step_indices:
            return 0.0

        avg_discovery_step = sum(rare_step_indices) / len(rare_step_indices)
        speed_score = 1.0 - (avg_discovery_step / total_steps)
        return max(0.0, min(1.0, speed_score))
