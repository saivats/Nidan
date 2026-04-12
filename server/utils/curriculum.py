from __future__ import annotations

from typing import Dict, List


EARLY_PHASE_THRESHOLD = 0.3
MID_PHASE_THRESHOLD = 0.7

EARLY_COVERAGE_BONUS = 0.10
FIRST_DISCOVERY_BONUS = 0.20
DIMINISHING_RETURNS_DECAY = 0.03


def get_budget_phase(budget_used: int, total_budget: int) -> str:
    if total_budget == 0:
        return "late"
    progress = budget_used / total_budget
    if progress < EARLY_PHASE_THRESHOLD:
        return "early"
    if progress < MID_PHASE_THRESHOLD:
        return "mid"
    return "late"


def compute_curriculum_multiplier(phase: str) -> float:
    phase_multipliers = {
        "early": 1.2,
        "mid": 1.0,
        "late": 0.8,
    }
    return phase_multipliers.get(phase, 1.0)


def compute_class_coverage_bonus(
    revealed_label: str,
    classes_discovered: List[str],
    all_classes: List[str],
) -> float:
    if revealed_label in classes_discovered:
        return 0.0
    return FIRST_DISCOVERY_BONUS


def compute_diminishing_returns_penalty(
    revealed_label: str,
    label_distribution: Dict[str, int],
) -> float:
    count = label_distribution.get(revealed_label, 0)
    if count <= 2:
        return 0.0
    return DIMINISHING_RETURNS_DECAY * (count - 2)


def compute_late_redundancy_amplifier(phase: str) -> float:
    if phase == "late":
        return 2.5
    if phase == "mid":
        return 1.5
    return 1.0
