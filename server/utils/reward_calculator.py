from __future__ import annotations

from typing import Dict, List

import numpy as np

from server.utils.curriculum import (
    compute_class_coverage_bonus,
    compute_curriculum_multiplier,
    compute_diminishing_returns_penalty,
    compute_late_redundancy_amplifier,
    get_budget_phase,
)

STEP_REWARD_CLIP_MIN = -0.1
STEP_REWARD_CLIP_MAX = 0.3
REDUNDANCY_SIMILARITY_THRESHOLD = 0.85
REDUNDANCY_PENALTY_SCALE = 0.05
RARE_CASE_BONUS = 0.15
DIVERSITY_BONUS_SCALE = 0.05


def compute_redundancy_penalty(
    new_embedding: np.ndarray, labeled_embeddings: np.ndarray
) -> float:
    if labeled_embeddings.shape[0] == 0:
        return 0.0

    mean_labeled = labeled_embeddings.mean(axis=0)
    norm_new = np.linalg.norm(new_embedding)
    norm_mean = np.linalg.norm(mean_labeled)

    if norm_new < 1e-10 or norm_mean < 1e-10:
        return 0.0

    cosine_sim = float(
        np.dot(new_embedding, mean_labeled) / (norm_new * norm_mean)
    )
    excess = max(0.0, cosine_sim - REDUNDANCY_SIMILARITY_THRESHOLD)
    return REDUNDANCY_PENALTY_SCALE * excess


def compute_rare_case_bonus(
    revealed_label: str, rare_classes: List[str]
) -> float:
    return RARE_CASE_BONUS if revealed_label in rare_classes else 0.0


def compute_step_reward(
    delta_auc: float,
    redundancy_penalty: float,
    rare_case_bonus: float,
    diversity_score: float = 0.0,
    budget_used: int = 0,
    total_budget: int = 1,
    revealed_label: str = "",
    classes_discovered: List[str] = None,
    all_classes: List[str] = None,
    label_distribution: Dict[str, int] = None,
) -> Dict[str, float]:
    if classes_discovered is None:
        classes_discovered = []
    if all_classes is None:
        all_classes = []
    if label_distribution is None:
        label_distribution = {}

    phase = get_budget_phase(budget_used, total_budget)
    curriculum_multiplier = compute_curriculum_multiplier(phase)

    diversity_bonus = DIVERSITY_BONUS_SCALE * diversity_score

    class_coverage_bonus = compute_class_coverage_bonus(
        revealed_label, classes_discovered, all_classes
    )

    diminishing_penalty = compute_diminishing_returns_penalty(
        revealed_label, label_distribution
    )

    redundancy_amplifier = compute_late_redundancy_amplifier(phase)
    adjusted_redundancy = redundancy_penalty * redundancy_amplifier

    raw_reward = (
        delta_auc
        + diversity_bonus
        - adjusted_redundancy
        + rare_case_bonus
        + class_coverage_bonus
        - diminishing_penalty
    )

    shaped_reward = raw_reward * curriculum_multiplier
    clipped_reward = float(np.clip(shaped_reward, STEP_REWARD_CLIP_MIN, STEP_REWARD_CLIP_MAX))

    return {
        "step_reward": clipped_reward,
        "diversity_bonus": diversity_bonus,
        "class_coverage_bonus": class_coverage_bonus,
        "curriculum_multiplier": curriculum_multiplier,
        "diminishing_penalty": diminishing_penalty,
    }
