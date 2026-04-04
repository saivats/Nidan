from __future__ import annotations

import numpy as np
from typing import List

STEP_REWARD_CLIP_MIN = -0.1
STEP_REWARD_CLIP_MAX = 0.3
REDUNDANCY_SIMILARITY_THRESHOLD = 0.85
REDUNDANCY_PENALTY_SCALE = 0.05
RARE_CASE_BONUS = 0.15


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
) -> float:
    raw_reward = delta_auc - redundancy_penalty + rare_case_bonus
    return float(np.clip(raw_reward, STEP_REWARD_CLIP_MIN, STEP_REWARD_CLIP_MAX))
