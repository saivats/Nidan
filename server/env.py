from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from server.data.dataset_loader import get_task_classes, get_task_class_weights, load_or_extract_embeddings
from server.graders.grader_task1 import Task1Grader
from server.graders.grader_task2 import Task2Grader
from server.graders.grader_task3 import Task3Grader
from server.models import (
    Action,
    CandidateImage,
    Observation,
    Reward,
    StepSummary,
)
from server.tasks.task1_binary import TASK1_CONFIG
from server.tasks.task2_multiclass import TASK2_CONFIG
from server.tasks.task3_rare import TASK3_CONFIG
from server.tasks.task_base import TaskConfig, TaskState
from server.utils.active_learning import (
    compute_binary_auc,
    compute_mean_diversity_score,
    compute_uncertainty_entropy,
    per_sample_diversity,
    per_sample_uncertainty,
)
from server.utils.reward_calculator import (
    compute_rare_case_bonus,
    compute_redundancy_penalty,
    compute_step_reward,
)

TASK_REGISTRY: Dict[str, TaskConfig] = {
    "task1": TASK1_CONFIG,
    "task2": TASK2_CONFIG,
    "task3": TASK3_CONFIG,
}

GRADER_REGISTRY = {
    "task1": Task1Grader(),
    "task2": Task2Grader(),
    "task3": Task3Grader(),
}

CANDIDATES_PER_STEP = 10
LR_MAX_ITER = 1000
SEED_PER_CLASS = 2

# Safety clamp: strictly inside (0, 1) exclusive
def _safe_score(s: float) -> float:
    return max(0.01, min(0.99, float(s)))


class Nidan:
    def __init__(self) -> None:
        self._state: Optional[TaskState] = None
        self._episode_history: List[StepSummary] = []
        self._label_encoder: Optional[LabelEncoder] = None
        self._model: Optional[LogisticRegression] = None
        self._rare_positives_found: int = 0
        self._last_annotation_result: Optional[str] = None

    def reset(self, task_id: str) -> Observation:
        config = TASK_REGISTRY[task_id]
        pool_emb, pool_labels, val_emb, val_labels = load_or_extract_embeddings(task_id)

        actual_pool_size = min(config.pool_size, pool_emb.shape[0])
        rng = np.random.RandomState(42)
        perm = rng.permutation(pool_emb.shape[0])[:actual_pool_size]

        pool_embeddings = pool_emb[perm].astype(np.float32)
        pool_labels_arr = pool_labels[perm]
        pool_image_ids = [f"{task_id}_img_{i:04d}" for i in range(actual_pool_size)]

        state = TaskState(
            config=config,
            pool_embeddings=pool_embeddings,
            pool_labels=pool_labels_arr,
            pool_image_ids=pool_image_ids,
            val_embeddings=val_emb.astype(np.float32),
            val_labels=val_labels,
        )

        state.unlabeled_indices = list(range(actual_pool_size))
        self._episode_history = []
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(config.classes)
        self._rare_positives_found = 0
        self._last_annotation_result = None

        seed_indices = self._select_seed_indices(state)
        for idx in seed_indices:
            state.labeled_indices.append(idx)
            state.labeled_labels.append(pool_labels_arr[idx])
            state.unlabeled_indices.remove(idx)
            if pool_labels_arr[idx] in config.rare_classes:
                self._rare_positives_found += 1

        self._model = self._train_model(state)
        state.current_auc = self._compute_auc(state)
        self._state = state

        return self._build_observation(state)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        state = self._state
        image_id = action.selected_image_id
        pool_idx = state.find_index_by_image_id(image_id)

        if pool_idx is None:
            raise ValueError(
                f"image_id '{image_id}' is not in the unlabeled pool."
            )

        old_auc = state.current_auc
        revealed_label = state.pool_labels[pool_idx]
        new_embedding = state.pool_embeddings[pool_idx]
        labeled_embeddings = state.get_labeled_embeddings()

        state.labeled_indices.append(pool_idx)
        state.labeled_labels.append(revealed_label)
        state.unlabeled_indices.remove(pool_idx)
        state.budget_used += 1
        state.step_count += 1

        if revealed_label in state.config.rare_classes:
            self._rare_positives_found += 1

        self._model = self._train_model(state)
        new_auc = self._compute_auc(state)
        state.current_auc = new_auc

        delta_auc = new_auc - old_auc
        redundancy_penalty = compute_redundancy_penalty(new_embedding, labeled_embeddings)
        rare_case_bonus = compute_rare_case_bonus(revealed_label, state.config.rare_classes)
        step_reward_val = compute_step_reward(delta_auc, redundancy_penalty, rare_case_bonus)
        state.cumulative_reward += step_reward_val

        self._last_annotation_result = revealed_label
        step_summary = StepSummary(
            step=state.step_count,
            selected_image_id=image_id,
            revealed_label=revealed_label,
            delta_auc=delta_auc,
            step_reward=step_reward_val,
            cumulative_auc=new_auc,
        )
        self._episode_history.append(step_summary)

        budget_exhausted = state.budget_used >= state.config.budget
        auc_reached = new_auc >= state.config.success_threshold
        done = budget_exhausted or auc_reached

        reward = Reward(
            delta_auc=delta_auc,
            redundancy_penalty=redundancy_penalty,
            rare_case_bonus=rare_case_bonus,
            step_reward=step_reward_val,
            cumulative_auc=new_auc,
        )

        # ALWAYS compute and include final_score in every step's info,
        # not just when done=True. The validator reads this field from
        # every step response and rejects 0.0 or 1.0.
        final_state_dict = self.state()
        final_state_dict["rare_positives_found"] = self._rare_positives_found
        grader = GRADER_REGISTRY[state.config.task_id]
        final_score = _safe_score(grader.grade(final_state_dict))

        info: Dict[str, Any] = {
            "revealed_label": revealed_label,
            "budget_used": state.budget_used,
            "budget_remaining": state.config.budget - state.budget_used,
            "rare_positives_found": self._rare_positives_found,
            "success_threshold": state.config.success_threshold,
            "final_score": final_score,  # always present, always strictly in (0,1)
            "done_reason": (
                "budget_exhausted" if budget_exhausted
                else ("auc_threshold_reached" if auc_reached else "in_progress")
            ),
        }

        observation = self._build_observation(state)
        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        if self._state is None:
            return {"status": "not_initialized"}

        s = self._state
        return {
            "task_id": s.config.task_id,
            "task_name": s.config.name,
            "step": s.step_count,
            "budget_used": s.budget_used,
            "budget_remaining": s.config.budget - s.budget_used,
            "current_auc": float(s.current_auc),
            "labeled_set_size": len(s.labeled_indices),
            "unlabeled_pool_size": len(s.unlabeled_indices),
            "label_distribution": self._get_label_distribution(s),
            "rare_positives_found": self._rare_positives_found,
            "success_threshold": s.config.success_threshold,
            "cumulative_reward": float(s.cumulative_reward),
            "episode_history": [h.model_dump() for h in self._episode_history],
        }

    def _select_seed_indices(self, state: TaskState) -> List[int]:
        selected: List[int] = []
        classes = state.config.classes
        labels = state.pool_labels
        rng = np.random.RandomState(42)

        for cls_name in classes:
            cls_indices = [i for i, lbl in enumerate(labels) if lbl == cls_name]
            n_seeds = min(SEED_PER_CLASS, len(cls_indices))
            if n_seeds > 0:
                chosen = rng.choice(cls_indices, size=n_seeds, replace=False)
                selected.extend(chosen.tolist())

        return selected

    def _train_model(self, state: TaskState) -> LogisticRegression:
        labeled_emb = state.get_labeled_embeddings()
        labeled_lbl = np.array(state.labeled_labels)

        if labeled_emb.shape[0] < 2 or len(np.unique(labeled_lbl)) < 2:
            dummy = LogisticRegression(max_iter=LR_MAX_ITER, random_state=42)
            n_classes = len(state.config.classes)
            dummy_X = np.random.randn(n_classes * 2, labeled_emb.shape[1]).astype(np.float32)
            dummy_y = state.config.classes * 2
            dummy.fit(dummy_X, dummy_y)
            return dummy

        model = LogisticRegression(
            max_iter=LR_MAX_ITER, random_state=42, solver="lbfgs", C=1.0
        )
        model.fit(labeled_emb, labeled_lbl)
        return model

    def _compute_auc(self, state: TaskState) -> float:
        if self._model is None:
            return 0.5

        val_emb = state.val_embeddings
        val_labels = state.val_labels

        try:
            proba = self._model.predict_proba(val_emb)
            classes = list(self._model.classes_)
            return compute_binary_auc(val_labels, proba, classes=classes)
        except Exception:
            return 0.5

    def _build_observation(self, state: TaskState) -> Observation:
        unlabeled_emb = state.get_unlabeled_embeddings()
        labeled_emb = state.get_labeled_embeddings()
        unlabeled_ids = state.get_unlabeled_image_ids()

        if unlabeled_emb.shape[0] == 0:
            candidates: List[CandidateImage] = []
            embedding_stats = {"mean_uncertainty": 0.0, "mean_diversity_score": 0.0}
        else:
            if self._model is not None:
                try:
                    proba = self._model.predict_proba(unlabeled_emb)
                    uncertainty_scores = per_sample_uncertainty(proba)
                except Exception:
                    uncertainty_scores = np.ones(unlabeled_emb.shape[0]) * 0.5
            else:
                uncertainty_scores = np.ones(unlabeled_emb.shape[0]) * 0.5

            diversity_scores = per_sample_diversity(unlabeled_emb, labeled_emb)
            diversity_scores = np.clip(diversity_scores, 0.0, 1.0)
            uncertainty_scores = np.clip(uncertainty_scores, 0.0, 1.0)

            prefilter_k = min(30, len(unlabeled_ids))
            top_uncertain_indices = np.argsort(uncertainty_scores)[-prefilter_k:][::-1]

            top_k = min(CANDIDATES_PER_STEP, len(top_uncertain_indices))
            selected_indices: List[int] = []
            remaining = list(top_uncertain_indices)

            selected_indices.append(remaining.pop(0))

            while len(selected_indices) < top_k and remaining:
                best_idx = max(remaining, key=lambda i: diversity_scores[i])
                remaining.remove(best_idx)
                selected_indices.append(best_idx)

            candidates = [
                CandidateImage(
                    image_id=unlabeled_ids[i],
                    uncertainty_score=float(uncertainty_scores[i]),
                    diversity_score=float(diversity_scores[i]),
                    modality=state.config.modality,
                    body_part=state.config.body_part,
                )
                for i in selected_indices
            ]

            mean_uncertainty = float(np.mean(uncertainty_scores))
            mean_diversity = compute_mean_diversity_score(unlabeled_emb, labeled_emb)
            embedding_stats = {
                "mean_uncertainty": mean_uncertainty,
                "mean_diversity_score": mean_diversity,
            }

        return Observation(
            task_id=state.config.task_id,
            step=state.step_count,
            budget_remaining=state.config.budget - state.budget_used,
            unlabeled_pool_size=len(state.unlabeled_indices),
            current_model_auc=float(state.current_auc),
            candidate_images=candidates,
            last_annotation_result=self._last_annotation_result,
            embedding_stats=embedding_stats,
            episode_history=list(self._episode_history),
        )

    def _get_label_distribution(self, state: TaskState) -> Dict[str, int]:
        distribution: Dict[str, int] = {cls: 0 for cls in state.config.classes}
        for lbl in state.labeled_labels:
            if lbl in distribution:
                distribution[lbl] += 1
        return distribution
