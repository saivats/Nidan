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
    ConfidenceHistogram,
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
    cosine_similarity_to_set,
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
SEED_PER_CLASS = 4

ANATOMICAL_REGIONS = [
    "left_lung_upper",
    "left_lung_lower",
    "right_lung_upper",
    "right_lung_lower",
    "cardiac_silhouette",
    "mediastinum",
    "costophrenic_angle",
    "hilar_region",
]


def _safe_score(s: float) -> float:
    return max(0.01, min(0.99, float(s)))


def _assign_anatomical_region(embedding: np.ndarray, seed: int = 0) -> str:
    hash_val = int(abs(np.sum(embedding * 1000))) + seed
    return ANATOMICAL_REGIONS[hash_val % len(ANATOMICAL_REGIONS)]


def _assign_patient_priority(uncertainty: float) -> str:
    if uncertainty > 0.8:
        return "critical"
    if uncertainty > 0.5:
        return "urgent"
    return "routine"


def _compute_confidence_histogram(proba: np.ndarray) -> ConfidenceHistogram:
    max_conf = np.max(proba, axis=1)
    return ConfidenceHistogram(
        very_low=int(np.sum(max_conf < 0.2)),
        low=int(np.sum((max_conf >= 0.2) & (max_conf < 0.4))),
        medium=int(np.sum((max_conf >= 0.4) & (max_conf < 0.6))),
        high=int(np.sum((max_conf >= 0.6) & (max_conf < 0.8))),
        very_high=int(np.sum(max_conf >= 0.8)),
    )


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
            if pool_labels_arr[idx] not in state.classes_discovered:
                state.classes_discovered.append(pool_labels_arr[idx])

        self._model = self._train_model(state)
        state.current_auc = self._compute_auc(state)
        state.auc_trajectory.append(state.current_auc)
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

        annotation_cost = state.compute_annotation_cost(revealed_label)

        state.labeled_indices.append(pool_idx)
        state.labeled_labels.append(revealed_label)
        state.unlabeled_indices.remove(pool_idx)
        state.budget_used += annotation_cost
        state.annotation_cost_spent += annotation_cost
        state.step_count += 1

        if revealed_label not in state.classes_discovered:
            state.classes_discovered.append(revealed_label)

        if revealed_label in state.config.rare_classes:
            self._rare_positives_found += 1

        self._model = self._train_model(state)
        new_auc = self._compute_auc(state)
        state.current_auc = new_auc
        state.auc_trajectory.append(new_auc)

        delta_auc = new_auc - old_auc
        redundancy_penalty = compute_redundancy_penalty(new_embedding, labeled_embeddings)
        rare_case_bonus = compute_rare_case_bonus(revealed_label, state.config.rare_classes)

        if labeled_embeddings.shape[0] > 0:
            cos_sim = cosine_similarity_to_set(new_embedding, labeled_embeddings)
            diversity_score = float(np.clip(1.0 - cos_sim, 0.0, 1.0))
        else:
            diversity_score = 1.0

        label_distribution = state.get_class_distribution()

        reward_components = compute_step_reward(
            delta_auc=delta_auc,
            redundancy_penalty=redundancy_penalty,
            rare_case_bonus=rare_case_bonus,
            diversity_score=diversity_score,
            budget_used=state.budget_used,
            total_budget=state.config.budget,
            revealed_label=revealed_label,
            classes_discovered=state.classes_discovered,
            all_classes=state.config.classes,
            label_distribution=label_distribution,
        )

        step_reward_val = reward_components["step_reward"]
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
            diversity_bonus=reward_components["diversity_bonus"],
            class_coverage_bonus=reward_components["class_coverage_bonus"],
            curriculum_multiplier=reward_components["curriculum_multiplier"],
            step_reward=step_reward_val,
            cumulative_auc=new_auc,
        )

        final_state_dict = self.state()
        final_state_dict["rare_positives_found"] = self._rare_positives_found
        grader = GRADER_REGISTRY[state.config.task_id]
        final_score = _safe_score(grader.grade(final_state_dict))

        info: Dict[str, Any] = {
            "revealed_label": revealed_label,
            "annotation_cost": annotation_cost,
            "budget_used": state.budget_used,
            "budget_remaining": max(0, state.config.budget - state.budget_used),
            "rare_positives_found": self._rare_positives_found,
            "success_threshold": state.config.success_threshold,
            "final_score": final_score,
            "budget_phase": state.get_budget_phase(),
            "class_coverage_ratio": state.get_class_coverage_ratio(),
            "auc_trajectory": [float(a) for a in state.auc_trajectory],
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
            "budget_remaining": max(0, s.config.budget - s.budget_used),
            "annotation_cost_spent": s.annotation_cost_spent,
            "current_auc": float(s.current_auc),
            "labeled_set_size": len(s.labeled_indices),
            "unlabeled_pool_size": len(s.unlabeled_indices),
            "label_distribution": s.get_class_distribution(),
            "class_coverage_ratio": s.get_class_coverage_ratio(),
            "rare_positives_found": self._rare_positives_found,
            "success_threshold": s.config.success_threshold,
            "cumulative_reward": float(s.cumulative_reward),
            "budget_phase": s.get_budget_phase(),
            "auc_trajectory": [float(a) for a in s.auc_trajectory],
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
            max_iter=LR_MAX_ITER, random_state=42, solver="lbfgs", C=0.1
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

        confidence_histogram = None

        if unlabeled_emb.shape[0] == 0:
            candidates: List[CandidateImage] = []
            embedding_stats = {"mean_uncertainty": 0.0, "mean_diversity_score": 0.0}
        else:
            if self._model is not None:
                try:
                    proba = self._model.predict_proba(unlabeled_emb)
                    uncertainty_scores = per_sample_uncertainty(proba)
                    confidence_histogram = _compute_confidence_histogram(proba)
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
                    region_of_interest=_assign_anatomical_region(
                        unlabeled_emb[i], seed=i
                    ),
                    acquisition_cost=state.config.rare_annotation_cost
                    if float(uncertainty_scores[i]) > 0.7 and state.config.variable_cost
                    else state.config.base_annotation_cost,
                    patient_priority=_assign_patient_priority(
                        float(uncertainty_scores[i])
                    ),
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
            budget_remaining=max(0, state.config.budget - state.budget_used),
            unlabeled_pool_size=len(state.unlabeled_indices),
            current_model_auc=float(state.current_auc),
            candidate_images=candidates,
            last_annotation_result=self._last_annotation_result,
            embedding_stats=embedding_stats,
            episode_history=list(self._episode_history),
            class_distribution=state.get_class_distribution(),
            model_confidence_histogram=confidence_histogram,
            budget_phase=state.get_budget_phase(),
            annotation_cost_spent=state.annotation_cost_spent,
        )
