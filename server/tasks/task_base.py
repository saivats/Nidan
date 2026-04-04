from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TaskConfig:
    task_id: str
    name: str
    classes: List[str]
    pool_size: int
    budget: int
    success_threshold: float
    rare_classes: List[str]
    modality: str = "xray"
    body_part: str = "chest"


@dataclass
class TaskState:
    config: TaskConfig
    pool_embeddings: np.ndarray
    pool_labels: np.ndarray
    pool_image_ids: List[str]
    val_embeddings: np.ndarray
    val_labels: np.ndarray
    labeled_indices: List[int] = field(default_factory=list)
    labeled_labels: List[str] = field(default_factory=list)
    unlabeled_indices: List[int] = field(default_factory=list)
    budget_used: int = 0
    current_auc: float = 0.5
    step_count: int = 0
    cumulative_reward: float = 0.0

    def get_labeled_embeddings(self) -> np.ndarray:
        if not self.labeled_indices:
            return np.zeros((0, self.pool_embeddings.shape[1]), dtype=np.float32)
        return self.pool_embeddings[self.labeled_indices]

    def get_unlabeled_embeddings(self) -> np.ndarray:
        if not self.unlabeled_indices:
            return np.zeros((0, self.pool_embeddings.shape[1]), dtype=np.float32)
        return self.pool_embeddings[self.unlabeled_indices]

    def get_unlabeled_image_ids(self) -> List[str]:
        return [self.pool_image_ids[i] for i in self.unlabeled_indices]

    def find_index_by_image_id(self, image_id: str) -> Optional[int]:
        try:
            pool_pos = self.pool_image_ids.index(image_id)
            if pool_pos in self.unlabeled_indices:
                return pool_pos
            return None
        except ValueError:
            return None
