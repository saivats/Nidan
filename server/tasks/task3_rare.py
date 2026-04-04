from __future__ import annotations

from server.tasks.task_base import TaskConfig

TASK3_CONFIG = TaskConfig(
    task_id="task3",
    name="Rare Pathology Detection",
    classes=["normal", "nodule", "effusion", "pneumothorax"],
    pool_size=600,
    budget=15,
    success_threshold=0.60,
    rare_classes=["nodule", "effusion", "pneumothorax"],
    modality="xray",
    body_part="chest",
)

RARE_POSITIVES_REQUIRED = 3
