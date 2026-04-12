from __future__ import annotations

from server.tasks.task_base import TaskConfig

TASK1_CONFIG = TaskConfig(
    task_id="task1",
    name="Binary Pneumonia Detection",
    classes=["normal", "pneumonia"],
    pool_size=200,
    budget=40,
    success_threshold=0.72,
    rare_classes=["pneumonia"],
    modality="xray",
    body_part="chest",
    variable_cost=False,
    base_annotation_cost=1,
)
