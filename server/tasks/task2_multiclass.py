from __future__ import annotations

from server.tasks.task_base import TaskConfig

TASK2_CONFIG = TaskConfig(
    task_id="task2",
    name="Multi-class Chest Conditions",
    classes=["normal", "pneumonia", "covid", "tuberculosis"],
    pool_size=400,
    budget=30,
    success_threshold=0.75,
    rare_classes=["covid", "tuberculosis"],
    modality="xray",
    body_part="chest",
)
