from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CandidateImage(BaseModel):
    image_id: str
    uncertainty_score: float = Field(ge=0.0, le=1.0)
    diversity_score: float = Field(ge=0.0, le=1.0)
    modality: str = Field(default="xray", pattern="^(xray|mri)$")
    body_part: str = Field(default="chest")
    thumbnail_b64: Optional[str] = None
    region_of_interest: str = Field(default="general")
    acquisition_cost: int = Field(default=1, ge=1)
    patient_priority: str = Field(default="routine", pattern="^(routine|urgent|critical)$")


class StepSummary(BaseModel):
    step: int
    selected_image_id: str
    revealed_label: str
    delta_auc: float
    step_reward: float
    cumulative_auc: float


class ConfidenceHistogram(BaseModel):
    very_low: int = Field(default=0, description="Confidence 0.0-0.2")
    low: int = Field(default=0, description="Confidence 0.2-0.4")
    medium: int = Field(default=0, description="Confidence 0.4-0.6")
    high: int = Field(default=0, description="Confidence 0.6-0.8")
    very_high: int = Field(default=0, description="Confidence 0.8-1.0")


class Observation(BaseModel):
    task_id: str
    step: int
    budget_remaining: int
    unlabeled_pool_size: int
    current_model_auc: float = Field(ge=0.0, le=1.0)
    candidate_images: List[CandidateImage]
    last_annotation_result: Optional[str] = None
    embedding_stats: Dict[str, float]
    episode_history: List[StepSummary]
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    model_confidence_histogram: Optional[ConfidenceHistogram] = None
    budget_phase: str = Field(default="early", pattern="^(early|mid|late)$")
    annotation_cost_spent: int = Field(default=0)


class Action(BaseModel):
    selected_image_id: str
    reasoning: Optional[str] = None


class Reward(BaseModel):
    delta_auc: float
    redundancy_penalty: float = Field(ge=0.0)
    rare_case_bonus: float = Field(ge=0.0)
    diversity_bonus: float = Field(default=0.0, ge=0.0)
    class_coverage_bonus: float = Field(default=0.0, ge=0.0)
    curriculum_multiplier: float = Field(default=1.0)
    step_reward: float
    cumulative_auc: float = Field(ge=0.0, le=1.0)


class ResetRequest(BaseModel):
    task_id: str = Field(default="task1", pattern="^task[123]$")


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict
