from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.utils.curriculum import get_budget_phase, compute_curriculum_multiplier
from server.utils.reward_calculator import compute_step_reward

def test_budget_phase():
    assert get_budget_phase(10, 100) == "early"
    assert get_budget_phase(40, 100) == "mid"
    assert get_budget_phase(80, 100) == "late"

def test_curriculum_multiplier():
    assert compute_curriculum_multiplier("early") > 1.0
    assert compute_curriculum_multiplier("mid") == 1.0
    assert compute_curriculum_multiplier("late") < 1.0

def test_compute_step_reward_basic():
    res = compute_step_reward(
        delta_auc=0.05,
        redundancy_penalty=0.0,
        rare_case_bonus=0.0,
        diversity_score=0.5,
        budget_used=10,
        total_budget=100,
        revealed_label="pneumonia",
        classes_discovered=["normal"],
        all_classes=["normal", "pneumonia"],
        label_distribution={"normal": 5}
    )
    assert "step_reward" in res
    assert "class_coverage_bonus" in res
    assert res["class_coverage_bonus"] > 0.0 # First time finding pneumonia

def test_compute_step_reward_diminishing_returns():
    res = compute_step_reward(
        delta_auc=0.01,
        redundancy_penalty=0.0,
        rare_case_bonus=0.0,
        diversity_score=0.1,
        budget_used=50,
        total_budget=100,
        revealed_label="normal",
        classes_discovered=["normal", "pneumonia"],
        all_classes=["normal", "pneumonia"],
        label_distribution={"normal": 10, "pneumonia": 2}
    )
    assert res["diminishing_penalty"] > 0.0 # Lots of 'normal' cases already
