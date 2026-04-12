from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.graders.grader_task1 import Task1Grader
from server.graders.grader_task2 import Task2Grader
from server.graders.grader_task3 import Task3Grader


def test_task1_grader_multi_metric():
    grader = Task1Grader()
    state = {
        "current_auc": 0.72,
        "budget_used": 20,
        "label_distribution": {"normal": 10, "pneumonia": 10}
    }
    score = grader.grade(state)
    assert 0.0 < score <= 1.0
    
    # Imbalanced state should score lower
    bad_state = {
        "current_auc": 0.72,
        "budget_used": 20,
        "label_distribution": {"normal": 19, "pneumonia": 1}
    }
    bad_score = grader.grade(bad_state)
    assert bad_score < score

def test_task2_grader_class_coverage():
    grader = Task2Grader()
    state_good = {
        "current_auc": 0.65,
        "budget_used": 15,
        "class_coverage_ratio": 1.0
    }
    score_good = grader.grade(state_good)
    
    state_bad = {
        "current_auc": 0.65,
        "budget_used": 15,
        "class_coverage_ratio": 0.5
    }
    score_bad = grader.grade(state_bad)
    
    assert score_good > score_bad

def test_task3_grader_speed():
    grader = Task3Grader()
    
    # Found rare classes early (index 0, 1)
    state_fast = {
        "current_auc": 0.70,
        "rare_positives_found": 3,
        "episode_history": [
            {"revealed_label": "nodule"},
            {"revealed_label": "effusion"},
            {"revealed_label": "pneumothorax"},
            {"revealed_label": "normal"},
            {"revealed_label": "normal"}
        ]
    }
    
    # Found rare classes late
    state_slow = {
        "current_auc": 0.70,
        "rare_positives_found": 3,
        "episode_history": [
            {"revealed_label": "normal"},
            {"revealed_label": "normal"},
            {"revealed_label": "nodule"},
            {"revealed_label": "effusion"},
            {"revealed_label": "pneumothorax"}
        ]
    }
    
    scorer_fast = grader.grade(state_fast)
    scorer_slow = grader.grade(state_slow)
    
    assert scorer_fast > scorer_slow
