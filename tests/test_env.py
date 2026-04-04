from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.env import Nidan
from server.graders.grader_task1 import Task1Grader
from server.graders.grader_task2 import Task2Grader
from server.graders.grader_task3 import Task3Grader
from server.models import Action, Observation, Reward


@pytest.fixture
def env():
    return Nidan()


@pytest.fixture
def reset_task1(env):
    observation = env.reset("task1")
    return env, observation


@pytest.fixture
def reset_task2(env):
    observation = env.reset("task2")
    return env, observation


@pytest.fixture
def reset_task3(env):
    observation = env.reset("task3")
    return env, observation


class TestReset:
    def test_reset_returns_observation(self, reset_task1):
        _, observation = reset_task1
        assert isinstance(observation, Observation)

    def test_reset_observation_has_correct_task_id(self, reset_task1):
        _, observation = reset_task1
        assert observation.task_id == "task1"

    def test_reset_sets_initial_step_to_zero(self, reset_task1):
        _, observation = reset_task1
        assert observation.step == 0

    def test_reset_budget_remaining_equals_full_budget(self, reset_task1):
        _, observation = reset_task1
        assert observation.budget_remaining == 40

    def test_reset_auc_is_valid_float(self, reset_task1):
        _, observation = reset_task1
        assert 0.0 <= observation.current_model_auc <= 1.0

    def test_reset_candidate_images_not_empty(self, reset_task1):
        _, observation = reset_task1
        assert len(observation.candidate_images) > 0

    def test_reset_candidate_images_limit(self, reset_task1):
        _, observation = reset_task1
        assert len(observation.candidate_images) <= 10

    def test_reset_embedding_stats_present(self, reset_task1):
        _, observation = reset_task1
        assert "mean_uncertainty" in observation.embedding_stats
        assert "mean_diversity_score" in observation.embedding_stats

    def test_reset_episode_history_empty(self, reset_task1):
        _, observation = reset_task1
        assert observation.episode_history == []

    def test_reset_task2_returns_valid_observation(self, reset_task2):
        _, observation = reset_task2
        assert isinstance(observation, Observation)
        assert observation.task_id == "task2"
        assert observation.budget_remaining == 30

    def test_reset_task3_returns_valid_observation(self, reset_task3):
        _, observation = reset_task3
        assert isinstance(observation, Observation)
        assert observation.task_id == "task3"
        assert observation.budget_remaining == 15


class TestStep:
    def test_step_returns_reward_in_valid_range(self, reset_task1):
        env, observation = reset_task1
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        _, reward, _, _ = env.step(action)
        assert -0.1 <= reward.step_reward <= 0.3

    def test_step_reward_is_reward_instance(self, reset_task1):
        env, observation = reset_task1
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        _, reward, _, _ = env.step(action)
        assert isinstance(reward, Reward)

    def test_step_decrements_budget(self, reset_task1):
        env, observation = reset_task1
        initial_budget = observation.budget_remaining
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        new_obs, _, _, _ = env.step(action)
        assert new_obs.budget_remaining == initial_budget - 1

    def test_step_removes_image_from_pool(self, reset_task1):
        env, observation = reset_task1
        initial_pool = observation.unlabeled_pool_size
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        new_obs, _, _, _ = env.step(action)
        assert new_obs.unlabeled_pool_size == initial_pool - 1

    def test_step_returns_observation_with_valid_auc(self, reset_task1):
        env, observation = reset_task1
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        new_obs, _, _, _ = env.step(action)
        assert 0.0 <= new_obs.current_model_auc <= 1.0

    def test_step_with_invalid_image_id_raises_value_error(self, reset_task1):
        env, _ = reset_task1
        action = Action(selected_image_id="nonexistent_bad_id_xyz")
        with pytest.raises(ValueError):
            env.step(action)

    def test_step_before_reset_raises_runtime_error(self, env):
        action = Action(selected_image_id="any_id")
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_step_sets_last_annotation_result(self, reset_task1):
        env, observation = reset_task1
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        new_obs, _, _, _ = env.step(action)
        assert new_obs.last_annotation_result is not None
        assert new_obs.last_annotation_result in ["normal", "pneumonia"]

    def test_step_cumulative_auc_is_non_negative(self, reset_task1):
        env, observation = reset_task1
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        _, reward, _, _ = env.step(action)
        assert reward.cumulative_auc >= 0.0

    def test_step_updates_episode_history(self, reset_task1):
        env, observation = reset_task1
        first_id = observation.candidate_images[0].image_id
        action = Action(selected_image_id=first_id)
        new_obs, _, _, _ = env.step(action)
        assert len(new_obs.episode_history) == 1


class TestDone:
    def test_done_true_when_budget_exhausted_task3(self, reset_task3):
        env, observation = reset_task3
        done = False
        steps = 0
        while not done and steps < 20:
            candidates = observation.candidate_images
            if not candidates:
                break
            image_id = candidates[0].image_id
            action = Action(selected_image_id=image_id)
            observation, _, done, _ = env.step(action)
            steps += 1
        assert done

    def test_done_info_contains_done_reason(self, reset_task3):
        env, observation = reset_task3
        done = False
        info = {}
        steps = 0
        while not done and steps < 20:
            candidates = observation.candidate_images
            if not candidates:
                break
            image_id = candidates[0].image_id
            action = Action(selected_image_id=image_id)
            observation, _, done, info = env.step(action)
            steps += 1
        assert "done_reason" in info
        assert info["done_reason"] in ["budget_exhausted", "auc_threshold_reached"]


class TestGraders:
    def _make_state(self, current_auc, budget_used, rare_positives_found=0):
        return {
            "current_auc": current_auc,
            "budget_used": budget_used,
            "rare_positives_found": rare_positives_found,
        }

    def test_task1_grader_score_in_range(self):
        grader = Task1Grader()
        for auc in [0.0, 0.5, 0.82, 1.0]:
            state = self._make_state(auc, 20)
            score = grader.grade(state)
            assert 0.0 <= score <= 1.0, f"Score out of range for AUC={auc}: {score}"

    def test_task1_grader_efficiency_bonus_applies(self):
        grader = Task1Grader()
        state_no_bonus = self._make_state(0.82, 40)
        state_with_bonus = self._make_state(0.82, 20)
        score_no_bonus = grader.grade(state_no_bonus)
        score_with_bonus = grader.grade(state_with_bonus)
        assert score_with_bonus >= score_no_bonus

    def test_task2_grader_score_in_range(self):
        grader = Task2Grader()
        for auc in [0.0, 0.4, 0.75, 1.0]:
            state = self._make_state(auc, 15)
            score = grader.grade(state)
            assert 0.0 <= score <= 1.0

    def test_task3_grader_score_in_range(self):
        grader = Task3Grader()
        for auc, rp in [(0.0, 0), (0.5, 1), (0.70, 3), (1.0, 5)]:
            state = self._make_state(auc, 10, rp)
            score = grader.grade(state)
            assert 0.0 <= score <= 1.0

    def test_task3_grader_zero_score_for_no_positives_no_auc(self):
        grader = Task3Grader()
        state = self._make_state(0.0, 15, 0)
        score = grader.grade(state)
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_task3_grader_max_score_for_perfect(self):
        grader = Task3Grader()
        state = self._make_state(1.0, 15, 3)
        score = grader.grade(state)
        assert score == pytest.approx(1.0, abs=1e-4)


class TestStateSerializable:
    def test_state_returns_dict(self, reset_task1):
        env, _ = reset_task1
        state_dict = env.state()
        assert isinstance(state_dict, dict)

    def test_state_is_json_serializable(self, reset_task1):
        env, _ = reset_task1
        state_dict = env.state()
        serialized = json.dumps(state_dict)
        assert isinstance(serialized, str)

    def test_state_after_step_is_json_serializable(self, reset_task1):
        env, observation = reset_task1
        first_id = observation.candidate_images[0].image_id
        env.step(Action(selected_image_id=first_id))
        state_dict = env.state()
        serialized = json.dumps(state_dict)
        assert isinstance(serialized, str)

    def test_state_not_initialized_returns_dict(self, env):
        state_dict = env.state()
        assert isinstance(state_dict, dict)
        assert "status" in state_dict
