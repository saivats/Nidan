from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
ENV_NAME = "nidan"

TASK_ORDER = ["task1", "task2", "task3"]
TASK_MAX_STEPS = {"task1": 40, "task2": 30, "task3": 15}

SYSTEM_PROMPT_EARLY = (
    "You are a radiologist assistant performing active learning in the EARLY phase. "
    "Your priority is CLASS COVERAGE — label at least one sample from every class. "
    "Look for candidates with HIGH DIVERSITY to explore the full feature space. "
    "Select ONE image_id from the candidates. Respond with ONLY the image_id."
)

SYSTEM_PROMPT_MID = (
    "You are a radiologist assistant performing active learning in the MID phase. "
    "You now have class coverage. Your priority is UNCERTAINTY REDUCTION — pick images "
    "the model is most confused about to improve AUC. Balance uncertainty with diversity. "
    "Avoid near-duplicates of already labeled images. "
    "Select ONE image_id from the candidates. Respond with ONLY the image_id."
)

SYSTEM_PROMPT_LATE = (
    "You are a radiologist assistant performing active learning in the LATE phase. "
    "Budget is almost exhausted. Every pick counts. Target the HIGHEST UNCERTAINTY "
    "candidate — go for the biggest AUC gain per annotation. Avoid any redundant picks. "
    "Select ONE image_id from the candidates. Respond with ONLY the image_id."
)

PHASE_PROMPTS = {
    "early": SYSTEM_PROMPT_EARLY,
    "mid": SYSTEM_PROMPT_MID,
    "late": SYSTEM_PROMPT_LATE,
}


def _safe_score(s: float) -> float:
    return max(0.01, min(0.99, float(s)))


def _get_budget_phase(observation: Dict[str, Any]) -> str:
    return observation.get("budget_phase", "mid")


def build_user_prompt(observation: Dict[str, Any]) -> str:
    phase = _get_budget_phase(observation)
    class_dist = observation.get("class_distribution", {})
    auc = observation.get("current_model_auc", 0.0)

    lines = [
        f"Task: {observation['task_id']}",
        f"Step: {observation['step']}",
        f"Budget remaining: {observation['budget_remaining']}",
        f"Budget phase: {phase}",
        f"Unlabeled pool size: {observation['unlabeled_pool_size']}",
        f"Current model AUC: {auc:.4f}",
        f"Mean uncertainty: {observation['embedding_stats'].get('mean_uncertainty', 0.0):.4f}",
        f"Mean diversity: {observation['embedding_stats'].get('mean_diversity_score', 0.0):.4f}",
    ]

    if class_dist:
        lines.append(f"\nLabeled class distribution: {class_dist}")
        missing = [cls for cls, count in class_dist.items() if count == 0]
        if missing:
            lines.append(f"MISSING CLASSES (priority): {missing}")

    histogram = observation.get("model_confidence_histogram")
    if histogram:
        lines.append(f"\nModel confidence histogram: {histogram}")

    lines.append("")
    lines.append("Candidate images (image_id | uncertainty | diversity | region | priority | cost):")
    for candidate in observation.get("candidate_images", []):
        lines.append(
            f"  {candidate['image_id']} | "
            f"uncertainty={candidate.get('uncertainty_score', 0.0):.4f} | "
            f"diversity={candidate.get('diversity_score', 0.0):.4f} | "
            f"region={candidate.get('region_of_interest', 'general')} | "
            f"priority={candidate.get('patient_priority', 'routine')} | "
            f"cost={candidate.get('acquisition_cost', 1)}"
        )

    if observation.get("last_annotation_result"):
        lines.append(f"\nLast annotation result: {observation['last_annotation_result']}")

    if phase == "early":
        lines.append("\nSTRATEGY: Prioritize diversity and class coverage — find all classes!")
    elif phase == "mid":
        lines.append("\nSTRATEGY: Target high-uncertainty samples to maximize AUC gain.")
    else:
        lines.append("\nSTRATEGY: Budget critical! Pick only the most impactful candidate.")

    lines.append("\nWhich image_id should be annotated next? Reply with ONLY the image_id.")
    return "\n".join(lines)


def select_image_via_llm(
    client: OpenAI, observation: Dict[str, Any], valid_ids: List[str]
) -> str:
    phase = _get_budget_phase(observation)
    system_prompt = PHASE_PROMPTS.get(phase, SYSTEM_PROMPT_MID)
    user_prompt = build_user_prompt(observation)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=64,
                temperature=0.0,
            )
            selected = response.choices[0].message.content.strip()
            if selected in valid_ids:
                return selected

            for vid in valid_ids:
                if vid in selected:
                    return vid
        except Exception:
            if attempt < 2:
                time.sleep(1)
            continue

    return _fallback_selection(observation, valid_ids)


def _fallback_selection(observation: Dict[str, Any], valid_ids: List[str]) -> str:
    candidates = observation.get("candidate_images", [])
    if not candidates:
        return valid_ids[0] if valid_ids else ""

    phase = _get_budget_phase(observation)

    if phase == "early":
        best = max(candidates, key=lambda c: c.get("diversity_score", 0.0))
    elif phase == "late":
        best = max(candidates, key=lambda c: c.get("uncertainty_score", 0.0))
    else:
        best = max(
            candidates,
            key=lambda c: c.get("uncertainty_score", 0.0) * 0.6 + c.get("diversity_score", 0.0) * 0.4,
        )

    return best["image_id"] if best["image_id"] in valid_ids else valid_ids[0]


def post_reset(http_client: httpx.Client, task_id: str) -> Dict[str, Any]:
    response = http_client.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=60.0)
    response.raise_for_status()
    return response.json()


def post_step(http_client: httpx.Client, image_id: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"selected_image_id": image_id}
    response = http_client.post(f"{ENV_BASE_URL}/step", json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()


def post_close(http_client: httpx.Client) -> None:
    try:
        http_client.post(f"{ENV_BASE_URL}/close", timeout=10.0)
    except Exception:
        pass


def run_task(
    task_id: str,
    llm_client: OpenAI,
    http_client: httpx.Client,
) -> Dict[str, Any]:
    max_steps = TASK_MAX_STEPS[task_id]

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards_list: List[float] = []
    done = False
    step_count = 0
    step_result: Dict[str, Any] = {}
    last_error: Optional[str] = None
    final_score = 0.01

    try:
        observation = post_reset(http_client, task_id)

        while not done and step_count < max_steps:
            candidates = observation.get("candidate_images", [])
            valid_ids = [c["image_id"] for c in candidates]

            if not valid_ids:
                break

            selected_id = select_image_via_llm(llm_client, observation, valid_ids)

            if not selected_id:
                break

            try:
                step_result = post_step(http_client, selected_id)
                last_error = None
            except Exception as exc:
                last_error = str(exc)
                step_count += 1
                print(
                    f"[STEP] step={step_count} action={selected_id} reward=0.00 done=false error={last_error}",
                    flush=True,
                )
                continue

            reward_data = step_result.get("reward", {})
            step_reward = reward_data.get("step_reward", 0.0)
            rewards_list.append(step_reward)
            done = step_result.get("done", False)
            observation = step_result.get("observation", observation)
            step_count += 1

            error_str = "null" if last_error is None else last_error
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={step_count} action={selected_id} reward={step_reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

        info = step_result.get("info", {})
        raw_score = info.get("final_score", None)
        if raw_score is not None:
            final_score = _safe_score(raw_score)
        elif rewards_list:
            raw_score = sum(rewards_list) / len(rewards_list)
            final_score = _safe_score(raw_score)

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} exception: {exc}", flush=True)

    finally:
        success = final_score >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"
        success_str = "true" if success else "false"

        post_close(http_client)

        print(
            f"[END] success={success_str} steps={step_count} score={final_score:.4f} rewards={rewards_str}",
            flush=True,
        )

    return {
        "task_id": task_id,
        "steps_taken": step_count,
        "rewards": rewards_list,
        "final_score": final_score,
        "success": success,
    }


def main() -> None:
    llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    http_client = httpx.Client()

    for task_id in TASK_ORDER:
        run_task(task_id, llm_client, http_client)

    http_client.close()


if __name__ == "__main__":
    main()
