from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

TASK_ORDER = ["task1", "task2", "task3"]
TASK_MAX_STEPS = {"task1": 40, "task2": 30, "task3": 15}
MAX_REWARD_PER_STEP = 0.3
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "You are a radiologist assistant performing active learning. You see a pool of unlabeled "
    "medical images. Each step, select ONE image_id to annotate — the one most likely to "
    "improve the diagnostic model. Prefer high uncertainty AND high diversity from already "
    "labeled images. Avoid near-duplicates. Respond with ONLY the image_id string, nothing else."
)


def build_user_prompt(observation: Dict[str, Any]) -> str:
    lines = [
        f"Task: {observation['task_id']}",
        f"Step: {observation['step']}",
        f"Budget remaining: {observation['budget_remaining']}",
        f"Unlabeled pool size: {observation['unlabeled_pool_size']}",
        f"Current model AUC: {observation['current_model_auc']:.4f}",
        f"Mean uncertainty: {observation['embedding_stats'].get('mean_uncertainty', 0.0):.4f}",
        f"Mean diversity: {observation['embedding_stats'].get('mean_diversity_score', 0.0):.4f}",
        "",
        "Candidate images (image_id | uncertainty | diversity):",
    ]
    for candidate in observation.get("candidate_images", []):
        lines.append(
            f"  {candidate['image_id']} | "
            f"uncertainty={candidate['uncertainty_score']:.4f} | "
            f"diversity={candidate['diversity_score']:.4f}"
        )

    if observation.get("last_annotation_result"):
        lines.append(f"\nLast annotation result: {observation['last_annotation_result']}")

    lines.append("\nWhich image_id should be annotated next? Reply with ONLY the image_id.")
    return "\n".join(lines)


def select_image_via_llm(
    client: OpenAI, observation: Dict[str, Any], valid_ids: List[str]
) -> str:
    user_prompt = build_user_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        selected = response.choices[0].message.content.strip()
        if selected in valid_ids:
            return selected
    except Exception as exc:
        log_event("WARNING", f"LLM call failed: {exc}. Falling back to top candidate.")

    return valid_ids[0] if valid_ids else ""


def log_event(event_type: str, message: str, data: Optional[Dict] = None) -> None:
    entry: Dict[str, Any] = {
        "event": event_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": message,
    }
    if data:
        entry["data"] = data
    print(json.dumps(entry), flush=True)


def post_reset(http_client: httpx.Client, task_id: str) -> Dict[str, Any]:
    response = http_client.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=60.0)
    response.raise_for_status()
    return response.json()


def post_step(http_client: httpx.Client, image_id: str, reasoning: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"selected_image_id": image_id}
    if reasoning:
        payload["reasoning"] = reasoning
    response = http_client.post(f"{ENV_BASE_URL}/step", json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()


def run_task(
    task_id: str,
    llm_client: OpenAI,
    http_client: httpx.Client,
) -> Dict[str, Any]:
    max_steps = TASK_MAX_STEPS[task_id]
    max_total_reward = max_steps * MAX_REWARD_PER_STEP

    log_event(
        "[START]",
        f"Starting task {task_id}",
        {"task_id": task_id, "max_steps": max_steps, "max_total_reward": max_total_reward},
    )

    observation = post_reset(http_client, task_id)

    task_total_reward = 0.0
    task_final_auc = observation.get("current_model_auc", 0.0)
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        candidates = observation.get("candidate_images", [])
        valid_ids = [c["image_id"] for c in candidates]

        if not valid_ids:
            log_event("WARNING", f"{task_id} step {step_count}: no candidates available, stopping.")
            break

        selected_id = select_image_via_llm(llm_client, observation, valid_ids)

        if not selected_id:
            log_event("WARNING", f"{task_id} step {step_count}: LLM returned empty id, stopping.")
            break

        reasoning = f"Uncertainty+diversity selection at step {step_count + 1}"
        step_result = post_step(http_client, selected_id, reasoning)

        reward_data = step_result.get("reward", {})
        step_reward = reward_data.get("step_reward", 0.0)
        task_total_reward += step_reward
        task_final_auc = reward_data.get("cumulative_auc", task_final_auc)
        done = step_result.get("done", False)
        observation = step_result.get("observation", observation)
        step_count += 1

        log_event(
            "[STEP]",
            f"{task_id} step {step_count}",
            {
                "task_id": task_id,
                "step": step_count,
                "selected_image_id": selected_id,
                "step_reward": round(step_reward, 4),
                "cumulative_auc": round(task_final_auc, 4),
                "delta_auc": round(reward_data.get("delta_auc", 0.0), 4),
                "done": done,
            },
        )

    info = step_result.get("info", {}) if step_count > 0 else {}
    final_score = info.get("final_score", task_total_reward / max(max_total_reward, 1e-9))
    success = final_score >= SUCCESS_SCORE_THRESHOLD

    log_event(
        "[END]",
        f"Task {task_id} complete",
        {
            "task_id": task_id,
            "steps_taken": step_count,
            "total_reward": round(task_total_reward, 4),
            "final_auc": round(task_final_auc, 4),
            "final_score": round(final_score, 4),
            "success": success,
            "done_reason": info.get("done_reason", "unknown"),
        },
    )

    return {
        "task_id": task_id,
        "steps_taken": step_count,
        "total_reward": task_total_reward,
        "final_auc": task_final_auc,
        "final_score": final_score,
        "success": success,
    }


def main() -> None:
    if not OPENAI_API_KEY:
        log_event("ERROR", "OPENAI_API_KEY is not set. Cannot proceed.")
        sys.exit(1)

    llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    http_client = httpx.Client()

    aggregate_results: List[Dict[str, Any]] = []

    for task_id in TASK_ORDER:
        log_event("INFO", f"=== Switching to {task_id} ===", {"task_id": task_id})
        result = run_task(task_id, llm_client, http_client)
        aggregate_results.append(result)

    total_score = sum(r["final_score"] for r in aggregate_results) / len(aggregate_results)
    log_event(
        "SUMMARY",
        "All tasks complete",
        {
            "task_results": aggregate_results,
            "aggregate_score": round(total_score, 4),
            "overall_success": total_score >= SUCCESS_SCORE_THRESHOLD,
        },
    )

    http_client.close()


if __name__ == "__main__":
    main()
