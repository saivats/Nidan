---
title: Nidan
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Nidan — Medical Image Active Learning Environment

Radiologists spend 60-80 % of annotation budgets labeling redundant, easy-to-classify images. **Nidan** is an OpenEnv-compatible reinforcement-learning environment that trains AI agents to select the *most informative* chest X-rays for expert annotation — cutting labeling cost while maximizing diagnostic AUC. Built with FastAPI, scikit-learn and PyTorch feature extractors, it exposes the full `reset() / step() / state()` API required by the OpenEnv specification.

---

## Environment Overview

| Concept | Description |
|---------|-------------|
| **Domain** | Medical imaging — chest X-ray classification |
| **Agent goal** | Maximize diagnostic model AUC while spending the fewest annotation labels |
| **Episode** | Agent queries one image per step from an unlabeled pool; the oracle reveals its ground-truth label; the internal classifier retrains and AUC is re-evaluated |
| **Termination** | Budget exhausted **or** AUC reaches the task-specific success threshold |

The agent must balance **uncertainty sampling** (pick images the model is confused about) with **diversity sampling** (pick images far from what has already been labeled) while hunting for **rare pathologies** that carry a bonus reward.

---

## Tasks

Three progressively harder tasks exercise different facets of the active-learning strategy:

| Task ID | Name | Difficulty | Classes | Budget | AUC Threshold |
|---------|------|------------|---------|--------|---------------|
| `task1` | Binary Pneumonia Detection | Easy | Normal, Pneumonia | 40 | ≥ 0.72 |
| `task2` | Multi-class Chest Conditions | Medium | Normal, Pneumonia, COVID, TB | 30 | ≥ 0.65 |
| `task3` | Rare Pathology Detection | Hard | Normal, Nodule, Effusion, Pneumothorax | 15 | ≥ 0.60 |

**Task 3** features extreme class imbalance (85 % normal, 5 % each rare class) and a tight 15-step budget, forcing the agent to prioritize rare-class discovery.

---

## Action Space

Each step the agent submits **one** action:

```json
{
  "selected_image_id": "task1_img_0042",
  "reasoning": "Highest uncertainty + high diversity from labeled set"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `selected_image_id` | `string` | ✅ | ID of the unlabeled image to annotate |
| `reasoning` | `string` | — | Optional explanation (logged but not scored) |

---

## Observation Space

After each step, the environment returns a structured observation:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `string` | Current task identifier |
| `step` | `int` | Current step number |
| `budget_remaining` | `int` | Remaining annotation budget |
| `unlabeled_pool_size` | `int` | Images still in the unlabeled pool |
| `current_model_auc` | `float` | Current diagnostic model AUC on a held-out validation set |
| `candidate_images` | `array` | Top-10 candidates ranked by uncertainty × diversity |
| `last_annotation_result` | `string` | Ground-truth label revealed in the previous step |
| `embedding_stats` | `object` | Pool-level mean uncertainty and mean diversity |
| `episode_history` | `array` | Summaries of all previous steps |

Each candidate image carries:

- **`uncertainty_score`** — prediction entropy normalised to [0, 1]; higher = model is less confident
- **`diversity_score`** — cosine distance from the mean of the labeled set; higher = more novel

---

## Reward Function

The shaped per-step reward provides dense feedback even when AUC moves slowly:

```
delta_auc          = auc_after - auc_before
diversity_bonus    = 0.05 × diversity_score          (in [0, 1])
redundancy_penalty = 0.05 × max(0, cos_sim - 0.85)   (penalises near-duplicates)
rare_case_bonus    = 0.15   if label ∈ rare classes, else 0

step_reward = clip(delta_auc + diversity_bonus − redundancy_penalty + rare_case_bonus, −0.1, 0.3)
```

| Component | Purpose |
|-----------|---------|
| `delta_auc` | Rewards every measurable improvement in diagnostic accuracy |
| `diversity_bonus` | Encourages exploring under-represented regions of the feature space |
| `redundancy_penalty` | Discourages selecting images very similar to already-labeled ones |
| `rare_case_bonus` | Incentivises finding rare pathologies that improve minority-class recall |

---

## Grading

Each task has a deterministic, reproducible grader returning a score in the exclusive interval **(0, 1)**.

| Task | Formula |
|------|---------|
| task1 | `min(auc / 0.72, 1.0) × efficiency_bonus` |
| task2 | `min(macro_auc / 0.65, 1.0) × efficiency_bonus` |
| task3 | `0.5 × min(rare_found / 3, 1.0) + 0.5 × min(auc / 0.70, 1.0)` |

The **efficiency bonus** rewards agents that reach the threshold using fewer steps, modelling real-world cost savings.

---

## Baseline Scores

Scores from running the bundled LLM inference agent (`gpt-4o-mini` via HuggingFace Router):

| Task | Agent | Steps | Final AUC | Grader Score |
|------|-------|-------|-----------|--------------|
| task1 | Random baseline | 40 | ~0.58 | ~0.35 |
| task1 | LLM (uncertainty+diversity) | 40 | ~0.74 | ~0.88 |
| task2 | Random baseline | 30 | ~0.52 | ~0.30 |
| task2 | LLM (uncertainty+diversity) | 30 | ~0.66 | ~0.78 |
| task3 | Random baseline | 15 | ~0.50 | ~0.18 |
| task3 | LLM (uncertainty+diversity) | 15 | ~0.61 | ~0.60 |

---

## Quick Start

### Docker (recommended)

```bash
docker build -t nidan .
docker run -p 7860:7860 nidan
```

### Without Docker

```bash
pip install -r requirements.txt
python server/data/feature_extractor.py   # pre-compute embeddings
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

### Validate

```bash
pip install openenv-core
openenv validate          # → [OK] nidan: Ready for multi-mode deployment
```

---

## API Endpoints

| Method | Endpoint | Description | Example |
|--------|----------|-------------|---------|
| `GET` | `/health` | Liveness check | `curl http://localhost:7860/health` |
| `POST` | `/reset` | Start a new episode | `curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' -d '{"task_id":"task1"}'` |
| `POST` | `/step` | Submit an annotation query | `curl -X POST http://localhost:7860/step -H 'Content-Type: application/json' -d '{"selected_image_id":"task1_img_0042"}'` |
| `GET` | `/state` | Current episode snapshot | `curl http://localhost:7860/state` |
| `POST` | `/close` | End the current episode | `curl -X POST http://localhost:7860/close` |
| `GET` | `/metadata` | Environment name + description | `curl http://localhost:7860/metadata` |
| `GET` | `/schema` | Action / Observation / State JSON schemas | `curl http://localhost:7860/schema` |

---

## Running the Inference Agent

```bash
export HF_TOKEN=<your_huggingface_token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=gpt-4o-mini
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

The inference agent prints `[START]`, `[STEP]`, and `[END]` lines to stdout in the format expected by the OpenEnv evaluator.

---

## Project Structure

```
nidan/
├── server/
│   ├── app.py              # OpenEnv-compatible FastAPI entry point
│   ├── main.py             # Application factory + routes
│   ├── env.py              # Core environment logic (reset/step/state)
│   ├── models.py           # Pydantic models — Observation, Action, Reward
│   ├── tasks/              # Task definitions (easy → medium → hard)
│   ├── graders/            # Deterministic graders (score in (0, 1))
│   ├── data/               # Dataset loading + embedding extraction
│   └── utils/              # Active-learning + reward utilities
├── inference.py            # LLM-based inference agent
├── openenv.yaml            # OpenEnv metadata & task registry
├── Dockerfile              # Production container
├── requirements.txt        # Python dependencies
└── tests/test_env.py       # Unit tests
```

## Stack

FastAPI · Pydantic · scikit-learn · PyTorch · HuggingFace Datasets · Docker · OpenAI SDK

---

**[Live Demo](https://saivats-nidan.hf.space)** · MIT License
