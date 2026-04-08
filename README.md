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

# Nidan

> Strategic medical image annotation through active learning. An environment where AI agents learn to select which X-rays to label for maximum diagnostic impact with minimal annotation budget.

## Problem

Medical AI models need labeled data. Expert annotation is expensive, slow, and bottlenecked by specialist availability. Nidan solves this by training agents to make smart annotation decisions — maximizing diagnostic model performance while minimizing labeling cost.

## How It Works

The agent observes an unlabeled medical image pool and strategically selects which images to request labels for. At each step:

1. View top candidates ranked by uncertainty and diversity
2. Select one image to annotate
3. Observe diagnostic model improvement
4. Earn reward based on AUC gain and annotation efficiency

Three progressively harder tasks test the agent's strategic thinking.

## Action Space

The agent submits a single action per step:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `selected_image_id` | string | Yes | ID of the unlabeled image to annotate |
| `reasoning` | string | No | Optional explanation for selection |

Example action:
```json
{"selected_image_id": "task1_img_0042", "reasoning": "Highest uncertainty + diversity"}
```

## Observation Space

Each observation returned by the environment contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `step` | integer | Current step number |
| `budget_remaining` | integer | Remaining annotation budget |
| `unlabeled_pool_size` | integer | Images still unlabeled |
| `current_model_auc` | float [0, 1] | Current diagnostic model AUC |
| `candidate_images` | array | Top-10 candidates with uncertainty and diversity scores |
| `last_annotation_result` | string | Label revealed in previous step |
| `embedding_stats` | object | Pool-level uncertainty and diversity statistics |
| `episode_history` | array | All previous step summaries |

Each candidate image includes:
- `image_id` — unique identifier
- `uncertainty_score` — model prediction entropy [0, 1]
- `diversity_score` — distance from labeled set [0, 1]

## Tasks

| Task | Difficulty | Classes | Budget | Threshold | Description |
|------|-----------|---------|--------|-----------|-------------|
| task1 | Easy | 2 (Normal, Pneumonia) | 40 | AUC >= 0.82 | Binary pneumonia detection |
| task2 | Medium | 4 (Normal, Pneumonia, COVID, TB) | 30 | AUC >= 0.75 | Multi-class chest conditions |
| task3 | Hard | 4 (Normal, Nodule, Effusion, Pneumothorax) | 15 | AUC >= 0.60 | Rare pathology detection with class imbalance |

## Reward Function

Each step returns a shaped reward providing incremental feedback:

```
delta_auc           = auc_after - auc_before
redundancy_penalty  = 0.05 * max(0, cosine_sim(new, mean_labeled) - 0.85)
rare_case_bonus     = 0.15 if label is rare class, else 0.0
step_reward         = clip(delta_auc - redundancy_penalty + rare_case_bonus, -0.1, 0.3)
```

- **Incremental progress**: rewards every AUC improvement
- **Penalizes redundancy**: selecting near-duplicates of labeled images
- **Rewards discovery**: bonus for finding rare pathologies

## Grading

Each task has a deterministic, reproducible grader returning a score in [0.0, 1.0]:

| Task | Grading Formula |
|------|----------------|
| task1 | `min(auc / 0.82, 1.0) * efficiency_bonus` |
| task2 | `min(macro_auc / 0.80, 1.0)` |
| task3 | `0.5 * min(rare_found / 3, 1.0) + 0.5 * min(auc / 0.70, 1.0)` |

## Baseline Performance Scores

Scores from running the inference agent with `gpt-4o-mini` via HuggingFace router:

| Task | Agent | Steps Used | Final AUC | Grader Score |
|------|-------|-----------|-----------|-------------|
| task1 | Random | 40 | ~0.68 | ~0.45 |
| task1 | LLM (uncertainty+diversity) | 40 | ~0.84 | ~0.92 |
| task2 | Random | 30 | ~0.58 | ~0.40 |
| task2 | LLM (uncertainty+diversity) | 30 | ~0.76 | ~0.85 |
| task3 | Random | 15 | ~0.52 | ~0.20 |
| task3 | LLM (uncertainty+diversity) | 15 | ~0.63 | ~0.65 |

## Quick Start

### Docker (recommended)
```bash
docker build -t nidan .
docker run -p 7860:7860 nidan
```

### Without Docker
```bash
pip install -r requirements.txt
python server/data/feature_extractor.py
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

### Test the API
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "task1"}'
```

## Running the Inference Agent

```bash
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=gpt-4o-mini
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| GET | `/metadata` | Environment name and description |
| GET | `/schema` | Action, observation, and state schemas |
| POST | `/reset` | Start a new task episode |
| POST | `/step` | Select an image to annotate |
| GET | `/state` | Current environment state snapshot |
| POST | `/close` | Close the current episode |

## Project Structure

- `server/app.py` — OpenEnv-compatible entry point
- `server/main.py` — FastAPI application
- `server/env.py` — Core environment logic
- `server/models.py` — Pydantic models (Observation, Action, Reward)
- `server/tasks/` — Task definitions (easy, medium, hard)
- `server/graders/` — Deterministic graders (score 0.0-1.0)
- `server/data/` — Dataset loading and embedding extraction
- `server/utils/` — Active learning and reward utilities
- `tests/test_env.py` — 33 unit tests
- `inference.py` — LLM inference agent
- `openenv.yaml` — OpenEnv metadata
- `pyproject.toml` — Package configuration

## Stack

FastAPI, Pydantic, scikit-learn, PyTorch, HuggingFace Datasets, Docker, OpenAI SDK

---

**[Live Demo](https://saivats-nidan.hf.space)** | MIT License
