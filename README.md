---
title: Nidan
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Nidan

**Medical Image Active Learning Environment** — Scaler Meta-PyTorch Hackathon Round 1

An OpenEnv-compatible environment where an AI agent learns to triage and annotate chest X-rays and MRIs to maximize diagnostic model performance with minimal labeled data.

---

## Overview

Nidan simulates the active learning loop for medical imaging. The agent selects which unlabeled images to request ground-truth annotations for, with a fixed labeling budget. The goal is to maximize AUC of a diagnostic classifier while spending as few labels as possible.

```
Agent → POST /reset  → Observation (pool of unlabeled images + uncertainty/diversity scores)
      → POST /step   → (Observation, Reward, done, info)
      → repeat until budget exhausted or AUC threshold reached
```

---

## Project Structure

```
nidan/
├── Dockerfile
├── openenv.yaml
├── README.md
├── inference.py
├── requirements.txt
├── server/
│   ├── main.py              # FastAPI server
│   ├── env.py               # Core Nidan class
│   ├── models.py            # Pydantic typed models
│   ├── tasks/
│   │   ├── task1_binary.py
│   │   ├── task2_multiclass.py
│   │   └── task3_rare.py
│   ├── graders/
│   │   ├── grader_base.py
│   │   ├── grader_task1.py
│   │   ├── grader_task2.py
│   │   └── grader_task3.py
│   ├── data/
│   │   ├── dataset_loader.py
│   │   ├── feature_extractor.py
│   │   └── synthetic_fallback.py
│   └── utils/
│       ├── active_learning.py
│       └── reward_calculator.py
└── tests/
    └── test_env.py
```

---

## Tasks

| Task | Difficulty | Classes | Budget | Success AUC |
|------|------------|---------|--------|-------------|
| task1 | Easy | normal, pneumonia | 40 | 0.82 |
| task2 | Medium | normal, pneumonia, covid, tuberculosis | 30 | 0.75 |
| task3 | Hard | normal, nodule, effusion, pneumothorax | 15 | 0.60 |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Environment name and version |
| `GET`  | `/health` | Health check |
| `POST` | `/reset` | `{"task_id": "task1"}` → Observation |
| `POST` | `/step`  | `{"selected_image_id": "..."}` → {observation, reward, done, info} |
| `GET`  | `/state` | Full current state snapshot |

---

## Reward Structure

```
delta_auc           = new_auc - old_auc
redundancy_penalty  = 0.05 * max(0, cosim(new_emb, mean_labeled) - 0.85)
rare_case_bonus     = 0.15 if revealed_label in rare_classes else 0.0
step_reward         = clip(delta_auc - redundancy_penalty + rare_case_bonus, -0.1, 0.3)
```

---

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Pre-extract embeddings (runs once, cached to disk)
python server/data/feature_extractor.py

# Start the server
uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t nidan .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  nidan
```

### Run Inference Agent

```bash
export OPENAI_API_KEY=your_openai_key
export MODEL_NAME=gpt-4o-mini
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for the inference agent | Yes (for agent) |
| `API_BASE_URL` | OpenAI-compatible API base URL | No (default: openai) |
| `MODEL_NAME` | LLM model name | No (default: gpt-4o-mini) |
| `ENV_BASE_URL` | Nidan server URL | No (default: localhost:7860) |
| `HF_TOKEN` | HuggingFace token for dataset access | No (falls back to synthetic) |

---

## Data Strategy

- **Primary**: Downloads `keremberke/chest-xray-classification` from HuggingFace and extracts ResNet18 embeddings (512-dim) offline.
- **Fallback**: If HuggingFace is unavailable, generates synthetic embeddings using Gaussian blobs that preserve realistic class distributions.
- **Caching**: All embeddings are cached to `server/data/embeddings_cache/` at build time. No images are loaded at inference time.

---

## Grading

| Task | Scoring Formula |
|------|-----------------|
| task1 | `min(auc / 0.82, 1.0) * efficiency_bonus` (bonus if AUC ≥ 0.82 using fewer labels) |
| task2 | `min(macro_auc / 0.80, 1.0)` |
| task3 | `0.5 * min(rare_found / 3, 1.0) + 0.5 * min(auc / 0.70, 1.0)` |

Expected baseline scores: task1 ~0.65 · task2 ~0.40 · task3 ~0.20

---

## Architecture

```
inference.py (OpenAI LLM Agent)
       │
       ▼
server/main.py (FastAPI)
       │
       ▼
server/env.py (Nidan)
    ├── data/dataset_loader.py   → embeddings (HF or synthetic)
    ├── utils/active_learning.py → AUC, entropy, diversity
    ├── utils/reward_calculator.py
    └── graders/                 → per-task scoring
```
