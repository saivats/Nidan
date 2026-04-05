---
title: Nidan
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<div align="center">

# 🩻 Nidan

### Medical Image Active Learning Environment

*An intelligent environment where AI agents learn to triage and annotate medical images — maximising diagnostic accuracy with the fewest possible labels.*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Live-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Saivats/Nidan)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## What is Nidan?

**Nidan** (二段 — *second stage*) is a structured environment for experimenting with **active learning on medical imaging data**. It simulates the real-world challenge faced by clinical AI teams: you have a large pool of unlabeled chest X-rays, a tight annotation budget, and need to build the best possible diagnostic model.

The environment exposes a clean HTTP API. An agent interacts with it by:
1. **Calling /reset** to start a task and receive an initial pool of candidate images
2. **Calling /step** repeatedly — selecting one image at a time to request its ground-truth label
3. **Receiving a reward** based on how much that label improved the underlying classifier's AUC

The better the agent's selection strategy, the higher the diagnostic model performance with fewer labels used.

`
Agent ──POST /reset──▶ Observation (unlabeled pool + uncertainty/diversity scores)
     ◀── Observation ─┘

Agent ──POST /step───▶ {selected_image_id}
     ◀── {observation, reward, done, info} ──┘

     repeat until budget exhausted or AUC threshold reached
`

---

## Why Active Learning for Medical Imaging?

Labeling medical images is expensive. It requires expert radiologists and hours of careful review per case. In practice, a team might have **hundreds of thousands of scans** but only budget to label a few hundred. Active learning solves this by being smart about *which* images to label — prioritising:

- Images the model is **most uncertain** about (high entropy predictions)
- Images that are **most different** from what has already been labeled (diversity sampling)
- **Rare pathologies** that are underrepresented in training data

Nidan lets you build, test, and compare active learning strategies in a controlled, reproducible way.

---

## Tasks

Nidan ships three progressively harder tasks:

| Task | Name | Classes | Budget | Success Threshold |
|------|------|---------|--------|-------------------|
| 	ask1 | Binary Triage | Normal, Pneumonia | 40 labels | AUC ≥ 0.82 |
| 	ask2 | Multi-class | Normal, Pneumonia, COVID-19, Tuberculosis | 30 labels | AUC ≥ 0.75 |
| 	ask3 | Rare Pathology | Normal, Nodule, Effusion, Pneumothorax | 15 labels | AUC ≥ 0.60 |

Each task has a different class imbalance profile, label budget, and scoring formula — designed to test different aspects of an active learning strategy.

---

## Reward Structure

Each /step call returns a shaped reward that incentivises genuinely useful label selections:

`
Δ_auc           = auc_after_labeling − auc_before_labeling
redundancy      = 0.05 × max(0, cosine_similarity(new, mean_labeled) − 0.85)
rare_bonus      = 0.15  if revealed label belongs to a rare class
                  0.00  otherwise

reward = clip(Δ_auc − redundancy + rare_bonus, −0.1, 0.3)
`

Agents are penalised for selecting images very similar to ones already labeled (redundancy), and rewarded for discovering rare pathologies.

---

## API Reference

The server runs on **port 7860** and exposes:

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| GET | / | — | Environment name and version |
| GET | /health | — | Health check |
| POST | /reset | {"task_id": "task1"} | Start a new episode, returns initial Observation |
| POST | /step | {"selected_image_id": "..."} | Label one image, returns (Observation, reward, done, info) |
| GET | /state | — | Full current episode state snapshot |

### Observation Schema

`json
{
  "task_id": "task1",
  "step": 3,
  "budget_remaining": 37,
  "current_model_auc": 0.74,
  "candidate_images": [
    {
      "image_id": "img_042",
      "uncertainty_score": 0.91,
      "diversity_score": 0.67,
      "composite_score": 0.79
    }
  ],
  "label_distribution": {"normal": 2, "pneumonia": 1},
  "episode_history": [...]
}
`

---

## Architecture

`
inference.py  (LLM-powered agent / custom strategy)
      │
      │  HTTP
      ▼
server/main.py          FastAPI server
      │
      ▼
server/env.py           Nidan core environment
      │
      ├── server/data/
      │   ├── dataset_loader.py      Downloads & caches HuggingFace dataset
      │   ├── feature_extractor.py   Extracts 512-dim ResNet18 embeddings
      │   └── synthetic_fallback.py  Gaussian blob fallback if HF unavailable
      │
      ├── server/tasks/
      │   ├── task1_binary.py        Easy task config
      │   ├── task2_multiclass.py    Medium task config
      │   └── task3_rare.py          Hard task config
      │
      ├── server/utils/
      │   ├── active_learning.py     Entropy, diversity, AUC computation
      │   └── reward_calculator.py   Reward shaping logic
      │
      └── server/graders/
          ├── grader_task1.py        Efficiency-weighted AUC scoring
          ├── grader_task2.py        Macro-average AUC scoring
          └── grader_task3.py        Composite recall + AUC scoring
`

**Key design decisions:**
- **No images at runtime** — only 512-dimensional float embeddings (extracted once at build time via ResNet18). This keeps the server fast and memory-efficient.
- **Synthetic fallback** — if HuggingFace is unreachable, the environment generates realistic Gaussian blob embeddings that preserve per-class distributions, so development works fully offline.
- **Classifier** — a lightweight LogisticRegression model is retrained from scratch on the labeled set after every /step call. This keeps AUC computation honest and fast.

---

## Running Locally

### With Docker (recommended)

`ash
docker build -t nidan .
docker run -p 7860:7860 nidan
`

### Without Docker

`ash
pip install -r requirements.txt

# Pre-extract embeddings once (cached to disk)
python server/data/feature_extractor.py

# Start server
uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload
`

### Quick API test

`ash
curl http://localhost:7860/health

curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'
`

---

## Inference Agent

inference.py ships a ready-to-run LLM-powered agent that:
- Connects to any OpenAI-compatible API (defaults to HuggingFace's router)
- Runs all three tasks sequentially
- Selects images based on composite_score (uncertainty × diversity)
- Logs structured [START], [STEP], [END] events

`ash
export OPENAI_API_KEY=your_key_here
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3   # or any compatible model
export ENV_BASE_URL=http://localhost:7860

python inference.py
`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| OPENAI_API_KEY | — | API key (required for inference agent) |
| API_BASE_URL | https://router.huggingface.co/v1 | OpenAI-compatible API base URL |
| MODEL_NAME | gpt-4o-mini | LLM model name |
| ENV_BASE_URL | http://localhost:7860 | Nidan server URL for the agent |
| HF_TOKEN | — | HuggingFace token (optional, for gated datasets) |

---

## Tests

`ash
pip install pytest
pytest tests/test_env.py -v
`

The test suite covers 33 cases across reset, step, state, reward shaping, grader logic, and edge cases.

---

## Grading

| Task | Formula |
|------|---------|
| 	ask1 | min(auc / 0.82, 1.0) × efficiency_bonus |
| 	ask2 | min(macro_auc / 0.80, 1.0) |
| 	ask3 |  .5 × min(rare_found / 3, 1.0) + 0.5 × min(auc / 0.70, 1.0) |

The efficiency bonus on Task 1 rewards agents that hit the AUC threshold using fewer labels than the full budget.

---

<div align="center">

Made with 🩻 by [Saivats](https://github.com/saivats)

</div>
