---
title: Nidan
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Nidan

> Strategic medical image annotation through active learning. An environment where AI agents learn to select which X-rays and MRIs to label for maximum diagnostic impact with minimal annotation budget.

## Problem

Medical AI models need labeled data. Expert annotation is expensive, slow, and bottlenecked. Nidan solves this by training agents to make smart annotation decisions — maximizing model performance while minimizing labeling cost.

## How It Works

The agent observes an unlabeled medical image pool and strategically selects which images to request labels for. At each step:

1. View top candidates ranked by uncertainty and diversity
2. Select one image to annotate
3. Observe diagnostic model improvement
4. Earn reward based on AUC gain and annotation efficiency

Three progressively harder tasks test the agent's strategic thinking.

## Quick Start
`ash
docker build -t nidan .
docker run -p 7860:7860 nidan
`

Test the API:
`ash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "task1"}'
`

## Project Structure

- server/main.py — FastAPI server
- server/env.py — Core environment
- server/tasks/ — 3 difficulty levels (binary, multi-class, rare pathology)
- server/graders/ — Task evaluation
- server/data/ — Dataset + embeddings
- 	ests/test_env.py — 33 unit tests (all passing)
- inference.py — LLM agent runner

## Tasks

| Task | Difficulty | Classes | Budget | Threshold |
|---|---|---|---|---|
| Task 1 | Easy | 2 | 40 | AUC 0.82 |
| Task 2 | Medium | 4 | 30 | AUC 0.75 |
| Task 3 | Hard | 4 (imbalanced) | 15 | Recall + AUC |

## API

- POST /reset — Start a task
- POST /step — Select image to annotate
- GET /state — Current environment state
- GET /health — Server status

## Stack

FastAPI • Pydantic • scikit-learn • PyTorch • HuggingFace • Docker

## Status

✅ 33/33 tests passing  
✅ All 3 tasks validated  
✅ Live on HuggingFace Spaces

---

**[Live Demo](https://saivats-nidan.hf.space)** • MIT License
