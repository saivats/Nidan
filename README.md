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

Radiologists spend 60-80% of annotation budgets labeling redundant, easy-to-classify images. **Nidan** is an OpenEnv-compatible reinforcement-learning environment that trains AI agents to act as **autonomous triagers**, selecting the *most informative* medical images for expert annotation. The goal: cut labeling cost while maximizing diagnostic AUC and discovering rare pathologies early.

Nidan is built with FastAPI, scikit-learn, and PyTorch feature extractors, exposing the full `reset() / step() / state()` API required by the OpenEnv specification.

---

## Why This Matters (Real-World Utility)

Unlike toy environments that assume all actions are equal, Nidan models the genuine complexity of medical annotation:
1. **Variable Annotation Cost**: Annotating a clear "normal" scan is fast (costs 1 budget). Annotating a complex rare pathology like Tuberculosis requires specialist review (costs 2 budget). Agents must perform cost-benefit analysis.
2. **Clinical Context**: Finding rare diseases faster saves lives. The environment assigns "patient priority" based on model uncertainty and maps embeddings to gross "anatomical regions", pulling the abstract RL task closer to a real clinical dashboard.
3. **The Active Learning Curriculum**: Our shaped reward function naturally guides agents through a realistic learning curriculum:
   - *Early Game*: Reward broad class coverage (find all diseases).
   - *Mid Game*: Reward uncertainty targeting (focus on confusing cases).
   - *Late Game*: Heavily penalize redundant picks as budget depletes.

---

## Environment Overview

| Concept | Description |
|---------|-------------|
| **Domain** | Medical imaging — chest X-ray classification |
| **Agent goal** | Maximize diagnostic model AUC while navigating variable annotation costs |
| **Episode** | Agent queries one image per step from an unlabeled pool; the oracle reveals its ground-truth label; the internal classifier retrains and AUC is re-evaluated |
| **Termination** | Budget exhausted **or** AUC reaches the task-specific success threshold |

---

## Tasks & Grading

Three progressively harder tasks exercise different facets of the active-learning strategy. Evaluated via multi-metric AI graders that return exclusive `(0, 1)` scores.

| Task ID | Name | Classes | Budget | Key Challenge | Grader Metrics |
|---------|------|---------|--------|---------------|----------------|
| `task1` | Binary Pneumonia Detection | Normal, Pneumonia | 40 | Establish baseline uncertainty vs diversity tradeoff | AUC (60%), Efficiency (20%), Class Balance (20%) |
| `task2` | Multi-class Chest Conditions | Normal, Pneumonia, COVID, TB | 30 | Variable annotation costs — rare diseases cost 2 budget units | Macro-AUC (50%), Efficiency (25%), Class Coverage (25%) |
| `task3` | Rare Pathology Detection | Normal, Nodule, Effusion, Pneumothorax | 15 | Extreme class imbalance (5% rare cases). Rare cases cost 2/15 budget! | Rare Discovery (40%), Speed of Discovery (30%), AUC (30%) |

---

## Observation Space

After each step, the environment returns a rich, clinically-inspired observation:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `string` | Current task identifier |
| `step` | `int` | Current step number |
| `budget_remaining` | `int` | Remaining annotation budget |
| `budget_phase` | `string` | Current curriculum phase (`early`, `mid`, `late`) |
| `current_model_auc` | `float` | Current diagnostic model AUC on a held-out validation set |
| `candidate_images` | `array` | Top-10 candidates ranked by uncertainty × diversity |
| `class_distribution` | `object` | Currently labeled class counts |
| `model_confidence_histogram`| `object` | 5-bucket distribution of model confidence across the pool |

Each candidate image carries:
- **`uncertainty_score`** — prediction entropy normalised to [0, 1]
- **`diversity_score`** — cosine distance from already labeled set
- **`region_of_interest`** — approximate anatomical attention (e.g., "left_lung_upper")
- **`patient_priority`** — inferred urgency (`routine`, `urgent`, `critical`)
- **`acquisition_cost`** — budget cost if selected (1 or 2)

---

## Action Space

Each step the agent submits **one** action:

```json
{
  "selected_image_id": "task_img_0042",
  "reasoning": "Early phase: Targeting left_lung_upper to expand class coverage."
}
```

---

## Reward Shaping

The per-step reward provides highly dense, curriculum-adjusted feedback:

1. **AUC Delta**: Baseline reward for measurable improvement.
2. **Diversity Bonus**: Encourages exploring under-represented regions.
3. **Class Coverage Bonus**: Large reward for discovering a new class for the first time.
4. **Diminishing Returns Penalty**: Weakens rewards for repeatedly sampling majority classes.
5. **Redundancy Penalty**: Punishes selecting near-duplicates. Amplifies by 2.5x in the `late` phase.

All multiplied by the `curriculum_multiplier` (1.2x early, 1.0x mid, 0.8x late).

---

## Inference Agent Strategies

The bundled inference agent uses the `OPENAI_API_KEY` to prompt `gpt-4o-mini` with phase-aware strategies:
- **Early**: Prioritize diversity and discover all classes.
- **Mid**: Pivot to uncertainty reduction to drive AUC.
- **Late**: Strict budget conservation; pick only high-impact samples.

### Baseline Scores (Phase-aware LLM)

| Task | Final AUC | Typical Grader Score |
|------|-----------|----------------------|
| `task1` | ~0.76 | ~0.89 |
| `task2` | ~0.68 | ~0.82 |
| `task3` | ~0.62 | ~0.68 |

*(Scores are significantly higher than random baselines due to variable cost management)*

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

## Running the Inference Agent

```bash
export HF_TOKEN=<your_huggingface_token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=gpt-4o-mini
export ENV_BASE_URL=http://localhost:7860

python inference.py
```
