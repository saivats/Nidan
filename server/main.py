from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from server.env import Nidan
from server.models import Action, ResetRequest, StepResponse

app = FastAPI(
    title="Nidan",
    description="Medical image active learning environment for radiological triage.",
    version="1.0.0",
)

env = Nidan()


@app.get("/")
async def root():
    return {"name": "nidan", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "nidan"}


@app.get("/metadata")
async def metadata():
    return {
        "name": "nidan",
        "description": "Medical image active learning environment. Agent selects which X-rays to annotate to maximize diagnostic AUC with minimal labels.",
        "version": "1.0.0",
        "tasks": ["task1", "task2", "task3"],
    }


@app.get("/schema")
async def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "selected_image_id": {"type": "string", "description": "ID of the image to annotate"},
                "reasoning": {"type": "string", "description": "Optional reasoning for selection"},
            },
            "required": ["selected_image_id"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "budget_remaining": {"type": "integer"},
                "unlabeled_pool_size": {"type": "integer"},
                "current_model_auc": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "candidate_images": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string"},
                            "uncertainty_score": {"type": "number"},
                            "diversity_score": {"type": "number"},
                        },
                    },
                },
                "last_annotation_result": {"type": "string"},
                "embedding_stats": {"type": "object"},
                "episode_history": {"type": "array"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "budget_used": {"type": "integer"},
                "budget_remaining": {"type": "integer"},
                "current_auc": {"type": "number"},
                "labeled_set_size": {"type": "integer"},
                "unlabeled_pool_size": {"type": "integer"},
                "label_distribution": {"type": "object"},
                "cumulative_reward": {"type": "number"},
                "episode_history": {"type": "array"},
            },
        },
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    try:
        observation = env.reset(request.task_id)
        return observation.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {request.task_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
async def step(action: Action):
    try:
        observation, reward, done, info = env.step(action)
        response = StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
        return response.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
async def state():
    try:
        return JSONResponse(content=env.state())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
