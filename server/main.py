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
    return {"name": "Nidan", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


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
