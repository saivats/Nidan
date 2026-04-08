from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from server.env import Nidan
from server.models import Action, ResetRequest, StepResponse

app = FastAPI(
    title="Nidan",
    description="Medical image active learning environment for radiological triage.",
    version="1.0.0",
)

env = Nidan()

MCP_TOOLS = [
    {
        "name": "reset",
        "description": "Start a new task episode",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "enum": ["task1", "task2", "task3"]},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "step",
        "description": "Select an image to annotate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selected_image_id": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["selected_image_id"],
        },
    },
    {
        "name": "state",
        "description": "Get current environment state",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _jsonrpc_response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _handle_mcp_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Any:
    if tool_name == "reset":
        task_id = arguments.get("task_id", "task1")
        observation = env.reset(task_id)
        return observation.model_dump()

    if tool_name == "step":
        action = Action(**arguments)
        observation, reward, done, info = env.step(action)
        response = StepResponse(observation=observation, reward=reward, done=done, info=info)
        return response.model_dump()

    if tool_name == "state":
        return env.state()

    raise ValueError(f"Unknown tool: {tool_name}")


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


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content=_jsonrpc_error(None, -32700, "Parse error"),
            status_code=200,
        )

    request_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    if body.get("jsonrpc") != "2.0":
        return JSONResponse(
            content=_jsonrpc_error(request_id, -32600, "Invalid JSON-RPC version"),
            status_code=200,
        )

    if method == "initialize":
        result = {
            "name": "nidan",
            "version": "1.0.0",
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "nidan",
                "version": "1.0.0",
            },
        }
        return JSONResponse(content=_jsonrpc_response(request_id, result), status_code=200)

    if method == "tools/list":
        result = {"tools": MCP_TOOLS}
        return JSONResponse(content=_jsonrpc_response(request_id, result), status_code=200)

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            tool_result = _handle_mcp_tool_call(tool_name, arguments)
            result = {"content": [{"type": "text", "text": str(tool_result)}]}
            return JSONResponse(content=_jsonrpc_response(request_id, result), status_code=200)
        except Exception as exc:
            return JSONResponse(
                content=_jsonrpc_error(request_id, -32603, str(exc)),
                status_code=200,
            )

    return JSONResponse(
        content=_jsonrpc_error(request_id, -32601, f"Method not found: {method}"),
        status_code=200,
    )


@app.post("/reset")
async def reset(request_body: ResetRequest):
    try:
        observation = env.reset(request_body.task_id)
        return observation.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {request_body.task_id}") from exc
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
async def get_state():
    try:
        return JSONResponse(content=env.state())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/close")
async def close_env():
    env._state = None
    env._episode_history = []
    env._model = None
    env._rare_positives_found = 0
    env._last_annotation_result = None
    return {"status": "closed"}

