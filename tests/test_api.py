from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.main import app


client = TestClient(app)


def test_reset_post_without_body_defaults_to_task1():
    response = client.post("/reset")
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "task1"


def test_reset_post_with_null_body_defaults_to_task1():
    response = client.post("/reset", content="null", headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "task1"


def test_reset_post_with_empty_json_defaults_to_task1():
    response = client.post("/reset", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "task1"


def test_reset_post_with_task_body_works():
    response = client.post("/reset", json={"task_id": "task2"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "task2"


def test_reset_post_with_invalid_body_returns_422():
    response = client.post("/reset", json={"task_id": "invalid"})
    assert response.status_code == 422
