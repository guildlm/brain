"""Tests for the FastAPI surface using TestClient with a fake backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.backends import FakeBackend
from src.core.orchestrator import BrainOrchestrator
from src.core.registry import GuildRegistry
from src.core.router import IntentRouter


@pytest.fixture
def client() -> TestClient:
    backend = FakeBackend(lambda model, msgs: "FAKE ANSWER")
    orch = BrainOrchestrator(
        registry=GuildRegistry.from_yaml(),
        router=IntentRouter(offline=True),
        backend=backend,
    )
    return TestClient(create_app(orchestrator=orch))


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["specialists"] >= 5


def test_classify_endpoint(client: TestClient):
    resp = client.post("/classify", json={"prompt": "Build an HTTP API in Go"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["domain"] == "code"
    assert body["language"] == "go"
    assert body["task"] == "generation"
    assert body["source"] == "heuristic"


def test_chat_endpoint_single_specialist(client: TestClient):
    resp = client.post("/chat", json={"prompt": "Build an HTTP API in Go"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["specialist"] == "guild-code/go_generator"
    assert body["answer"] == "FAKE ANSWER"
    assert body["classification"]["domain"] == "code"
    assert len(body["steps"]) == 1


def test_chat_endpoint_pipeline(client: TestClient):
    resp = client.post("/chat", json={"prompt": "Fix the deadlock bug in my Go goroutine"})
    assert resp.status_code == 200
    body = resp.json()
    assert [s["action"] for s in body["steps"]] == ["analyze", "fix", "verify"]


def test_guilds_endpoint(client: TestClient):
    resp = client.get("/guilds")
    assert resp.status_code == 200
    body = resp.json()
    ids = {s["id"] for s in body["specialists"]}
    assert "guild-code/go_generator" in ids
    assert body["fallback"] == "brain/generalist"


def test_classify_validation_error(client: TestClient):
    resp = client.post("/classify", json={"prompt": ""})
    assert resp.status_code == 422  # empty prompt rejected by pydantic


def test_app_imports_without_orchestrator():
    # The production app builds lazily and must import cleanly with no backend.
    app = create_app()
    assert app.title == "GuildLM Brain"
