"""Tests for intent classification: heuristic cases, parser robustness, fallback."""

from __future__ import annotations

import pytest

from src.core.backends import FakeBackend
from src.core.models import Classification
from src.core.router import IntentRouter, parse_classification


# ---------------------------------------------------------------- heuristics
@pytest.mark.parametrize(
    ("prompt", "domain", "language", "task"),
    [
        ("Build a high performance HTTP API in Go", "code", "go", "generation"),
        ("Please review this Go code for race conditions", "code", "go", "review"),
        ("Write a table-driven unit test for my Go function", "code", "go", "testing"),
        ("There is a bug causing a deadlock in my goroutine", "code", "go", "bug_fix"),
        ("Explain what this Go channel code does", "code", "go", "explanation"),
        ("What is the capital of France?", "general", "unknown", "general_qa"),
    ],
)
def test_heuristic_classification(prompt, domain, language, task):
    router = IntentRouter(offline=True)
    result = router.classify(prompt)
    assert isinstance(result, Classification)
    assert result.source == "heuristic"
    assert result.domain == domain
    assert result.language == language
    assert result.task == task


def test_offline_router_never_calls_backend():
    backend = FakeBackend()
    router = IntentRouter(backend=backend, offline=True)
    router.classify("Build something in Go")
    assert backend.calls == []  # offline must not touch the backend


# ------------------------------------------------------------- parser robust
def test_parse_clean_json():
    raw = '{"domain":"code","language":"go","task":"generation","subtask":"http","confidence":0.9}'
    result = parse_classification(raw)
    assert result is not None
    assert result.domain == "code"
    assert result.subtask == "http"
    assert result.confidence == pytest.approx(0.9)
    assert result.source == "llm"


def test_parse_json_in_code_fence_and_prose():
    raw = "Sure! Here is the classification:\n```json\n{\"domain\": \"code\", \"task\": \"review\", \"confidence\": 0.8}\n```\nHope that helps."
    result = parse_classification(raw)
    assert result is not None
    assert result.domain == "code"
    assert result.task == "review"


def test_parse_garbled_json_returns_none():
    assert parse_classification("not json at all") is None
    assert parse_classification("{domain: code, oops") is None
    assert parse_classification("") is None


def test_parse_clamps_and_coerces_bad_confidence():
    raw = '{"domain":"code","confidence":"high"}'
    result = parse_classification(raw)
    assert result is not None
    assert result.confidence == 0.0  # non-numeric coerced to 0


# --------------------------------------------------------------- LLM + fallback
def _responder_factory(payload: str):
    def responder(model, messages):
        return payload

    return responder


def test_llm_classification_used_when_confident():
    backend = FakeBackend(
        _responder_factory('{"domain":"code","language":"go","task":"testing","confidence":0.95}')
    )
    router = IntentRouter(backend=backend)
    result = router.classify("write tests")
    assert result.source == "llm"
    assert result.task == "testing"
    assert len(backend.calls) == 1


def test_low_confidence_falls_back_to_heuristic():
    backend = FakeBackend(
        _responder_factory('{"domain":"legal","language":"english","task":"review","confidence":0.05}')
    )
    router = IntentRouter(backend=backend, confidence_threshold=0.5)
    result = router.classify("Build an HTTP API in Go")
    assert result.source == "heuristic"
    assert result.domain == "code"


def test_garbled_llm_output_falls_back_to_heuristic():
    backend = FakeBackend(_responder_factory("the model rambled and produced no json"))
    router = IntentRouter(backend=backend)
    result = router.classify("review my Go code")
    assert result.source == "heuristic"
    assert result.task == "review"


def test_backend_exception_falls_back_to_heuristic():
    def boom(model, messages):
        raise RuntimeError("connection refused")

    backend = FakeBackend(boom)
    router = IntentRouter(backend=backend)
    result = router.classify("Build something in Go")
    assert result.source == "heuristic"
