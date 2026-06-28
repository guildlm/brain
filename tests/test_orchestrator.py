"""Tests for the orchestrator: single-specialist and multi-step pipelines."""

from __future__ import annotations

from src.core.backends import FakeBackend, ModelManager
from src.core.models import Classification
from src.core.orchestrator import BrainOrchestrator
from src.core.registry import GuildRegistry
from src.core.router import IntentRouter


def _orchestrator(backend: FakeBackend) -> BrainOrchestrator:
    return BrainOrchestrator(
        registry=GuildRegistry.from_yaml(),
        router=IntentRouter(offline=True),
        backend=backend,
    )


def test_single_specialist_execution():
    backend = FakeBackend(lambda model, msgs: "GENERATED CODE")
    orch = _orchestrator(backend)
    # An explanation request maps to a single specialist (generation is now a
    # multi-step guild pipeline, so it would no longer exercise the single path).
    result = orch.execute_request("Explain what a goroutine is in Go")
    assert result.specialist == "guild-code/go_explainer"
    assert result.answer == "GENERATED CODE"
    assert len(result.steps) == 1
    assert result.steps[0].action == "answer"
    # The system prompt of the selected specialist must be injected.
    sys_msg = backend.calls[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "Go" in sys_msg["content"]


def test_generalist_fallback_execution():
    backend = FakeBackend(lambda model, msgs: "general answer")
    orch = _orchestrator(backend)
    result = orch.execute_request("What is the weather today?")
    assert result.specialist == "brain/generalist"
    assert result.answer == "general answer"


def test_multistep_bugfix_pipeline_threads_outputs():
    # Each step returns a marker so we can verify ordering and threading.
    seq = {"n": 0}

    def responder(model, messages):
        seq["n"] += 1
        user = messages[-1]["content"]
        return f"STEP{seq['n']}::sees<{user[:40]}>"

    backend = FakeBackend(responder)
    orch = _orchestrator(backend)
    result = orch.execute_request("Fix the deadlock bug in my Go goroutine")

    # Three-step pipeline: analyze -> fix -> verify
    assert [s.action for s in result.steps] == ["analyze", "fix", "verify"]
    assert [s.specialist for s in result.steps] == [
        "guild-code/go_reviewer",
        "guild-code/go_generator",
        "guild-code/go_reviewer",
    ]
    # Step 2 prompt must include step 1's output (threaded via {previous}).
    fix_prompt = backend.calls[1]["messages"][-1]["content"]
    assert "STEP1" in fix_prompt
    # Final answer is the last step's output.
    assert result.answer == result.steps[-1].output
    assert result.specialist == "guild-code/go_reviewer"


def test_backend_failure_is_caught_and_reported():
    def boom(model, messages):
        raise RuntimeError("backend down")

    orch = _orchestrator(FakeBackend(boom))
    result = orch.execute_request("Build an HTTP API in Go")
    assert "Error: inference backend unavailable" in result.answer


def test_model_manager_tracks_loaded_adapters():
    backend = FakeBackend(lambda model, msgs: "ok")
    mm = ModelManager()
    orch = BrainOrchestrator(
        registry=GuildRegistry.from_yaml(),
        router=IntentRouter(offline=True),
        backend=backend,
        model_manager=mm,
    )
    orch.execute_request("Build an HTTP API in Go")
    assert any("go-generator-lora" in key for key in mm.loaded)


def test_classify_only_does_not_execute():
    backend = FakeBackend(lambda model, msgs: "should not be called")
    orch = _orchestrator(backend)
    classification = orch.classify("review my Go code")
    assert isinstance(classification, Classification)
    assert classification.task == "review"
    assert backend.calls == []  # no execution happened
