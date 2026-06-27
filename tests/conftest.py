"""Shared pytest fixtures for the GuildLM brain test-suite."""

from __future__ import annotations

import pytest

from src.core.backends import FakeBackend
from src.core.orchestrator import BrainOrchestrator
from src.core.registry import GuildRegistry
from src.core.router import IntentRouter


@pytest.fixture
def registry() -> GuildRegistry:
    """The real bundled guild registry loaded from configs/guilds.yaml."""
    return GuildRegistry.from_yaml()


@pytest.fixture
def fake_backend() -> FakeBackend:
    """A deterministic backend that echoes the user prompt."""
    return FakeBackend()


@pytest.fixture
def offline_orchestrator(registry: GuildRegistry, fake_backend: FakeBackend) -> BrainOrchestrator:
    """An orchestrator using the heuristic router and a fake backend (no network)."""
    router = IntentRouter(offline=True)
    return BrainOrchestrator(registry=registry, router=router, backend=fake_backend)
