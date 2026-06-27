"""FastAPI surface for the GuildLM brain.

Exposes routing and orchestration over HTTP:

* ``POST /classify`` — classify a prompt (no execution).
* ``POST /chat``     — route + execute, returning the specialist used and answer.
* ``GET  /guilds``   — dump the loaded registry.
* ``GET  /health``   — liveness probe.

The orchestrator is provided via a FastAPI dependency so tests can inject one
backed by :class:`~src.core.backends.FakeBackend` (no real LLM required).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.core.backends import OpenAIBackend
from src.core.models import ChatResult, Classification, Specialist
from src.core.orchestrator import BrainOrchestrator
from src.core.registry import GuildRegistry
from src.core.router import IntentRouter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- schemas
class ClassifyRequest(BaseModel):
    """Request body for ``/classify`` and ``/chat``."""

    prompt: str = Field(..., min_length=1, description="The user request to route.")


class ChatResponse(BaseModel):
    """Response body for ``/chat``."""

    specialist: str
    answer: str
    classification: Classification
    steps: List[dict]


class GuildsResponse(BaseModel):
    """Response body for ``/guilds``."""

    base_model: str
    fallback: str
    specialists: List[Specialist]


class HealthResponse(BaseModel):
    """Response body for ``/health``."""

    status: str = "ok"
    specialists: int = 0


def create_app(orchestrator: Optional[BrainOrchestrator] = None) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    orchestrator:
        Injected orchestrator. When ``None`` a production orchestrator backed by
        :class:`OpenAIBackend` is lazily constructed on first use. Tests pass a
        fake-backed orchestrator here.
    """
    app = FastAPI(
        title="GuildLM Brain",
        version="0.1.0",
        description="Central router/orchestrator for the GuildLM guild of specialist SLMs.",
    )
    app.state.orchestrator = orchestrator

    def get_orchestrator() -> BrainOrchestrator:
        """Resolve (lazily building) the request orchestrator."""
        if app.state.orchestrator is None:
            backend = OpenAIBackend()
            registry = GuildRegistry.from_yaml()
            app.state.orchestrator = BrainOrchestrator(
                registry=registry,
                router=IntentRouter(backend=backend),
                backend=backend,
            )
        return app.state.orchestrator

    @app.get("/health", response_model=HealthResponse)
    def health(orch: BrainOrchestrator = Depends(get_orchestrator)) -> HealthResponse:
        """Liveness probe with a quick registry sanity count."""
        return HealthResponse(status="ok", specialists=len(orch.registry.specialists))

    @app.post("/classify", response_model=Classification)
    def classify(
        body: ClassifyRequest,
        orch: BrainOrchestrator = Depends(get_orchestrator),
    ) -> Classification:
        """Classify a prompt without executing it."""
        return orch.classify(body.prompt)

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        body: ClassifyRequest,
        orch: BrainOrchestrator = Depends(get_orchestrator),
    ) -> ChatResponse:
        """Route and execute a prompt, returning the specialist and answer."""
        result: ChatResult = orch.execute_request(body.prompt)
        return ChatResponse(
            specialist=result.specialist,
            answer=result.answer,
            classification=result.classification,
            steps=[s.model_dump() for s in result.steps],
        )

    @app.get("/guilds", response_model=GuildsResponse)
    def guilds(orch: BrainOrchestrator = Depends(get_orchestrator)) -> GuildsResponse:
        """Dump the loaded guild registry."""
        reg = orch.registry
        return GuildsResponse(
            base_model=reg.base_model,
            fallback=reg.fallback_id,
            specialists=reg.specialists,
        )

    return app


# Module-level app for ``uvicorn src.api.server:app`` (production entrypoint).
app = create_app()
