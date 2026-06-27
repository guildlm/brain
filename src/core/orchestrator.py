"""Central orchestration for the GuildLM brain.

:class:`BrainOrchestrator` ties the pieces together:

1. The :class:`~src.core.router.IntentRouter` classifies the request.
2. The :class:`~src.core.registry.GuildRegistry` selects a specialist and/or a
   multi-step pipeline.
3. The injected :class:`~src.core.backends.Backend` runs inference for each
   step, threading outputs through the pipeline.

The brain itself never answers domain questions — it only routes and
orchestrates. The backend is injectable so tests run a :class:`FakeBackend`
with no network.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.core.backends import Backend, ModelManager, OpenAIBackend
from src.core.models import ChatResult, Classification, PipelineStep, StepResult
from src.core.registry import GuildRegistry
from src.core.router import IntentRouter

logger = logging.getLogger(__name__)


class BrainOrchestrator:
    """Coordinate routing and (multi-step) execution of a user request."""

    def __init__(
        self,
        registry: Optional[GuildRegistry] = None,
        router: Optional[IntentRouter] = None,
        backend: Optional[Backend] = None,
        model_manager: Optional[ModelManager] = None,
    ) -> None:
        self.registry = registry or GuildRegistry.from_yaml()
        self.backend = backend or OpenAIBackend()
        # Router shares the execution backend by default so both speak to the
        # same endpoint; callers may inject a dedicated router backend instead.
        self.router = router or IntentRouter(backend=self.backend)
        self.model_manager = model_manager or ModelManager()

    # ----------------------------------------------------------------- public
    def classify(self, prompt: str) -> Classification:
        """Classify a request without executing it."""
        return self.router.classify(prompt)

    def execute_request(self, prompt: str) -> ChatResult:
        """Classify, route and execute a request end to end."""
        logger.info("Received request: %r", prompt)
        classification = self.router.classify(prompt)

        pipeline = self.registry.pipeline_for(classification)
        if pipeline:
            logger.info(
                "Executing %d-step pipeline for %s:%s",
                len(pipeline),
                classification.domain,
                classification.task,
            )
            return self._run_pipeline(prompt, classification, pipeline)

        specialist = self.registry.select_specialist(classification)
        logger.info("Routing to single specialist: %s", specialist.id)
        step = self._run_step(
            specialist_id=specialist.id,
            action="answer",
            prompt_text=prompt,
        )
        return ChatResult(
            classification=classification,
            specialist=specialist.id,
            steps=[step],
            answer=step.output,
        )

    # ---------------------------------------------------------------- pipeline
    def _run_pipeline(
        self,
        prompt: str,
        classification: Classification,
        pipeline: List[PipelineStep],
    ) -> ChatResult:
        """Run a multi-step pipeline, threading each step's output forward."""
        results: List[StepResult] = []
        previous = ""
        for step in pipeline:
            prompt_text = step.prompt_template.format(input=prompt, previous=previous)
            result = self._run_step(
                specialist_id=step.specialist,
                action=step.action,
                prompt_text=prompt_text,
            )
            results.append(result)
            previous = result.output

        primary = pipeline[0].specialist
        return ChatResult(
            classification=classification,
            specialist=primary,
            steps=results,
            answer=results[-1].output if results else "",
        )

    # -------------------------------------------------------------------- step
    def _run_step(self, specialist_id: str, action: str, prompt_text: str) -> StepResult:
        """Invoke a single specialist via the backend and return its output."""
        specialist = self.registry.get(specialist_id)
        # Bookkeeping: ensure the specialist's adapter is notionally resident.
        self.model_manager.ensure_loaded(specialist.model, specialist.lora)

        try:
            output = self.backend.chat(
                model=specialist.model,
                messages=[
                    {"role": "system", "content": specialist.system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface a useful message, do not crash
            logger.error("Inference failed for %s: %s", specialist_id, exc)
            output = (
                f"Error: inference backend unavailable for '{specialist_id}'. "
                f"Details: {exc}"
            )
        return StepResult(specialist=specialist_id, action=action, output=output)
