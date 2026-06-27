"""Shared data models for the GuildLM brain.

These light-weight Pydantic models are passed between the router, the
registry and the orchestrator. Keeping them in one module avoids circular
imports and gives a single source of truth for the wire/JSON shapes used by
the API layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Classification(BaseModel):
    """Structured result of intent classification.

    This is the contract the router promises to produce regardless of whether
    the underlying decision came from an LLM or the deterministic heuristic
    fallback.
    """

    domain: str = Field("general", description="High-level domain, e.g. 'code'.")
    language: str = Field("unknown", description="Programming/natural language, e.g. 'go'.")
    task: str = Field("general_qa", description="Coarse task, e.g. 'generation'.")
    subtask: Optional[str] = Field(None, description="Optional finer-grained task hint.")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Classifier confidence [0,1].")
    source: str = Field("heuristic", description="Which classifier produced this: 'llm' or 'heuristic'.")


class Specialist(BaseModel):
    """A single guild specialist (an SLM with a focused system prompt)."""

    id: str = Field(..., description="Stable id, e.g. 'guild-code/go_generator'.")
    guild: str = Field(..., description="Owning guild, e.g. 'code'.")
    domain: str = Field(..., description="Domain the specialist serves.")
    languages: List[str] = Field(default_factory=list, description="Languages handled.")
    tasks: List[str] = Field(default_factory=list, description="Tasks handled.")
    system_prompt: str = Field(..., description="System prompt injected at inference time.")
    model: str = Field(..., description="Base model id served by the backend.")
    lora: Optional[str] = Field(None, description="Optional LoRA adapter id to hot-swap.")


class PipelineStep(BaseModel):
    """One step in a multi-step orchestration pipeline."""

    specialist: str = Field(..., description="Specialist id to invoke for this step.")
    action: str = Field(..., description="Human-readable label, e.g. 'analyze'.")
    prompt_template: str = Field(
        "{input}",
        description="Template for the step prompt. Supports {input} and {previous}.",
    )


class StepResult(BaseModel):
    """The output of a single executed pipeline step."""

    specialist: str
    action: str
    output: str


class ChatResult(BaseModel):
    """Full result of routing and executing a user request."""

    classification: Classification
    specialist: str = Field(..., description="Primary specialist selected.")
    steps: List[StepResult] = Field(default_factory=list)
    answer: str = Field(..., description="Final answer produced by the pipeline.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict (handy for logging / JSON responses)."""
        return self.model_dump()
