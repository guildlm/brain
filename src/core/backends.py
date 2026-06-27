"""Inference backend abstraction for the GuildLM brain.

The brain never embeds a hard dependency on a concrete LLM server. Instead it
talks to a small :class:`Backend` protocol. The production implementation
(:class:`OpenAIBackend`) targets any OpenAI-compatible endpoint (Ollama, vLLM,
llama.cpp, OpenAI itself). Tests use :class:`FakeBackend` so the whole stack is
exercisable with no network access.

:class:`ModelManager` is a VRAM-aware bookkeeping stub. It tracks which LoRA
adapters are notionally "loaded" so the orchestrator can reason about hot-swaps
today and we can wire real adapter loading (vLLM multi-LoRA) later without
changing call sites.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# A chat message is the familiar OpenAI {"role": ..., "content": ...} dict.
Message = Dict[str, str]


@runtime_checkable
class Backend(Protocol):
    """Minimal chat-completion interface the orchestrator depends on."""

    def chat(
        self,
        model: str,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Return the assistant's text reply for ``messages``."""
        ...


class OpenAIBackend:
    """Backend backed by any OpenAI-compatible HTTP endpoint.

    Defaults target a local Ollama server. The ``api_key`` is required by the
    OpenAI SDK but ignored by most local servers.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # Imported lazily so that test environments / the heuristic-only path do
        # not require the openai package to be importable at module load.
        from openai import OpenAI

        self.base_url = base_url or os.getenv("BRAIN_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("BRAIN_API_KEY", "ollama")
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        logger.info("OpenAIBackend initialised (base_url=%s)", self.base_url)

    def chat(
        self,
        model: str,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Call the chat-completions endpoint and return the reply text."""
        logger.debug("OpenAIBackend.chat model=%s msgs=%d", model, len(messages))
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class FakeBackend:
    """Deterministic in-memory backend for tests and offline development.

    Supply ``responder`` to compute a reply from the request, or rely on the
    default which echoes a structured, inspectable string. Every call is
    recorded on :attr:`calls` for assertions.
    """

    def __init__(
        self,
        responder: Optional[Callable[[str, List[Message]], str]] = None,
    ) -> None:
        self._responder = responder
        self.calls: List[Dict[str, object]] = []

    def chat(
        self,
        model: str,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Record the call and return a deterministic reply."""
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        if self._responder is not None:
            return self._responder(model, messages)
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[fake:{model}] {user}"


class ModelManager:
    """VRAM-aware adapter bookkeeping stub.

    Tracks which (base model, LoRA adapter) pairs are notionally resident.
    Today this is logging + a small LRU cap; later it can drive real vLLM
    multi-LoRA load/unload calls without touching the orchestrator.
    """

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        # Maintains insertion order so we can evict the least-recently loaded.
        self._loaded: Dict[str, str] = {}

    @staticmethod
    def _key(model: str, lora: Optional[str]) -> str:
        return f"{model}::{lora or 'base'}"

    def is_loaded(self, model: str, lora: Optional[str] = None) -> bool:
        """Return whether the (model, lora) pair is currently resident."""
        return self._key(model, lora) in self._loaded

    def ensure_loaded(self, model: str, lora: Optional[str] = None) -> str:
        """Ensure a (model, lora) pair is resident, evicting if necessary.

        Returns the resolved adapter key. This is pure bookkeeping for now.
        """
        key = self._key(model, lora)
        if key in self._loaded:
            # Refresh recency.
            self._loaded.pop(key)
            self._loaded[key] = lora or "base"
            logger.debug("Adapter already loaded: %s", key)
            return key

        while len(self._loaded) >= self.capacity:
            evicted, _ = next(iter(self._loaded.items()))
            self._loaded.pop(evicted)
            logger.info("Evicting adapter to free VRAM: %s", evicted)

        self._loaded[key] = lora or "base"
        logger.info("Hot-swapped adapter into VRAM: %s", key)
        return key

    @property
    def loaded(self) -> List[str]:
        """List currently resident adapter keys (LRU order, oldest first)."""
        return list(self._loaded.keys())
