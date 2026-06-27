"""Intent classification / routing for the GuildLM brain.

The router decides *what kind of request* the user made — never the answer. It
produces a :class:`~src.core.models.Classification` describing the domain,
language, task, subtask and a confidence score.

Two classifiers are layered:

* An **LLM classifier** that asks an OpenAI-compatible model for structured
  JSON. This is the production path and handles nuance the keywords miss.
* A deterministic **heuristic classifier** (improved keyword logic) used as a
  fallback whenever the LLM is unavailable, returns garbage, reports low
  confidence, or when ``offline=True``.

The seam between the two is explicit so the heuristic path is fully unit
testable with no network.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

from src.core.backends import Backend
from src.core.models import Classification

logger = logging.getLogger(__name__)

# Below this confidence the LLM result is discarded in favour of the heuristic.
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

SUPPORTED_DOMAINS = ["code", "legal", "finance", "medical", "creative", "general"]

_CLASSIFIER_SYSTEM_PROMPT = (
    "You are the routing classifier for GuildLM, a system of specialist models. "
    "You do NOT answer the user's question. You ONLY classify their intent. "
    "Respond with a single minified JSON object and nothing else, using exactly "
    "these keys: domain, language, task, subtask, confidence.\n"
    f"- domain: one of {SUPPORTED_DOMAINS}.\n"
    "- language: the programming or natural language involved (e.g. 'go', "
    "'python', 'english'), or 'unknown'.\n"
    "- task: a coarse action such as 'generation', 'review', 'testing', "
    "'bug_fix', 'explanation', or 'general_qa'.\n"
    "- subtask: an optional finer hint, or null.\n"
    "- confidence: a float in [0,1] for how sure you are.\n"
    "Example: {\"domain\":\"code\",\"language\":\"go\",\"task\":\"generation\","
    "\"subtask\":\"http_server\",\"confidence\":0.92}"
)


def parse_classification(raw: str, *, source: str = "llm") -> Optional[Classification]:
    """Robustly parse a model's text into a :class:`Classification`.

    Handles clean JSON, JSON wrapped in markdown code fences, and JSON embedded
    in surrounding prose. Returns ``None`` when nothing usable can be salvaged
    so callers can fall back to the heuristic.
    """
    if not raw or not raw.strip():
        return None

    candidate = _extract_json_object(raw)
    if candidate is None:
        logger.warning("No JSON object found in classifier output: %r", raw[:120])
        return None

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("Classifier output was not valid JSON: %r", candidate[:120])
        return None

    if not isinstance(data, dict):
        return None

    # Normalise / coerce fields defensively; the model may omit or mistype keys.
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    subtask = data.get("subtask")
    if subtask in ("", "null", "none"):
        subtask = None

    return Classification(
        domain=str(data.get("domain") or "general").lower(),
        language=str(data.get("language") or "unknown").lower(),
        task=str(data.get("task") or "general_qa").lower(),
        subtask=str(subtask).lower() if subtask else None,
        confidence=confidence,
        source=source,
    )


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first balanced ``{...}`` JSON object substring from text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class IntentRouter:
    """Classify user prompts into routing decisions.

    Parameters
    ----------
    backend:
        Optional inference backend used for LLM classification. When ``None``
        the router operates purely on heuristics.
    model:
        Model id to use for classification (env ``BRAIN_ROUTER_MODEL``).
    offline:
        Force the deterministic heuristic path (no backend calls).
    confidence_threshold:
        LLM results below this are replaced by the heuristic result.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        model: Optional[str] = None,
        *,
        offline: bool = False,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.backend = backend
        self.model = model or os.getenv("BRAIN_ROUTER_MODEL", "qwen2.5:7b-instruct")
        self.offline = offline
        self.confidence_threshold = confidence_threshold

    def classify(self, prompt: str) -> Classification:
        """Return the best available classification for ``prompt``."""
        if self.offline or self.backend is None:
            logger.info("Routing via heuristic classifier (offline=%s)", self.offline)
            return self.heuristic_classify(prompt)

        llm_result = self._llm_classify(prompt)
        if llm_result is None:
            logger.info("LLM classification unavailable; using heuristic fallback")
            return self.heuristic_classify(prompt)

        if llm_result.confidence < self.confidence_threshold:
            logger.info(
                "LLM confidence %.2f below threshold %.2f; using heuristic fallback",
                llm_result.confidence,
                self.confidence_threshold,
            )
            return self.heuristic_classify(prompt)

        logger.info("Classified via LLM: %s", llm_result.model_dump())
        return llm_result

    # Backwards-compatible alias for the original public method name.
    def classify_intent(self, prompt: str) -> Classification:
        """Deprecated alias of :meth:`classify`."""
        return self.classify(prompt)

    def _llm_classify(self, prompt: str) -> Optional[Classification]:
        """Ask the backend model for a structured classification."""
        if self.backend is None:
            return None
        try:
            raw = self.backend.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any backend error
            logger.warning("Router backend call failed: %s", exc)
            return None
        return parse_classification(raw, source="llm")

    def heuristic_classify(self, prompt: str) -> Classification:
        """Deterministic keyword-based classifier (the offline fallback).

        An improved version of the original keyword logic: it detects the Go
        language, distinguishes generation / review / testing / bug-fix /
        explanation tasks, and assigns a modest confidence so that callers can
        still reason about reliability.
        """
        text = prompt.lower()
        tokens = set(re.findall(r"[a-z0-9_+#.]+", text))

        domain = "general"
        language = "unknown"
        task = "general_qa"
        subtask: Optional[str] = None
        confidence = 0.3

        if self._mentions(tokens, text, {"go", "golang", "goroutine", "goroutines"}):
            domain = "code"
            language = "go"
            confidence = 0.7
            task, subtask = self._classify_code_task(text)
        elif self._mentions(tokens, text, {"code", "function", "program", "refactor", "compile"}):
            domain = "code"
            confidence = 0.5
            task, subtask = self._classify_code_task(text)

        result = Classification(
            domain=domain,
            language=language,
            task=task,
            subtask=subtask,
            confidence=confidence,
            source="heuristic",
        )
        logger.info("Classified via heuristic: %s", result.model_dump())
        return result

    @staticmethod
    def _mentions(tokens: set, text: str, needles: set) -> bool:
        """True if any needle appears as a token or substring phrase."""
        if tokens & needles:
            return True
        return any(" " in n and n in text for n in needles)

    @staticmethod
    def _classify_code_task(text: str) -> tuple[str, Optional[str]]:
        """Map free text to a (task, subtask) pair for the code domain.

        Order matters: explicit review/testing/explanation intents take
        precedence over bug-fix keywords (e.g. "review for race conditions" is a
        review, not a bug fix).
        """
        if any(k in text for k in ("test", "unit test", "coverage", "table-driven")):
            return "testing", None
        if any(k in text for k in ("review", "audit", "security", "vet", "lint")):
            return "review", None
        if any(k in text for k in ("explain", "understand", "what does", "how does", "document")):
            return "explanation", None
        if any(k in text for k in ("race condition", "deadlock", "bug", "fix", "broken", "panic")):
            return "bug_fix", None
        if any(k in text for k in ("build", "write", "create", "implement", "generate", "api", "server")):
            return "generation", None
        return "review", None
