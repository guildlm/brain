"""Guild & specialist registry for the GuildLM brain.

The registry is the single source of truth for *who* can serve a request. It is
loaded from ``configs/guilds.yaml`` so guilds/specialists can be added without
code changes. :meth:`GuildRegistry.select_specialist` resolves a
:class:`~src.core.models.Classification` to a concrete specialist via the data,
replacing the old hardcoded if/else routing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.core.models import Classification, PipelineStep, Specialist

logger = logging.getLogger(__name__)

# Default config path: <repo>/configs/guilds.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "guilds.yaml"


class GuildRegistry:
    """In-memory view over the guild configuration.

    Use :meth:`from_yaml` to construct from disk, or pass a parsed config dict
    directly (handy for tests).
    """

    def __init__(self, config: Dict) -> None:
        defaults = config.get("defaults", {}) or {}
        self.base_model: str = defaults.get("base_model", "qwen2.5:7b-instruct")
        self.fallback_id: str = defaults.get("fallback_specialist", "brain/generalist")

        self._specialists: Dict[str, Specialist] = {}
        for raw in config.get("specialists", []) or []:
            spec = Specialist(**raw)
            self._specialists[spec.id] = spec

        if self.fallback_id not in self._specialists:
            raise ValueError(
                f"Fallback specialist '{self.fallback_id}' is not defined in the registry."
            )

        self._pipelines: Dict[str, List[PipelineStep]] = {}
        for key, steps in (config.get("pipelines", {}) or {}).items():
            self._pipelines[key] = [PipelineStep(**s) for s in steps]

        logger.info(
            "Loaded registry: %d specialists, %d pipelines",
            len(self._specialists),
            len(self._pipelines),
        )

    @classmethod
    def from_yaml(cls, path: Optional[Path | str] = None) -> "GuildRegistry":
        """Load the registry from a YAML file (defaults to bundled config)."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Guild config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        logger.info("Loading guild registry from %s", path)
        return cls(config)

    # ------------------------------------------------------------------ access
    @property
    def specialists(self) -> List[Specialist]:
        """All registered specialists."""
        return list(self._specialists.values())

    def get(self, specialist_id: str) -> Specialist:
        """Return a specialist by id, or raise ``KeyError``."""
        return self._specialists[specialist_id]

    @property
    def fallback(self) -> Specialist:
        """The generalist fallback specialist."""
        return self._specialists[self.fallback_id]

    # --------------------------------------------------------------- selection
    def select_specialist(self, classification: Classification) -> Specialist:
        """Resolve a classification to the best-matching specialist.

        Scoring prefers specialists whose declared domain, language and task all
        match. Domain is mandatory; ties break by most-specific (language + task)
        match. Falls back to the generalist when nothing scores.
        """
        best: Optional[Specialist] = None
        best_score = 0

        for spec in self._specialists.values():
            if spec.id == self.fallback_id:
                continue
            if spec.domain != classification.domain:
                continue

            score = 1  # domain match
            lang_ok = (
                not spec.languages
                or classification.language in spec.languages
                or classification.language == "unknown"
            )
            if not lang_ok:
                continue
            if classification.language in spec.languages:
                score += 2
            if classification.task in spec.tasks:
                score += 2

            if score > best_score:
                best_score = score
                best = spec

        if best is None:
            logger.info("No specialist matched %s; using fallback", classification.model_dump())
            return self.fallback

        logger.info("Selected specialist %s (score=%d)", best.id, best_score)
        return best

    def pipeline_for(self, classification: Classification) -> Optional[List[PipelineStep]]:
        """Return the multi-step pipeline for a classification, if any."""
        key = f"{classification.domain}:{classification.task}"
        return self._pipelines.get(key)
