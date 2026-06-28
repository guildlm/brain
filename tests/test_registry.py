"""Tests for the guild registry: loading and specialist selection."""

from __future__ import annotations

import pytest

from src.core.models import Classification
from src.core.registry import GuildRegistry


def test_registry_loads_bundled_config(registry: GuildRegistry):
    ids = {s.id for s in registry.specialists}
    assert "guild-code/go_generator" in ids
    assert "guild-code/go_reviewer" in ids
    assert "guild-code/go_tester" in ids
    assert "guild-code/go_explainer" in ids
    assert registry.fallback_id == "brain/generalist"


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("generation", "guild-code/go_generator"),
        ("review", "guild-code/go_reviewer"),
        ("testing", "guild-code/go_tester"),
        ("explanation", "guild-code/go_explainer"),
    ],
)
def test_select_specialist_for_go_tasks(registry: GuildRegistry, task, expected):
    c = Classification(domain="code", language="go", task=task, confidence=0.9)
    assert registry.select_specialist(c).id == expected


def test_registry_loads_sql_guild(registry: GuildRegistry):
    ids = {s.id for s in registry.specialists}
    assert "guild-sql/sql_generator" in ids
    assert "guild-sql/sql_reviewer" in ids
    assert "guild-sql/sql_optimizer" in ids
    assert "guild-sql/sql_explainer" in ids


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("generation", "guild-sql/sql_generator"),
        ("review", "guild-sql/sql_reviewer"),
        ("optimization", "guild-sql/sql_optimizer"),
        ("explanation", "guild-sql/sql_explainer"),
    ],
)
def test_select_specialist_for_sql_tasks(registry: GuildRegistry, task, expected):
    c = Classification(domain="sql", language="sql", task=task, confidence=0.9)
    assert registry.select_specialist(c).id == expected


def test_sql_optimization_pipeline_lookup(registry: GuildRegistry):
    c = Classification(domain="sql", language="sql", task="optimization", confidence=0.9)
    pipeline = registry.pipeline_for(c)
    assert pipeline is not None
    assert [s.action for s in pipeline] == ["analyze", "rewrite", "verify"]


def test_select_specialist_falls_back_to_generalist(registry: GuildRegistry):
    c = Classification(domain="general", language="unknown", task="general_qa", confidence=0.5)
    assert registry.select_specialist(c).id == "brain/generalist"


def test_unknown_domain_falls_back(registry: GuildRegistry):
    c = Classification(domain="legal", language="english", task="review", confidence=0.9)
    assert registry.select_specialist(c).id == "brain/generalist"


def test_pipeline_lookup(registry: GuildRegistry):
    c = Classification(domain="code", language="go", task="bug_fix", confidence=0.9)
    pipeline = registry.pipeline_for(c)
    assert pipeline is not None
    assert [s.action for s in pipeline] == ["analyze", "fix", "verify"]
    # code:generation is also a guild pipeline now (generate -> test -> review).
    gen = registry.pipeline_for(
        Classification(domain="code", language="go", task="generation", confidence=0.9)
    )
    assert [s.action for s in gen] == ["generate", "test", "review"]
    # A single-specialist task (explanation) has no pipeline.
    assert registry.pipeline_for(
        Classification(domain="code", language="go", task="explanation", confidence=0.9)
    ) is None


def test_registry_rejects_missing_fallback():
    with pytest.raises(ValueError):
        GuildRegistry({"defaults": {"fallback_specialist": "nope"}, "specialists": []})


def test_get_specialist_by_id(registry: GuildRegistry):
    spec = registry.get("guild-code/go_generator")
    assert spec.lora == "guildlm/go-generator-lora"
    assert "go" in spec.languages
