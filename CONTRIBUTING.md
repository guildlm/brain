# Contributing to GuildLM Brain

Thanks for your interest in improving the GuildLM brain — the central
router/orchestrator that classifies user intent and dispatches to specialist
guild SLMs. The brain never answers domain questions itself; it only routes and
orchestrates. Please keep contributions aligned with that responsibility.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The entire test-suite runs offline with a `FakeBackend` — no LLM server or
network access is required, and CI enforces this.

## Project layout

| Path | Responsibility |
| --- | --- |
| `src/core/router.py` | Intent classification (LLM + heuristic fallback) |
| `src/core/registry.py` | Guild/specialist registry loaded from YAML |
| `src/core/orchestrator.py` | Single- and multi-step orchestration |
| `src/core/backends.py` | Inference backend abstraction + `ModelManager` |
| `src/core/models.py` | Shared Pydantic models |
| `src/api/server.py` | FastAPI surface |
| `src/cli.py` | `brain` CLI (typer) |
| `configs/guilds.yaml` | Guild registry data |

## Guidelines

- **Type hints, docstrings, logging.** Public functions and classes are
  documented; no dead code.
- **Keep the brain a router.** Do not add domain-answering logic here — add a
  specialist to `configs/guilds.yaml` instead.
- **Tests must pass offline.** Use the injectable backend (`FakeBackend`) in
  tests; never hit a real endpoint in CI.
- **Add tests** for new routing rules, specialists, pipelines, or endpoints.

## Adding a guild specialist

1. Add an entry under `specialists:` in `configs/guilds.yaml` (id, guild,
   domain, languages, tasks, model, optional `lora`, system_prompt).
2. (Optional) Add a `pipelines:` entry keyed by `<domain>:<task>` for
   multi-step orchestration.
3. Extend `IntentRouter.heuristic_classify` if the new domain needs offline
   classification.
4. Add tests covering selection and (if applicable) the pipeline.

## Commit messages

Describe the change and its motivation. By contributing you agree your work is
licensed under the project's Apache-2.0 license.
