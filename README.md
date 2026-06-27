# GuildLM Brain

The **brain** is the central coordinator of [GuildLM](https://github.com/guildlm/guildlm.github.io).
It does one job and does it well: **classify a user's intent and route the
request to the right guild specialist** (a small, focused SLM). The brain never
answers domain questions itself — it only routes and orchestrates.

```
                       ┌─────────────────────────────────────────────┐
   user prompt  ─────▶ │                  BRAIN                       │
                       │                                             │
                       │  1. IntentRouter.classify()                 │
                       │     ├─ LLM classifier (structured JSON) ─┐  │
                       │     └─ heuristic fallback ◀──────────────┘  │   (low confidence /
                       │            │                                │    offline / garbled)
                       │            ▼                                │
                       │  2. GuildRegistry.select_specialist()       │
                       │     └─ pipeline_for()  (multi-step?)        │
                       │            │                                │
                       │            ▼                                │
                       │  3. BrainOrchestrator runs step(s) via      │
                       │     the injected Backend (OpenAI-compatible)│
                       └────────────┬────────────────────────────────┘
                                    ▼
              ┌──────────────────────────────────────────────┐
              │  Code Guild: go_generator / go_reviewer /     │
              │  go_tester / go_explainer   ·   brain/generalist
              └──────────────────────────────────────────────┘
```

## Why a router?

A single large model is expensive and mediocre at everything. GuildLM instead
runs many **specialist SLMs** (base model + LoRA adapter + focused system
prompt). The brain is the thin, fast layer that decides *who* should handle each
request and, when needed, chains specialists into a pipeline.

## Routing flow

1. **Classify** (`src/core/router.py`). The `IntentRouter` asks an
   OpenAI-compatible model for structured JSON
   `{domain, language, task, subtask, confidence}`. The parser is defensive
   (handles code fences, surrounding prose, bad types). If the LLM is
   unavailable, returns garbage, reports **low confidence**, or `offline=True`,
   it falls back to a deterministic **heuristic** classifier.
2. **Select** (`src/core/registry.py`). The `GuildRegistry` (loaded from
   `configs/guilds.yaml`) scores specialists by domain/language/task and picks
   the best match, or the `brain/generalist` fallback. It also resolves
   multi-step **pipelines** keyed by `<domain>:<task>`.
3. **Execute** (`src/core/orchestrator.py`). The `BrainOrchestrator` runs the
   single specialist or threads the pipeline steps, calling the injected
   `Backend` for inference.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI

The package installs a `brain` console script (typer):

```bash
brain classify "Build an HTTP API in Go"            # route decision (uses backend)
brain classify "Review my Go code" --offline        # heuristic only, no network
brain chat "Fix the deadlock bug in my Go goroutine" # route + execute (pipeline)
brain guilds                                         # list registered specialists
brain serve --host 0.0.0.0 --port 8000               # run the API
```

## HTTP API

`src/api/server.py` exposes a FastAPI app (run via `brain serve` or
`uvicorn src.api.server:app`).

| Method & path | Description | Body / response |
| --- | --- | --- |
| `POST /classify` | Classify a prompt (no execution) | `{"prompt": "..."}` → `Classification` |
| `POST /chat` | Route + execute | `{"prompt": "..."}` → `{specialist, answer, classification, steps}` |
| `GET /guilds` | Dump the registry | `{base_model, fallback, specialists[]}` |
| `GET /health` | Liveness probe | `{status, specialists}` |

`Classification` shape:

```json
{"domain": "code", "language": "go", "task": "generation",
 "subtask": "http_server", "confidence": 0.92, "source": "llm"}
```

## Guild registry schema (`configs/guilds.yaml`)

```yaml
defaults:
  base_model: qwen2.5:7b-instruct       # default backend model
  fallback_specialist: brain/generalist  # used when nothing matches

specialists:
  - id: guild-code/go_generator   # stable unique id
    guild: code                   # owning guild
    domain: code                  # domain it serves
    languages: [go]               # languages handled
    tasks: [generation, bug_fix]  # tasks handled
    model: qwen2.5:7b-instruct    # base model served by the backend
    lora: guildlm/go-generator-lora  # optional LoRA adapter (or null)
    system_prompt: >-             # injected at inference time
      You are an expert Go programmer ...

pipelines:                        # multi-step orchestration, keyed <domain>:<task>
  code:bug_fix:
    - { specialist: guild-code/go_reviewer,  action: analyze, prompt_template: "...{input}" }
    - { specialist: guild-code/go_generator, action: fix,     prompt_template: "...{previous}" }
    - { specialist: guild-code/go_reviewer,  action: verify,  prompt_template: "...{previous}" }
```

`prompt_template` supports `{input}` (the original user prompt) and `{previous}`
(the prior step's output).

## Running with Ollama (default)

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
brain serve            # talks to http://localhost:11434/v1
```

## Running with vLLM (or any OpenAI-compatible endpoint)

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8001
export BRAIN_BASE_URL=http://localhost:8001/v1
export BRAIN_API_KEY=dummy
export BRAIN_ROUTER_MODEL=Qwen/Qwen2.5-7B-Instruct
brain serve
```

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRAIN_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `BRAIN_API_KEY` | `ollama` | API key (ignored by most local servers) |
| `BRAIN_ROUTER_MODEL` | `qwen2.5:7b-instruct` | Model used for classification |

## Architecture & extensibility

- `src/core/backends.py` — `Backend` protocol with `OpenAIBackend` (production)
  and `FakeBackend` (tests). `ModelManager` is a VRAM-aware bookkeeping stub
  that tracks/hot-swaps LoRA adapters (logging today; real load/unload later).
- The backend is **injectable** everywhere, so the whole stack — router,
  registry, orchestrator, API — is unit-tested with **no network**.

### Extending the guilds

Add a specialist (and optionally a pipeline) to `configs/guilds.yaml`, extend
the heuristic classifier in `src/core/router.py` if the new domain needs offline
routing, and add tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Testing

```bash
pytest -q     # 38 tests, fully offline
```

## License

[Apache-2.0](LICENSE).
