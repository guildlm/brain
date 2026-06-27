"""Command-line interface for the GuildLM brain.

Entrypoint ``brain`` exposing:

* ``brain classify "..."`` — print the routing classification as JSON.
* ``brain chat "..."``     — route + execute, print the specialist and answer.
* ``brain guilds``         — list registered guild specialists.
* ``brain serve``          — run the FastAPI app with uvicorn.

By default these build a production orchestrator (OpenAI-compatible backend,
default Ollama). Use ``--offline`` on ``classify`` to force the heuristic
router with no backend calls.
"""

from __future__ import annotations

import json
import logging

import typer

from src.core.orchestrator import BrainOrchestrator
from src.core.registry import GuildRegistry
from src.core.router import IntentRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = typer.Typer(help="GuildLM brain: classify and route requests to specialist guilds.")


@app.command()
def classify(
    prompt: str = typer.Argument(..., help="The request to classify."),
    offline: bool = typer.Option(False, "--offline", help="Use the heuristic router only."),
) -> None:
    """Classify a prompt and print the structured result."""
    if offline:
        router = IntentRouter(offline=True)
        result = router.classify(prompt)
    else:
        result = BrainOrchestrator().classify(prompt)
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def chat(prompt: str = typer.Argument(..., help="The request to route and execute.")) -> None:
    """Route and execute a prompt against the selected specialist."""
    result = BrainOrchestrator().execute_request(prompt)
    typer.echo(f"Specialist: {result.specialist}")
    for step in result.steps:
        typer.echo(f"  - {step.action} via {step.specialist}")
    typer.echo("\nAnswer:\n" + result.answer)


@app.command()
def guilds() -> None:
    """List registered guild specialists."""
    registry = GuildRegistry.from_yaml()
    for spec in registry.specialists:
        langs = ",".join(spec.languages) or "-"
        tasks = ",".join(spec.tasks) or "-"
        typer.echo(f"{spec.id:28s} guild={spec.guild:8s} langs=[{langs}] tasks=[{tasks}]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
