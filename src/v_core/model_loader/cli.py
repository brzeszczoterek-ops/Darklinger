from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from ..config import load_config
from ..llm import LLM
from .discovery import discover_models, human_size
from .interactive import default_profile
from .models import LoaderState, LocalModel
from .qualification import ModelQualifier
from .router import ModelRouteCandidate, ModelRouter
from .runtime import find_llama_server, start_llama_server
from .storage import ModelLoaderStore


app = typer.Typer(
    add_completion=False,
    help="Qualify local GGUF models and configure DARKLINGER's verified router.",
)


def _context() -> tuple[Path, ModelLoaderStore, LoaderState, list[LocalModel]]:
    root = load_config().model_runtime_root
    store = ModelLoaderStore(root)
    state = store.load()
    return root, store, state, discover_models(state.model_directories)


def _resolve_model(value: str, models: list[LocalModel], state: LoaderState) -> LocalModel:
    requested = value.strip()
    if not requested:
        requested = state.last_model_path
    exact: list[LocalModel] = []
    for model in models:
        if requested in {str(model.path), model.path.name, model.name}:
            exact.append(model)
    if len(exact) == 1:
        return exact[0]
    if not requested and len(models) == 1:
        return models[0]
    if len(exact) > 1:
        raise typer.BadParameter("model name is ambiguous; provide its full path")
    raise typer.BadParameter("model was not found in configured model directories")


@app.command("list")
def list_models() -> None:
    """Show discovered models, card freshness, and routing-pool membership."""

    _, _, state, models = _context()
    if not models:
        typer.echo("No local GGUF models were found.")
        raise typer.Exit(1)
    for model in models:
        path = str(model.path)
        profile = state.profiles.get(path)
        card = state.qualifications.get(path)
        freshness = "unqualified"
        score = "-"
        if profile is not None and card is not None:
            if card.is_current(model.path, profile):
                freshness = "qualified"
                score = str(card.overall_score)
            else:
                freshness = "stale"
        pool = "pool" if path in state.routing_model_paths else "-"
        active = "last" if path == state.last_model_path else "-"
        typer.echo(
            f"{model.path.name} | {human_size(model.size_bytes)} | {freshness} "
            f"{score}/100 | {pool} | {active}"
        )
    typer.echo(f"routing={'on' if state.routing_enabled else 'off'}")
    strategy = (
        state.routing_strategy
        if load_config().edition.is_full
        else "automatic"
    )
    typer.echo(f"routing_strategy={strategy}")


@app.command("qualify")
def qualify_model(
    model: str = typer.Argument("", help="Full path, filename, or unique model stem."),
    add_to_pool: bool = typer.Option(
        True,
        "--add-to-pool/--no-add-to-pool",
        help="Add the qualified model to the routing pool when a slot is available.",
    ),
) -> None:
    """Load one GGUF, run all bounded probes, and save its verified card."""

    asyncio.run(_qualify_model(model, add_to_pool=add_to_pool))


async def _qualify_model(model: str, *, add_to_pool: bool) -> None:
    root, store, state, models = _context()
    selected = _resolve_model(model, models, state)
    key = str(selected.path)
    profile = state.profiles.get(key) or default_profile(selected)
    binary = find_llama_server(state.server_binary)
    if binary is None:
        raise typer.BadParameter("llama-server was not found; configure it in DARKLINGER first")

    typer.echo(f"Loading {selected.path.name} for qualification...")
    session = await start_llama_server(binary, profile, root, status=typer.echo)
    try:
        llm = LLM()
        card = await ModelQualifier(llm).qualify(profile)
        await llm.client.close()
    finally:
        await session.stop()

    state.server_binary = str(binary)
    state.profiles[key] = profile
    state.qualifications[key] = card
    if add_to_pool and key not in state.routing_model_paths:
        if len(state.routing_model_paths) >= 3:
            typer.echo("Routing pool is already full; card was saved without adding it.")
        else:
            state.routing_model_paths.append(key)
    if state.routing_model_paths:
        state.routing_enabled = True
    store.save(state)

    typer.echo(f"Qualified {selected.path.name}: overall {card.overall_score}/100")
    for capability, score in card.capabilities.items():
        typer.echo(f"  {capability}: {score}/100")


@app.command("pool")
def configure_pool(
    models: list[str] = typer.Argument(
        ..., help="One to three full paths, filenames, or unique model stems."
    ),
) -> None:
    """Set the exact one-to-three-model automatic routing pool."""

    if not 1 <= len(models) <= 3:
        raise typer.BadParameter("routing pool must contain between one and three models")
    _, store, state, discovered = _context()
    resolved = [_resolve_model(value, discovered, state) for value in models]
    paths = list(dict.fromkeys(str(model.path) for model in resolved))
    if len(paths) != len(models):
        raise typer.BadParameter("routing pool cannot contain duplicate models")
    missing_cards = [
        Path(path).name
        for path in paths
        if path not in state.qualifications
        or path not in state.profiles
        or not state.qualifications[path].is_current(Path(path), state.profiles[path])
    ]
    if missing_cards:
        raise typer.BadParameter(
            "qualify these models with their current profiles first: "
            + ", ".join(missing_cards)
        )
    state.routing_model_paths = paths
    state.routing_enabled = True
    state.routing_strategy = "automatic"
    store.save(state)
    typer.echo("Automatic local routing enabled for:")
    for path in paths:
        typer.echo(f"  {Path(path).name}")


@app.command("routing")
def set_routing(enabled: str = typer.Argument(..., help="on or off")) -> None:
    """Enable or disable switching without deleting cards or the pool."""

    normalized = enabled.strip().casefold()
    if normalized not in {"on", "off"}:
        raise typer.BadParameter("routing must be 'on' or 'off'")
    _, store, state, _ = _context()
    if normalized == "on" and not state.routing_model_paths:
        raise typer.BadParameter("configure a qualified routing pool first")
    state.routing_enabled = normalized == "on"
    store.save(state)
    typer.echo(f"Automatic model routing is {normalized}.")


@app.command("route")
def inspect_route(prompt: str = typer.Argument(..., help="Task to classify.")) -> None:
    """Show the deterministic decision without starting or switching a model."""

    _, _, state, _ = _context()
    candidates = []
    stale: list[str] = []
    for path in state.routing_model_paths:
        card = state.qualifications.get(path)
        profile = state.profiles.get(path)
        if card is None or profile is None:
            stale.append(f"{Path(path).name}: qualification or profile missing")
            continue
        reasons = card.stale_reasons(Path(path), profile)
        if reasons:
            stale.append(f"{Path(path).name}: {', '.join(reasons)}")
            continue
        candidates.append(ModelRouteCandidate(path, card))
    decision = ModelRouter().choose(
        prompt,
        candidates,
        current_model_path=state.last_model_path,
        strategy=(
            state.routing_strategy
            if load_config().edition.is_full
            else "automatic"
        ),
    )
    if decision is None:
        typer.echo("No current qualified candidate is available.")
        for item in stale:
            typer.echo(f"  {item}")
        raise typer.Exit(1)
    typer.echo(
        f"{decision.task_kind}: {Path(decision.selected_model_path).name} "
        f"({decision.score}/100)"
    )
    if decision.fallback_model_paths:
        typer.echo(
            "fallbacks: "
            + ", ".join(Path(path).name for path in decision.fallback_model_paths)
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
