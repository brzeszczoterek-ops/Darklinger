from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Callable

from ..llm import LLM
from .discovery import discover_models, human_size
from .chat_templates import CHAT_TEMPLATE_PROFILES, infer_chat_template
from .models import LoaderState, LocalModel, ModelProfile
from .qualification import ModelQualifier
from .runtime import LlamaServerSession, find_llama_server, start_llama_server
from .storage import ModelLoaderStore


Input = Callable[[str], str]
Output = Callable[[str], None]


class ModelLoaderInteractionError(RuntimeError):
    pass


async def bootstrap_interactive_model(
    runtime_root: Path,
    *,
    mode: str = "prompt",
    input_fn: Input = input,
    output: Output = print,
    stdin_is_tty: bool | None = None,
    allow_manual_hierarchy: bool = False,
    llm_factory: Callable[[], Any] = LLM,
    qualifier_factory: Callable[[Any], Any] = ModelQualifier,
) -> LlamaServerSession | None:
    selected_mode = mode.strip().casefold()
    if selected_mode not in {"off", "prompt", "required"}:
        raise ValueError("model loader mode must be off, prompt, or required")
    if selected_mode == "off":
        return None
    interactive = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty
    if not interactive:
        if selected_mode == "required":
            raise ModelLoaderInteractionError(
                "local model loader requires an interactive terminal"
            )
        return None

    store = ModelLoaderStore(runtime_root)
    state = store.load()
    models = discover_models(state.model_directories)
    while not models:
        output("No local GGUF models were found in the configured directories.")
        directory = _read(
            input_fn,
            "Directory containing GGUF models (Enter = use external server): ",
        ).strip()
        if not directory:
            if selected_mode == "required":
                output("A local model is required for this startup mode.")
                continue
            return None
        resolved = Path(directory).expanduser().resolve()
        if not resolved.is_dir():
            output(f"Directory does not exist: {resolved}")
            continue
        rendered = str(resolved)
        if rendered not in state.model_directories:
            state.model_directories.append(rendered)
        store.save(state)
        models = discover_models(state.model_directories)

    while True:
        action = choose_startup_action(
            state,
            models,
            input_fn=input_fn,
            output=output,
            allow_external=(selected_mode != "required"),
            allow_manual_hierarchy=allow_manual_hierarchy,
        )
        if action == "external":
            return None
        if action == "qualify":
            try:
                await qualify_model_interactively(
                    runtime_root,
                    state,
                    models,
                    store=store,
                    input_fn=input_fn,
                    output=output,
                    llm_factory=llm_factory,
                    qualifier_factory=qualifier_factory,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as error:
                output(
                    "Model qualification failed: "
                    f"{type(error).__name__}: {error}"
                )
            state = store.load()
            models = discover_models(state.model_directories)
            continue
        if action == "pool":
            configure_routing_pool_interactively(
                state,
                models,
                store=store,
                input_fn=input_fn,
                output=output,
            )
            state = store.load()
            continue
        if action == "hierarchy":
            configure_manual_hierarchy_interactively(
                state,
                models,
                store=store,
                input_fn=input_fn,
                output=output,
                allow_manual_hierarchy=allow_manual_hierarchy,
            )
            state = store.load()
            continue
        break

    model = choose_model(
        models,
        last_model_path=state.last_model_path,
        input_fn=input_fn,
        output=output,
        allow_external=(selected_mode != "required"),
    )
    if model is None:
        return None

    key = str(model.path)
    saved_profile = state.profiles.get(key)
    profile = (
        saved_profile
        if saved_profile is not None
        and Path(saved_profile.model_path).resolve() == model.path
        else default_profile(model)
    )
    output(render_profile(profile))
    edit = _read(input_fn, "Edit these parameters? [y/N]: ").strip().casefold()
    if edit in {"y", "yes", "t", "tak"}:
        profile = edit_profile(profile, input_fn=input_fn, output=output)

    binary = find_llama_server(state.server_binary)
    while binary is None:
        supplied = _read(
            input_fn,
            "Path to the llama-server executable (Enter = cancel): ",
        ).strip()
        if not supplied:
            if selected_mode == "required":
                output("llama-server is required for this startup mode.")
                continue
            return None
        binary = find_llama_server(supplied)
        if binary is None:
            output("That file is missing or is not executable.")

    state.server_binary = str(binary)
    state.profiles[key] = profile
    store.save(state)
    session = await start_llama_server(binary, profile, runtime_root, status=output)
    try:
        state.last_model_path = key
        store.save(state)
    except BaseException:
        await session.stop()
        raise
    return session


def choose_startup_action(
    state: LoaderState,
    models: list[LocalModel],
    *,
    input_fn: Input,
    output: Output,
    allow_external: bool,
    allow_manual_hierarchy: bool = False,
) -> str:
    """Choose startup work before any multi-gigabyte model is loaded."""

    current_cards = sum(
        1
        for model in models
        if str(model.path) in state.profiles
        and str(model.path) in state.qualifications
        and state.qualifications[str(model.path)].is_current(
            model.path,
            state.profiles[str(model.path)],
        )
    )
    current_pool_cards = 0
    stale_pool: list[str] = []
    for path in state.routing_model_paths:
        profile = state.profiles.get(path)
        card = state.qualifications.get(path)
        if profile is None:
            stale_pool.append(f"{Path(path).name}: profile missing")
            continue
        if card is None:
            stale_pool.append(f"{Path(path).name}: qualification missing")
            continue
        reasons = card.stale_reasons(Path(path), profile)
        if reasons:
            stale_pool.append(f"{Path(path).name}: {', '.join(reasons)}")
            continue
        current_pool_cards += 1

    output("\nDARKLINGER startup:")
    output("  1. Start V")
    output(
        "  2. Qualify or requalify a local model "
        f"({current_cards}/{len(models)} current)"
    )
    output(
        "  3. Configure automatic model routing pool "
        f"({current_pool_cards}/{len(state.routing_model_paths)} current; "
        f"{len(state.routing_model_paths)}/3 selected)"
    )
    if allow_manual_hierarchy:
        mode = (
            "active"
            if state.routing_strategy == "manual_hierarchy"
            else "automatic scoring active"
        )
        output(f"  4. Configure Full manual model hierarchy ({mode})")
    if state.routing_enabled and state.routing_model_paths and stale_pool:
        output(
            "  WARNING: automatic model routing is not fully available. "
            f"{current_pool_cards}/{len(state.routing_model_paths)} selected "
            "models have current qualification cards."
        )
        for item in stale_pool:
            output(f"    - {item}")
        output(
            "  Requalify stale models with option 2 before relying on "
            "automatic switching."
        )
    if allow_external:
        output("  0. Use the server configured in .env")
    while True:
        raw = _read(input_fn, "Select startup action [Enter = 1]: ").strip()
        if not raw or raw == "1":
            return "start"
        if raw == "2":
            return "qualify"
        if raw == "3":
            return "pool"
        if raw == "4" and allow_manual_hierarchy:
            return "hierarchy"
        if raw == "0" and allow_external:
            return "external"
        output("Enter a startup action number from the list.")


async def qualify_model_interactively(
    runtime_root: Path,
    state: LoaderState,
    models: list[LocalModel],
    *,
    store: ModelLoaderStore,
    input_fn: Input,
    output: Output,
    llm_factory: Callable[[], Any] = LLM,
    qualifier_factory: Callable[[Any], Any] = ModelQualifier,
) -> bool:
    """Qualify one exact GGUF/profile and return to the startup menu."""

    model = choose_model(
        models,
        last_model_path=state.last_model_path,
        input_fn=input_fn,
        output=output,
        allow_external=False,
    )
    assert model is not None
    key = str(model.path)
    saved_profile = state.profiles.get(key)
    profile = (
        saved_profile
        if saved_profile is not None
        and Path(saved_profile.model_path).resolve() == model.path
        else default_profile(model)
    )
    output(render_profile(profile))
    edit = _read(
        input_fn,
        "Edit this model profile before qualification? [y/N]: ",
    ).strip().casefold()
    if edit in {"y", "yes", "t", "tak"}:
        profile = edit_profile(profile, input_fn=input_fn, output=output)

    binary = find_llama_server(state.server_binary)
    while binary is None:
        supplied = _read(
            input_fn,
            "Path to llama-server for qualification (Enter = cancel): ",
        ).strip()
        if not supplied:
            output("Qualification cancelled; no card was changed.")
            return False
        binary = find_llama_server(supplied)
        if binary is None:
            output("That file is missing or is not executable.")

    state.server_binary = str(binary)
    state.profiles[key] = profile
    store.save(state)
    output(
        f"\nQualifying {model.path.name}. This runs bounded local tests and may "
        "take several minutes..."
    )
    session = await start_llama_server(binary, profile, runtime_root, status=output)
    llm: Any | None = None
    try:
        llm = llm_factory()
        card = await qualifier_factory(llm).qualify(profile)
    finally:
        if llm is not None:
            client = getattr(llm, "client", None)
            close = getattr(client, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        await session.stop()

    state = store.load()
    state.server_binary = str(binary)
    state.profiles[key] = profile
    state.qualifications[key] = card
    output(f"\nQualification complete: {model.path.name} = {card.overall_score}/100")
    for capability, score in card.capabilities.items():
        output(f"  {capability}: {score}/100")

    if key in state.routing_model_paths:
        output("This model is already in the automatic routing pool.")
    elif len(state.routing_model_paths) >= 3:
        output(
            "The routing pool is full. The card was saved; use startup option 3 "
            "to replace a pool member."
        )
    else:
        add = _read(
            input_fn,
            "Add this qualified model to the automatic routing pool? [Y/n]: ",
        ).strip().casefold()
        if add not in {"n", "no", "nie"}:
            state.routing_model_paths.append(key)
            state.routing_enabled = True
            output("Model added to the automatic routing pool.")
    store.save(state)
    return True


def configure_routing_pool_interactively(
    state: LoaderState,
    models: list[LocalModel],
    *,
    store: ModelLoaderStore,
    input_fn: Input,
    output: Output,
) -> bool:
    """Configure one to three current cards without a separate CLI command."""

    eligible: list[tuple[LocalModel, int]] = []
    for model in models:
        key = str(model.path)
        profile = state.profiles.get(key)
        card = state.qualifications.get(key)
        if profile is None or card is None or not card.is_current(model.path, profile):
            continue
        eligible.append((model, card.overall_score))
    if not eligible:
        output("No current qualification cards exist. Run startup option 2 first.")
        return False

    output("\nQualified models available for automatic routing:")
    for index, (model, score) in enumerate(eligible, start=1):
        marker = " (current pool)" if str(model.path) in state.routing_model_paths else ""
        output(f"  {index}. {model.path.name} — {score}/100{marker}")
    output("  0. Disable automatic routing")
    while True:
        raw = _read(
            input_fn,
            "Select 1-3 model numbers separated by commas [Enter = keep current]: ",
        ).strip()
        if not raw:
            return False
        if raw == "0":
            state.routing_enabled = False
            state.routing_strategy = "automatic"
            store.save(state)
            output("Automatic model routing disabled; saved cards were kept.")
            return True
        try:
            indices = [int(item.strip()) - 1 for item in raw.split(",")]
        except ValueError:
            output("Enter one to three comma-separated model numbers.")
            continue
        if (
            not 1 <= len(indices) <= 3
            or len(set(indices)) != len(indices)
            or any(index < 0 or index >= len(eligible) for index in indices)
        ):
            output("Choose one to three distinct model numbers from the list.")
            continue
        state.routing_model_paths = [str(eligible[index][0].path) for index in indices]
        state.routing_enabled = True
        state.routing_strategy = "automatic"
        store.save(state)
        output("Automatic model routing pool updated.")
        return True


def configure_manual_hierarchy_interactively(
    state: LoaderState,
    models: list[LocalModel],
    *,
    store: ModelLoaderStore,
    input_fn: Input,
    output: Output,
    allow_manual_hierarchy: bool,
) -> bool:
    """Set a hard owner-defined fallback order for DARKLINGER-Full."""

    if not allow_manual_hierarchy:
        output("Manual model hierarchy requires DARKLINGER-Full.")
        return False
    by_path = {str(model.path): model for model in models}
    ordered: list[LocalModel] = []
    for path in state.routing_model_paths:
        model = by_path.get(path)
        profile = state.profiles.get(path)
        card = state.qualifications.get(path)
        if (
            model is None
            or profile is None
            or card is None
            or not card.is_current(model.path, profile)
        ):
            output(
                "Every model in the routing pool must have a current "
                "qualification before a manual hierarchy can be enabled."
            )
            return False
        ordered.append(model)
    if not ordered:
        output("Configure the qualified routing pool with option 3 first.")
        return False

    output("\nDARKLINGER-Full manual model hierarchy:")
    for index, model in enumerate(ordered, start=1):
        output(f"  {index}. {model.path.name}")
    output("  0. Return to automatic capability scoring")
    while True:
        raw = _read(
            input_fn,
            "Enter every model number in priority order [Enter = keep current]: ",
        ).strip()
        if not raw:
            return False
        if raw == "0":
            state.routing_strategy = "automatic"
            store.save(state)
            output("Automatic capability scoring restored.")
            return True
        try:
            indices = [int(item.strip()) - 1 for item in raw.split(",")]
        except ValueError:
            output("Enter every model number once, separated by commas.")
            continue
        if (
            len(indices) != len(ordered)
            or set(indices) != set(range(len(ordered)))
        ):
            output("The hierarchy must contain every pool model exactly once.")
            continue
        state.routing_model_paths = [str(ordered[index].path) for index in indices]
        state.routing_enabled = True
        state.routing_strategy = "manual_hierarchy"
        store.save(state)
        output("DARKLINGER-Full manual model hierarchy enabled.")
        return True


def choose_model(
    models: list[LocalModel],
    *,
    last_model_path: str,
    input_fn: Input,
    output: Output,
    allow_external: bool,
) -> LocalModel | None:
    if not models:
        raise ValueError("cannot choose from an empty model list")
    output("\nLocal GGUF models:")
    default_index = 0
    for index, model in enumerate(models, start=1):
        marker = " (last used)" if str(model.path) == last_model_path else ""
        if marker:
            default_index = index - 1
        output(f"  {index}. {model.path.name} — {human_size(model.size_bytes)}{marker}")
    if allow_external:
        output("  0. Use the server configured in .env")
    while True:
        raw = _read(
            input_fn,
            f"Select model [Enter = {default_index + 1}]: ",
        ).strip()
        if not raw:
            return models[default_index]
        if raw == "0" and allow_external:
            return None
        try:
            index = int(raw) - 1
        except ValueError:
            output("Enter a model number from the list.")
            continue
        if 0 <= index < len(models):
            return models[index]
        output("Selected model number is outside the list.")


def default_profile(model: LocalModel) -> ModelProfile:
    alias = re.sub(r"[^A-Za-z0-9._-]+", "-", model.name).strip("-._")
    alias = (alias or "local-model")[:128]
    return ModelProfile(
        model_path=str(model.path),
        alias=alias,
        chat_template=infer_chat_template(str(model.path), alias),
        context_size=_environment_int("V_CORE_CONTEXT", 32_768),
        temperature=_environment_float("V_CORE_TEMPERATURE", 0.2),
        top_p=_environment_float("V_CORE_TOP_P", 0.95),
        port=_base_url_port(os.getenv("V_CORE_BASE_URL", ""), 5_001),
    )


def edit_profile(
    profile: ModelProfile,
    *,
    input_fn: Input,
    output: Output,
) -> ModelProfile:
    values = profile.to_dict()
    values["alias"] = _prompt_text(input_fn, "API alias", profile.alias)
    values["context_size"] = _prompt_int(
        input_fn, "Context size", profile.context_size, 256, 10_000_000
    )
    values["gpu_layers"] = _prompt_choice_or_integer(
        input_fn, "GPU layers", profile.gpu_layers, {"auto", "all"}, 0, 9_999
    )
    values["threads"] = _prompt_int(
        input_fn, "CPU threads (0 = llama.cpp default)", profile.threads, 0, 4_096
    )
    values["batch_size"] = _prompt_int(
        input_fn, "Batch size", profile.batch_size, 1, 1_048_576
    )
    values["ubatch_size"] = _prompt_int(
        input_fn, "Micro-batch size", profile.ubatch_size, 1, values["batch_size"]
    )
    values["parallel"] = _prompt_int(
        input_fn, "Parallel slots", profile.parallel, 1, 1_024
    )
    values["flash_attention"] = _prompt_choice(
        input_fn,
        "Flash attention",
        profile.flash_attention,
        {"auto", "on", "off"},
    )
    values["reasoning"] = _prompt_choice(
        input_fn,
        "Reasoning mode",
        profile.reasoning,
        {"auto", "on", "off"},
    )
    values["chat_template"] = _prompt_choice(
        input_fn,
        "Chat template",
        profile.chat_template,
        CHAT_TEMPLATE_PROFILES,
    )
    values["anti_repetition"] = _prompt_choice(
        input_fn,
        "Anti-repetition",
        profile.anti_repetition,
        {"off", "balanced", "strong"},
    )
    kv_cache_types = {
        "f32",
        "f16",
        "bf16",
        "q8_0",
        "q5_0",
        "q5_1",
        "q4_0",
        "q4_1",
        "iq4_nl",
    }
    values["cache_type_k"] = _prompt_choice(
        input_fn,
        "KV cache K type",
        profile.cache_type_k,
        kv_cache_types,
    )
    values["cache_type_v"] = _prompt_choice(
        input_fn,
        "KV cache V type",
        profile.cache_type_v,
        kv_cache_types,
    )
    values["temperature"] = _prompt_float(
        input_fn, "Temperature", profile.temperature, 0.0, 5.0
    )
    values["top_p"] = _prompt_float(
        input_fn, "Top-p", profile.top_p, 0.000001, 1.0
    )
    values["port"] = _prompt_int(input_fn, "Local API port", profile.port, 1_024, 65_535)
    values["startup_timeout_seconds"] = _prompt_float(
        input_fn,
        "Startup timeout in seconds",
        profile.startup_timeout_seconds,
        1.0,
        3_600.0,
    )
    rendered_extras = shlex.join(profile.extra_args)
    extra_text = _prompt_text(
        input_fn,
        "Additional llama.cpp arguments",
        rendered_extras,
    )
    try:
        values["extra_args"] = tuple(shlex.split(extra_text)) if extra_text else ()
        return ModelProfile.from_dict(values)
    except (ValueError, TypeError) as error:
        output(f"Invalid profile: {error}")
        output("Keeping the previous profile.")
        return profile


def render_profile(profile: ModelProfile) -> str:
    return (
        "\nSelected profile:\n"
        f"  alias={profile.alias}\n"
        f"  context={profile.context_size}, gpu_layers={profile.gpu_layers}, "
        f"threads={profile.threads or 'auto'}\n"
        f"  batch={profile.batch_size}, ubatch={profile.ubatch_size}, "
        f"parallel={profile.parallel}, flash_attn={profile.flash_attention}\n"
        f"  reasoning={profile.reasoning}, "
        f"chat_template={profile.chat_template}, "
        f"anti_repetition={profile.anti_repetition}\n"
        f"  kv_cache_k={profile.cache_type_k}, "
        f"kv_cache_v={profile.cache_type_v}\n"
        f"  temperature={profile.temperature:g}, top_p={profile.top_p:g}, "
        f"port={profile.port}\n"
        f"  extra_args={shlex.join(profile.extra_args) or '(none)'}"
    )


def _read(input_fn: Input, prompt: str) -> str:
    try:
        return input_fn(prompt)
    except EOFError:
        return ""


def _prompt_text(input_fn: Input, label: str, default: str) -> str:
    value = _read(input_fn, f"{label} [{default or 'none'}]: ").strip()
    return value or default


def _prompt_int(
    input_fn: Input,
    label: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    while True:
        raw = _read(input_fn, f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            continue
        if minimum <= value <= maximum:
            return value


def _prompt_float(
    input_fn: Input,
    label: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    while True:
        raw = _read(input_fn, f"{label} [{default:g}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            continue
        if minimum <= value <= maximum:
            return value


def _prompt_choice(
    input_fn: Input,
    label: str,
    default: str,
    choices: set[str],
) -> str:
    while True:
        value = _read(input_fn, f"{label} [{default}]: ").strip().casefold()
        if not value:
            return default
        if value in choices:
            return value


def _prompt_choice_or_integer(
    input_fn: Input,
    label: str,
    default: str,
    choices: set[str],
    minimum: int,
    maximum: int,
) -> str:
    while True:
        value = _read(input_fn, f"{label} [{default}]: ").strip().casefold()
        if not value:
            return default
        if value in choices:
            return value
        try:
            parsed = int(value)
        except ValueError:
            continue
        if minimum <= parsed <= maximum:
            return str(parsed)


def _environment_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _environment_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _base_url_port(value: str, default: int) -> int:
    try:
        from urllib.parse import urlparse

        return urlparse(value).port or default
    except ValueError:
        return default
