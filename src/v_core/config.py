from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from .branding import env_value
from .edition import Edition, resolve_edition


@dataclass(slots=True)
class Config:
    workspace: Path

    memory_root: Path

    learning_root: Path

    model_runtime_root: Path

    autonomy_root: Path

    voice_root: Path

    evm_profile: str

    learning_profile: str

    model_loader_mode: str

    filesystem_server: list[str]

    browser_server: list[str]

    edition: Edition

    project_read_root: Path | None = None


def load_config() -> Config:

    load_dotenv()

    project_root = Path.cwd()

    edition = resolve_edition(env_value("DARKLINGER_EDITION", "auto"))

    workspace_value = os.getenv("V_CORE_MCP_FILESYSTEM", "agent_workspace")
    workspace = Path(workspace_value).expanduser()
    if not workspace.is_absolute():
        workspace = project_root / workspace

    workspace.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_root_value = env_value("DARKLINGER_MEMORY_ROOT", "memory")
    memory_root = Path(memory_root_value).expanduser()
    if not memory_root.is_absolute():
        memory_root = project_root / memory_root
    memory_root.mkdir(parents=True, exist_ok=True)

    learning_root_value = env_value("DARKLINGER_LEARNING_ROOT", "learning")
    learning_root = Path(learning_root_value).expanduser()
    if not learning_root.is_absolute():
        learning_root = project_root / learning_root
    learning_root.mkdir(parents=True, exist_ok=True)

    model_runtime_root_value = env_value(
        "DARKLINGER_MODEL_RUNTIME_ROOT", "model_runtime"
    )
    model_runtime_root = Path(model_runtime_root_value).expanduser()
    if not model_runtime_root.is_absolute():
        model_runtime_root = project_root / model_runtime_root
    model_runtime_root.mkdir(parents=True, exist_ok=True)

    autonomy_root_value = env_value("DARKLINGER_AUTONOMY_ROOT", "autonomy")
    autonomy_root = Path(autonomy_root_value).expanduser()
    if not autonomy_root.is_absolute():
        autonomy_root = project_root / autonomy_root
    autonomy_root.mkdir(parents=True, exist_ok=True)

    voice_root_value = env_value("DARKLINGER_VOICE_ROOT", "voice")
    voice_root = Path(voice_root_value).expanduser()
    if not voice_root.is_absolute():
        voice_root = project_root / voice_root
    voice_root.mkdir(parents=True, exist_ok=True)

    evm_profile = env_value(
        "DARKLINGER_EVM_PROFILE", edition.default_evm_profile
    ).strip().lower()
    if evm_profile not in {"client", "owner_lab"}:
        raise ValueError("DARKLINGER_EVM_PROFILE must be 'client' or 'owner_lab'")

    learning_profile = env_value(
        "DARKLINGER_LEARNING_PROFILE", edition.default_learning_profile
    ).strip().lower()
    if learning_profile not in {"client", "owner_lab"}:
        raise ValueError(
            "DARKLINGER_LEARNING_PROFILE must be 'client' or 'owner_lab'"
        )
    if not edition.is_full and (
        evm_profile == "owner_lab" or learning_profile == "owner_lab"
    ):
        raise ValueError(
            "owner_lab profiles require DARKLINGER-Full; public DARKLINGER supports "
            "the client profiles only"
        )

    model_loader_mode = env_value(
        "DARKLINGER_MODEL_LOADER", "prompt"
    ).strip().lower()
    if model_loader_mode not in {"off", "prompt", "required"}:
        raise ValueError(
            "DARKLINGER_MODEL_LOADER must be 'off', 'prompt', or 'required'"
        )

    project_read_value = env_value("DARKLINGER_PROJECT_READ_ROOT", "").strip()
    project_read_root = (
        Path(project_read_value).expanduser().resolve()
        if project_read_value
        else None
    )
    filesystem_roots = [str(workspace)]
    if project_read_root is not None and project_read_root.is_dir():
        filesystem_roots.append(str(project_read_root))

    return Config(
        workspace=workspace,

        memory_root=memory_root,

        learning_root=learning_root,

        model_runtime_root=model_runtime_root,

        autonomy_root=autonomy_root,

        voice_root=voice_root,

        evm_profile=evm_profile,

        learning_profile=learning_profile,

        model_loader_mode=model_loader_mode,

        filesystem_server=[
            "npx",
            "@modelcontextprotocol/server-filesystem",
            *filesystem_roots,
        ],

        browser_server=[
            "npx",
            "@playwright/mcp",
            "--browser=firefox",
        ],
        edition=edition,
        project_read_root=project_read_root,
    )
