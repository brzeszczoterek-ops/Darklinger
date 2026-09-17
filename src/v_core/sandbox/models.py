from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class NetworkMode(str, Enum):
    """Network policies understood by DARKLINGER sandboxes.

    Only ``OFFLINE`` is currently executable. The remaining values are explicit
    design contracts so a caller cannot accidentally turn networking on while a
    backend is still missing the necessary enforcement.
    """

    OFFLINE = "offline"
    LOCAL_TESTNET = "local_testnet"
    FETCH_THEN_OFFLINE = "fetch_then_offline"
    ALLOWLIST_PROXY = "allowlist_proxy"


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    timeout_seconds: float = 120.0
    cpu_seconds: int = 60
    memory_mb: int = 1_024
    max_output_bytes: int = 2_000_000
    max_file_bytes: int = 256 * 1024 * 1024
    max_workspace_bytes: int = 512 * 1024 * 1024
    max_open_files: int = 256
    max_processes: int = 64

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.cpu_seconds <= 0:
            raise ValueError("cpu_seconds must be positive")
        if self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if self.max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive")
        if self.max_open_files <= 2:
            raise ValueError("max_open_files must be greater than 2")
        if self.max_processes <= 0:
            raise ValueError("max_processes must be positive")


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    command: tuple[str, ...]
    workspace: Path
    network: NetworkMode = NetworkMode.OFFLINE
    read_only_inputs: tuple[Path, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    working_directory: str = "."
    limits: SandboxLimits = field(default_factory=SandboxLimits)

    def __post_init__(self) -> None:
        if not self.command or not self.command[0]:
            raise ValueError("command must contain an executable")
        if any("\x00" in argument for argument in self.command):
            raise ValueError("command contains a NUL byte")
        if Path(self.working_directory).is_absolute():
            raise ValueError("working_directory must be relative to /workspace")
        if ".." in Path(self.working_directory).parts:
            raise ValueError("working_directory cannot escape /workspace")


@dataclass(frozen=True, slots=True)
class SandboxResult:
    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    backend: str
    timed_out: bool = False
    output_limited: bool = False
    workspace_limited: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.output_limited
            and not self.workspace_limited
        )
