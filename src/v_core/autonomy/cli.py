from __future__ import annotations

import os
from pathlib import Path

import typer

from ..branding import env_value
from rich.console import Console

from .control import ControlChannel, GlobalControlChannel
from .hotkey import LinuxHotkeyWatcher, input_devices, parse_chord
from .journal import CheckpointStore
from .models import ControlSignal
from .runtime import RuntimeRegistry


app = typer.Typer(add_completion=False)
console = Console()


def default_root() -> Path:
    return Path(env_value("DARKLINGER_AUTONOMY_ROOT", "autonomy")).expanduser()


@app.command("signal")
def send_signal(
    task_id: str,
    action: ControlSignal,
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    channel = ControlChannel(root / "control", task_id)
    channel.request(action)
    console.print(f"{action.value.upper()} requested for {task_id}")


@app.command()
def status(
    task_id: str,
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    loaded = CheckpointStore(root / "checkpoints").load(task_id)
    if loaded is None:
        raise typer.Exit(code=1)

    task, envelope = loaded
    console.print(f"Task      : {task.task_id}")
    console.print(f"Status    : {task.status.value}")
    console.print(f"Mode      : {envelope.mode.value}")
    console.print(f"Actions   : {task.action_count}/{envelope.budget.max_actions}")
    console.print(f"Failures  : {task.failure_count}/{envelope.budget.max_failures}")
    console.print(f"Objective : {task.objective}")
    if task.blocked_reason:
        console.print(f"Reason    : {task.blocked_reason}")


@app.command("panic-all")
def panic_all(
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    GlobalControlChannel(root / "control").request_panic()
    terminated = RuntimeRegistry(root / "runtime").terminate_all()
    console.print("GLOBAL PANIC latched")
    if terminated:
        console.print(f"Termination requested for PID(s): {', '.join(map(str, terminated))}")


@app.command("reset-panic")
def reset_panic(
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    GlobalControlChannel(root / "control").clear_panic()
    console.print("Global panic latch cleared")


@app.command("input-devices")
def list_input_devices() -> None:
    devices = input_devices()
    if not devices:
        console.print("No Linux input event devices found.")
        raise typer.Exit(code=1)
    for path, name in devices:
        if not path.exists():
            access = "device node not visible"
        elif os.access(path, os.R_OK):
            access = "readable"
        else:
            access = "permission denied"
        console.print(f"{path}  {name}  [{access}]")


@app.command("hotkey")
def hotkey(
    device: Path = typer.Option(
        ...,
        help="Keyboard event device, for example /dev/input/event3.",
    ),
    chord: str = typer.Option("Q+P+0", help="Simultaneous emergency chord."),
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    try:
        parsed_chord = parse_chord(chord)
    except ValueError as exc:
        console.print(f"Invalid chord: {exc}")
        raise typer.Exit(code=2) from exc

    global_control = GlobalControlChannel(root / "control")
    runtimes = RuntimeRegistry(root / "runtime")

    def trigger() -> None:
        global_control.request_panic()
        terminated = runtimes.terminate_all()
        console.print(
            f"EMERGENCY CHORD {chord.upper()} — GLOBAL PANIC; "
            f"termination requested for {len(terminated)} runtime(s)"
        )

    console.print(f"Watching {device} for {chord.upper()} (Ctrl+C exits watcher)")
    try:
        LinuxHotkeyWatcher(device, parsed_chord).watch(trigger)
    except PermissionError as exc:
        console.print(
            f"Cannot read {device}. Grant this user read access to the keyboard "
            "event device (normally through an input/uaccess rule)."
        )
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        console.print(f"Input device does not exist: {device}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
