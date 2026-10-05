"""Runtime-owned job state; elapsed time is observation, never a kill condition."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Awaitable, Callable
from uuid import uuid4


@dataclass
class ExecutionSignals:
    cancel: threading.Event
    report: Callable[..., None]


execution_signals: ContextVar[ExecutionSignals | None] = ContextVar("execution_signals", default=None)


def report_progress(phase: str, **measurements: Any) -> None:
    signals = execution_signals.get()
    if signals is not None:
        signals.report(phase, **measurements)


@dataclass
class ToolJob:
    id: str
    tool: str
    arguments: dict[str, Any]
    interaction: str
    state: str = "queued"
    phase: str = "queued"
    started: float = field(default_factory=time.monotonic)
    changed: float = field(default_factory=time.monotonic)
    finished: float | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task | None = None
    owns_task: bool = True
    outcome: Any = None
    error: str = ""
    collected: bool = False


class ToolSupervisor:
    def __init__(self, root: Path):
        self.root = root
        self.jobs: dict[str, ToolJob] = {}
        self.lock = threading.RLock()
        self.last_heartbeat = time.monotonic()
        self.stop_monitor = threading.Event()
        self.heartbeat: asyncio.Task | None = None
        self.monitor: threading.Thread | None = None

    def start_monitoring(self) -> None:
        """Monitor a running application even before its first tool call."""
        if self.heartbeat is None and not self.stop_monitor.is_set():
            # Called from the owning event loop, never from the monitor thread.
            asyncio.get_running_loop()
            self.last_heartbeat = time.monotonic()
            self.heartbeat = asyncio.create_task(self._beat())
            self.monitor = threading.Thread(target=self._monitor, daemon=True,
                                            name="darklinger-tool-monitor")
            self.monitor.start()

    def _new_job(self, tool: str, arguments: dict, interaction: str) -> ToolJob:
        with self.lock:
            if sum(j.state in {"queued", "running", "stopping"} for j in self.jobs.values()) >= 32:
                raise ValueError("All background execution slots are occupied; existing jobs continue.")
            # Keep a bounded session history without dropping active work.
            for key in list(self.jobs):
                if len(self.jobs) < 128:
                    break
                if self.jobs[key].state in {"succeeded", "failed", "cancelled"}:
                    del self.jobs[key]
            job = ToolJob(uuid4().hex, tool, dict(arguments), interaction)
            self.jobs[job.id] = job
        self.start_monitoring()
        return job

    def start(self, tool: str, arguments: dict, interaction: str,
              call: Callable[[], Awaitable[Any]]) -> ToolJob:
        job = self._new_job(tool, arguments, interaction)
        job.task = asyncio.create_task(self._execute(job, call))
        def cancelled_before_start(task):
            if task.cancelled() and not job.measurements.get("process_alive"):
                with self.lock:
                    job.state = job.phase = "cancelled"
                    job.finished = job.changed = time.monotonic()
        job.task.add_done_callback(cancelled_before_start)
        return job

    async def _beat(self):
        while not self.stop_monitor.is_set():
            self.last_heartbeat = time.monotonic()
            await asyncio.sleep(.25)

    async def observe_call(self, tool: str, arguments: dict, interaction: str,
                           call: Callable[[], Awaitable[Any]]) -> Any:
        """Observe ordinary calls in their owning task (MCP sessions are task-affine)."""
        job = self._new_job(tool, arguments, interaction)
        job.task = asyncio.current_task()
        job.owns_task = False
        job.state = "running"
        token = execution_signals.set(ExecutionSignals(
            job.cancel, lambda phase, **values: self.update(job, phase, **values)))
        try:
            outcome = await call()
            with self.lock:
                job.outcome = outcome
                job.error = str(getattr(outcome, "error", "") or "")
                job.state = "failed" if job.error else "succeeded"
                job.collected = True  # Caller already receives this receipt.
            return outcome
        except asyncio.CancelledError:
            job.cancel.set()
            job.state = "cancelled"
            raise
        except Exception as error:
            job.error = f"{type(error).__name__}: {error}"
            job.state = "failed"
            raise
        finally:
            execution_signals.reset(token)
            with self.lock:
                job.phase = job.state
                job.finished = job.changed = time.monotonic()

    def _monitor(self):
        # This thread continues writing diagnostic state if the model fails or
        # the application's event loop stops advancing. It never cancels work.
        self.root.mkdir(parents=True, exist_ok=True)
        while not self.stop_monitor.wait(.5):
            payload = self.snapshot()
            try:
                temporary = self.root / "latest.tmp"
                temporary.write_text(json.dumps(payload, ensure_ascii=False))
                temporary.replace(self.root / "latest.json")
            except OSError:
                pass

    def update(self, job: ToolJob, phase: str, **measurements):
        with self.lock:
            if job.state not in {"running", "stopping"}:
                return
            job.phase = str(phase)[:80]
            # Only fixed adapter measurements, no arguments, page text or model
            # declarations of success enter the persistent monitoring snapshot.
            job.measurements.update({k: v for k, v in measurements.items() if k in {
                "pid", "process_alive", "bytes_received", "headers_received",
                "pages_completed", "returncode", "process_state", "wait_channel",
            }})
            if phase == "process_stopped" and job.cancel.is_set():
                job.state = "cancelled"
            job.changed = time.monotonic()
            if job.state in {"succeeded", "failed", "cancelled"}:
                job.finished = job.changed

    async def _execute(self, job: ToolJob, call):
        with self.lock:
            job.state = "running"
        token = execution_signals.set(ExecutionSignals(
            job.cancel, lambda phase, **values: self.update(job, phase, **values)))
        try:
            outcome = await call()
            with self.lock:
                job.outcome = outcome
                job.error = str(getattr(outcome, "error", "") or "")
                job.state = "failed" if job.error else "succeeded"
        except asyncio.CancelledError:
            job.cancel.set()
            with self.lock:
                job.state = "stopping" if job.measurements.get("process_alive") else "cancelled"
        except Exception as error:
            with self.lock:
                job.error = f"{type(error).__name__}: {error}"
                job.state = "failed"
        finally:
            execution_signals.reset(token)
            with self.lock:
                job.phase = job.state
                job.changed = time.monotonic()
                if job.state in {"succeeded", "failed", "cancelled"}:
                    job.finished = job.changed

    def get(self, job_id: str) -> ToolJob:
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("Unknown job in this runtime session")
            return self.jobs[job_id]

    def status(self, job_id: str) -> dict:
        job = self.get(job_id)
        with self.lock:
            now = time.monotonic()
            return {
                "job_id": job.id, "tool": job.tool, "state": job.state,
                "phase": job.phase,
                "elapsed_seconds": round((job.finished if job.finished is not None else now) - job.started, 2),
                "last_signal_age_seconds": round(now - job.changed, 2),
                "signals": dict(job.measurements),
                "completion_verified": job.state == "succeeded",
                "automatic_deadline": False,
                "interpretation": (
                    "No completion evidence yet; waiting is not proof of a hang."
                    if job.state in {"queued", "running", "stopping"} else job.state),
            }

    def snapshot(self) -> dict:
        with self.lock:
            age = time.monotonic() - self.last_heartbeat if self.heartbeat is not None else None
            return {"runtime_pid": os.getpid(), "heartbeat_age_seconds": round(age, 2) if age is not None else None,
                    "event_loop_state": ("not_started" if age is None else
                                         "responding" if age < 3 else "unresponsive_suspected"),
                    "automatic_cancellation": False,
                    "jobs": [self.status(key) for key in list(self.jobs)]}

    async def wait_status(self, job_id: str) -> dict:
        job = self.get(job_id)
        # This is a control-plane observation interval, NOT an execution limit.
        # shield prevents timeout of the status request from cancelling the job.
        if job.state in {"queued", "running", "stopping"} and job.task and not job.task.done():
            await asyncio.wait({job.task}, timeout=1)
        return self.status(job_id)

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        with self.lock:
            if job.state in {"queued", "running"}:
                job.state = "stopping"
                job.cancel.set()
                if job.task:
                    job.task.cancel()
        return self.status(job_id)

    async def close(self):
        for job in list(self.jobs.values()):
            self.cancel(job.id)
        await asyncio.gather(*(j.task for j in self.jobs.values() if j.task and j.owns_task), return_exceptions=True)
        self.stop_monitor.set()
        if self.heartbeat:
            self.heartbeat.cancel()
            await asyncio.gather(self.heartbeat, return_exceptions=True)
        if self.monitor:
            await asyncio.to_thread(self.monitor.join, 2)
