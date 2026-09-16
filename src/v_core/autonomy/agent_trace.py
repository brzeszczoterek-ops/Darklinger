from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .journal import TaskJournal
from .models import utc_now


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


@dataclass(slots=True)
class AgentTaskTrace:
    """Durable evidence for one interactive agent task.

    This is deliberately separate from model-authored conversation. Only the
    runtime can add tool checkpoints, so a model cannot turn a promise into an
    apparent execution record by printing JSON in its final answer.
    """

    root: Path
    objective: str
    task_id: str = field(default_factory=lambda: f"interactive-{uuid4().hex}")
    status: str = "running"
    started_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    runtime_pid: int = field(default_factory=os.getpid)
    requirements: dict[str, bool] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    context_rollovers: list[dict[str, Any]] = field(default_factory=list)
    owner_checkpoint: dict[str, Any] = field(default_factory=dict)
    owner_authorization: dict[str, Any] = field(default_factory=dict)
    _journal: TaskJournal = field(init=False, repr=False)
    _checkpoint_root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._journal = TaskJournal(self.root / "journal")
        self._checkpoint_root = self.root / "checkpoints"
        self._checkpoint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._checkpoint_root, 0o700)
        self._journal.append(
            self.task_id,
            "task_started",
            {"objective": self.objective},
        )
        self._save()

    def record_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        self.updated_at = utc_now()
        self._journal.append(self.task_id, event, data)
        self._save()

    def set_requirements(self, requirements: dict[str, Any]) -> None:
        self.requirements = dict(requirements)
        self.record_event("task_contract_created", {"requirements": self.requirements})

    def tool_started(self, name: str, arguments: dict | str) -> int:
        index = len(self.tool_calls) + 1
        call = {
            "sequence": index,
            "tool": name,
            "arguments": _json_safe(arguments),
            "status": "running",
            "started_at": utc_now(),
        }
        self.tool_calls.append(call)
        self.updated_at = utc_now()
        self._journal.append(
            self.task_id,
            "tool_started",
            call,
        )
        self._save()
        return index

    def tool_finished(
        self,
        sequence: int,
        result: str,
        *,
        error: str | None = None,
        evidence_excerpt: str | None = None,
        provider_tool: str = "",
        capabilities: list[str] | tuple[str, ...] = (),
        provider_attempts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        recovery_ticket: dict[str, Any] | None = None,
    ) -> None:
        call = self.tool_calls[sequence - 1]
        call["status"] = "failed" if error else "succeeded"
        call["finished_at"] = utc_now()
        call["result_sha256"] = _digest(result)
        call["result_excerpt"] = (
            result if evidence_excerpt is None else evidence_excerpt
        )[:6_000]
        if error:
            call["error"] = error
        if provider_tool:
            call["provider_tool"] = str(provider_tool)[:128]
        if capabilities:
            call["capabilities"] = [str(item)[:96] for item in capabilities[:16]]
        if provider_attempts:
            call["provider_attempts"] = _json_safe(list(provider_attempts)[:16])
        if recovery_ticket is not None:
            call["recovery_ticket"] = _json_safe(recovery_ticket)
        self.updated_at = utc_now()
        self._journal.append(
            self.task_id,
            "tool_failed" if error else "tool_completed",
            call,
        )
        self._save()

    def context_rolled(
        self,
        *,
        step: int,
        estimated_tokens_before: int,
        estimated_tokens_after: int,
        context_size: int,
        summary: dict[str, Any],
        evidence_count: int,
        emergency: bool = False,
    ) -> None:
        rollover = {
            "sequence": len(self.context_rollovers) + 1,
            "step": step,
            "estimated_tokens_before": estimated_tokens_before,
            "estimated_tokens_after": estimated_tokens_after,
            "context_size": context_size,
            "evidence_count": evidence_count,
            "emergency": emergency,
            "summary": _json_safe(summary),
            "timestamp": utc_now(),
        }
        self.context_rollovers.append(rollover)
        self.updated_at = utc_now()
        self._journal.append(self.task_id, "context_rolled", rollover)
        self._save()

    def await_owner(
        self,
        *,
        reason: str,
        step_limit: int,
        successful_tool_count: int,
        failed_tool_count: int,
        missing: list[str],
        progress_summary: dict[str, Any] | None = None,
        accepted_commands: list[str] | None = None,
    ) -> None:
        if self.status != "running":
            return
        self.status = "awaiting_owner"
        self.updated_at = utc_now()
        self.finished_at = None
        self.owner_checkpoint = {
            "reason": reason[:2_000],
            "step_limit": step_limit,
            "successful_tool_count": successful_tool_count,
            "failed_tool_count": failed_tool_count,
            "missing": [str(item)[:256] for item in missing[:32]],
            "progress_summary": _json_safe(progress_summary or {}),
            "requested_at": self.updated_at,
            "accepted_commands": accepted_commands or ["/continue", "/stop"],
        }
        self._journal.append(
            self.task_id,
            "task_awaiting_owner",
            self.owner_checkpoint,
        )
        self._save(result={"owner_decision_required": self.owner_checkpoint})

    def resume_from_owner(self) -> None:
        if self.status != "awaiting_owner":
            raise RuntimeError("task is not awaiting an owner decision")
        self.status = "running"
        self.updated_at = utc_now()
        self.finished_at = None
        self._journal.append(
            self.task_id,
            "task_resumed_by_owner",
            {"previous_checkpoint": self.owner_checkpoint},
        )
        self._save()

    def authorize_continuous_from_owner(self) -> None:
        if self.status != "awaiting_owner":
            raise RuntimeError("task is not awaiting an owner decision")
        self.status = "running"
        self.updated_at = utc_now()
        self.finished_at = None
        self.owner_authorization = {
            "mode": "continuous",
            "granted_at": self.updated_at,
            "scope": "this_task_only",
            "silent_batch_checkpoints": True,
            "emergency_controls_remain_active": True,
            "stalled_batch_limit": 3,
            "stalled_batches": 0,
        }
        self._journal.append(
            self.task_id,
            "task_continuous_authorized_by_owner",
            {
                "previous_checkpoint": self.owner_checkpoint,
                "authorization": self.owner_authorization,
            },
        )
        self._save()

    @property
    def continuous_authorized(self) -> bool:
        return self.owner_authorization.get("mode") == "continuous"

    def continuous_batch_checkpoint(
        self,
        *,
        step_limit: int,
        successful_tool_count: int,
        failed_tool_count: int,
        new_evidence_count: int,
        missing: list[str],
        progress_summary: dict[str, Any] | None = None,
    ) -> int:
        if self.status != "running" or not self.continuous_authorized:
            raise RuntimeError("task has no active continuous authorization")
        stalled = int(self.owner_authorization.get("stalled_batches", 0))
        stalled = 0 if new_evidence_count > 0 else stalled + 1
        self.owner_authorization["stalled_batches"] = stalled
        self.updated_at = utc_now()
        checkpoint = {
            "step_limit": step_limit,
            "successful_tool_count": successful_tool_count,
            "failed_tool_count": failed_tool_count,
            "new_evidence_count": new_evidence_count,
            "missing": [str(item)[:256] for item in missing[:32]],
            "progress_summary": _json_safe(progress_summary or {}),
            "stalled_batches": stalled,
            "timestamp": self.updated_at,
        }
        self.owner_checkpoint = checkpoint
        self._journal.append(
            self.task_id,
            "task_continuous_batch_checkpoint",
            checkpoint,
        )
        self._save(result={"continuous_checkpoint": checkpoint})
        return stalled

    def stop_from_owner(self) -> None:
        if self.status != "awaiting_owner":
            raise RuntimeError("task is not awaiting an owner decision")
        self.status = "stopped"
        self.updated_at = utc_now()
        self.finished_at = self.updated_at
        result = {
            "reason": "owner stopped task at the step-budget checkpoint",
            "owner_checkpoint": self.owner_checkpoint,
        }
        self._journal.append(self.task_id, "task_stopped_by_owner", result)
        self._save(result=result)

    @classmethod
    def load(cls, root: Path, task_id: str) -> "AgentTaskTrace | None":
        """Restore an existing interactive trace without creating a new task."""

        root = Path(root)
        path = root / "checkpoints" / f"{task_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        trace = cls.__new__(cls)
        trace.root = root
        trace.objective = str(payload.get("objective", ""))
        trace.task_id = str(payload.get("task_id", task_id))
        trace.status = str(payload.get("status", "running"))
        trace.started_at = str(payload.get("started_at", utc_now()))
        trace.updated_at = str(payload.get("updated_at", utc_now()))
        trace.finished_at = payload.get("finished_at")
        trace.runtime_pid = int(payload.get("runtime_pid") or 0)
        trace.requirements = dict(payload.get("requirements", {}))
        trace.tool_calls = list(payload.get("tool_calls", []))
        trace.context_rollovers = list(payload.get("context_rollovers", []))
        trace.owner_checkpoint = dict(payload.get("owner_checkpoint", {}))
        trace.owner_authorization = dict(payload.get("owner_authorization", {}))
        trace._journal = TaskJournal(root / "journal")
        trace._checkpoint_root = root / "checkpoints"
        return trace

    @staticmethod
    def recover_stale_running(root: Path) -> list[str]:
        """Mark checkpoints abandoned by a dead PALADYN process as interrupted.

        A live PID is left untouched so two explicitly separate runtimes do not
        rewrite each other's active task. Older checkpoints had no PID; a new
        runtime can safely recover those because no old process identity exists.
        """

        recovered: list[str] = []
        checkpoints = Path(root) / "checkpoints"
        try:
            candidates = list(checkpoints.glob("interactive-*.json"))
        except OSError:
            return recovered
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("status") != "running":
                continue
            runtime_pid = int(payload.get("runtime_pid") or 0)
            if runtime_pid > 0:
                try:
                    os.kill(runtime_pid, 0)
                except (OSError, ProcessLookupError):
                    pass
                else:
                    continue
            task_id = str(payload.get("task_id", path.stem))
            trace = AgentTaskTrace.load(root, task_id)
            if trace is None:
                continue
            now = utc_now()
            for call in trace.tool_calls:
                if call.get("status") != "running":
                    continue
                call["status"] = "failed"
                call["finished_at"] = now
                call["error"] = "PALADYN runtime exited before the tool completed."
            trace.status = "interrupted"
            trace.updated_at = now
            trace.finished_at = now
            data = {
                "reason": "previous PALADYN runtime exited without closing the task"
            }
            trace._journal.append(task_id, "task_recovered_as_interrupted", data)
            trace._save(result=data)
            recovered.append(task_id)
        return recovered

    def complete(self, answer: str) -> None:
        self.status = "completed"
        self.updated_at = utc_now()
        self.finished_at = self.updated_at
        result = {
            "result_sha256": _digest(answer),
            "result_excerpt": answer[:2_000],
        }
        self._journal.append(self.task_id, "task_completed", result)
        self._save(result=result)

    def stop(self, reason: str) -> None:
        self._terminate("stopped", "task_stopped", reason)

    def fail(self, reason: str) -> None:
        self._terminate("failed", "task_failed", reason)

    def block(self, reason: str) -> None:
        self._terminate("blocked", "task_blocked", reason)

    def evidence(self) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        for call in self.tool_calls:
            arguments = json.dumps(
                call.get("arguments", {}),
                ensure_ascii=False,
                default=str,
            )
            calls.append(
                {
                    "sequence": call.get("sequence"),
                    "tool": call.get("tool"),
                    "provider_tool": call.get("provider_tool", call.get("tool")),
                    "capabilities": call.get("capabilities", []),
                    "provider_attempts": call.get("provider_attempts", []),
                    "recovery_ticket": call.get("recovery_ticket"),
                    "status": call.get("status"),
                    "arguments_sha256": _digest(arguments),
                    "arguments_excerpt": arguments[:2_000],
                    "result_sha256": call.get("result_sha256", ""),
                    "result_excerpt": call.get("result_excerpt", ""),
                    "error": call.get("error", ""),
                }
            )
        return {
            "task_id": self.task_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_pid": self.runtime_pid,
            "tool_calls": calls,
            "context_rollovers": self.context_rollovers,
            "owner_checkpoint": self.owner_checkpoint,
            "owner_authorization": self.owner_authorization,
            "requirements": self.requirements,
            "successful_tool_count": sum(
                call.get("status") == "succeeded" for call in self.tool_calls
            ),
            "failed_tool_count": sum(
                call.get("status") == "failed" for call in self.tool_calls
            ),
        }

    @staticmethod
    def latest_context(root: Path) -> dict[str, Any] | None:
        """Load a bounded runtime-authored summary for follow-up recovery."""

        checkpoints = Path(root) / "checkpoints"
        try:
            candidates = sorted(
                checkpoints.glob("interactive-*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return None
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            calls = []
            for call in payload.get("tool_calls", [])[-6:]:
                summary = {
                    "tool": str(call.get("tool", ""))[:128],
                    "status": str(call.get("status", ""))[:32],
                    "error": str(call.get("error", ""))[:500],
                }
                if call.get("created_tool_name"):
                    summary["created_tool_name"] = str(
                        call["created_tool_name"]
                    )[:128]
                calls.append(summary)
            rollovers = payload.get("context_rollovers", [])
            latest_rollover = (
                rollovers[-1]
                if isinstance(rollovers, list) and rollovers
                and isinstance(rollovers[-1], dict)
                else {}
            )
            return {
                "task_id": str(payload.get("task_id", ""))[:128],
                "objective": str(payload.get("objective", ""))[:2_000],
                "status": str(payload.get("status", ""))[:32],
                "requirements": payload.get("requirements", {}),
                "tool_calls": calls,
                "latest_context_rollover": {
                    "sequence": latest_rollover.get("sequence"),
                    "step": latest_rollover.get("step"),
                    "summary": latest_rollover.get("summary", {}),
                } if latest_rollover else {},
                "owner_checkpoint": payload.get("owner_checkpoint", {}),
                "owner_authorization": payload.get("owner_authorization", {}),
            }
        return None

    @staticmethod
    def latest_action_context(
        root: Path,
        *,
        exclude_task_id: str = "",
    ) -> dict[str, Any] | None:
        """Load the latest checkpoint that describes concrete runtime work.

        A short follow-up such as ``continue`` can itself produce a checkpoint
        with no contract and no calls. Walking back to the latest checkpoint
        with runtime requirements prevents that empty turn from severing the
        user's task, while still keeping prior tool results out of the new
        interaction's evidence ledger.
        """

        checkpoints = Path(root) / "checkpoints"
        try:
            candidates = sorted(
                checkpoints.glob("interactive-*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return None

        parsed: list[dict[str, Any]] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task_id = str(payload.get("task_id", ""))[:128]
            if exclude_task_id and task_id == exclude_task_id:
                continue
            requirements = payload.get("requirements", {})
            if not isinstance(requirements, dict):
                requirements = {}
            raw_calls = payload.get("tool_calls", [])
            if not isinstance(raw_calls, list):
                raw_calls = []
            calls = [
                {
                    **{
                        "tool": str(call.get("tool", ""))[:128],
                        "status": str(call.get("status", ""))[:32],
                        "error": str(call.get("error", ""))[:500],
                    },
                    **(
                        {
                            "created_tool_name": str(
                                call["created_tool_name"]
                            )[:128]
                        }
                        if call.get("created_tool_name")
                        else {}
                    ),
                }
                for call in raw_calls[-6:]
                if isinstance(call, dict)
            ]
            parsed.append(
                {
                    "task_id": task_id,
                    "objective": str(payload.get("objective", ""))[:2_000],
                    "status": str(payload.get("status", ""))[:32],
                    "requirements": requirements,
                    "tool_calls": calls,
                    "latest_context_rollover": (
                        payload.get("context_rollovers", [])[-1]
                        if isinstance(payload.get("context_rollovers", []), list)
                        and payload.get("context_rollovers", [])
                        and isinstance(payload.get("context_rollovers", [])[-1], dict)
                        else {}
                    ),
                    "owner_checkpoint": payload.get("owner_checkpoint", {}),
                    "owner_authorization": payload.get("owner_authorization", {}),
                }
            )

        # A runtime-owned contract is stronger routing evidence than a model's
        # attempted call. Only fall back to calls for older checkpoints that
        # predate task contracts or for actions the deterministic parser did
        # not yet classify.
        for context in parsed:
            if AgentTaskTrace.context_has_action_route(
                {**context, "tool_calls": []}
            ):
                return context
        for context in parsed:
            if AgentTaskTrace.context_has_action_route(
                {**context, "requirements": {}}
            ):
                return context
        return None

    @staticmethod
    def latest_created_tool(root: Path) -> dict[str, str] | None:
        """Return the newest runtime-verified generated tool activation.

        Follow-up turns may refer to "that tool" after one or more ordinary
        turns.  Resolve that reference from durable lifecycle metadata rather
        than model-authored chat or a copied test result.
        """

        checkpoints = Path(root) / "checkpoints"
        try:
            candidates = sorted(
                checkpoints.glob("interactive-*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return None
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            calls = payload.get("tool_calls", [])
            if not isinstance(calls, list):
                continue
            for call in reversed(calls):
                if not isinstance(call, dict):
                    continue
                if call.get("status") != "succeeded":
                    continue
                if call.get("tool") not in {
                    "learning_create_tool",
                    "learning_create_snapshot_extractor",
                }:
                    continue
                name = str(call.get("created_tool_name", "")).strip()
                if not name:
                    continue
                return {
                    "name": name[:128],
                    "task_id": str(payload.get("task_id", ""))[:128],
                }
        return None

    @staticmethod
    def context_has_action_route(context: dict[str, Any] | None) -> bool:
        """Return whether a checkpoint is safe to resume as executable work.

        Reporting flags alone are not an execution route. A runtime-review-only
        checkpoint is accepted only when its own objective explicitly describes
        diagnostics; this keeps a bad semantic classification from becoming the
        parent of every later ``continue`` turn.
        """

        if not isinstance(context, dict):
            return False
        requirements = context.get("requirements", {})
        if not isinstance(requirements, dict):
            requirements = {}
        concrete_flags = (
            "requires_browser_navigation",
            "requires_browser_snapshot",
            "requires_web_discovery",
            "requires_distinct_detail_page",
            "requires_file_read",
            "requires_file_mutation",
            "requires_command_execution",
            "requires_created_tool",
            "requires_created_tool_execution",
            "requires_created_skill",
            "requires_created_artifact",
            "allows_artifact_fallback",
        )
        if any(bool(requirements.get(name)) for name in concrete_flags):
            return True
        if requirements.get("required_tools"):
            return True
        if requirements.get("required_capabilities"):
            return True
        if requirements.get("requires_runtime_review"):
            # Import locally to keep the persistence module lightweight and to
            # avoid making checkpoint loading depend on intent-router startup.
            from .task_contract import TaskContract

            objective = str(context.get("objective", ""))
            if TaskContract.from_prompt(objective).requires_runtime_review:
                return True

        calls = context.get("tool_calls", [])
        if not isinstance(calls, list):
            return False
        return any(
            isinstance(call, dict)
            and str(call.get("tool", ""))
            and str(call.get("tool", "")) != "runtime_review_task"
            for call in calls
        )

    def _save(self, *, result: dict[str, Any] | None = None) -> None:
        path = self._checkpoint_root / f"{self.task_id}.json"
        temporary = self._checkpoint_root / f".{self.task_id}.tmp"
        payload = {
            "schema_version": 4,
            "task_id": self.task_id,
            "objective": self.objective,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "requirements": self.requirements,
            "tool_calls": self.tool_calls,
            "context_rollovers": self.context_rollovers,
            "owner_checkpoint": self.owner_checkpoint,
            "owner_authorization": self.owner_authorization,
        }
        if result is not None:
            payload["result"] = result

        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)

    def _terminate(self, status: str, event: str, reason: str) -> None:
        if self.status != "running":
            return
        self.status = status
        self.updated_at = utc_now()
        self.finished_at = self.updated_at
        data = {"reason": reason[:2_000]}
        self._journal.append(self.task_id, event, data)
        self._save(result=data)
