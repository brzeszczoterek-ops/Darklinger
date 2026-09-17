from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener


_DIRECT_OPENER = build_opener(ProxyHandler({}))
_LIVE_GENERATION = re.compile(
    r"n_decoded\s*=\s*(\d+).*?tg_3s\s*=\s*([\d.]+)\s*t/s",
    re.IGNORECASE,
)
_LIVE_PROMPT = re.compile(
    r"prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([\d.]+),"
    r".*?/\s*([\d.]+)\s+tokens per second",
    re.IGNORECASE,
)
_COMPLETED_PROMPT = re.compile(
    r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens.*?"
    r"([\d.]+) tokens per second",
    re.IGNORECASE,
)
_COMPLETED_GENERATION = re.compile(
    r"(?<!prompt )eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens.*?"
    r"([\d.]+) tokens per second",
    re.IGNORECASE,
)


def _http_json(port: int, path: str) -> Any:
    """Read a bounded llama.cpp localhost endpoint without using proxy settings."""

    try:
        with _DIRECT_OPENER.open(
            f"http://127.0.0.1:{int(port)}{path}", timeout=0.35
        ) as response:
            return json.loads(
                response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
            )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


def _tail(path: Path, maximum: int = 128 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum))
            return handle.read(maximum).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _latest_checkpoint(
    root: Path,
    *,
    not_before: float | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    checkpoint_root = root / "interactive" / "checkpoints"
    try:
        candidates = sorted(
            checkpoint_root.glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None, {}
    for path in candidates:
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            continue
        if not_before is not None and modified_at < not_before:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return path, payload
    return None, {}


def _parse_timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event_text(event: str, data: dict[str, Any]) -> tuple[str, str]:
    tool = str(data.get("tool", "")).strip()
    mappings = {
        "task_started": ("TASK", "Task accepted by runtime"),
        "task_completed": ("DONE", "Task completed"),
        "task_blocked": ("BLOCKED", "Task stopped with a blocker"),
        "task_awaiting_owner": ("WAIT", "Waiting for owner decision"),
        "task_resumed_by_owner": ("RESUME", "Owner resumed task"),
        "context_rolled": (
            "CONTEXT",
            f"Context rollover #{data.get('sequence', '?')}",
        ),
        "candidate_rejected": (
            "REJECT",
            "Model output rejected: " + str(data.get("reason", "unverified output")),
        ),
        "learning_evidence_recorded": (
            "LEARN",
            "Runtime recorded "
            + str(data.get("outcome", "tool"))
            + " evidence for "
            + str(data.get("tool", "learning")),
        ),
        "task_contract_created": ("CONTRACT", "Task completion contract created"),
    }
    if event in mappings:
        return mappings[event]
    if event == "tool_started":
        return "TOOL", f"Started {tool or 'tool'}"
    if event == "tool_completed":
        return "TOOL", f"Completed {tool or 'tool'}"
    if event == "tool_failed":
        return "FAIL", f"Failed {tool or 'tool'}"
    if event == "model_phase_routed":
        previous = str(data.get("from_phase", "model"))
        current = str(data.get("to_phase", "model"))
        return "ROUTE", f"Model phase {previous} → {current}"
    if event == "model_protocol_fallback":
        return "ROUTE", "Fallback model selected after protocol failure"
    if event == "final_answer_loop_cut_off":
        return "STOP", "Invalid final-answer loop stopped"
    return event.replace("_", " ").upper()[:16], event.replace("_", " ")


def _journal_activity(
    autonomy_root: Path,
    task_id: str,
    *,
    limit: int = 12,
) -> tuple[list[dict[str, str]], str, datetime | None]:
    if not task_id or not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", task_id):
        return [], "", None
    path = autonomy_root / "interactive" / "journal" / f"{task_id}.jsonl"
    events: list[dict[str, str]] = []
    phase = ""
    started_at: datetime | None = None
    for raw_line in _tail(path, 256 * 1024).splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        event = str(record.get("event", ""))
        data = record.get("data", {})
        if not isinstance(data, dict):
            data = {}
        timestamp = str(record.get("timestamp", ""))
        if event == "task_started" and started_at is None:
            started_at = _parse_timestamp(timestamp)
        if event == "model_phase_routed":
            phase = str(data.get("to_phase", ""))
        label, text = _event_text(event, data)
        events.append(
            {
                "id": f"{timestamp}:{event}:{data.get('sequence', '')}",
                "timestamp": timestamp,
                "label": label,
                "text": text[:220],
            }
        )
    return events[-limit:], phase, started_at


def runtime_activity(
    autonomy_root: Path | str | None,
    session: Any | None,
    *,
    ready: bool,
    session_started_at: float | None = None,
    sample_state: dict[str, Any] | None = None,
    sample_now: float | None = None,
    http_json: Callable[[int, str], Any] = _http_json,
) -> dict[str, Any]:
    """Build one factual live view from llama.cpp and DARKLINGER's task ledger."""

    root = Path(autonomy_root).expanduser() if autonomy_root else None
    checkpoint: dict[str, Any] = {}
    if root is not None:
        _, checkpoint = _latest_checkpoint(
            root,
            not_before=session_started_at,
        )
    task_id = str(checkpoint.get("task_id", ""))
    task_status = str(checkpoint.get("status", "idle"))
    events: list[dict[str, str]] = []
    phase = ""
    started_at: datetime | None = None
    if root is not None:
        events, phase, started_at = _journal_activity(root, task_id)

    profile = getattr(session, "profile", None)
    port = int(getattr(profile, "port", 0) or 0)
    slots = http_json(port, "/slots") if port else None
    slot = (
        next((item for item in slots if isinstance(item, dict)), None)
        if isinstance(slots, list)
        else None
    )
    processing = bool(slot and slot.get("is_processing"))
    prompt_tokens = int(slot.get("n_prompt_tokens", 0) or 0) if slot else 0
    prompt_processed = (
        int(slot.get("n_prompt_tokens_processed", 0) or 0) if slot else 0
    )
    next_token = slot.get("next_token", []) if slot else []
    token_state = (
        next((item for item in next_token if isinstance(item, dict)), {})
        if isinstance(next_token, list)
        else {}
    )
    generated_tokens = int(token_state.get("n_decoded", 0) or 0)
    remaining_tokens = int(token_state.get("n_remain", 0) or 0)
    context_size = int(
        (slot.get("n_ctx", 0) if slot else 0)
        or getattr(profile, "context_size", 0)
        or 0
    )
    context_used = min(context_size, prompt_tokens + generated_tokens)

    log_path = getattr(session, "log_path", None)
    log_text = _tail(Path(log_path)) if log_path else ""
    generation_matches = list(_LIVE_GENERATION.finditer(log_text))
    prompt_matches = list(_LIVE_PROMPT.finditer(log_text))
    completed_generation = list(_COMPLETED_GENERATION.finditer(log_text))
    completed_prompt = list(_COMPLETED_PROMPT.finditer(log_text))
    generation_rate = (
        float(generation_matches[-1].group(2))
        if processing and generated_tokens and generation_matches
        else None
    )
    prompt_rate = (
        float(prompt_matches[-1].group(3))
        if processing and not generated_tokens and prompt_matches
        else None
    )
    now = time.monotonic() if sample_now is None else float(sample_now)
    if sample_state is not None:
        task_key = (
            str(getattr(getattr(session, "process", None), "pid", "external")),
            str(slot.get("id_task", "")) if slot else "",
        )
        previous_key = sample_state.get("task_key")
        previous_at = float(sample_state.get("sample_at", now) or now)
        elapsed_sample = now - previous_at
        if processing and previous_key == task_key and elapsed_sample > 0.15:
            generated_delta = generated_tokens - int(
                sample_state.get("generated_tokens", generated_tokens) or 0
            )
            prompt_delta = prompt_processed - int(
                sample_state.get("prompt_processed", prompt_processed) or 0
            )
            if generated_delta > 0:
                generation_rate = generated_delta / elapsed_sample
            elif prompt_delta > 0 and not generated_tokens:
                prompt_rate = prompt_delta / elapsed_sample
        sample_state.update(
            {
                "task_key": task_key,
                "sample_at": now,
                "generated_tokens": generated_tokens,
                "prompt_processed": prompt_processed,
            }
        )

    # Short local responses can complete between two UI polls and never emit
    # llama.cpp's periodic ``tg_3s`` line. Preserve their final timing instead
    # of replacing useful telemetry with an em dash as soon as the slot idles.
    if not processing:
        if completed_generation:
            generated_tokens = int(completed_generation[-1].group(1))
            generation_rate = float(completed_generation[-1].group(2))
        if completed_prompt:
            prompt_tokens = int(completed_prompt[-1].group(1))
            prompt_rate = float(completed_prompt[-1].group(2))
        context_used = min(context_size, prompt_tokens + generated_tokens)

    calls = checkpoint.get("tool_calls", [])
    if not isinstance(calls, list):
        calls = []
    last_call = calls[-1] if calls and isinstance(calls[-1], dict) else {}
    last_tool = str(last_call.get("tool", ""))
    last_tool_status = str(last_call.get("status", ""))
    running_tool = next(
        (
            call
            for call in reversed(calls)
            if isinstance(call, dict) and call.get("status") == "running"
        ),
        None,
    )

    if running_tool is not None:
        state = "tool_running"
        detail = f"Tool running: {running_tool.get('tool', 'unknown')}"
    elif processing and generated_tokens:
        state = "repairing" if last_tool_status == "failed" else "generating"
        detail = (
            f"Model generating a retry after {last_tool} failed"
            if state == "repairing" and last_tool
            else "Model generating the next action"
        )
    elif processing:
        state = "prompt"
        detail = "Model processing prompt and runtime context"
    elif task_status == "running" and not ready:
        updated = _parse_timestamp(checkpoint.get("updated_at", ""))
        age = (
            (datetime.now(timezone.utc) - updated).total_seconds()
            if updated is not None
            else 0.0
        )
        if age >= 20:
            state = "stalled"
            detail = "No model or tool activity detected"
        else:
            state = "orchestrating"
            detail = "Runtime selecting and validating the next action"
    else:
        state = "idle"
        detail = "No active task"

    elapsed = (
        max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))
        if started_at is not None and task_status == "running"
        else 0
    )
    return {
        "state": state,
        "detail": detail,
        "phase": (
            phase or "execution" if task_status == "running" else "idle"
        ),
        "task_id": task_id,
        "task_status": task_status,
        "task_elapsed_seconds": elapsed,
        "objective": " ".join(str(checkpoint.get("objective", "")).split())[:260],
        "step": len(calls),
        "last_tool": last_tool,
        "last_tool_status": last_tool_status,
        "model_processing": processing,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_processed": prompt_processed,
        "generated_tokens": generated_tokens,
        "remaining_tokens": remaining_tokens,
        "generation_tokens_per_second": generation_rate,
        "prompt_tokens_per_second": prompt_rate,
        "last_response_generation_tokens_per_second": (
            float(completed_generation[-1].group(2))
            if completed_generation
            else None
        ),
        "last_response_prompt_tokens_per_second": (
            float(completed_prompt[-1].group(2)) if completed_prompt else None
        ),
        "context_used": context_used,
        "context_size": context_size,
        "context_percent": (
            round(100.0 * context_used / context_size, 2) if context_size else 0.0
        ),
        "events": events,
    }
