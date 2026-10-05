"""Read-only preview evidence for local qualification, independent of scoring."""
from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

_RUN_ID = re.compile(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{12}")
_MAX_EVENT_BYTES = 512_000
_MAX_RUN_BYTES = 32_000_000


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class QualificationTrace:
    def __init__(self, runtime_root: Path, output: Callable[[str], None] | None = None):
        self.root = Path(runtime_root) / "qualifications"
        self.output = output
        self.run_id = ""
        self.path: Path | None = None
        self.error = ""
        self.summary: dict[str, Any] = {}

    def start(self, model_path: str, alias: str, total: int, harness_version: int) -> None:
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S-") + uuid4().hex[:12]
        self.path = self.root / f"{self.run_id}.jsonl"
        self.summary = {"run_id": self.run_id, "model_path": model_path, "alias": alias,
                        "started_at": _now(), "state": "unfinished", "total": total, "completed": 0}
        self.emit("run_started", model_path=model_path, alias=alias, total=total,
                  harness_version=harness_version, simulated_tools=True)
        self._print(f"Qualification transcript: {self.path}")

    def _print(self, text: str) -> None:
        if self.output is not None:
            # Preserve visible text without allowing a model to control a terminal.
            text = "".join(c if c in "\n\t" or ord(c) >= 32 and not 127 <= ord(c) <= 159
                           else f"\\x{ord(c):02x}" for c in text)
            try:
                self.output(text)
            except Exception as error:
                self.error = str(error)

    def emit(self, event: str, **data: Any) -> None:
        if self.path is None:
            return
        payload = {"event": event, "timestamp": _now(), "run_id": self.run_id, **data}
        try:
            encoded = json.dumps(payload, ensure_ascii=False, default=str)
            if len(encoded.encode()) > _MAX_EVENT_BYTES:
                # Keep lifecycle evidence; never display clipped text as complete.
                payload = {"event": event, "timestamp": payload["timestamp"],
                           "run_id": self.run_id, "probe": data.get("probe", ""),
                           "request_id": data.get("request_id"), "truncated": True,
                           "detail": "Preview event exceeded 512 KB; payload omitted."}
                encoded = json.dumps(payload)
            self.root.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            self.summary["updated_at"] = payload["timestamp"]
            if data.get("probe"):
                self.summary["current_probe"] = data["probe"]
            if event == "probe_finished":
                self.summary["completed"] = data["index"]
            if event == "run_finished":
                self.summary.update({key: value for key, value in data.items()
                                     if key in {"state", "overall_score", "error"}})
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.summary, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.path.with_suffix(".json"))
        except (OSError, ValueError) as error:
            self.error = str(error)
            self._print(f"Qualification preview unavailable: {error}")
        # Observation failures must never change model grading.
        try:
            if event == "probe_started":
                self._print(f"\nTEST {data['index']}/{data['total']}: {data['probe']}")
            elif event == "request_started":
                self._print(f"Request {data['request_id']} (attempt {data['attempt']}) — waiting for model")
                for message in data.get("messages", []):
                    self._print(f"[{message.get('role', 'message')}]\n{message.get('content', '')}")
                if data.get("tools"):
                    self._print("Available simulated tools:\n" + json.dumps(data["tools"], ensure_ascii=False, indent=2))
            elif event == "response_received":
                reply = data["response"]
                self._print("[model answer]\n" + str(reply.get("content", "")))
                if reply.get("tool_calls"):
                    self._print("[requested simulated actions]\n" + json.dumps(reply["tool_calls"], ensure_ascii=False, indent=2))
                self._print(f"Finish: {reply.get('finish_reason') or 'unknown'}; elapsed {data['latency_ms']} ms")
            elif event == "request_failed":
                self._print(f"Request failed: {data.get('error', '')}")
            elif event == "probe_finished":
                result = data["result"]
                self._print(f"RESULT: {result['score']}/100 — {result['detail']}")
        except Exception:
            pass


def read_qualification_run(runtime_root: Path | None, run_id: str) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid qualification run ID")
    if runtime_root is None:
        raise FileNotFoundError("qualification storage is unavailable")
    path = Path(runtime_root) / "qualifications" / f"{run_id}.jsonl"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("qualification transcript not found")
    events = []
    incomplete = False
    with path.open("rb") as handle:
        raw = handle.read(_MAX_RUN_BYTES + 1)
    truncated = len(raw) > _MAX_RUN_BYTES
    for line in raw[:_MAX_RUN_BYTES].splitlines(keepends=True):
        if not line.endswith(b"\n"):
            incomplete = True
            continue
        if len(line) > _MAX_EVENT_BYTES + 1:
            truncated = True
            continue
        try:
            item = json.loads(line)
        except (ValueError, UnicodeError):
            incomplete = True
            continue
        if isinstance(item, dict) and item.get("run_id") == run_id:
            events.append(item)
    start = next((item for item in events if item.get("event") == "run_started"), {})
    finish = next((item for item in reversed(events) if item.get("event") == "run_finished"), {})
    results = [item for item in events if item.get("event") == "probe_finished"]
    current = next((item.get("probe", "") for item in reversed(events) if item.get("probe")), "")
    return {"run_id": run_id, "model_path": start.get("model_path", ""), "alias": start.get("alias", ""),
            "started_at": start.get("timestamp", ""), "updated_at": events[-1].get("timestamp", "") if events else "",
            "state": finish.get("state", "unfinished"), "total": start.get("total", 0),
            "completed": len(results), "current_probe": current, "overall_score": finish.get("overall_score"),
            "error": finish.get("error", ""), "events": events,
            "truncated": truncated or any(item.get("truncated") for item in events), "incomplete_line": incomplete}


def qualification_preview(runtime_root: Path | None, run_id: str = "") -> dict[str, Any]:
    if run_id and not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid qualification run ID")
    root = Path(runtime_root) / "qualifications" if runtime_root is not None else None
    if root is None or not root.is_dir():
        return {"runs": [], "selected": None}
    paths = sorted((p for p in root.glob("*.jsonl") if _RUN_ID.fullmatch(p.stem) and not p.is_symlink()),
                   key=lambda p: p.name, reverse=True)[:50]
    runs = []
    for path in paths:
        try:
            metadata = path.with_suffix(".json")
            if metadata.is_symlink():
                continue
            if metadata.is_file():
                with metadata.open("rb") as handle:
                    run = json.loads(handle.read(32_000))
                if not isinstance(run, dict) or run.get("run_id") != path.stem:
                    continue
            else:
                run = {key: value for key, value in read_qualification_run(runtime_root, path.stem).items()
                       if key != "events"}
        except (OSError, ValueError):
            continue
        runs.append(run)
    runs.sort(key=lambda run: (str(run.get("started_at", "")), run["run_id"]), reverse=True)
    selected_id = run_id or (runs[0]["run_id"] if runs else "")
    selected = read_qualification_run(runtime_root, selected_id) if selected_id else None
    return {"runs": runs, "selected": selected}
