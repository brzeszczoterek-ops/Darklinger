"""Read-only, bounded views of this runtime's memoirs and persona.

No arbitrary path API, model calls, memory reconstruction, or disk writes.
Archive access is enforced here, not by hiding a frontend control.
"""
from __future__ import annotations

import json
import os
import re
import stat
from typing import Any


_SESSION_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}")
_MAX_BYTES = 2 * 1024 * 1024


def _text(value: Any, limit: int = 12000) -> str:
    if not isinstance(value, str):
        return ""
    return value if len(value) <= limit else value[:limit] + "\n[… view truncated]"


def _view(data: dict) -> dict:
    turns = data.get("turns", [])
    turns = turns if isinstance(turns, list) else []
    return {
        "session_id": _text(data.get("session_id"), 80),
        "started_at": _text(data.get("started_at"), 80),
        "source": _text(data.get("source"), 80),
        # Model grounding checks never upgrade prose into verified evidence.
        "verified": False,
        "memory": _text(data.get("memory")),
        "fallback_reason": _text(data.get("fallback_reason"), 200),
        "partial_excerpts": data.get("partial_excerpts") is True,
        "turn_ids": [i for i in data.get("turn_ids", []) if type(i) is int][:100]
                    if isinstance(data.get("turn_ids"), list) else [],
        "omitted_turns": max(0, len(turns) - 50),
        "turns": [{"id": t.get("id") if type(t.get("id")) is int else None,
                   "at": _text(t.get("at"), 80), "status": _text(t.get("status"), 80),
                   "user": _text(t.get("user")), "assistant": _text(t.get("assistant"))}
                  for t in turns[-50:] if isinstance(t, dict)],
    }


def _read_archive(memoir: Any, session_id: str) -> dict:
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError("Invalid session ID.")
    # Pin the directory and reject symlinks, including a swapped leaf file.
    directory = os.open(memoir.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(session_id + ".json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                raise ValueError("The file is not a regular file or exceeds the 2 MiB view limit.")
            raw = handle.read(_MAX_BYTES + 1)
            if len(raw) > _MAX_BYTES:
                raise ValueError("The 2 MiB view limit was exceeded.")
            data = json.loads(raw)
    finally:
        os.close(directory)
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("kind") != "session_memoir" or data.get("session_id") != session_id):
        raise ValueError("Unsupported or corrupt memoir archive.")
    return _view(data)


def browse(runtime: Any, *, view: str, session_id: str = "current") -> dict:
    edition = getattr(runtime.config.edition, "name", "public")
    archive_allowed = edition in {"full", "full_access"}
    memoir = runtime.memoir
    if view == "persona":
        agent = getattr(runtime.core, "agent", None)
        persona = getattr(agent, "persona", None)
        sections = []
        if persona is not None:
            for field, label in (("identity", "Identity"), ("voice", "Voice"),
                                 ("constitution", "Persona principles")):
                component = getattr(persona, field, None)
                if component is not None:
                    sections.append({"title": label, "text": _text(component.render(), 32000)})
            relationship = getattr(getattr(agent, "memory", None), "relationship_state", None)
            if archive_allowed and relationship is not None:
                sections.append({"title": "Current persona context for the model (preview)",
                                 "text": _text(persona.build_runtime(relationship), 32000)})
        return {"sections": sections, "delivery_verified": False,
                "notice": "Configuration preview, not the full prompt or proof of delivery to the model. "
                          "Public does not show relationship data from previous sessions."}
    if view == "sessions":
        sessions = []
        limited = False
        if memoir is not None:
            sessions.append({"id": "current", "label": "Current session", "started_at": memoir.started})
            if archive_allowed:
                try:
                    if memoir.root.is_symlink():
                        raise ValueError("The archive must not be a symbolic link.")
                    with os.scandir(memoir.root) as entries:
                        names = []
                        for index, entry in enumerate(entries):
                            if index >= 1000:
                                limited = True
                                break
                            sid = entry.name.removesuffix(".json")
                            if (entry.name.endswith(".json") and _SESSION_ID.fullmatch(sid)
                                    and entry.is_file(follow_symlinks=False)
                                    and sid != memoir.session_id):
                                names.append(sid)
                        limited = limited or len(names) > 100
                        sessions.extend({"id": sid, "label": sid, "started_at": ""}
                                        for sid in sorted(names, reverse=True)[:100])
                except FileNotFoundError:
                    pass
        return {"sessions": sessions, "archive_allowed": archive_allowed, "limited": limited}
    if view != "session":
        raise ValueError("Unknown view.")
    if session_id != "current" and not archive_allowed:
        raise PermissionError("Public provides access to the current session only.")
    if memoir is None:
        return {"available": False}
    if session_id != "current":
        return {"available": True, "record": _read_archive(memoir, session_id), "save_state": "saved"}
    record = _view({"session_id": memoir.session_id, "started_at": memoir.started,
                    "turns": memoir.turns, "source": "current_ui_session"})
    if memoir.saved:
        record = _read_archive(memoir, memoir.session_id)
    return {"available": True, "record": record,
            "save_state": _text(memoir.state.get("state"), 80)}
