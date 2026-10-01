"""Explicit owner preferences, separate from the installer's working defaults."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def read_preferences(root: Path) -> dict[str, Any]:
    path = root / "speech_settings.json"
    if not path.exists():
        return {}
    if path.stat().st_size > 32_768:
        raise ValueError("Speech settings file is too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Speech settings must be an object")
    return data


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
