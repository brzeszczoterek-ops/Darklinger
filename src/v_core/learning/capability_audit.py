from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import json
from typing import Iterable

from .models import ArtifactKind, SkillManifest, ToolManifest, utc_now


class CapabilityRisk(IntEnum):
    ROUTINE = 0
    REVIEW = 1
    CRITICAL = 2

    @property
    def label(self) -> str:
        return self.name.casefold()


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    kind: ArtifactKind
    name: str
    risk: CapabilityRisk
    reasons: tuple[str, ...]
    primitives: tuple[str, ...]
    creation_origin: str
    owner_notification: bool
    assessed_at: str
    source_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["risk"] = self.risk.label
        return data


_NETWORK_IMPORTS = {
    "aiohttp",
    "ftplib",
    "http",
    "requests",
    "smtplib",
    "socket",
    "ssl",
    "urllib3",
    "websockets",
}
_PROCESS_IMPORTS = {"multiprocessing", "pty", "subprocess"}
_HOST_IMPORTS = {"ctypes", "fcntl", "grp", "pwd", "resource", "signal"}
_FILESYSTEM_IMPORTS = {"os", "pathlib", "shutil", "tempfile"}
_DYNAMIC_CALLS = {"compile", "eval", "exec", "__import__"}
_PROCESS_CALLS = {"popen", "run", "call", "check_call", "check_output", "system"}
_NETWORK_CALLS = {"connect", "create_connection", "request", "urlopen"}
_WRITE_MODES = {"a", "ab", "a+", "w", "wb", "w+", "x", "xb", "x+"}
_SENSITIVE_SKILL_TOOLS = {
    "browser_type",
    "edit_file",
    "full_tor_browser_inventory",
    "full_tor_fetch",
    "full_tor_inventory",
    "full_tor_search",
    "move_file",
    "sandbox_execute_offline",
    "write_file",
}


def _root(module: str) -> str:
    return str(module or "").split(".", 1)[0].casefold()


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id.casefold()
    if isinstance(node, ast.Attribute):
        return node.attr.casefold()
    return ""


def _imports(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(_root(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(_root(node.module or ""))
    result.discard("")
    return result


def _writes_files(node: ast.Call) -> bool:
    if _call_name(node.func) != "open":
        return False
    mode: object = "r"
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = keyword.value.value
    return isinstance(mode, str) and mode.casefold() in _WRITE_MODES


def assess_tool(
    manifest: ToolManifest,
    source: str,
    *,
    privileged: bool,
    creation_origin: str = "unknown",
) -> CapabilityAssessment:
    """Classify executable primitives, not the topic or prose description."""

    source_text = str(source)
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    reasons: set[str] = set()
    primitives: set[str] = set()
    risk = CapabilityRisk.ROUTINE
    try:
        tree = ast.parse(source_text, mode="exec")
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
        reasons.add("source_not_parseable")
        risk = CapabilityRisk.REVIEW

    imported = _imports(tree)
    if imported & _NETWORK_IMPORTS:
        primitives.add("network_access")
        reasons.add("network_primitive")
        risk = max(risk, CapabilityRisk.CRITICAL)
    if imported & _PROCESS_IMPORTS:
        primitives.add("process_control")
        reasons.add("process_primitive")
        risk = max(risk, CapabilityRisk.CRITICAL)
    if imported & _HOST_IMPORTS:
        primitives.add("host_control")
        reasons.add("host_primitive")
        risk = max(risk, CapabilityRisk.CRITICAL)
    if imported & _FILESYSTEM_IMPORTS:
        primitives.add("filesystem_access")
        reasons.add("filesystem_primitive")
        risk = max(risk, CapabilityRisk.REVIEW)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in _DYNAMIC_CALLS:
            primitives.add("dynamic_code")
            reasons.add("dynamic_execution_primitive")
            risk = max(risk, CapabilityRisk.CRITICAL)
        if name in _PROCESS_CALLS:
            primitives.add("process_control")
            reasons.add("process_primitive")
            risk = max(risk, CapabilityRisk.CRITICAL)
        if name in _NETWORK_CALLS:
            primitives.add("network_access")
            reasons.add("network_primitive")
            risk = max(risk, CapabilityRisk.CRITICAL)
        if _writes_files(node):
            primitives.add("filesystem_write")
            reasons.add("filesystem_write_primitive")
            risk = max(risk, CapabilityRisk.REVIEW)

    capabilities = tuple(sorted(manifest.provides_capabilities))
    if capabilities:
        primitives.update(capabilities)
    if privileged and risk is not CapabilityRisk.ROUTINE:
        reasons.add("owner_privileged_generated_code")
    origin = creation_origin if creation_origin in {
        "owner_requested",
        "agent_initiated",
        "unknown",
    } else "unknown"
    if origin == "agent_initiated" and risk >= CapabilityRisk.REVIEW:
        reasons.add("agent_initiated_sensitive_capability")

    return CapabilityAssessment(
        kind=ArtifactKind.TOOL,
        name=manifest.name,
        risk=risk,
        reasons=tuple(sorted(reasons)),
        primitives=tuple(sorted(primitives)),
        creation_origin=origin,
        owner_notification=risk >= CapabilityRisk.REVIEW,
        assessed_at=utc_now(),
        source_sha256=digest,
    )


def assess_skill(
    manifest: SkillManifest,
    *,
    creation_origin: str = "unknown",
) -> CapabilityAssessment:
    required = set(manifest.required_tools)
    sensitive = sorted(required & _SENSITIVE_SKILL_TOOLS)
    risk = CapabilityRisk.REVIEW if sensitive else CapabilityRisk.ROUTINE
    reasons = {"orchestrates_sensitive_tools"} if sensitive else set()
    origin = creation_origin if creation_origin in {
        "owner_requested",
        "agent_initiated",
        "unknown",
    } else "unknown"
    if origin == "agent_initiated" and risk >= CapabilityRisk.REVIEW:
        reasons.add("agent_initiated_sensitive_capability")
    return CapabilityAssessment(
        kind=ArtifactKind.SKILL,
        name=manifest.name,
        risk=risk,
        reasons=tuple(sorted(reasons)),
        primitives=tuple(sensitive),
        creation_origin=origin,
        owner_notification=risk >= CapabilityRisk.REVIEW,
        assessed_at=utc_now(),
        source_sha256=hashlib.sha256(
            json.dumps(manifest.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    )


def notification_events(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [
        dict(record)
        for record in records
        if record.get("owner_notification") is True
    ]


__all__ = [
    "CapabilityAssessment",
    "CapabilityRisk",
    "assess_skill",
    "assess_tool",
    "notification_events",
]
