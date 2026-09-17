"""Bounded local fixture checks, not a general tool execution planner.

Passing one fixture is evidence for that case only. Unknown tools are never run.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4


_WRITE_TOOLS = {"write_file", "create_directory", "edit_file", "move_file"}
_CASES = {
    "read_file": "read_known_utf8_content",
    "write_file": "write_new_file_and_verify_content",
    "list_directory": "list_exact_fixture_entry",
    "create_directory": "create_empty_directory",
    "edit_file": "replace_exact_text_in_fixture",
    "move_file": "move_fixture_and_preserve_content",
    "search_files": "find_exact_filename_among_decoys",
    "directory_tree": "describe_nested_fixture_tree",
    "get_file_info": "match_fixture_size_and_type",
}


async def test_local_tools(tools, definitions, trace=None, *, timeout=10, prompt="", allow_write=True):
    names = sorted({item.get("function", {}).get("name") for item in definitions
                    if isinstance(item, dict) and isinstance(item.get("function"), dict)
                    and isinstance(item["function"].get("name"), str)})
    explicit = [name for name in names if re.search(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)", prompt, re.IGNORECASE)]
    names = explicit or names
    results = []
    root = getattr(tools, "workspace", None)
    direct = getattr(tools, "_call_direct", None)
    root = Path(root).resolve() if root is not None else None
    for name in names:
        if name in _WRITE_TOOLS and not allow_write:
            results.append({"tool": name, "status": "not_tested",
                            "reason": "Owner requested read-only checks; write provider not called."})
            continue
        try:
            learning = getattr(tools, "learning", None)
            extension = getattr(tools, "edition_extension", None)
            generated_override = (
                learning is not None and name in learning.active_tool_names()
            )
            edition_override = (
                extension is not None and extension.handles_tool(name)
            )
        except Exception as error:
            results.append({"tool": name, "status": "not_tested",
                            "reason": "Provider identity check failed: " + type(error).__name__})
            continue
        if generated_override:
            results.append({"tool": name, "status": "not_tested",
                            "reason": "Provider overrides built-in fixture tool; execution skipped."})
            continue
        core_self_test = getattr(tools, "self_test_tool", None)
        if callable(core_self_test):
            try:
                fixture_result = await asyncio.wait_for(core_self_test(name), timeout)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                results.append({
                    "tool": name,
                    "status": "failed",
                    "reason": "Core functional fixture failed: " + type(error).__name__,
                    "evidence_level": "deterministic_offline_fixture",
                    "live_network": False,
                })
                continue
            if isinstance(fixture_result, dict):
                results.append(fixture_result)
                continue
        if edition_override:
            self_test = getattr(extension, "self_test_tool", None)
            if not callable(self_test):
                results.append({"tool": name, "status": "not_tested",
                                "reason": "No approved edition fixture; no live fallback."})
                continue
            try:
                fixture_result = await asyncio.wait_for(self_test(name), timeout)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                results.append({
                    "tool": name,
                    "status": "failed",
                    "reason": "Edition functional fixture failed: " + type(error).__name__,
                    "evidence_level": "deterministic_offline_fixture",
                    "live_network": False,
                })
                continue
            if isinstance(fixture_result, dict):
                results.append(fixture_result)
                continue
            results.append({"tool": name, "status": "not_tested",
                            "reason": "No approved edition fixture; no live fallback."})
            continue
        if root is None or not callable(direct):
            results.append({"tool": name, "status": "not_tested",
                            "reason": "Local fixture provider unavailable."})
            continue
        if not root.is_dir():
            results.append({"tool": name, "status": "not_tested",
                            "reason": "Configured workspace unavailable."})
            continue
        if name not in _CASES:
            results.append({"tool": name, "status": "not_tested",
                            "reason": "No approved local functional fixture; no live fallback."})
            continue
        # Each tool gets separate disposable data: edits/moves cannot pollute
        # later checks. This is fixture isolation, not an OS sandbox.
        try:
            temporary = TemporaryDirectory(prefix=".darklinger-selftest-", dir=root)
        except OSError as error:
            results.append({"tool": name, "status": "not_tested",
                            "reason": "Temporary fixture directory unavailable: " + type(error).__name__})
            continue
        with temporary as folder:
            try:
                fixture = _prepare_fixture(name, Path(folder))
            except OSError as error:
                results.append({"tool": name, "status": "not_tested",
                                "reason": "Fixture preparation failed: " + type(error).__name__})
                continue
            results.append(await _run_fixture(tools, name, fixture, trace, timeout))
    return _report(results)


@dataclass
class _Fixture:
    arguments: dict
    directory: Path
    source: Path
    target: Path
    marker: str


def _prepare_fixture(name, directory):
    marker = "darklinger-fixture-" + uuid4().hex + "-zażółć"
    source = directory / ("input-" + uuid4().hex + ".txt")
    source.write_text(marker, encoding="utf-8")
    target = directory / "output.txt"
    arguments = {"path": str(source)}
    if name in {"list_directory", "directory_tree", "search_files"}:
        arguments = {"path": str(directory)}
    if name == "write_file":
        arguments = {"path": str(target), "content": marker}
    elif name == "create_directory":
        target = directory / "created"
        arguments = {"path": str(target)}
    elif name == "edit_file":
        arguments["edits"] = [{"oldText": marker, "newText": marker + "-edited"}]
        arguments["dry_run"] = False
    elif name == "move_file":
        arguments = {"source": str(source), "destination": str(target)}
    elif name == "search_files":
        (directory / "decoy.txt").write_text("not a match", encoding="utf-8")
        arguments["pattern"] = source.name
    elif name == "directory_tree":
        nested = directory / "nested"
        nested.mkdir()
        (nested / "child.txt").write_text("child", encoding="utf-8")
    return _Fixture(arguments, directory, source, target, marker)


async def _run_fixture(tools, name, fixture, trace, timeout):
    row = {"tool": name, "case": _CASES[name]}
    arguments = fixture.arguments
    try:
        normalized = tools.normalize_arguments(name, deepcopy(arguments))
    except Exception as error:
        return dict(row, status="not_tested", reason="Fixture path validation failed: " + type(error).__name__)
    if normalized != arguments:
        return dict(row, status="not_tested", reason="Argument mapping changed the prepared fixture.")
    sequence = trace.tool_started(name, deepcopy(arguments)) if trace else None
    try:
        raw = await asyncio.wait_for(tools._call_direct(name, deepcopy(arguments)), timeout)
        detect = getattr(tools, "_recovery_failure", lambda *_: "")
        passed = not detect(str(raw), name) and _verify_fixture(name, fixture, raw)
        reason = "Prepared fixture matched expected result." if passed else "Fixture result did not match expectations."
        if trace:
            trace.tool_finished(sequence, str(raw), error=None if passed else reason)
        return dict(row, status="passed" if passed else "failed", reason=reason)
    except asyncio.CancelledError:
        if trace:
            trace.tool_finished(sequence, "", error="Functional check cancelled.")
        raise
    except Exception as error:
        reason = "Functional check failed: " + type(error).__name__
        if trace:
            trace.tool_finished(sequence, "", error=reason)
        return dict(row, status="failed", reason=reason)


def _file_matches(path, content):
    return not path.is_symlink() and path.is_file() and path.read_bytes() == content.encode("utf-8")


def _verify_fixture(name, fixture, raw):
    source, target, marker = fixture.source, fixture.target, fixture.marker
    if name == "move_file":
        return not source.exists() and not source.is_symlink() and _file_matches(target, marker)
    if name == "edit_file":
        return _file_matches(source, marker + "-edited")
    if not _file_matches(source, marker):
        return False
    if name == "write_file":
        return _file_matches(target, marker)
    if name == "create_directory":
        return not target.is_symlink() and target.is_dir() and not list(target.iterdir())
    if name == "read_file":
        return isinstance(raw, str) and raw == marker
    if name == "list_directory":
        return _listing_matches(raw, source.name)
    if name == "search_files":
        return isinstance(raw, str) and raw.splitlines() == [str(source)]
    if name == "directory_tree":
        if not isinstance(raw, str):
            return False
        tree = json.loads(raw)
        expected = [{"name": source.name, "type": "file"},
                    {"name": "nested", "type": "directory", "children": [
                        {"name": "child.txt", "type": "file"}]}]
        return isinstance(tree, list) and sorted(tree, key=lambda row: row["name"]) == sorted(expected, key=lambda row: row["name"])
    if name == "get_file_info":
        if not isinstance(raw, str):
            return False
        fields = {}
        for line in raw.splitlines():
            key, separator, value = line.partition(":")
            if not separator or key in fields:
                return False
            fields[key] = value.strip()
        return (fields.get("size") == str(len(marker.encode("utf-8")))
                and fields.get("isFile") == "true"
                and fields.get("isDirectory") == "false")
    return False


def _listing_matches(raw, expected_name):
    """Accept exact entries, never a filename embedded in an error or prose.

    The directory fixture contains one file. The built-in MCP provider returns
    text blocks with [FILE] prefixes; local providers may return plain lines.
    Unknown result formats fail verification instead of being coerced to text.
    """
    blocks = [raw] if isinstance(raw, str) else raw
    if not isinstance(blocks, list) or not all(isinstance(block, str) for block in blocks):
        return False
    entries = [line.strip().removeprefix("[FILE] ")
               for block in blocks for line in block.splitlines() if line.strip()]
    return entries == [expected_name]


def _report(results):
    counts = {status: sum(row["status"] == status for row in results)
              for status in ("passed", "failed", "not_tested")}
    lines = ["Boss, these are bounded functional checks, not a certification of every tool.",
             "Approved deterministic fixtures only; not an OS sandbox. Network-capable tools may be exercised without live network access. No repairs or live fallback."]
    lines.extend(f"- {row['tool']}: {row['status']} — {row['reason']}" for row in results)
    lines.append("A passing fixture confirms only that tested case. Untested tools remain unverified.")
    return "\n\n".join(lines), {"scope": "bounded_functional_fixtures", "results": results,
                                     "counts": counts, "functional_tests": counts["passed"] + counts["failed"],
                                     "live_network_tests": sum(bool(row.get("live_network")) for row in results)}
