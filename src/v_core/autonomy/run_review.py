from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

from .task_contract import TaskContract


_TASK_ID = re.compile(r"^interactive-[A-Za-z0-9_-]{1,160}$")
_HTTP_ERROR = re.compile(r"(?:http(?: status)?\D{0,12})([45]\d\d)\b", re.IGNORECASE)


def _bounded(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _normal_url(value: Any) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path.rstrip("/") or "/",
            parsed.query,
            "",
        )
    )


def _checkpoint_paths(root: Path) -> list[Path]:
    checkpoint_root = Path(root) / "checkpoints"
    try:
        return sorted(
            checkpoint_root.glob("interactive-*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return []


def load_task_checkpoint(
    root: Path,
    *,
    task_id: str = "",
    exclude_task_id: str = "",
) -> dict[str, Any] | None:
    """Load one DARKLINGER-owned checkpoint without accepting arbitrary paths."""

    candidates = _checkpoint_paths(root)
    if task_id:
        if not _TASK_ID.fullmatch(task_id):
            raise ValueError("invalid DARKLINGER interactive task ID")
        candidates = [
            Path(root) / "checkpoints" / f"{task_id}.json"
        ]
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        current_id = str(payload.get("task_id", ""))
        if current_id == exclude_task_id:
            continue
        if not _TASK_ID.fullmatch(current_id):
            continue
        return payload
    return None


def review_task_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a bounded, deterministic post-mortem from runtime evidence.

    The model may explain this report, but it cannot add findings here. Every
    finding is derived from checkpoint fields written by DARKLINGER itself.
    """

    raw_calls = payload.get("tool_calls", [])
    calls = [item for item in raw_calls if isinstance(item, dict)] if isinstance(raw_calls, list) else []
    raw_rollovers = payload.get("context_rollovers", [])
    rollovers = (
        [item for item in raw_rollovers if isinstance(item, dict)]
        if isinstance(raw_rollovers, list)
        else []
    )
    findings: list[dict[str, Any]] = []

    def add(
        code: str,
        severity: str,
        summary: str,
        *,
        tool_calls: list[int] | None = None,
        rollovers: list[int] | None = None,
        evidence: str = "",
    ) -> None:
        finding: dict[str, Any] = {
            "code": code,
            "severity": severity,
            "summary": summary,
        }
        if tool_calls:
            finding["tool_calls"] = tool_calls[:24]
        if rollovers:
            finding["rollovers"] = rollovers[:24]
        if evidence:
            finding["evidence"] = _bounded(evidence, 700)
        findings.append(finding)

    status = str(payload.get("status", "unknown"))
    if status == "running":
        add(
            "unfinished_checkpoint",
            "high",
            "The process ended or the review began while the checkpoint still claimed that the task was running.",
            evidence=f"status={status}; finished_at={payload.get('finished_at')!r}",
        )
    elif status == "interrupted":
        add(
            "interrupted_checkpoint",
            "high",
            "The task ended because its DARKLINGER runtime exited before normal completion.",
            evidence=(
                f"status={status}; finished_at={payload.get('finished_at')!r}"
            ),
        )

    failed_http: list[int] = []
    snapshots_after_failure: list[int] = []
    latest_navigation_failed = False
    for call in calls:
        sequence = int(call.get("sequence") or 0)
        tool = str(call.get("tool", ""))
        status_value = str(call.get("status", ""))
        error = str(call.get("error", ""))
        excerpt = str(call.get("result_excerpt", ""))
        if tool in {"browser_navigate", "web_read"}:
            latest_navigation_failed = status_value == "failed"
        elif tool == "browser_snapshot" and latest_navigation_failed:
            snapshots_after_failure.append(sequence)
        if status_value == "failed" and _HTTP_ERROR.search(error + " " + excerpt):
            failed_http.append(sequence)
    if failed_http:
        add(
            "http_error",
            "medium",
            "One or more browser requests reached an HTTP error page.",
            tool_calls=failed_http,
            evidence="Failed browser calls contained an HTTP 4xx/5xx status.",
        )
    if snapshots_after_failure:
        add(
            "snapshot_after_failed_navigation",
            "high",
            "DARKLINGER requested a page snapshot after the latest navigation had failed, so it could only observe the error page.",
            tool_calls=snapshots_after_failure,
        )

    by_result: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_failure: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_navigation: dict[str, list[int]] = defaultdict(list)
    for call in calls:
        sequence = int(call.get("sequence") or 0)
        tool = str(call.get("tool", ""))
        digest = str(call.get("result_sha256", ""))
        if digest:
            by_result[(tool, digest)].append(sequence)
        if str(call.get("status", "")) == "failed":
            signature = _bounded(call.get("error", ""), 350)
            if signature:
                by_failure[(tool, signature)].append(sequence)
        if tool in {"browser_navigate", "web_read"}:
            arguments = call.get("arguments", {})
            url = arguments.get("url", "") if isinstance(arguments, dict) else ""
            normalized = _normal_url(url)
            if normalized:
                by_navigation[normalized].append(sequence)

    repeated_results = [seqs for seqs in by_result.values() if len(seqs) >= 2]
    if repeated_results:
        flattened = sorted({seq for seqs in repeated_results for seq in seqs})
        add(
            "repeated_identical_result",
            "medium",
            "Different steps produced byte-identical results, but the run continued without treating the repetition as a stall.",
            tool_calls=flattened,
            evidence=f"{len(repeated_results)} repeated result signature(s).",
        )

    repeated_failures = [seqs for seqs in by_failure.values() if len(seqs) >= 2]
    if repeated_failures:
        add(
            "repeated_identical_failure",
            "high",
            "The same tool failure was repeated instead of changing strategy or stopping.",
            tool_calls=sorted({seq for seqs in repeated_failures for seq in seqs}),
        )

    repeated_urls = [seqs for seqs in by_navigation.values() if len(seqs) >= 2]
    if repeated_urls:
        add(
            "revisited_navigation_target",
            "medium",
            "The browser revisited a URL already used in the same task.",
            tool_calls=sorted({seq for seqs in repeated_urls for seq in seqs}),
        )

    oversized_queries: list[int] = []
    for call in calls:
        tool = str(call.get("tool", ""))
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        if tool == "web_search":
            query = str(arguments.get("query", ""))
        elif tool == "browser_navigate":
            url = str(arguments.get("url", ""))
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            if not (parsed.hostname or "").casefold().endswith("duckduckgo.com"):
                continue
            query = parse_qs(parsed.query).get("q", [""])[0]
        else:
            continue
        if len(query) > 180:
            oversized_queries.append(int(call.get("sequence") or 0))
    if oversized_queries:
        add(
            "oversized_search_query",
            "high",
            "A full instruction or other oversized text was used as the search query instead of a focused query.",
            tool_calls=oversized_queries,
        )

    empty_rollovers = [
        int(item.get("sequence") or 0)
        for item in rollovers
        if int(item.get("evidence_count") or 0) == 0
    ]
    if empty_rollovers:
        add(
            "rollover_without_evidence",
            "high",
            "Context rollover recorded no tool evidence, so verified findings could disappear from the model's working context.",
            rollovers=empty_rollovers,
        )

    overflowing_rollovers = [
        int(item.get("sequence") or 0)
        for item in rollovers
        if int(item.get("context_size") or 0) > 0
        and int(item.get("estimated_tokens_after") or 0)
        >= int(item.get("context_size") or 0)
    ]
    if overflowing_rollovers:
        add(
            "rollover_still_over_context",
            "high",
            "Compaction still estimated a prompt at or above the configured context window.",
            rollovers=overflowing_rollovers,
        )

    contract = TaskContract.from_dict(payload.get("requirements"))
    first_satisfied = 0
    for index in range(1, len(calls) + 1):
        if not contract.unmet(calls[:index]):
            first_satisfied = int(calls[index - 1].get("sequence") or index)
            break
    later_calls = [
        int(call.get("sequence") or 0)
        for call in calls
        if first_satisfied and int(call.get("sequence") or 0) > first_satisfied
    ]
    if later_calls:
        add(
            "tooling_after_contract_satisfied",
            "high",
            "The runtime evidence contract was satisfied, but the model was still allowed to call more tools instead of being forced to produce the final report.",
            tool_calls=later_calls,
            evidence=f"Contract first became satisfied after tool call {first_satisfied}.",
        )

    failed_count = sum(call.get("status") == "failed" for call in calls)
    succeeded_count = sum(call.get("status") == "succeeded" for call in calls)
    return {
        "schema_version": 1,
        "task_id": _bounded(payload.get("task_id", ""), 180),
        "objective": _bounded(payload.get("objective", ""), 1_000),
        "status": status,
        "metrics": {
            "tool_calls": len(calls),
            "successful_tool_calls": succeeded_count,
            "failed_tool_calls": failed_count,
            "context_rollovers": len(rollovers),
            "finding_count": len(findings),
        },
        "findings": findings[:32],
        "grounding_rule": (
            "Only describe faults listed in findings. Cite their tool_calls or "
            "rollovers. Do not invent missing events or claim that a repair was applied."
        ),
    }


def review_task(
    root: Path,
    *,
    task_id: str = "",
    exclude_task_id: str = "",
) -> dict[str, Any]:
    payload = load_task_checkpoint(
        root,
        task_id=task_id,
        exclude_task_id=exclude_task_id,
    )
    if payload is None:
        return {
            "schema_version": 1,
            "error": "No matching DARKLINGER interactive task checkpoint was found.",
            "findings": [],
        }
    return review_task_checkpoint(payload)
