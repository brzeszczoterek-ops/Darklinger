"""Runtime-owned, claim-specific provenance for model-written memories.

Exact normalized evidence is deliberately conservative: paraphrases remain
proposals. A tool result is an observation, never automatic truth verification.
Legacy source labels alone cannot grant authority to newly derived content.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .models import MemorySource


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _get(entry: Any, name: str, default: Any = None) -> Any:
    return entry.get(name, default) if isinstance(entry, dict) else getattr(entry, name, default)


def claims(entry: Any) -> list[str]:
    if _get(entry, "content") is not None:
        values = [_get(entry, "content")]
    else:
        values = [_get(entry, "summary", ""), _get(entry, "lesson", "")]
        values += _get(entry, "lessons", []) if isinstance(_get(entry, "lessons", []), list) else []
    return list(dict.fromkeys(text for value in values if (text := _text(value))))


def _receipt(claim: str, source: MemorySource, reference: str) -> dict[str, str]:
    return {"claim": claim, "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
            "source": source.value, "reference": reference}


def ground_reflection(entry: Any, task: str, execution: dict[str, Any] | None) -> None:
    supports = [_receipt(_text(task), MemorySource.DIRECTLY_TOLD, "owner_task")]
    for index, call in enumerate((execution or {}).get("tool_calls", [])):
        if not isinstance(call, dict) or call.get("status") != "succeeded":
            continue
        observed = _text(call.get("result_excerpt", ""))
        if observed:
            supports.append(_receipt(observed, MemorySource.OBSERVED,
                                     f"tool_call:{index}:{call.get('tool', '')}"))
    _assign(entry, supports)


def ground_derived(entry: Any, parents: Iterable[Any]) -> None:
    supports = []
    for parent in parents:
        parent_source = _get(parent, "source")
        if parent_source not in {MemorySource.DIRECTLY_TOLD, MemorySource.OBSERVED}:
            continue
        parent_claims = claims(parent)
        for ref in _get(parent, "evidence_refs", []):
            if not isinstance(ref, dict):
                continue
            claim = ref.get("claim", "")
            if (claim in parent_claims and ref.get("source") == MemorySource(parent_source).value
                    and ref.get("claim_sha256") == hashlib.sha256(claim.encode()).hexdigest()
                    and ref.get("reference")):
                supports.append(ref)
    _assign(entry, supports)


def _assign(entry: Any, supports: list[dict[str, str]]) -> None:
    wanted = claims(entry)
    matched = []
    for claim in wanted:
        ref = next((ref for ref in supports if ref["claim"] == claim), None)
        if ref is None:
            entry.source, entry.evidence_refs = MemorySource.SELF_GENERATED, []
            return
        matched.append(dict(ref))
    # Mixed origins cannot be flattened into a stronger single source label.
    origins = {ref["source"] for ref in matched}
    if not wanted or len(origins) != 1:
        entry.source, entry.evidence_refs = MemorySource.SELF_GENERATED, []
        return
    entry.source = MemorySource(next(iter(origins)))
    entry.evidence_refs = matched
