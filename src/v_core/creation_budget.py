"""Bounded retry accounting from executor receipts, never model-written reports.

Novel code/model identity can buy a single probe, not a claim of progress.
Only comparable, independently specified offline tests can reset stagnation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from .generated_tool_contract import GeneratedToolContract
from .learning.source_builder import build_source_blueprint, json_assignments


CREATION_TOOLS = frozenset({"learning_create_tool", "learning_create_skill",
    "learning_create_snapshot_extractor", "learning_create_repair_adapter"})
SOURCE_DRAFT_TOOL = "runtime_validate_generated_tool_source"
CREATION_ATTEMPT_TOOLS = CREATION_TOOLS | frozenset({SOURCE_DRAFT_TOOL})
STAGNATION_LIMIT = 3
TOTAL_ATTEMPT_LIMIT = 8
TOTAL_SECONDS = 600.0


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_candidate(
    source: str,
    objective: str,
    *,
    generated_contract: GeneratedToolContract | None = None,
) -> dict[str, Any] | None:
    """Read-only eligibility: immutable user fixture, no candidate-derived oracle."""
    try:
        if len(source) > 100_000:
            return None
        blueprint = build_source_blueprint(
            source=source,
            objective=objective,
            generated_contract=generated_contract,
        )
        assignments = json_assignments(objective)
        if blueprint.expected is None or (
            generated_contract is None
            and any(key not in assignments for key in blueprint.arguments)
        ):
            return None
        tree = ast.parse(source)
        # Reject executable dependencies from this narrow local progress path.
        # This is eligibility, not a replacement for the existing source policy.
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
            return None
        # Comments/formatting, docstrings and mere diagnostic wording are not
        # novel approaches. A changed AST is only permission for one probe.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                node.body = [part for part in node.body if not (
                    isinstance(part, ast.Expr) and isinstance(part.value, ast.Constant)
                    and isinstance(part.value.value, str))]
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = "<text>"
        test_contract = [
            {"arguments": arguments, "expected": expected}
            for arguments, expected in (
                blueprint.tests or ((blueprint.arguments, blueprint.expected),)
            )
        ]
        expected_digests = [digest(item["expected"]) for item in test_contract]
        return {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "approach_sha256": digest(ast.dump(tree, include_attributes=False)),
                "contract_sha256": digest(test_contract),
                "expected_sha256": (
                    expected_digests[0]
                    if len(expected_digests) == 1
                    else digest(expected_digests)
                ),
                "test_expected_sha256": expected_digests}
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return None


def validation_receipt(candidate: dict | None, report: dict) -> dict | None:
    if not candidate or report.get("network") != "offline" or report.get("sandbox") != "bubblewrap":
        return None
    tests = report.get("tests", [])
    expected_digests = candidate.get("test_expected_sha256", [])
    if (
        not isinstance(expected_digests, list)
        or len(tests) != len(expected_digests)
        or not tests
        or not all(isinstance(test, dict) for test in tests)
    ):
        return None
    ranks = []
    for test, expected_digest in zip(tests, expected_digests, strict=True):
        rank = {"fixture_validation": 1, "execution": 2, "comparison": 3}.get(test.get("stage"), 0)
        if rank >= 2 and test.get("expected_sha256") != expected_digest:
            return None
        if test.get("passed") is True and test.get("status") == "passed":
            rank = 4
        ranks.append(rank)
    rank = min(ranks)
    return {**candidate, "rank": rank, "origin": "local_executor"}


@dataclass
class Budget:
    attempts: int = 0
    failures: int = 0
    stagnant: int = 0
    contract: str = ""
    highwater: int = 0
    progress_resets: int = 0
    approaches: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    code_probe_used: bool = False
    model_probe_used: bool = False
    remaining_seconds: float = TOTAL_SECONDS

    @property
    def hard_reason(self) -> str:
        if self.attempts >= TOTAL_ATTEMPT_LIMIT:
            return "attempt_budget"
        return ""

    @property
    def stop_reason(self) -> str:
        if self.hard_reason:
            return self.hard_reason
        if self.stagnant >= STAGNATION_LIMIT and (
            not self.contract or (self.code_probe_used and self.model_probe_used)
        ):
            return "no_verified_progress"
        return ""

    def admit(self, candidate: dict | None, model: str) -> tuple[bool, str]:
        if self.hard_reason:
            return False, self.hard_reason
        if self.stagnant < STAGNATION_LIMIT:
            return True, "ordinary_attempt"
        if not candidate or candidate.get("contract_sha256") != self.contract:
            return False, "no_comparable_contract"
        if (not self.code_probe_used and candidate.get("approach_sha256")
                and candidate["approach_sha256"] not in self.approaches):
            return True, "changed_code_probe"
        if not self.model_probe_used and self.models and model and model not in self.models:
            return True, "changed_model_probe"
        return False, "no_new_approach"


def evaluate(calls: list[dict], *, now: datetime | None = None) -> Budget:
    state = Budget()
    started = None
    for call in calls:
        if call.get("tool") not in CREATION_ATTEMPT_TOOLS:
            continue
        state.attempts += 1
        try:
            timestamp = datetime.fromisoformat(call.get("started_at", ""))
            if timestamp.tzinfo is not None:
                started = timestamp if started is None else min(started, timestamp)
        except (TypeError, ValueError):
            pass
        probe = call.get("creation_probe")
        state.code_probe_used |= probe == "changed_code_probe"
        state.model_probe_used |= probe == "changed_model_probe"
        receipt = call.get("creation_validation")
        improved = False
        if isinstance(receipt, dict) and receipt.get("origin") == "local_executor":
            contract = receipt.get("contract_sha256", "")
            rank = receipt.get("rank", 0)
            if isinstance(rank, int) and 0 <= rank <= 4 and contract:
                if not state.contract:
                    state.contract = contract
                    state.highwater = rank
                elif contract == state.contract and rank > state.highwater:
                    state.highwater = rank
                    state.stagnant = 0
                    state.progress_resets += 1
                    improved = True
                if contract == state.contract:
                    state.approaches.add(receipt.get("approach_sha256", ""))
        candidate = call.get("creation_candidate")
        if isinstance(candidate, dict) and candidate.get("contract_sha256") == state.contract:
            state.approaches.add(candidate.get("approach_sha256", ""))
        model = call.get("creation_model", "")
        if model:
            state.models.add(model)
        if call.get("status") == "failed":
            state.failures += 1
            if not improved:
                state.stagnant += 1
    if started is not None:
        elapsed = ((now or datetime.now(timezone.utc)) - started).total_seconds()
        state.remaining_seconds = max(0.0, TOTAL_SECONDS - max(0.0, elapsed))
    return state
