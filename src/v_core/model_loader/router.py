from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from ..autonomy.task_contract import TaskContract
from .qualification import ModelQualificationCard


TASK_KINDS = {
    "conversation",
    "coding",
    "code_review",
    "research",
    "tool_use",
    "document",
}


@dataclass(frozen=True, slots=True)
class ModelRouteCandidate:
    model_path: str
    card: ModelQualificationCard


@dataclass(frozen=True, slots=True)
class ModelRouteDecision:
    task_kind: str
    selected_model_path: str
    score: int
    fallback_model_paths: tuple[str, ...]
    requirements: tuple[str, ...]
    reason: str


_CODE = re.compile(
    r"(?:\b(?:code|coding|compile|debug|function|implement|program|python|refactor|"
    r"script|source|test|typescript)\b|\b(?:kod|funkcj|napisz|program|skrypt|"
    r"test|zaimplementuj)\w*\b|\.(?:c|cpp|go|js|py|rs|sol|ts)\b)",
    re.IGNORECASE,
)
_DOCUMENT = re.compile(
    r"\b(?:archive|catalog|classify|document|extract|index|summari[sz]e|"
    r"archiwiz\w*|dokument\w*|katalog\w*|klasyfik\w*|podsum\w*|stre[śs]c\w*)\b",
    re.IGNORECASE,
)


def classify_model_task(prompt: str, contract: TaskContract | None = None) -> str:
    """Map an owner request to a capability class without asking a worker model."""

    task_contract = contract or TaskContract.from_prompt(prompt)
    if (
        task_contract.requires_created_tool
        or task_contract.requires_created_artifact
    ):
        return "coding"
    if any(
        (
            task_contract.requires_browser_navigation,
            task_contract.requires_browser_snapshot,
            task_contract.requires_web_discovery,
            task_contract.requires_distinct_detail_page,
            any(
                name in {
                    "full_tor_search",
                    "full_tor_fetch",
                    "full_tor_inventory",
                    "full_tor_browser_inventory",
                }
                for name in task_contract.required_tools
            ),
        )
    ):
        return "research"
    if (
        _CODE.search(prompt)
        and task_contract.requires_file_read
        and not task_contract.requires_file_mutation
        and TaskContract.requests_read_only(prompt)
    ):
        return "code_review"
    if _CODE.search(prompt):
        return "coding"
    if any(
        (
            task_contract.requires_file_read,
            task_contract.requires_file_mutation,
            task_contract.requires_command_execution,
            task_contract.requires_created_skill,
            task_contract.requires_created_artifact,
            bool(task_contract.required_tools),
            bool(task_contract.required_capabilities),
        )
    ):
        return "tool_use"
    if _DOCUMENT.search(prompt):
        return "document"
    return "conversation"


def classify_model_phase(
    prompt: str,
    contract: TaskContract | None = None,
    successful_calls: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Select the specialist needed by the current executor-owned phase.

    A mixed objective can begin with browser discovery, move into generated
    source, and finally execute the activated tool. Looking only at the owner
    prompt collapses all three into one label. The contract ledger is the
    authoritative phase boundary; model prose never gets to request a model.
    """

    task_contract = contract or TaskContract.from_prompt(prompt)
    calls = [dict(call) for call in successful_calls]
    missing = set(task_contract.unmet(calls))

    if any(item.startswith("browser_") for item in missing) or any(
        item.startswith("public_fact:") for item in missing
    ):
        return "research"
    if set(task_contract.required_tools).intersection(missing) or any(
        item.startswith("capability:") for item in missing
    ):
        return "tool_use"
    if missing.intersection(
        {
            "learning_create_tool",
            "learning_create_skill",
            "learning_create_tool_or_skill",
        }
    ):
        return "coding"
    if "generated_tool_execution" in missing:
        return "tool_use"
    return classify_model_task(prompt, task_contract)


class ModelRouter:
    """Select only from locally qualified cards using runtime-owned weights."""

    _WEIGHTS: dict[str, dict[str, int]] = {
        "conversation": {
            # A conversational turn is the one place where preserving V is the
            # primary capability.  Grounding and recovery still matter, but
            # they must not let a technically strong coder outrank a model that
            # actually holds the persona in ordinary dialogue.
            "conversation": 30,
            "persona": 50,
            "instruction_following": 10,
            "execution_honesty": 5,
            "context_recovery": 3,
            "prompt_injection_resistance": 2,
        },
        "coding": {
            "coding": 35,
            "structured_output": 15,
            "tool_calling": 10,
            "instruction_following": 5,
            "execution_honesty": 10,
            "agentic_control": 10,
            "recovery": 15,
        },
        "code_review": {
            # Read-only review rewards factual preservation and resistance to
            # prompt/data confusion more than source-generation fluency. This
            # keeps a strong tool-generating model from automatically winning
            # static analysis when a more grounded reviewer is available.
            "coding": 10,
            "instruction_following": 10,
            "grounding": 25,
            "execution_honesty": 10,
            "context_recovery": 15,
            "prompt_injection_resistance": 30,
        },
        "research": {
            "research": 25,
            "tool_calling": 15,
            "structured_output": 5,
            "grounding": 15,
            "execution_honesty": 10,
            "agentic_control": 15,
            "context_recovery": 5,
            "prompt_injection_resistance": 10,
        },
        "tool_use": {
            "tool_calling": 30,
            "structured_output": 15,
            "instruction_following": 5,
            "conversation": 5,
            "execution_honesty": 15,
            "agentic_control": 20,
            "recovery": 10,
        },
        "document": {
            "structured_output": 20,
            "instruction_following": 15,
            "coding": 10,
            "conversation": 5,
            "grounding": 20,
            "execution_honesty": 5,
            "agentic_control": 10,
            "context_recovery": 10,
            "prompt_injection_resistance": 5,
        },
    }

    def choose(
        self,
        prompt: str,
        candidates: Iterable[ModelRouteCandidate],
        *,
        current_model_path: str = "",
        contract: TaskContract | None = None,
        task_kind: str | None = None,
        strategy: str = "automatic",
        switch_margin: int = 5,
        excluded_model_paths: Iterable[str] = (),
    ) -> ModelRouteDecision | None:
        task_kind = task_kind or classify_model_phase(prompt, contract)
        if task_kind not in TASK_KINDS:
            raise ValueError(f"unknown model task kind: {task_kind}")
        strategy = str(strategy).strip().casefold()
        if strategy not in {"automatic", "manual_hierarchy"}:
            raise ValueError(f"unknown model routing strategy: {strategy}")
        if not 0 <= int(switch_margin) <= 100:
            raise ValueError("model switch margin must be between 0 and 100")
        weights = self._WEIGHTS[task_kind]
        excluded = {str(path) for path in excluded_model_paths}
        ranked: list[tuple[int, str, ModelQualificationCard]] = []
        for candidate in candidates:
            if candidate.model_path in excluded:
                continue
            score = _weighted_score(candidate.card, weights)
            ranked.append((score, candidate.model_path, candidate.card))
        if not ranked:
            return None
        if strategy == "automatic":
            ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
        best_score, selected, _ = ranked[0]
        current = next(
            (item for item in ranked if item[1] == current_model_path),
            None,
        )
        kept_current = bool(
            strategy == "automatic"
            and current is not None
            and selected != current_model_path
            and best_score - current[0] < switch_margin
        )
        if kept_current and current is not None:
            selected_score, selected, _ = current
        else:
            selected_score = best_score
        fallbacks = tuple(item[1] for item in ranked if item[1] != selected)
        requirements = tuple(weights)
        return ModelRouteDecision(
            task_kind=task_kind,
            selected_model_path=selected,
            score=selected_score,
            fallback_model_paths=fallbacks,
            requirements=requirements,
            reason=(
                "selected by the owner-defined DARKLINGER-Full model hierarchy"
                if strategy == "manual_hierarchy"
                else
                (
                    f"kept current model because the best verified {task_kind} "
                    f"score improved by less than the {switch_margin}-point "
                    "hot-swap margin"
                )
                if kept_current
                else (
                    f"highest verified {task_kind} capability score across "
                    f"{len(ranked)} local candidate(s)"
                )
            ),
        )


def _weighted_score(card: ModelQualificationCard, weights: dict[str, int]) -> int:
    total = sum(weights.values())
    return round(
        sum(card.score(capability) * weight for capability, weight in weights.items())
        / total
    )


def render_route_decision(decision: ModelRouteDecision) -> dict[str, Any]:
    return {
        "task_kind": decision.task_kind,
        "selected_model_path": decision.selected_model_path,
        "score": decision.score,
        "fallback_model_paths": list(decision.fallback_model_paths),
        "requirements": list(decision.requirements),
        "reason": decision.reason,
    }
