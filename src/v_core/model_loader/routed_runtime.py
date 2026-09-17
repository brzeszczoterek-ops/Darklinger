from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import ModelProfile
from .qualification import ModelQualificationCard
from .router import (
    ModelRouteCandidate,
    ModelRouteDecision,
    ModelRouter,
    classify_model_phase,
)
from .runtime import (
    LlamaServerSession,
    LlamaServerStartError,
    find_llama_server,
    start_llama_server,
)
from .storage import ModelLoaderStore


@dataclass(frozen=True, slots=True)
class ModelSwitchResult:
    decision: ModelRouteDecision | None
    previous_model_path: str
    active_model_path: str
    switched: bool
    failures: tuple[str, ...] = ()
    requested_task_kind: str = ""
    configured_candidates: int = 0
    eligible_candidates: int = 0
    routing_strategy: str = "automatic"


class RoutedModelRuntime:
    """Own one local server and replace it only between top-level user turns."""

    def __init__(
        self,
        session: LlamaServerSession,
        runtime_root: Path,
        llm: Any,
        *,
        status: Callable[[str], None] = print,
        allow_manual_hierarchy: bool = False,
    ) -> None:
        self.session = session
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.store = ModelLoaderStore(self.runtime_root)
        self.llm = llm
        self.status = status
        self.allow_manual_hierarchy = bool(allow_manual_hierarchy)
        self.router = ModelRouter()
        self._journal_path = self.runtime_root / "routing.jsonl"
        self._last_unavailable_notice = ""

    @property
    def active_model_path(self) -> str:
        return str(Path(self.session.profile.model_path).expanduser().resolve())

    async def ensure_for(
        self,
        prompt: str,
        *,
        task_kind: str | None = None,
        excluded_model_paths: Iterable[str] = (),
        trigger: str = "task_route",
    ) -> ModelSwitchResult:
        state = self.store.load()
        previous = self.active_model_path
        requested_task_kind = task_kind or classify_model_phase(prompt)
        inference = getattr(self.llm, "inference", None)
        if inference is not None:
            if trigger == "task_route":
                inference.begin_turn(requested_task_kind)
            else:
                inference.select_profile(
                    inference.profile_for_task(requested_task_kind)
                )
        server_tuning = (
            inference.consume_server_tuning()
            if inference is not None
            else None
        )
        if not state.routing_enabled or not state.routing_model_paths:
            if server_tuning is not None:
                return await self._restart_active_with_tuning(
                    state,
                    server_tuning.values(),
                    prompt=prompt,
                    task_kind=requested_task_kind,
                    trigger=trigger,
                )
            return ModelSwitchResult(None, previous, previous, False)

        routing_strategy = (
            state.routing_strategy
            if self.allow_manual_hierarchy
            and state.routing_strategy == "manual_hierarchy"
            else "automatic"
        )
        candidates, rejected = self._current_candidates_with_rejections(state)
        configured_count = len(state.routing_model_paths[:3])
        eligible_count = len(candidates)
        if not candidates:
            detail = ", ".join(
                f"{Path(path).name} ({'|'.join(reasons)})"
                for path, reasons in rejected.items()
            )
            reason = (
                "automatic model routing unavailable: none of the "
                f"{configured_count} configured model(s) has a current "
                "qualification card"
                + (f"; {detail}" if detail else "")
                + ". Requalify the selected models before relying on routing."
            )
            result = ModelSwitchResult(
                None,
                previous,
                previous,
                False,
                (reason,),
                requested_task_kind=requested_task_kind,
                configured_candidates=configured_count,
                eligible_candidates=eligible_count,
                routing_strategy=routing_strategy,
            )
            if reason != self._last_unavailable_notice:
                self.status(f"DARKLINGER WARNING: {reason}")
                self._last_unavailable_notice = reason
            self._record(prompt, result, trigger=trigger)
            return result

        self._last_unavailable_notice = ""
        decision = self.router.choose(
            prompt,
            candidates,
            current_model_path=previous,
            task_kind=requested_task_kind,
            strategy=routing_strategy,
            excluded_model_paths=excluded_model_paths,
        )
        if decision is None or decision.selected_model_path == previous:
            if server_tuning is not None:
                return await self._restart_active_with_tuning(
                    state,
                    server_tuning.values(),
                    prompt=prompt,
                    task_kind=requested_task_kind,
                    trigger=trigger,
                    decision=decision,
                    configured_candidates=configured_count,
                    eligible_candidates=eligible_count,
                    routing_strategy=routing_strategy,
                )
            result = ModelSwitchResult(
                decision,
                previous,
                previous,
                False,
                requested_task_kind=requested_task_kind,
                configured_candidates=configured_count,
                eligible_candidates=eligible_count,
                routing_strategy=routing_strategy,
            )
            self._record(prompt, result, trigger=trigger)
            return result

        binary = find_llama_server(state.server_binary)
        if binary is None:
            failure = "configured llama-server is unavailable"
            result = ModelSwitchResult(
                decision,
                previous,
                previous,
                False,
                (failure,),
                requested_task_kind=requested_task_kind,
                configured_candidates=configured_count,
                eligible_candidates=eligible_count,
                routing_strategy=routing_strategy,
            )
            self._record(prompt, result, trigger=trigger)
            return result

        profiles = state.profiles
        ordered = list(
            dict.fromkeys(
                (
                    decision.selected_model_path,
                    *decision.fallback_model_paths,
                    previous,
                )
            )
        )
        previous_profile = self.session.profile
        await self.session.stop()
        failures: list[str] = []
        for path in ordered:
            profile = profiles.get(path)
            if path == previous and profile is None:
                profile = previous_profile
            if profile is None:
                failures.append(f"{Path(path).name}: saved profile is missing")
                continue
            if server_tuning is not None:
                profile = ModelProfile.from_dict(
                    {**profile.to_dict(), **server_tuning.values()}
                )
            session: LlamaServerSession | None = None
            try:
                session = await start_llama_server(
                    binary,
                    profile,
                    self.runtime_root,
                    status=self.status,
                )
                await self.llm.reconfigure()
            except Exception as error:
                if session is not None:
                    await session.stop()
                failures.append(f"{Path(path).name}: {type(error).__name__}: {error}")
                continue
            self.session = session
            state.profiles[path] = profile
            state.last_model_path = path
            self.store.save(state)
            result = ModelSwitchResult(
                decision,
                previous,
                path,
                path != previous,
                tuple(failures),
                requested_task_kind=requested_task_kind,
                configured_candidates=configured_count,
                eligible_candidates=eligible_count,
                routing_strategy=routing_strategy,
            )
            if path != previous:
                self.status(
                    f"V routed this {decision.task_kind} task to "
                    f"{Path(path).name} (verified score {decision.score}/100)."
                )
            self._record(prompt, result, trigger=trigger)
            return result

        raise LlamaServerStartError(
            "model routing stopped the active server and every qualified fallback "
            "failed to start: " + "; ".join(failures)
        )

    async def _restart_active_with_tuning(
        self,
        state: Any,
        values: dict[str, Any],
        *,
        prompt: str,
        task_kind: str,
        trigger: str,
        decision: ModelRouteDecision | None = None,
        configured_candidates: int = 0,
        eligible_candidates: int = 0,
        routing_strategy: str = "automatic",
    ) -> ModelSwitchResult:
        previous = self.active_model_path
        original = self.session.profile
        tuned = ModelProfile.from_dict({**original.to_dict(), **values})
        if tuned == original:
            result = ModelSwitchResult(
                decision,
                previous,
                previous,
                False,
                requested_task_kind=task_kind,
                configured_candidates=configured_candidates,
                eligible_candidates=eligible_candidates,
                routing_strategy=routing_strategy,
            )
            self._record(prompt, result, trigger=trigger)
            return result
        binary = find_llama_server(state.server_binary)
        if binary is None:
            failure = "configured llama-server is unavailable"
            result = ModelSwitchResult(
                decision,
                previous,
                previous,
                False,
                (failure,),
                requested_task_kind=task_kind,
                configured_candidates=configured_candidates,
                eligible_candidates=eligible_candidates,
                routing_strategy=routing_strategy,
            )
            self._record(prompt, result, trigger=trigger)
            return result
        await self.session.stop()
        try:
            session = await start_llama_server(
                binary,
                tuned,
                self.runtime_root,
                status=self.status,
            )
            await self.llm.reconfigure()
        except Exception as error:
            # A tuning failure must not strand Darklinger without its model.
            restored = await start_llama_server(
                binary,
                original,
                self.runtime_root,
                status=self.status,
            )
            self.session = restored
            await self.llm.reconfigure()
            failure = f"model tuning rejected: {type(error).__name__}: {error}"
            result = ModelSwitchResult(
                decision,
                previous,
                previous,
                False,
                (failure,),
                requested_task_kind=task_kind,
                configured_candidates=configured_candidates,
                eligible_candidates=eligible_candidates,
                routing_strategy=routing_strategy,
            )
            self._record(prompt, result, trigger=trigger)
            return result
        self.session = session
        state.profiles[previous] = tuned
        state.last_model_path = previous
        self.store.save(state)
        self.status(
            "V applied a new Full inference configuration and safely restarted "
            "the active local model."
        )
        result = ModelSwitchResult(
            decision,
            previous,
            previous,
            False,
            requested_task_kind=task_kind,
            configured_candidates=configured_candidates,
            eligible_candidates=eligible_candidates,
            routing_strategy=routing_strategy,
        )
        self._record(prompt, result, trigger=trigger)
        return result

    async def ensure_for_phase(
        self,
        prompt: str,
        task_kind: str,
    ) -> ModelSwitchResult:
        """Route at a runtime-owned phase boundary inside one mixed task."""

        return await self.ensure_for(
            prompt,
            task_kind=task_kind,
            trigger="phase_route",
        )

    async def retry_after_rejection(
        self,
        prompt: str,
        task_kind: str,
        excluded_model_paths: Iterable[str] = (),
    ) -> ModelSwitchResult:
        """Retry a failed response with the next qualified local specialist.

        Model output cannot name its own replacement. The runtime excludes the
        model that just failed and selects the next model from persisted,
        current qualification cards for the same task class.
        """

        rejected = self.active_model_path
        excluded = tuple(
            dict.fromkeys((*tuple(excluded_model_paths), rejected))
        )
        return await self.ensure_for(
            prompt,
            task_kind=task_kind,
            excluded_model_paths=excluded,
            trigger="response_rejection",
        )

    async def stop(self) -> None:
        await self.session.stop()

    def _current_candidates(self, state: Any) -> list[ModelRouteCandidate]:
        candidates, _ = self._current_candidates_with_rejections(state)
        return candidates

    def _current_candidates_with_rejections(
        self,
        state: Any,
    ) -> tuple[list[ModelRouteCandidate], dict[str, tuple[str, ...]]]:
        candidates: list[ModelRouteCandidate] = []
        rejected: dict[str, tuple[str, ...]] = {}
        for path in state.routing_model_paths[:3]:
            profile: ModelProfile | None = state.profiles.get(path)
            card: ModelQualificationCard | None = state.qualifications.get(path)
            if profile is None:
                rejected[path] = ("profile_missing",)
                continue
            if card is None:
                rejected[path] = ("qualification_missing",)
                continue
            model_path = Path(path).expanduser()
            reasons = card.stale_reasons(model_path, profile)
            if reasons:
                rejected[path] = reasons
                continue
            candidates.append(ModelRouteCandidate(path, card))
        return candidates, rejected

    def _record(
        self,
        prompt: str,
        result: ModelSwitchResult,
        *,
        trigger: str,
    ) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "trigger": trigger,
            "task_kind": (
                result.decision.task_kind
                if result.decision
                else result.requested_task_kind or "unrouted"
            ),
            "routing_status": (
                "unavailable"
                if result.decision is None and result.failures
                else "ready"
            ),
            "configured_candidates": result.configured_candidates,
            "eligible_candidates": result.eligible_candidates,
            "routing_strategy": result.routing_strategy,
            "previous_model_path": result.previous_model_path,
            "active_model_path": result.active_model_path,
            "switched": result.switched,
            "score": result.decision.score if result.decision else 0,
            "failures": list(result.failures),
        }
        descriptor = os.open(
            self._journal_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(
                descriptor,
                (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
