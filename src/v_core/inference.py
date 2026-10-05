from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping
import math


@dataclass(frozen=True, slots=True)
class InferenceParameters:
    """Validated request-time sampling controls understood by llama.cpp."""

    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    typical_p: float = 1.0
    repeat_penalty: float = 1.05
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = -1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {"top_k", "seed"}:
                if type(value) is not int:
                    raise ValueError(f"{name} must be an integer")
            elif type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= float(self.temperature) <= 5.0:
            raise ValueError("temperature must be between 0 and 5")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if not 0 <= int(self.top_k) <= 10_000:
            raise ValueError("top_k must be between 0 and 10000")
        if not 0.0 <= float(self.min_p) <= 1.0:
            raise ValueError("min_p must be between 0 and 1")
        if not 0.0 < float(self.typical_p) <= 1.0:
            raise ValueError("typical_p must be between 0 and 1")
        if not 0.0 < float(self.repeat_penalty) <= 10.0:
            raise ValueError("repeat_penalty must be between 0 and 10")
        if not -2.0 <= float(self.presence_penalty) <= 2.0:
            raise ValueError("presence_penalty must be between -2 and 2")
        if not -2.0 <= float(self.frequency_penalty) <= 2.0:
            raise ValueError("frequency_penalty must be between -2 and 2")
        if not -1 <= int(self.seed) <= 2_147_483_647:
            raise ValueError("seed must be -1 or a 32-bit non-negative integer")

    def merged(self, values: Mapping[str, Any]) -> "InferenceParameters":
        allowed = set(asdict(self))
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                "unknown request tuning fields: " + ", ".join(sorted(unknown))
            )
        return replace(self, **dict(values))


PUBLIC_INFERENCE_PROFILES: dict[str, InferenceParameters] = {
    "balanced": InferenceParameters(temperature=0.2),
    "conversation": InferenceParameters(
        temperature=0.65,
        top_p=0.95,
        top_k=50,
        min_p=0.04,
        repeat_penalty=1.02,
    ),
    "creative": InferenceParameters(
        temperature=0.9,
        top_p=0.98,
        top_k=80,
        min_p=0.03,
        repeat_penalty=1.0,
    ),
    "coding": InferenceParameters(
        temperature=0.08,
        top_p=0.82,
        top_k=20,
        min_p=0.02,
        repeat_penalty=1.08,
    ),
    "analysis": InferenceParameters(
        temperature=0.12,
        top_p=0.88,
        top_k=30,
        min_p=0.03,
        repeat_penalty=1.06,
    ),
    "research": InferenceParameters(
        temperature=0.18,
        top_p=0.9,
        top_k=35,
        min_p=0.03,
        repeat_penalty=1.06,
    ),
    "tool_use": InferenceParameters(
        temperature=0.05,
        top_p=0.8,
        top_k=20,
        min_p=0.02,
        repeat_penalty=1.08,
    ),
    "document": InferenceParameters(
        temperature=0.22,
        top_p=0.92,
        top_k=40,
        min_p=0.04,
        repeat_penalty=1.05,
    ),
}


TASK_INFERENCE_PROFILE = {
    "conversation": "conversation",
    "coding": "coding",
    "code_review": "analysis",
    "research": "research",
    "tool_use": "tool_use",
    "document": "document",
}


@dataclass(frozen=True, slots=True)
class ServerTuning:
    """Full-only server controls applied at the next safe model boundary."""

    context_size: int | None = None
    gpu_layers: str | None = None
    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    parallel: int | None = None
    flash_attention: str | None = None
    reasoning: str | None = None
    chat_template: str | None = None
    anti_repetition: str | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    extra_args: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        # Reuse the loader's actual types, flags and supported cache formats.
        from .model_loader.models import ModelProfile

        values = self.values()
        for name in ("context_size", "threads", "batch_size", "ubatch_size", "parallel"):
            if name in values and type(values[name]) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.extra_args is not None:
            if not isinstance(self.extra_args, (list, tuple)) or any(
                not isinstance(item, str) for item in self.extra_args
            ):
                raise ValueError("extra_args must be a list of strings")
        defaults = {"model_path": ".", "alias": "validation"}
        # Validate individual controls here; validate the final merged profile
        # again at the runtime boundary before stopping the active server.
        if "batch_size" in values and "ubatch_size" not in values:
            defaults["ubatch_size"] = min(512, values["batch_size"])
        if "ubatch_size" in values and "batch_size" not in values:
            defaults["batch_size"] = max(2048, values["ubatch_size"])
        try:
            validated = ModelProfile(**{**defaults, **values})
        except (TypeError, AttributeError) as error:
            raise ValueError("invalid server tuning types") from error
        for name in values:
            object.__setattr__(self, name, getattr(validated, name))

    def values(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(slots=True)
class InferenceController:
    """Select public presets and accept validated Full persona tuning."""

    edition_name: str = "public"
    active_profile: str = "balanced"
    _turn_request_overrides: dict[str, Any] = field(default_factory=dict)
    _session_request_overrides: dict[str, Any] = field(default_factory=dict)
    _pending_server: ServerTuning | None = None
    outcome_store: Any = None
    identity_provider: Any = None
    task_kind: str = "conversation"
    last_task_id: str = ""
    last_recommendation: Any = None
    _model_fallback: InferenceParameters | None = None

    def bind_model_defaults(self, *, temperature: float, top_p: float) -> None:
        """Bind validated loader defaults to Full's balanced fallback only.

        Named task profiles remain deterministic.  This gives the model
        profile editor a real, documented effect without allowing a saved
        model profile to rewrite Public presets or coding/tool parameters.
        """

        self._model_fallback = PUBLIC_INFERENCE_PROFILES["balanced"].merged(
            {"temperature": temperature, "top_p": top_p}
        )

    def outcome_identity(self):
        if not callable(self.identity_provider):
            return None
        from .model_loader.storage import ModelLoaderStorageError
        try:
            return self.identity_provider()
        except (OSError, ValueError, ModelLoaderStorageError):
            return None

    def adapt_request(self, request: dict[str, Any], *, explicit_temperature: bool = False):
        from .model_loader.outcome_runtime import current_run, request_snapshot
        run = current_run()
        identity = self.outcome_identity()
        if run is not None:
            self.last_recommendation = None
        if (run is None or self.outcome_store is None or identity is None or not identity[2]
                or explicit_temperature or self._turn_request_overrides or self._session_request_overrides):
            return
        values, context = request_snapshot(request)
        recommendation = self.outcome_store.recommend(task_kind=self.task_kind, context=run.context,
            model_fingerprint=identity[0], server_profile_fingerprint=identity[1], edition=self.edition_name,
            default_profile=self.active_profile, default_request=values, request_context=context)
        self.last_recommendation = recommendation
        if not recommendation.allowed:
            raise RuntimeError(recommendation.reason)
        for key, value in recommendation.request.items():
            if key in request.get("extra_body", {}):
                request["extra_body"][key] = value
            else:
                request[key] = value


    @property
    def is_full(self) -> bool:
        return self.edition_name == "full"

    def begin_turn(self, task_kind: str, *, creative: bool = False) -> str:
        self.task_kind = task_kind
        self.last_recommendation = None
        self._turn_request_overrides.clear()
        profile = "creative" if creative else self.profile_for_task(task_kind)
        self.select_profile(profile)
        return self.active_profile

    @staticmethod
    def profile_for_task(task_kind: str) -> str:
        return TASK_INFERENCE_PROFILE.get(task_kind, "balanced")

    def select_profile(self, name: str) -> None:
        normalized = str(name).strip().casefold()
        if normalized not in PUBLIC_INFERENCE_PROFILES:
            raise ValueError(f"unknown inference profile: {name}")
        self.active_profile = normalized

    def parameters(self) -> InferenceParameters:
        selected = PUBLIC_INFERENCE_PROFILES[self.active_profile]
        if (
            self.is_full
            and self.active_profile == "balanced"
            and self._model_fallback is not None
        ):
            selected = self._model_fallback
        if self._session_request_overrides:
            selected = selected.merged(self._session_request_overrides)
        if self._turn_request_overrides:
            selected = selected.merged(self._turn_request_overrides)
        return selected

    def configure_full(
        self,
        *,
        profile: str | None = None,
        request: Mapping[str, Any] | None = None,
        server: Mapping[str, Any] | None = None,
        scope: str = "turn",
    ) -> dict[str, Any]:
        if not self.is_full:
            raise PermissionError(
                "public Darklinger exposes named inference profiles only"
            )
        return self._configure_validated(profile=profile, request=request, server=server, scope=scope)

    def configure_approved(self, **settings: Any) -> dict[str, Any]:
        """Called by the authenticated operator decision endpoint, never a model tool."""
        if self.edition_name != "full_access":
            raise PermissionError("operator tuning requires Full Access")
        return self._configure_validated(**settings)

    def _configure_validated(self, *, profile=None, request=None, server=None, scope="turn"):
        normalized_scope = str(scope).strip().casefold()
        if normalized_scope not in {"turn", "session"}:
            raise ValueError("scope must be turn or session")
        normalized_profile = str(profile or self.active_profile).strip().casefold()
        if normalized_profile not in PUBLIC_INFERENCE_PROFILES:
            raise ValueError(f"unknown inference profile: {profile}")
        request_values = dict(request or {})
        PUBLIC_INFERENCE_PROFILES[normalized_profile].merged(request_values)
        pending = self._pending_server
        if server:
            try:
                pending = ServerTuning(**dict(server))
            except TypeError as error:
                raise ValueError("unknown server tuning field") from error
        # Commit only after ALL request and server controls have validated.
        self.active_profile = normalized_profile
        if normalized_scope == "session":
            self._session_request_overrides.update(request_values)
        else:
            self._turn_request_overrides.update(request_values)
        self._pending_server = pending
        return self.status()

    def consume_server_tuning(self) -> ServerTuning | None:
        tuning = self._pending_server
        self._pending_server = None
        return tuning

    def status(self) -> dict[str, Any]:
        return {
            "outcome_memory": self.outcome_store.inspect() if self.outcome_store else None,
            "last_task_id": self.last_task_id,
            "adaptive_decision": asdict(self.last_recommendation) if self.last_recommendation else None,
            "parameter_source": (
                "named task preset; Full balanced fallback may inherit the model "
                "profile temperature/top_p; then Full session and turn overrides; "
                "explicit call temperature wins"
            ),
            "edition": self.edition_name,
            "profile": self.active_profile,
            "request": asdict(self.parameters()),
            "server_restart_pending": self._pending_server is not None,
            "server": self._pending_server.values() if self._pending_server else {},
        }
