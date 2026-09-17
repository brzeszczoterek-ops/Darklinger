from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping


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
        if self.context_size is not None and not 256 <= self.context_size <= 10_000_000:
            raise ValueError("context_size is outside the supported range")
        if self.gpu_layers is not None:
            value = str(self.gpu_layers).strip().casefold()
            if value not in {"auto", "all"}:
                try:
                    int(value)
                except ValueError as exc:
                    raise ValueError("gpu_layers must be auto, all, or an integer") from exc
            object.__setattr__(self, "gpu_layers", value)
        if self.threads is not None and not 0 <= self.threads <= 4_096:
            raise ValueError("threads is outside the supported range")
        if self.batch_size is not None and not 1 <= self.batch_size <= 1_048_576:
            raise ValueError("batch_size is outside the supported range")
        if self.ubatch_size is not None and not 1 <= self.ubatch_size <= 1_048_576:
            raise ValueError("ubatch_size is outside the supported range")
        if self.parallel is not None and not 1 <= self.parallel <= 1_024:
            raise ValueError("parallel is outside the supported range")
        for field_name, allowed in {
            "flash_attention": {"auto", "on", "off"},
            "reasoning": {"auto", "on", "off"},
            "chat_template": {"auto", "embedded", "hermes_3_tool_use"},
            "anti_repetition": {"off", "balanced", "strong"},
        }.items():
            value = getattr(self, field_name)
            if value is not None:
                normalized = str(value).strip().casefold()
                if normalized not in allowed:
                    raise ValueError(
                        f"{field_name} must be one of: {', '.join(sorted(allowed))}"
                    )
                object.__setattr__(self, field_name, normalized)
        for field_name in ("cache_type_k", "cache_type_v"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = str(value).strip().casefold()
                if normalized not in {"f16", "bf16", "q8_0", "q4_0"}:
                    raise ValueError(f"{field_name} is not supported")
                object.__setattr__(self, field_name, normalized)
        if self.extra_args is not None:
            extras = tuple(str(item).strip() for item in self.extra_args)
            if (
                len(extras) > 128
                or sum(len(item.encode("utf-8")) for item in extras) > 64 * 1024
                or any(
                    not item or "\x00" in item or len(item.encode("utf-8")) > 4_096
                    for item in extras
                )
            ):
                raise ValueError("extra llama.cpp arguments are invalid")
            object.__setattr__(self, "extra_args", extras)

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

    @property
    def is_full(self) -> bool:
        return self.edition_name == "full"

    def begin_turn(self, task_kind: str, *, creative: bool = False) -> str:
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
        normalized_scope = str(scope).strip().casefold()
        if normalized_scope not in {"turn", "session"}:
            raise ValueError("scope must be turn or session")
        if profile:
            self.select_profile(profile)
        request_values = dict(request or {})
        # Validate the complete merged result before retaining any override.
        PUBLIC_INFERENCE_PROFILES[self.active_profile].merged(request_values)
        if normalized_scope == "session":
            self._session_request_overrides.update(request_values)
        else:
            self._turn_request_overrides.update(request_values)
        if server:
            values = dict(server)
            if "extra_args" in values:
                values["extra_args"] = tuple(values["extra_args"])
            pending = ServerTuning(**values)
            if (
                pending.batch_size is not None
                and pending.ubatch_size is not None
                and pending.ubatch_size > pending.batch_size
            ):
                raise ValueError(
                    "ubatch_size must be no larger than batch_size"
                )
            self._pending_server = pending
        return self.status()

    def consume_server_tuning(self) -> ServerTuning | None:
        tuning = self._pending_server
        self._pending_server = None
        return tuning

    def status(self) -> dict[str, Any]:
        return {
            "edition": self.edition_name,
            "profile": self.active_profile,
            "request": asdict(self.parameters()),
            "server_restart_pending": self._pending_server is not None,
            "server": self._pending_server.values() if self._pending_server else {},
        }
