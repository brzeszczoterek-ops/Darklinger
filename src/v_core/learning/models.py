from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any
from uuid import uuid4


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def clean_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:maximum].strip()


def bounded_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    return value[:maximum]


def finite_score(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(1.0, parsed))


def validate_name(value: str) -> str:
    value = clean_text(value, maximum=64)
    if not _NAME.fullmatch(value):
        raise ValueError(
            "artifact names must match ^[a-z][a-z0-9_]{2,63}$"
        )
    return value


def validate_version(value: str) -> str:
    value = clean_text(value, maximum=32)
    if not _VERSION.fullmatch(value):
        raise ValueError("artifact version must be semantic MAJOR.MINOR.PATCH")
    return value


class EvidenceSource(str, Enum):
    USER_CORRECTION = "user_correction"
    TOOL_RESULT = "tool_result"
    TEST_RESULT = "test_result"
    TASK_RUNTIME = "task_runtime"
    SELF_REVIEW = "self_review"


class EvidenceOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTION = "correction"
    REGRESSION = "regression"


@dataclass(slots=True)
class LearningEvidence:
    task_id: str
    source: EvidenceSource
    outcome: EvidenceOutcome
    summary: str
    evidence_id: str = field(default_factory=lambda: uuid4().hex)
    expected: str = ""
    actual: str = ""
    confidence: float = 0.0
    verified: bool = False
    fingerprint: str = ""
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_id = clean_text(self.task_id, maximum=128)
        if not self.task_id:
            raise ValueError("task_id is required")
        self.summary = clean_text(self.summary, maximum=2_000)
        if not self.summary:
            raise ValueError("evidence summary is required")
        # Expected/actual are evidence, not prose. Preserve significant CSV,
        # code, whitespace, and line breaks within a fixed journal bound.
        self.expected = bounded_text(self.expected, maximum=20_000)
        self.actual = bounded_text(self.actual, maximum=20_000)
        self.confidence = finite_score(self.confidence)
        raw_metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}
        try:
            encoded_metadata = json.dumps(
                raw_metadata,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("evidence metadata must contain JSON values") from error
        if len(encoded_metadata.encode("utf-8")) > 20_000:
            raise ValueError("evidence metadata exceeds 20 KB")
        self.metadata = json.loads(encoded_metadata)
        if not self.fingerprint:
            material = {
                "source": self.source.value,
                "outcome": self.outcome.value,
                "summary": self.summary.casefold(),
                "expected": self.expected.casefold(),
                "actual": self.actual.casefold(),
            }
            self.fingerprint = hashlib.sha256(
                json.dumps(material, sort_keys=True).encode("utf-8")
            ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        data["outcome"] = self.outcome.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningEvidence":
        copied = dict(data)
        copied["source"] = EvidenceSource(copied["source"])
        copied["outcome"] = EvidenceOutcome(copied["outcome"])
        return cls(**copied)


class LessonStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    RETIRED = "retired"


@dataclass(slots=True)
class LearnedLesson:
    title: str
    hypothesis: str
    trigger: str
    action: str
    evidence_ids: list[str]
    lesson_id: str = field(default_factory=lambda: uuid4().hex)
    status: LessonStatus = LessonStatus.CANDIDATE
    confidence: float = 0.0
    revision: int = 1
    successful_uses: int = 0
    failed_uses: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.title = clean_text(self.title, maximum=160)
        self.hypothesis = clean_text(self.hypothesis, maximum=1_000)
        self.trigger = clean_text(self.trigger, maximum=500)
        self.action = clean_text(self.action, maximum=1_000)
        if not all((self.title, self.hypothesis, self.trigger, self.action)):
            raise ValueError("lesson title, hypothesis, trigger, and action are required")
        self.evidence_ids = list(dict.fromkeys(self.evidence_ids))
        if not self.evidence_ids:
            raise ValueError("a lesson requires evidence")
        self.confidence = finite_score(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedLesson":
        copied = dict(data)
        copied["status"] = LessonStatus(copied["status"])
        return cls(**copied)


class ArtifactKind(str, Enum):
    TOOL = "tool"
    SKILL = "skill"


class ArtifactScope(str, Enum):
    TASK = "task"
    PERSISTENT = "persistent"


class ArtifactStatus(str, Enum):
    QUARANTINED = "quarantined"
    VALIDATED = "validated"
    ACTIVE = "active"
    REJECTED = "rejected"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class ToolTestCase:
    name: str
    arguments: dict[str, Any]
    expected: dict[str, Any]

    def __post_init__(self) -> None:
        name = clean_text(self.name, maximum=120)
        if not name:
            raise ValueError("test case name is required")
        object.__setattr__(self, "name", name)
        if not isinstance(self.arguments, dict) or not isinstance(self.expected, dict):
            raise ValueError("tool test arguments and expected output must be objects")
        try:
            encoded = json.dumps(
                {"arguments": self.arguments, "expected": self.expected},
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("tool test contains non-JSON data") from error
        if len(encoded.encode("utf-8")) > 200_000:
            raise ValueError("tool test exceeds 200 KB")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolManifest:
    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    tests: tuple[ToolTestCase, ...]
    scope: ArtifactScope = ArtifactScope.TASK
    lesson_ids: tuple[str, ...] = ()
    timeout_seconds: float = 10.0
    provides_capabilities: tuple[str, ...] = ()
    repair_ticket_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(self, "version", validate_version(self.version))
        description = clean_text(self.description, maximum=500)
        if not description:
            raise ValueError("tool description is required")
        object.__setattr__(self, "description", description)
        if not isinstance(self.input_schema, dict) or not isinstance(
            self.output_schema, dict
        ):
            raise ValueError("tool schemas must be objects")
        if not self.tests:
            raise ValueError("a generated tool requires at least one test")
        if len(self.tests) > 100:
            raise ValueError("a generated tool may define at most 100 tests")
        if len(self.lesson_ids) > 64:
            raise ValueError("a generated tool may link at most 64 lessons")
        if not 0.05 <= float(self.timeout_seconds) <= 120:
            raise ValueError("tool timeout must be between 0.05 and 120 seconds")
        from v_core.tool_recovery import capabilities_for_tool

        capabilities = capabilities_for_tool(self.name, self.provides_capabilities)
        if len(capabilities) > 8:
            raise ValueError("a generated tool may provide at most 8 capabilities")
        object.__setattr__(self, "provides_capabilities", capabilities)
        ticket_id = clean_text(self.repair_ticket_id, maximum=64)
        if ticket_id and not re.fullmatch(r"[0-9a-f]{32}", ticket_id):
            raise ValueError("repair_ticket_id must be a DARKLINGER recovery ticket ID")
        object.__setattr__(self, "repair_ticket_id", ticket_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scope"] = self.scope.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolManifest":
        copied = dict(data)
        copied["scope"] = ArtifactScope(copied.get("scope", "task"))
        copied["tests"] = tuple(
            ToolTestCase(**item) for item in copied.get("tests", [])
        )
        copied["lesson_ids"] = tuple(copied.get("lesson_ids", []))
        copied["provides_capabilities"] = tuple(
            copied.get("provides_capabilities", [])
        )
        return cls(**copied)


@dataclass(frozen=True, slots=True)
class SkillTestCase:
    user_input: str
    should_match: bool

    def __post_init__(self) -> None:
        user_input = clean_text(self.user_input, maximum=1_000)
        if not user_input:
            raise ValueError("skill test input is required")
        object.__setattr__(self, "user_input", user_input)
        if not isinstance(self.should_match, bool):
            raise ValueError("skill should_match must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SkillManifest:
    name: str
    version: str
    description: str
    triggers: tuple[str, ...]
    steps: tuple[str, ...]
    required_tools: tuple[str, ...]
    tests: tuple[SkillTestCase, ...]
    scope: ArtifactScope = ArtifactScope.TASK
    lesson_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(self, "version", validate_version(self.version))
        description = clean_text(self.description, maximum=500)
        if not description:
            raise ValueError("skill description is required")
        object.__setattr__(self, "description", description)
        triggers = tuple(
            dict.fromkeys(clean_text(item, maximum=80).casefold() for item in self.triggers)
        )
        if not triggers or any(len(item) < 2 for item in triggers):
            raise ValueError("a skill requires non-empty triggers")
        if len(triggers) > 20:
            raise ValueError("a skill may define at most 20 triggers")
        object.__setattr__(self, "triggers", triggers)
        steps = tuple(clean_text(item, maximum=500) for item in self.steps)
        if not steps or any(not item for item in steps) or len(steps) > 20:
            raise ValueError("a skill requires between 1 and 20 non-empty steps")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "required_tools",
            tuple(dict.fromkeys(validate_name(item) for item in self.required_tools)),
        )
        if len(self.required_tools) > 64:
            raise ValueError("a skill may require at most 64 tools")
        if len(self.tests) > 100:
            raise ValueError("a skill may define at most 100 tests")
        if len(self.lesson_ids) > 64:
            raise ValueError("a skill may link at most 64 lessons")
        if (
            not self.tests
            or not any(case.should_match for case in self.tests)
            or not any(not case.should_match for case in self.tests)
        ):
            raise ValueError(
                "a skill requires at least one positive and one negative trigger test"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scope"] = self.scope.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillManifest":
        copied = dict(data)
        copied["scope"] = ArtifactScope(copied.get("scope", "task"))
        copied["triggers"] = tuple(copied.get("triggers", []))
        copied["steps"] = tuple(copied.get("steps", []))
        copied["required_tools"] = tuple(copied.get("required_tools", []))
        copied["lesson_ids"] = tuple(copied.get("lesson_ids", []))
        copied["tests"] = tuple(
            SkillTestCase(**item) for item in copied.get("tests", [])
        )
        return cls(**copied)


@dataclass(slots=True)
class ArtifactRecord:
    kind: ArtifactKind
    name: str
    version: str
    scope: ArtifactScope
    scope_key: str
    digest: str
    manifest_path: str
    artifact_id: str = field(default_factory=lambda: uuid4().hex)
    status: ArtifactStatus = ArtifactStatus.QUARANTINED
    source_path: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    activation_count: int = 0
    successful_runs: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.name = validate_name(self.name)
        self.version = validate_version(self.version)
        if self.scope is ArtifactScope.PERSISTENT:
            if self.scope_key != "persistent":
                raise ValueError("persistent artifact has invalid scope key")
        elif not re.fullmatch(r"[0-9a-f]{32}", self.scope_key):
            raise ValueError("task artifact has invalid scope key")
        if not re.fullmatch(r"[0-9a-f]{64}", self.digest):
            raise ValueError("artifact digest must be SHA-256")
        for name in ("activation_count", "successful_runs", "failure_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def transition(self, status: ArtifactStatus) -> None:
        self.status = status
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["scope"] = self.scope.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        copied = dict(data)
        copied["kind"] = ArtifactKind(copied["kind"])
        copied["scope"] = ArtifactScope(copied["scope"])
        copied["status"] = ArtifactStatus(copied["status"])
        return cls(**copied)
