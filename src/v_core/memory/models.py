from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class MemoryKind(str, Enum):

    FACT = "fact"

    PREFERENCE = "preference"

    LESSON = "lesson"

    EXPERIENCE = "experience"

    RELATIONSHIP_EVENT = "relationship_event"


class MemorySource(str, Enum):

    OBSERVED = "observed"

    DIRECTLY_TOLD = "directly_told"

    INFERRED = "inferred"

    SELF_GENERATED = "self_generated"

    VERIFIED = "verified"


@dataclass
class ReflectionEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    task: str = ""

    result: str = ""

    summary: str = ""

    lesson: str = ""

    importance: str = "low"

    remember: bool = False

    kind: MemoryKind = MemoryKind.LESSON

    source: MemorySource = MemorySource.SELF_GENERATED

    evidence_refs: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ExperienceEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    summary: str = ""

    lesson: str = ""

    confidence: float = 0.0

    importance: str = "low"

    kind: MemoryKind = MemoryKind.EXPERIENCE

    source: MemorySource = MemorySource.SELF_GENERATED

    evidence_refs: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SummaryEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    summary: str = ""

    lessons: list[str] = field(
        default_factory=list
    )

    confidence: float = 0.0

    importance: str = "low"

    kind: MemoryKind = MemoryKind.LESSON

    source: MemorySource = MemorySource.SELF_GENERATED

    evidence_refs: list[dict[str, str]] = field(default_factory=list)


@dataclass
class KnowledgeEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    title: str = ""

    content: str = ""

    reason: str = ""

    confidence: float = 0.0

    kind: MemoryKind = MemoryKind.FACT

    source: MemorySource = MemorySource.INFERRED

    evidence_refs: list[dict[str, str]] = field(default_factory=list)

    # "always" is reserved for stable preferences and generic reversible
    # behavioural lessons. Topic memories remain dormant until explicitly
    # recalled so they cannot steer an unrelated new conversation.
    activation_mode: str = "on_recall"


@dataclass
class ProposalEntry:
    """A model suggestion awaiting runtime or owner disposition."""

    proposal_id: str = field(default_factory=lambda: uuid4().hex)

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    title: str = ""

    suggestion: str = ""

    reason: str = ""

    confidence: float = 0.0

    kind: MemoryKind = MemoryKind.LESSON

    source: MemorySource = MemorySource.SELF_GENERATED

    status: str = "pending"

    decided_at: str = ""

    decision_mode: str = "owner_review"

    triage_reason: str = ""

    triage_scope: str = ""


@dataclass
class SessionEvent:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    event_type: str = ""

    data: dict = field(
        default_factory=dict
    )
