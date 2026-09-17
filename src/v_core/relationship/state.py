from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _text(value: object, *, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit].strip()


def _unique_texts(value: object, *, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _text(item, limit=item_limit)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        result.append(cleaned)
        seen.add(key)
    return result[-limit:]


@dataclass
class RelationshipState:

    MAX_SHARED_HISTORY = 200
    MAX_FORMS_OF_ADDRESS = 12

    created_at: str = field(
        default_factory=_now
    )

    updated_at: str = field(
        default_factory=_now
    )

    familiarity: float = 0.0

    trust: float = 0.0

    emotional_bond: float = 0.0

    relationship_depth: float = 0.0

    understanding_of_boss: float = 0.0

    shared_history: list[str] = field(
        default_factory=list
    )

    preferred_forms_of_address: list[str] = field(
        default_factory=list
    )

    # Empty means DARKLINGER's configured default (currently English). This is
    # runtime-owned state: the relationship model never writes it.
    preferred_response_language: str = ""

    def __post_init__(self) -> None:
        self.created_at = _text(self.created_at, limit=80) or _now()
        self.updated_at = _text(self.updated_at, limit=80) or self.created_at
        self.familiarity = _score(self.familiarity)
        self.trust = _score(self.trust)
        self.emotional_bond = _score(self.emotional_bond)
        self.relationship_depth = _score(self.relationship_depth)
        self.understanding_of_boss = _score(self.understanding_of_boss)
        self.shared_history = _unique_texts(
            self.shared_history,
            limit=self.MAX_SHARED_HISTORY,
            item_limit=500,
        )
        self.preferred_forms_of_address = _unique_texts(
            self.preferred_forms_of_address,
            limit=self.MAX_FORMS_OF_ADDRESS,
            item_limit=80,
        )
        self.preferred_response_language = _text(
            self.preferred_response_language,
            limit=48,
        )

    def touch(self) -> None:

        self.updated_at = _now()

    def add_shared_experience(
        self,
        experience: str,
    ) -> bool:

        experience = _text(experience, limit=500)
        if not experience:
            return False

        if experience.casefold() not in {
            item.casefold() for item in self.shared_history
        }:

            self.shared_history.append(
                experience
            )
            self.shared_history = self.shared_history[-self.MAX_SHARED_HISTORY:]

            self.touch()
            return True

        return False

    def add_form_of_address(
        self,
        form: str,
    ) -> bool:

        form = _text(form, limit=80)
        if not form:
            return False

        if form.casefold() not in {
            item.casefold() for item in self.preferred_forms_of_address
        }:

            self.preferred_forms_of_address.append(
                form
            )
            self.preferred_forms_of_address = (
                self.preferred_forms_of_address[-self.MAX_FORMS_OF_ADDRESS:]
            )

            self.touch()
            return True

        return False

    def set_response_language(self, language: str) -> bool:
        language = _text(language, limit=48)
        if language.casefold() == self.preferred_response_language.casefold():
            return False
        self.preferred_response_language = language
        self.touch()
        return True
