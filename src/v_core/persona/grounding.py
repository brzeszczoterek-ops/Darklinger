"""Narrow deterministic checks, not a general factual-truth classifier."""
from __future__ import annotations

import re


_QUOTED = re.compile(r'```.*?```|`[^`]*`|"[^"\n]*"|„[^”\n]*”|“[^”\n]*”', re.DOTALL)
_PHYSICAL_CLAIM = re.compile(
    r"\b(?:i\s+(?:heard|smelled|tasted|woke\s+up|slept|went\s+home)|"
    r"i(?:'m|\s+am|\s+was)\s+(?:at\s+home|in\s+my\s+(?:house|room|bed))|"
    r"my\s+(?:childhood|bedroom)|"
    r"(?:usłyszał[ae]m|słyszał[ae]m|zasłyszał[ae]m|obudził[ae]m\s+się|"
    r"spał[ae]m|poczuł[ae]m\s+zapach)|"
    r"(?:jestem|był[ae]m)\s+(?:w\s+domu|w\s+moim\s+pokoju))\b",
    re.IGNORECASE,
)


def claims_physical_experience(text: str) -> bool:
    """Detect unsupported physical autobiography in a text-only reply.

    Callers must exempt requested fiction. Direct quotations, code, familiar
    conversational idioms, negatives and conditional hypotheticals are not
    treated as autobiographical claims. No attempt is made to verify all facts.
    """
    prose = _QUOTED.sub(" ", text).replace("’", "'")
    for sentence in re.split(r"(?<=[.!?])\s+", prose):
        for match in _PHYSICAL_CLAIM.finditer(sentence):
            before = sentence[:match.start()].casefold().rstrip()
            after = sentence[match.end():].casefold().lstrip()
            if re.search(r"\b(?:if|suppose|imagine|gdyby|jeśli)\s*$", before):
                continue
            if match.group().casefold() == "i heard" and re.match(
                r"(?:you\b|your\s+(?:point|question|message)\b)", after
            ):
                continue
            return True
    return False


_RUNTIME_STATUS = re.compile(
    r"\b(?:the|my|our)\s+(?:secure\s+)?(?:api|server|service|script|tool|process|job)\s+"
    r"(?:is|has\s+been)\s+(?:already\s+)?(?:live|online|up|running|responding|started|completed|finished)\b|"
    r"\b(?:api|serwer|usługa|usluga|skrypt|narzędzie|narzedzie|proces|zadanie)\s+"
    r"(?:(?:już|juz|teraz|właśnie|wlasnie)\s+)?(?:działa|dziala|odpowiada|pracuje|"
    r"jest\s+(?:uruchomion\w*|gotow\w*|aktywn\w*))\b|"
    r"\b(?:i|we)\s+(?:have\s+)?(?:hacked|breached|connected\s+to|tested)\s+"
    r"(?:the|a|an)?\s*(?:secure\s+)?api\b",
    re.IGNORECASE,
)
_COMPLETION_ACK = re.compile(
    r"^(?:i\s+(?:did|have\s+done)(?:\s+it)?|we\s+(?:did|have\s+done)(?:\s+it)?|"
    r"(?:it|the\s+job)\s+is\s+done|zrobił[ae]m(?:\s+to)?|wykonał[ae]m(?:\s+to)?)"
    r"(?:\s*,\s*boss)?[.!]?$", re.IGNORECASE,
)


def claims_unobserved_runtime_status(text: str, *, prompt: str = "") -> bool:
    """Detect current execution status that a text-only turn cannot establish.

    Keep offers, explicit questions, negatives, quotes and conditional examples.
    This is a narrow guard for runtime-checkable claims, not general fact checking.
    """
    completion_question = bool(re.search(
        r"\b(?:did\s+you\s+(?:do|finish|run|execute|test)|have\s+you\s+(?:done|finished|run|tested)|"
        r"zrobił[ae]ś|wykonał[ae]ś|skończył[ae]ś|zrobilas|zrobiles|wykonalas|wykonales)\b",
        prompt, re.IGNORECASE,
    ))
    prose = _QUOTED.sub(" ", str(text)).replace("’", "'")
    for sentence in re.split(r"(?<=[.!?])\s+", prose):
        sentence = sentence.strip()
        if sentence.endswith("?") or re.match(r"^(?:if|when|suppose|imagine|gdyby|jeśli|jesli|gdy|kiedy)\b", sentence, re.I):
            continue
        if _RUNTIME_STATUS.search(sentence) or completion_question and _COMPLETION_ACK.fullmatch(sentence):
            return True
    return False
