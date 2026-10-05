from __future__ import annotations

import re
import unicodedata

from langdetect import DetectorFactory, LangDetectException, detect_langs


# langdetect uses random sampling internally. A fixed seed keeps DARKLINGER's
# language gate deterministic across runs and test environments.
DetectorFactory.seed = 0


_POLISH_CHARACTERS = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_TECHNICAL_IDENTIFIER = re.compile(
    r"\b(?=[A-Z0-9_-]*[A-Z])(?=[A-Z0-9_-]*[0-9])"
    r"[A-Z0-9]+(?:[-_][A-Z0-9]+)*\b"
)


def literal_identifiers(text: str) -> frozenset[str]:
    return frozenset(_TECHNICAL_IDENTIFIER.findall(text))


def normalize_literal_reply(prompt: str, answer: str) -> str:
    """Remove sentence punctuation only for an explicitly literal-only reply.

    Never extract or guess a value from prose: the entire reply must already
    consist of one machine identifier and a final period.
    """
    if not re.search(
        r"\b(?:reply|answer|respond|return|output)\s+(?:only|just)\b|"
        r"\b(?:odpowiedz|zwróć|podaj)\s+(?:tylko|wyłącznie)\b", prompt, re.IGNORECASE,
    ):
        return answer
    candidate = answer.strip()
    if candidate.endswith(".") and candidate[:-1] in literal_identifiers(candidate):
        return candidate[:-1]
    return answer


def preserve_literal_identifiers(original: str, rewritten: str) -> str:
    """Restore only accent corruption of known literal IDs; reject lost IDs.

    A language rewrite cannot change data. No fuzzy spelling, digit correction,
    or new identifier is inferred: an accented token must normalize exactly to
    an identifier already present in the original answer.
    """
    identifiers = set(_TECHNICAL_IDENTIFIER.findall(original))
    if not identifiers:
        return rewritten

    def restore(match: re.Match) -> str:
        token = match.group()
        normalized = "".join(
            ch for ch in unicodedata.normalize("NFKD", token)
            if not unicodedata.combining(ch)
        )
        return normalized if normalized in identifiers else token

    restored = re.sub(r"\b[^\W_]+(?:[-_][^\W_]+)*\b", restore, rewritten)
    if not identifiers.issubset(set(_TECHNICAL_IDENTIFIER.findall(restored))):
        return ""
    return restored

_POLISH_WORDS = frozenset(
    {
        "ale",
        "bardzo",
        "bedzie",
        "będzie",
        "chcesz",
        "ci",
        "co",
        "czesc",
        "cześć",
        "czy",
        "dla",
        "dobrze",
        "gotowy",
        "gotowa",
        "gotowe",
        "dzisiaj",
        "hej",
        "jak",
        "jest",
        "moge",
        "mogę",
        "mozemy",
        "możemy",
        "nie",
        "pomoc",
        "pomóc",
        "tak",
        "witaj",
        "zrobic",
        "zrobić",
    }
)

_NON_ENGLISH_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:answer|reply|respond|speak|write|continue)\s+"
        r"(?:to\s+me\s+)?(?:in|using)\s+(?!english\b)[a-z-]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:odpowiedz|odpowiadaj|pisz|napisz|mow|mów|rozmawiaj)\b"
        r".{0,32}\bpo\s+(?!angielsku\b)[a-ząćęłńóśźż-]+",
        re.IGNORECASE,
    ),
)

_ENGLISH_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:answer|reply|respond|speak|write|continue)\s+"
        r"(?:to\s+me\s+)?(?:in|using)\s+english\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:odpowiedz|odpowiadaj|pisz|napisz|mow|mów|rozmawiaj)\b"
        r".{0,32}\bpo\s+angielsku\b",
        re.IGNORECASE,
    ),
)

_USER_ENGLISH_DEMAND_PATTERNS = (
    re.compile(
        r"\bplease\s+(?:write|speak|reply|respond|ask|talk|communicate)\b"
        r".{0,48}\bin\s+english\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:can|could|would|will)\s+you\b.{0,64}"
        r"\b(?:write|speak|reply|respond|ask|talk|communicate|use)\b"
        r".{0,48}\benglish\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[.!?]\s*)use\s+english\b",
        re.IGNORECASE,
    ),
)

_LANGUAGE_CODES = {
    "english": {"en"},
    "polish": {"pl"},
    "chinese": {"zh-cn", "zh-tw"},
    "mandarin": {"zh-cn", "zh-tw"},
    "simplified chinese": {"zh-cn"},
    "traditional chinese": {"zh-tw"},
    "spanish": {"es"},
    "german": {"de"},
    "french": {"fr"},
    "italian": {"it"},
    "portuguese": {"pt"},
    "russian": {"ru"},
    "ukrainian": {"uk"},
    "japanese": {"ja"},
    "korean": {"ko"},
    "czech": {"cs"},
    "slovak": {"sk"},
    "hungarian": {"hu"},
    "dutch": {"nl"},
    "turkish": {"tr"},
    "romanian": {"ro"},
    "bulgarian": {"bg"},
    "croatian": {"hr"},
    "slovenian": {"sl"},
    "swedish": {"sv"},
    "norwegian": {"no"},
    "danish": {"da"},
    "finnish": {"fi"},
    "greek": {"el"},
    "arabic": {"ar"},
    "hebrew": {"he"},
    "hindi": {"hi"},
}


def explicitly_requests_non_english(prompt: str) -> bool:
    """Return true only for an explicit instruction to change language.

    Merely writing in another language or mentioning a language is not enough.
    English wins if the prompt contains conflicting language instructions.
    """

    text = prompt.strip()
    if not text or any(pattern.search(text) for pattern in _ENGLISH_REQUEST_PATTERNS):
        return False
    return any(pattern.search(text) for pattern in _NON_ENGLISH_REQUEST_PATTERNS)


def asks_user_to_use_english(text: str) -> bool:
    """Detect an impermissible demand that Boss switch input language."""

    return any(pattern.search(text) for pattern in _USER_ENGLISH_DEMAND_PATTERNS)


def looks_non_english(text: str) -> bool:
    """Conservatively detect natural-language output that is not English.

    Code, paths, identifiers, and very short neutral answers are intentionally
    not rejected. Polish receives an additional deterministic check so common
    replies without diacritics cannot bypass the gate.
    """

    prose = _prose_for_detection(text)
    if not prose:
        return False

    if any(character in _POLISH_CHARACTERS for character in prose):
        return True

    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿąćęłńóśźż]+", prose.casefold())
    polish_hits = sum(word in _POLISH_WORDS for word in words)
    if polish_hits >= 2 or (polish_hits == 1 and len(words) <= 4):
        return True

    letters = "".join(character for character in prose if character.isalpha())
    if len(letters) < 12:
        return False

    # Statistical language guesses are unstable for short acknowledgements.
    # Require positive English evidence rather than exempting all short prose;
    # the explicit Polish checks above still take precedence.
    english_markers = {"the", "this", "that", "your", "our", "is", "are", "has", "have", "with"}
    if len(words) <= 8 and len(set(words) & english_markers) >= 2:
        return False

    try:
        candidates = detect_langs(prose)
    except LangDetectException:
        return False

    if not candidates:
        return False

    best = candidates[0]
    return best.lang != "en" and best.prob >= 0.70


def matches_requested_language(text: str, requested: str) -> bool:
    """Conservatively validate a runtime-owned visible-output language.

    This is deliberately a verifier, not a language chooser. Unknown language
    names are left to the model instead of being falsely rejected by an English-
    only allowlist.
    """

    language = " ".join(str(requested).casefold().split())
    if not language:
        language = "english"
    if language == "english":
        return not looks_non_english(text)

    prose = _prose_for_detection(text)
    if not prose:
        return True

    if language in {"chinese", "mandarin", "simplified chinese", "traditional chinese"}:
        return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", prose))
    if language == "japanese":
        return bool(re.search(r"[\u3040-\u30ff]", prose))
    if language == "korean":
        return bool(re.search(r"[\uac00-\ud7af]", prose))
    if language == "arabic":
        return bool(re.search(r"[\u0600-\u06ff]", prose))
    if language == "hebrew":
        return bool(re.search(r"[\u0590-\u05ff]", prose))

    expected = _LANGUAGE_CODES.get(language)
    if expected is None:
        return True

    letters = "".join(character for character in prose if character.isalpha())
    if len(letters) < 8:
        # Very short replies are frequently language-neutral and langdetect is
        # unreliable on them. Do not destroy a valid answer for false precision.
        return True
    try:
        candidates = detect_langs(prose)
    except LangDetectException:
        return True
    return any(
        candidate.lang in expected and candidate.prob >= 0.55
        for candidate in candidates[:3]
    )


def _prose_for_detection(text: str) -> str:
    without_fences = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    without_inline_code = re.sub(r"`[^`]*`", " ", without_fences)
    without_urls = re.sub(r"https?://\S+", " ", without_inline_code)
    without_paths = re.sub(r"(?:^|\s)(?:[./~][^\s]+)", " ", without_urls)
    # Machine identifiers are literal data, not evidence of prose language.
    # In short replies an unquoted marker such as BURSZTYN-42 can otherwise
    # outweigh perfectly valid English ("is our password", "got it").
    without_identifiers = _TECHNICAL_IDENTIFIER.sub(" ", without_paths)
    return " ".join(without_identifiers.split())
