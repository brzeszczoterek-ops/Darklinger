from __future__ import annotations

import json
import re
from urllib.parse import urlsplit


_FULL_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_BARE_DOMAIN = re.compile(
    # Do not treat dotted JSON/Python field names (``request.url``) as web
    # targets when they are embedded in quoted task data. Full URLs remain
    # accepted by the dedicated matcher above.
    r"(?<![@\w\"'])"
    r"(?:www\.)?"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}"
    r"(?::[0-9]{1,5})?"
    r"(?:/[^\s<>]*)?",
    re.IGNORECASE,
)
_WEB_INTENT = re.compile(
    r"\b(?:"
    r"browse|check|explore|inspect|navigate|open|research|review|site|visit|website|"
    r"przejrzyj|sprawdz|sprawdź|strona|strone|stronę|wejdz|wejdź|witryna"
    r")\b",
    re.IGNORECASE,
)

# Speech-to-text engines commonly spell URL punctuation out instead of
# preserving it. Keep this table deliberately small and structural: it maps
# punctuation, never names or likely destinations. Additional languages can be
# added without changing the routing code.
_SPOKEN_URL_SEPARATORS = {
    "colon": ":",
    "dwukropek": ":",
    "doppelpunkt": ":",
    "slash": "/",
    "ukosnik": "/",
    "ukośnik": "/",
    "lamane": "/",
    "łamane": "/",
    "lamany": "/",
    "łamany": "/",
    "dot": ".",
    "period": ".",
    "punkt": ".",
    "kropka": ".",
    "dash": "-",
    "hyphen": "-",
    "minus": "-",
    "myslnik": "-",
    "myślnik": "-",
}
_SPOKEN_URL_TOKEN = re.compile(
    r"https?|[^\W_]+|[:/.-]",
    re.IGNORECASE | re.UNICODE,
)


def _embedded_contract_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Locate JSON carried as tool input so its example URLs stay data."""
    spans: list[tuple[int, int]] = []
    for marker in re.finditer(r"(?<![\w.])tool_contract\s*=\s*", prompt):
        try:
            _, end = json.JSONDecoder().raw_decode(prompt[marker.end():])
        except (TypeError, ValueError):
            continue
        spans.append((marker.start(), marker.end() + end))
    return tuple(spans)


def _extract_spoken_web_target(prompt: str) -> str | None:
    """Rebuild an HTTP(S) URL whose punctuation was spoken aloud.

    A syntactically valid host is required, so ordinary sentences mentioning
    ``HTTP`` do not become browser actions. Sentence-ending punctuation after
    the final host token terminates the reconstruction instead of leaking the
    following instruction into the URL.
    """

    matches = list(_SPOKEN_URL_TOKEN.finditer(prompt))
    start = next(
        (
            index
            for index, match in enumerate(matches)
            if match.group(0).casefold() in {"http", "https"}
        ),
        None,
    )
    if start is None:
        return None

    scheme = matches[start].group(0).casefold()
    rendered = scheme
    saw_spoken_separator = False
    for match in matches[start + 1 : start + 64]:
        token = match.group(0)
        folded = token.casefold()
        separator = _SPOKEN_URL_SEPARATORS.get(folded)
        if separator is not None:
            rendered += separator
            saw_spoken_separator = True
            continue
        if token in {":", "/", ".", "-"}:
            rendered += token
            continue

        # A URL dictated as a sentence normally ends with a literal period.
        # Stop there once the reconstructed value already has a real host.
        rendered += folded
        following = prompt[match.end() :]
        if re.match(r"\s*[.!?](?:\s|$)", following):
            try:
                parsed = urlsplit(rendered)
            except ValueError:
                parsed = None
            if parsed is not None and parsed.hostname and "." in parsed.hostname:
                break

    if not saw_spoken_separator:
        return None
    try:
        parsed = urlsplit(rendered)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or "." not in parsed.hostname
        or any(character.isspace() for character in rendered)
    ):
        return None
    return rendered


def extract_web_targets(prompt: str) -> tuple[str, ...]:
    """Return every explicit web address without duplicating URL hosts.

    The runtime used to preserve only the first address in an owner's request.
    Comparisons therefore became lossy before the model had even seen them.
    Full URL spans take precedence over the bare-domain matcher so a value such
    as ``https://example.com/a`` is returned once, with its path intact.
    """

    candidates: list[tuple[int, str]] = []
    full_spans: list[tuple[int, int]] = []
    embedded = _embedded_contract_spans(prompt)
    for match in _FULL_URL.finditer(prompt):
        if any(start <= match.start() < end for start, end in embedded):
            continue
        full_spans.append(match.span())
        target = match.group(0).rstrip(").,;]}\"'")
        if target:
            candidates.append((match.start(), target))

    for match in _BARE_DOMAIN.finditer(prompt):
        start, end = match.span()
        if any(span_start <= start < span_end for span_start, span_end in embedded):
            continue
        if any(start < full_end and end > full_start for full_start, full_end in full_spans):
            continue
        target = match.group(0).rstrip(").,;]}\"'")
        if target.rsplit(".", 1)[-1].casefold() in {
            "entries", "url", "status", "version", "method", "request",
            "response", "json",
        }:
            continue
        normalized = f"https://{target}"
        if target:
            candidates.append((start, normalized))

    found: list[str] = []
    for _, target in sorted(candidates):
        if target not in found:
            found.append(target)

    if found:
        return tuple(found)
    spoken = _extract_spoken_web_target(prompt)
    return (spoken,) if spoken is not None else ()


def extract_web_target(prompt: str) -> str | None:
    targets = extract_web_targets(prompt)
    return targets[0] if targets else None


def requests_web_access(prompt: str) -> bool:
    embedded = _embedded_contract_spans(prompt)
    if any(
        not any(start <= match.start() < end for start, end in embedded)
        for match in _FULL_URL.finditer(prompt)
    ) or _extract_spoken_web_target(prompt) is not None:
        return True
    return bool(_WEB_INTENT.search(prompt) and extract_web_targets(prompt))
