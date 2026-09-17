from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any
from html import unescape
from urllib.parse import unquote, urlsplit

from ..capabilities.web_target import (
    extract_web_target,
    extract_web_targets,
    requests_web_access,
)
from ..tool_recovery import capabilities_for_tool, normalize_capabilities


_ONLINE_ACTION = re.compile(
    r"\b(?:aggregate|browse|check|collect|extract|find|gather|inspect|list|log\s+in|look\s+for|monitor|open|"
    r"register|research|scan|search|visit|crawl|scrape|"
    r"ekstrakc\w*|gromad\w*|monitor\w*|przejr\w*|przeszuk\w*|sprawd\w*|(?:po)?szuk\w*|wejd\w*|"
    r"wej[śsćc]\w*|wesz\w*|wyszuk\w*|wyciagn\w*|wyciągn\w*|zalog\w*|za[łl]o[żz]\w*|"
    r"zbier\w*|znajd\w*|znalaz\w*|znale\w*)\b",
    re.IGNORECASE,
)
_ONLINE_RESOURCE = re.compile(
    r"\b(?:browser|darkne\w*|forum\w*|internet|online|page|repository|repo|site|web|website|"
    r"github|facebook\w*|instagram\w*|linkedin\w*|osint|profile|social\s+media|"
    r"internet\w*|interne\w*|market\w*|profil\w*|sie[cć]\w*|stron\w*|"
    r"witryn\w*)\b",
    re.IGNORECASE,
)
_TOR_ACCESS_REQUEST = re.compile(
    r"(?:\b(?:w|przez)\s+(?:darkne\w*|sieci\s+tor)\b|"
    r"\b(?:forum\w*|market\w*|stron\w*|witryn\w*)\b.{0,48}"
    r"\b(?:na|przez|w)\s+torze\b|"
    r"\b(?:in|on|through|via)\s+(?:the\s+)?"
    r"(?:darknet|dark\s*web|tor\s+network)\b|"
    r"\b(?:forum|market|site|website)s?\b.{0,48}\b(?:on|through|via)\s+tor\b|"
    r"\b(?:przeszuk\w*|wejd\w*|wej[śsćc]\w*)\s+(?:do\s+|w\s+)?(?:darkne\w*|sie[cć]\s+tor)\b|"
    r"\b(?:search|browse|visit|open|inspect)\s+(?:the\s+)?"
    r"(?:darknet|dark\s*web|tor\s+network)\b|"
    r"\b(?:using|używ\w*|uzyw\w*)\s+(?:the\s+)?tor(?:\s+browser)?\b|"
    r"\.onion\b)",
    re.IGNORECASE,
)
_GITHUB_SURFACE = re.compile(r"\bgithub\w*(?:\.com)?\b", re.IGNORECASE)
_GITHUB_CANDIDATE_RESEARCH = re.compile(
    r"\b(?:github\w*(?:\.com)?|repo(?:sitory|sitories)?|repozytori\w*)\b"
    r"(?:(?![.!?;\n]).){0,180}\b"
    r"(?:tool\w*|skill\w*|repo(?:sitory|sitories)?|"
    r"narz[eę]dzi\w*|umiej[eę]tno\w*|repozytori\w*)\b|"
    r"\b(?:tool\w*|skill\w*|repo(?:sitory|sitories)?|"
    r"narz[eę]dzi\w*|umiej[eę]tno\w*|repozytori\w*)\b"
    r"(?:(?![.!?;\n]).){0,180}\bgithub\w*(?:\.com)?\b",
    re.IGNORECASE,
)
_TOR_SURFACE = re.compile(
    r"\b(?:darkne\w*|dark\s*web|torze|tor\s+network|sie[cć]\w*\s+tor)\b",
    re.IGNORECASE,
)
_EXPLICIT_TOR_TRANSPORT = re.compile(
    r"(?:\b(?:through|via|using)\s+(?:the\s+)?tor(?:\s+browser)?\b|"
    r"\bprzez\s+(?:sie[cć]\s+)?tor\b|"
    r"\bu[żz]yw\w*\s+(?:sieci\s+)?tor\w*\b|\.onion\b)",
    re.IGNORECASE,
)
_PURPOSE_ACTION_PREFIX = re.compile(
    r"(?:\b(?:do|for|für|pour|para|per|pro|для)\s+)$",
    re.IGNORECASE,
)
_EXPLICIT_ONION_ADDRESS = re.compile(
    r"https?://(?:[a-z0-9-]+\.)*[a-z2-7]{56}\.onion(?:[^\s<>]*)?",
    re.IGNORECASE,
)
_TOR_INTERACTIVE_BROWSER = re.compile(
    r"\b(?:captcha|kapcza|kapczę|kapcze|private\s+window|private\s+browser|"
    r"prywatn\w*\s+okn\w*|"
    r"interactive\s+browser|browser\s+session)\b",
    re.IGNORECASE,
)
_TOR_SITE_INVENTORY = re.compile(
    r"\b(?:crawl|crawler|inventory|map\s+(?:the\s+)?site|site\s+map|"
    r"z?inwentaryz\w*|map\w*\s+stron\w*|katalog\w*\s+stron\w*)\b",
    re.IGNORECASE,
)
_EXPLICIT_WEB_ADDRESS = re.compile(
    r"(?:https?://|www\.|\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}(?:[/:?#]|\b))",
    re.IGNORECASE,
)
_DETAIL_PAGE = re.compile(
    r"\b(?:first|top)\s+(?:result|repository|repo|link)|"
    r"\b(?:inspect|open|visit)\s+(?:the\s+)?(?:first|top)\s+"
    r"(?:result|repository|repo|link)\b|"
    r"\bpierwsz\w*\s+(?:wynik\w*|repozytori\w*|link\w*)|"
    r"\b(?:otworz|otwórz|sprawdz|sprawdź|wejdz|wejdź)\w*\s+"
    r"(?:w\s+)?pierwsz\w*\s+(?:wynik\w*|repozytori\w*|link\w*)\b",
    re.IGNORECASE,
)
_CREATE_TOOL = re.compile(
    r"\b(?:create|build|implement|write|generate|stworz|stwórz|utworz|utwórz|zbuduj|napisz|"
    r"wygeneruj|zaimplementuj)\w*\b(?:(?![.!?;\n]).){0,180}?\b"
    r"(?:tool\w*|narzedzi\w*|narzędzi\w*)\b",
    re.IGNORECASE,
)
_CREATE_SKILL = re.compile(
    r"\b(?:create|build|implement|write|generate|stworz|stwórz|utworz|utwórz|zbuduj|napisz|"
    r"wygeneruj|zaimplementuj)\w*\b(?:(?![.!?;\n]).){0,180}?\b"
    r"(?:skill\w*|umiejetn\w*|umiejętn\w*)\b",
    re.IGNORECASE,
)
_CAPABILITY_QUESTION_PREFIX = re.compile(
    r"(?:\b(?:tell\s+me\s+(?:whether|if)\s+)?(?:are\s+you\s+able\s+to|"
    r"do\s+you\s+know\s+how\s+to|can\s+you\s+even|"
    r"what\s+(?:can|could)\s+you)\b|"
    r"\b(?:powiedz\s+mi\s*,?\s*)?czy\s+(?:(?:ty|v)\s+)?"
    r"(?:w\s+og[oó]le\s+)?(?:potraf\w*|um(?:iesz|ie\w*|ia\w*)|"
    r"(?:jeste\w*|by[łl]\w*)\s+w\s+stanie)\b)",
    re.IGNORECASE,
)
_CREATION_TOPIC = re.compile(
    r"\b(?:create|build|implement|write|generate|creating|building|"
    r"stworz|stwórz|tworz|utworz|utwórz|zbuduj|napisz|wygeneruj|"
    r"zaimplementuj)\w*\b(?:(?![.!?;\n]).){0,180}?\b"
    r"(?:tool\w*|narzedzi\w*|narzędzi\w*|skill\w*|"
    r"umiejetn\w*|umiejętn\w*)\b",
    re.IGNORECASE,
)
_ARTIFACT_DISJUNCTION = re.compile(
    r"\b(?:or|either|alternatively|albo|lub|b[aą]d[źz]|oder|ou|oppure|"
    r"или|либо|або)\b",
    re.IGNORECASE,
)
_QUOTED_TEXT = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|„[^”\n]*(?:”|")|“[^”\n]*”',
    re.DOTALL,
)
_CONDITIONAL_FALLBACK = re.compile(
    r"\b(?:if|unless|otherwise|when\s+nothing|"
    r"je(?:sli|śli|zeli|żeli)|gdy|wtedy|w\s+przeciwnym\s+razie|"
    r"wenn|falls|sonst|si|sinon|se|altrimenti|"
    r"ha|amennyiben|különben|pokud|jinak|ak|inak|"
    r"если|иначе|якщо|інакше)\b",
    re.IGNORECASE,
)
_RUNTIME_REVIEW = re.compile(
    r"(?:\b(?:analy[sz]e|audit|inspect|review)\w*\b.{0,80}"
    r"\b(?:execution|logs?|run|runtime|session|trace)\w*\b|"
    r"\b(?:analizuj|przeanalizuj|przejrzyj|sprawd[źz])\w*\b.{0,80}"
    r"\b(?:b[łl][ęe]d\w*|log\w*|przebieg\w*|sesj\w*)\b|"
    r"\bruntime_review_task\b|^/review-last-run\b)",
    re.IGNORECASE,
)
_USE_CREATED_TOOL = re.compile(
    r"\b(?:and\s+then\s+(?:execute|invoke|run|use)|"
    r"then\s+(?:really\s+)?(?:execute|invoke|run|use)|"
    r"(?:really\s+)?(?:execute|invoke|run|use)\s+it|"
    r"(?:really\s+)?(?:execute|invoke|run|use)\s+(?:the\s+)?"
    r"(?:(?:new|newly\s+created|created)\s+)?tool|"
    r"a\s+nastepnie\s+uzyj|a\s+następnie\s+użyj|potem\s+uzyj|"
    r"potem\s+użyj|uzyj\s+go|użyj\s+go|"
    r"(?:show|give|present|demonstrate)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:result|results|output|demo)|"
    r"(?:pokaz|pokaż|przedstaw|podaj)\s+(?:mi\s+)?(?:jego\s+|jej\s+)?"
    r"(?:rezultat\w*|wynik\w*|dzialani\w*|działani\w*)|"
    r"a\s+nast[eę]pnie\s+(?:wykonaj|uruchom)|"
    r"potem\s+(?:wykonaj|uruchom)|(?:wykonaj|uruchom)\s+(?:go|je))\b",
    re.IGNORECASE,
)
_DISABLE_WEB = re.compile(
    r"\b(?:(?:do\s+not|don't|never)\s+(?:use|run|execute)\s+(?:any\s+)?(?:network|web|online)\s+tools?|"
    r"nie\s+(?:uruchamiaj|używaj|uzywaj)\s+narz[eę]dzi\s+sieciowych|"
    r"nie\s+(?:korzystaj|używaj|uzywaj|łącz|lacz|wchodź|wchodz)\w*"
    r"(?:\s+z)?\s+(?:internetu|sieci|webu)|"
    r"(?:do\s+not|don't|never|without)\s+"
    r"(?:live\s+)?(?:browse|browsing|contact|crawl|crawling|navigate|network|"
    r"scrape|scraping|search|visit)|"
    r"(?:bez|nigdy\s+nie|nie)\s+(?:kontakt\w*|laczeni\w*|łączeni\w*|"
    r"nawig\w*|przeglad\w*|przegląd\w*|sieci\w*|wyszuk\w*)|"
    r"offline[- ]only)\b",
    re.IGNORECASE,
)

_READ_ONLY = re.compile(
    r"\b(?:read[- ]only|no\s+(?:writes|modifications)|"
    r"(?:do\s+not|don't)\s+(?:write|modify|change)\s+(?:any\s+)?files|"
    r"tylko\s+odczyt|nie\s+(?:zapisuj|modyfikuj|zmieniaj)\s+"
    r"(?:tego\s+)?plik(?:u|[oó]w))\b",
    re.IGNORECASE,
)


def _search_outside_quoted_text(
    pattern: re.Pattern[str], prompt: str
) -> re.Match[str] | None:
    """Ignore command-like words that belong to a quoted input fixture."""

    quoted_spans = [match.span() for match in _QUOTED_TEXT.finditer(prompt)]
    for match in pattern.finditer(prompt):
        if not any(start <= match.start() < end for start, end in quoted_spans):
            return match
    return None


def _capability_question_spans(prompt: str) -> list[tuple[int, int]]:
    """Return question clauses that ask about ability rather than execution."""

    spans: list[tuple[int, int]] = []
    for match in _CAPABILITY_QUESTION_PREFIX.finditer(prompt):
        endings = [
            index
            for marker in (".", "?", "!", ";", "\n")
            if (index := prompt.find(marker, match.end())) >= 0
        ]
        end = min(endings) + 1 if endings else len(prompt)
        spans.append((match.start(), end))
    return spans


def _search_actionable_creation(
    pattern: re.Pattern[str], prompt: str
) -> re.Match[str] | None:
    """Ignore quoted fixtures and creation mentioned only as a capability question."""

    quoted_spans = [match.span() for match in _QUOTED_TEXT.finditer(prompt)]
    capability_spans = _capability_question_spans(prompt)
    for match in pattern.finditer(prompt):
        if any(start <= match.start() < end for start, end in quoted_spans):
            continue
        if any(start <= match.start() < end for start, end in capability_spans):
            continue
        return match
    return None
_READ_FILE = re.compile(
    r"\b(?:analy[sz]\w*|cat|inspect|open|read|review|show|"
    r"analiz\w*|odczyt\w*|otworz\w*|otwórz\w*|przeanaliz\w*|"
    r"przeczyt\w*|przejr\w*|pokaz\w*|pokaż\w*)\b",
    re.IGNORECASE,
)
_STRONG_READ_FILE = re.compile(
    r"\b(?:cat|open|read|odczyt\w*|otworz\w*|otwórz\w*|przeczyt\w*)\b",
    re.IGNORECASE,
)
_PROJECT_CODE_TARGET = re.compile(
    r"\b(?:code|codebase|project|repository|repo|source(?:\s+tree)?|"
    r"kod\w*|projekt\w*|repozytori\w*|zrodl\w*|źródł\w*)\b",
    re.IGNORECASE,
)
_MUTATE_FILE = re.compile(
    r"\b(?:append|create|delete|edit|move|rename|replace|save|write|"
    r"dodaj\w*|edytuj\w*|napisz\w*|przenies\w*|przenieś\w*|"
    r"usun\w*|usuń\w*|utworz\w*|utwórz\w*|zapisz\w*)\b",
    re.IGNORECASE,
)
_RUN_COMMAND = re.compile(
    r"\b(?:execute|run|test|uruchom\w*|wykonaj\w*|przetestuj\w*)\b",
    re.IGNORECASE,
)
_COMMAND_TARGET = re.compile(
    r"\b(?:command|script|shell|test|tests|pytest|foundry|forge|"
    r"komend\w*|skrypt\w*|test\w*)\b",
    re.IGNORECASE,
)


def _requests_command_execution(prompt: str) -> bool:
    """Require a command action and target that are distinct lexical spans.

    ``test`` can be both a verb and a noun. Matching the two patterns
    independently made passive content such as a filename or Markdown heading
    containing "smoke test" require an unrelated sandbox execution. Genuine
    requests such as "run the test" and "test this script" still contain two
    distinct spans.
    """

    # Path components are data: two paths beneath test/ do not constitute
    # two independent verb/noun matches authorizing command execution.
    command_text = re.sub(
        r"(?<![\w/.-])(?:\.?/?[\w.-]+/)+[\w.-]+",
        " ",
        prompt,
    )
    actions = tuple(_RUN_COMMAND.finditer(command_text))
    targets = tuple(_COMMAND_TARGET.finditer(command_text))
    return any(
        action.span() != target.span()
        for action in actions
        for target in targets
    )
_FILE_TARGET = re.compile(
    r"(?:\b(?:file|plik\w*)\b|(?:^|[\s/])[\w.-]+\."
    r"(?:cfg|conf|csv|docx?|html?|ini|json|log|md|pdf|py|sh|sol|toml|ts|txt|"
    r"xml|ya?ml)\b)",
    re.IGNORECASE,
)
_EXPLICIT_LOCAL_PATH = re.compile(
    r"(?:^|[\s'\"`])(?:[a-z]:[\\/]|/|~/|\.\.?/)[^\s'\"`<>]+",
    re.IGNORECASE,
)
_REPORT_RESULT = re.compile(
    r"\b(?:answer|describe|explain|extract|find|give|identify|list|report|summari[sz]e|tell|what|which|"
    r"co|jakie|które|ktore|opisz\w*|podaj\w*|powiedz\w*|przedstaw\w*|stre[śs]c\w*|wyciagn\w*|wyciągn\w*|"
    r"wymien\w*|wymień\w*|znajd\w*|znale[źz]\w*)\b",
    re.IGNORECASE,
)
_PUBLIC_FACT_FIELDS = (
    ("count", re.compile(r"\b(?:how\s+many|number\s+of|count|ile)\b", re.IGNORECASE)),
    ("opening_hours", re.compile(
        r"\b(?:opening\s+hours?|business\s+hours?|hours?|"
        r"godzin\w*|otwar\w*)\b",
        re.IGNORECASE,
    )),
    ("address", re.compile(
        r"\b(?:address\w*|location\w*|adres\w*|lokaliz\w*)\b",
        re.IGNORECASE,
    )),
    ("contact", re.compile(
        r"\b(?:contact\w*|phone\w*|telephone\w*|kontakt\w*|telefon\w*)\b",
        re.IGNORECASE,
    )),
)
_NETWORK_ADDRESS_CONTEXT = re.compile(
    r"(?:\b(?:darkne\w*|dark\s*web|tor\s+network|onion|url|uri|link|domain|"
    r"web\s*address|network\s+address|website\s+address|site\s+address)\w*\b|"
    r"\b(?:adres\w*|address\w*)\b[^.!?;\n]{0,64}\b(?:"
    r"darkne\w*|domen\w*|internet\w*|link\w*|onion\w*|sie[cć]\w*|"
    r"stron\w*|tor\w*|url\w*|witryn\w*|web\w*)\b|"
    r"\b(?:darkne\w*|domen\w*|internet\w*|link\w*|onion\w*|sie[cć]\w*|"
    r"stron\w*|tor\w*|url\w*|witryn\w*|web\w*)\b[^.!?;\n]{0,64}"
    r"\b(?:adres\w*|address\w*)\b)",
    re.IGNORECASE,
)
_PHYSICAL_ADDRESS_CONTEXT = re.compile(
    r"\b(?:building|business|city|headquarters|mailing|office|physical|postal|"
    r"premises|shop|store|street|venue|"
    r"budyn\w*|fizyczn\w*|lokal\w*|miast\w*|poczt\w*|siedzib\w*|"
    r"sklep\w*|ulic\w*)\b",
    re.IGNORECASE,
)
_PUBLIC_WHERE_FIELD = re.compile(r"\b(?:where|gdzie)\b", re.IGNORECASE)
_PUBLIC_PRICE_CONTEXT = re.compile(
    r"\b(?:how\s+much|ile)\b[^,.!?;\n]{0,64}\b(?:cost\w*|price\w*|cen\w*|koszt\w*)\b",
    re.IGNORECASE,
)
_PUBLIC_EXPLICIT_COUNT_CONTEXT = re.compile(
    r"\b(?:how\s+many|number\s+of|count|liczb\w*|"
    r"ile\b[^,.!?;\n]{0,48}\b(?:jest|s[aą]|istniej\w*|znajduj\w*))\b",
    re.IGNORECASE,
)
_PURCHASE_SOURCE_CONTEXT = re.compile(
    r"\b(?:where|gdzie)\b(?=[^.!?;\n]{0,80}\b(?:"
    r"acquir\w*|buy\w*|get\w*|obtain\w*|order\w*|purchas\w*|"
    r"kupi\w*|naby\w*|zam[oó]wi\w*|zdob[yą]\w*)\b)",
    re.IGNORECASE,
)
_ADDRESS_EVIDENCE = re.compile(
    r"(?:\b\d{2}-\d{3}\b|"
    r"\b(?:address|adres|aleja|avenue|boulevard|location|osiedle|plac|road|"
    r"square|street|ulica|ul\.)\b[^\n]{0,100}\b\d{1,5}[a-z]?(?:/\d+)?\b|"
    r"\b[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż.-]{3,}(?:ska|owa|cka|ego|skiej|owej)\s+"
    r"\d{1,5}[a-z]?(?:/\d+)?\b)",
    re.IGNORECASE,
)
_CONTACT_EVIDENCE = re.compile(
    r"(?:\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|"
    r"(?:\+?\d[\d\s().-]{6,}\d))",
    re.IGNORECASE,
)
_OPENING_HOURS_EVIDENCE = re.compile(
    r"(?:\b(?:mon|tue|wed|thu|fri|sat|sun|pon|wt|śr|sr|czw|pt|sob|niedz)"
    r"[\wąćęłńóśźż.-]*\b[^\n]{0,80}\b(?:[01]?\d|2[0-3])[:.]\d{2}\b|"
    r"\b(?:[01]?\d|2[0-3])[:.]\d{2}\s*[-–—]\s*"
    r"(?:[01]?\d|2[0-3])[:.]\d{2}\b)",
    re.IGNORECASE,
)
_PRICE_EVIDENCE = re.compile(
    r"(?:[$€£¥]\s*\d[\d\s.,]*|"
    r"\d[\d\s.,]*\s*(?:USD|EUR|GBP|PLN|CHF|CZK|HUF|JPY|CAD|AUD|zł|€|\$|£)\b)",
    re.IGNORECASE,
)
_IMAGE_EVIDENCE = re.compile(
    r"(?:\bimg\s+[\"']|\bimage\s+[\"']|!\[[^\]]*\]\(|"
    r"https?://[^\s<>]+\.(?:avif|gif|jpe?g|png|webp)(?:[?#][^\s<>]*)?)",
    re.IGNORECASE,
)
_RESEARCH_FACET_NAMES = frozenset(
    {
        "price",
        "purchase_source",
        "item_list",
        "item_descriptions",
        "images",
        "exhaustive_coverage",
        "github_repositories",
    }
)
_FIRST_HEADING = re.compile(
    r"\b(?:first\s+(?:heading|header|title)|"
    r"pierwsz\w*\s+(?:naglow\w*|nagłów\w*|tytul\w*|tytuł\w*))\b",
    re.IGNORECASE,
)
_GROUNDING_STOPWORDS = {
    "about", "after", "also", "and", "browser", "content", "data", "file",
    "first", "from", "have", "heading", "into", "page", "result", "snapshot",
    "that", "their", "there", "these", "this", "tool", "verified", "with",
}
_RAW_BROWSER_SCAFFOLD = re.compile(
    r"(?:\bgeneric\s*\[(?:active\]\s*)?\[?ref=|\[ref=[^\]]+\]|"
    r"\bcursor=(?:pointer|text)\b)",
    re.IGNORECASE,
)
_GROUNDING_ENTITY_STOPWORDS = {
    "another", "based", "boss", "description", "feature", "features",
    "first", "finally", "here", "however", "key",
    "english", "finding", "findings", "lastly", "next", "okay", "open",
    "darklinger", "please", "response", "result", "results", "second",
    "section", "sections", "source", "sources", "still", "the", "therefore",
    "this", "third", "verified", "would",
}
_HTTP_URL = re.compile(r"https?://[^\s<>\[\](){}\"']+", re.IGNORECASE)
_GITHUB_REPOSITORY_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9._/-])"
    r"([A-Za-z0-9](?:[A-Za-z0-9._-]{0,38})/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99}))"
    r"(?![A-Za-z0-9._/-])"
)
_GITHUB_RESERVED_ROOTS = frozenset(
    {
        "about",
        "apps",
        "collections",
        "customer-stories",
        "enterprise",
        "events",
        "explore",
        "features",
        "issues",
        "marketplace",
        "new",
        "notifications",
        "orgs",
        "pricing",
        "pulls",
        "search",
        "security",
        "settings",
        "site",
        "sponsors",
        "topics",
        "trending",
    }
)


def _grounding_url_key(value: str) -> str:
    """Normalize an observed HTTP URL without weakening its host/path identity."""

    parsed = urlsplit(unescape(value.rstrip(".,;:!?")))
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    path = unquote(parsed.path).rstrip("/").casefold()
    return hostname + path if hostname else ""


def _github_repository_identifier(value: str) -> str:
    """Return the exact owner/repository pair from a GitHub repository URL."""

    parsed = urlsplit(unescape(value.rstrip(".,;:!?")))
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    if hostname != "github.com":
        return ""
    segments = [unquote(part) for part in parsed.path.split("/") if part]
    if len(segments) < 2 or segments[0].casefold() in _GITHUB_RESERVED_ROOTS:
        return ""
    owner, repository = segments[:2]
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        return ""
    return f"{owner}/{repository}"


def _claimed_github_repository_identifiers(answer: str) -> set[str]:
    """Extract repository claims without treating arbitrary prose paths as repos."""

    claims = {
        identifier
        for url in _HTTP_URL.findall(answer)
        if (identifier := _github_repository_identifier(url))
    }
    answer_without_urls = _HTTP_URL.sub("", answer)
    for line in answer_without_urls.splitlines():
        identifiers = {
            match.group(1)
            for match in _GITHUB_REPOSITORY_IDENTIFIER.finditer(line)
        }
        if not identifiers:
            continue
        line_folded = line.casefold()
        if (
            "github" in line_folded
            or "repo" in line_folded
            or line.lstrip().startswith(("-", "*"))
            or re.match(r"\s*\d{1,3}[.)]\s", line)
        ):
            claims.update(identifiers)
    return claims


def _grounding_entity_is_present(entity: str, grounding_text: str) -> bool:
    """Accept a source-backed entity, including the safe ``X-based`` form."""

    normalized = entity.casefold()
    if normalized in grounding_text:
        return True
    # Product and protocol names are frequently pluralized in natural prose
    # while source pages use the singular label (CAPTCHAs vs CAPTCHA). This is
    # morphology, not a new entity claim.
    if normalized.endswith("s") and len(normalized) >= 5:
        singular = normalized[:-1]
        if singular in grounding_text:
            return True
    for suffix in ("-based", "_based"):
        if normalized.endswith(suffix):
            stem = normalized[: -len(suffix)].strip("-_")
            # This permits ``Python-based`` when the source says ``Python`` or
            # an inflected form such as Polish ``Pythonie``. It does not grant
            # semantic equivalence to unrelated product names.
            return len(stem) >= 3 and stem in grounding_text
    return False


def _research_item_labels(text: str) -> set[str]:
    """Extract repeated item-like labels from observed page structure."""

    labels: set[str] = set()
    patterns = (
        r'\b(?:heading|img|image)\s+["\']([^"\']{3,160})["\']',
        r"^\s{0,3}#{2,6}\s+(.{3,160})$",
        r"^\s{0,3}\d{1,3}[.)]\s+(.{3,160})$",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            label = " ".join(match.group(1).split()).strip(" .:-")
            if label:
                labels.add(label.casefold())
    return labels


def _is_search_listing_url(url: str) -> bool:
    """Return whether a navigation target is a search/listing page, not a source."""

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/").casefold()
    if hostname.endswith("duckduckgo.com"):
        return True
    if hostname in {"google.com", "www.google.com", "bing.com", "www.bing.com"}:
        return path in {"", "/search"}
    if hostname in {"search.brave.com", "search.yahoo.com"}:
        return True
    if path == "/search":
        return True

    # A marketplace category is useful discovery evidence, not the concrete
    # offer/detail evidence required by a research contract. Use structural
    # path markers instead of a domain allowlist so unseen markets behave the
    # same way and Shopify-like product paths are not rejected as collections.
    segments = [
        unquote(segment).casefold()
        for segment in parsed.path.split("/")
        if segment
    ]
    detail_markers = {
        "ad",
        "article",
        "blog",
        "doc",
        "docs",
        "item",
        "items",
        "offer",
        "oferta",
        "post",
        "product",
        "products",
    }
    if any(segment in detail_markers for segment in segments):
        return False
    listing_markers = {
        "auction",
        "auctions",
        "aukcja",
        "aukcje",
        "catalog",
        "catalogue",
        "categories",
        "category",
        "collection",
        "collections",
        "kategoria",
        "kategorie",
        "kolekcje",
        "listing",
        "listings",
        "marketplace",
        "search",
        "tag",
        "tags",
        "temat",
        "topic",
        "topics",
    }
    return any(
        segment in listing_markers
        or re.fullmatch(r"q-[^/]+", segment) is not None
        for segment in segments
    )


def _snapshot_mentions_url(snapshot: str, url: str) -> bool:
    """Verify that a later navigation target was exposed by earlier evidence.

    Browser snapshots may render a full URL, a decoded redirect, or only a
    hostname plus path. Requiring both host and a non-root path prevents a model
    from satisfying discovery by guessing an unrelated address from memory.
    """

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    if not hostname:
        return False
    haystack = unquote(unescape(snapshot)).casefold().replace("www.", "")
    path = unquote(parsed.path).rstrip("/").casefold()
    if hostname not in haystack:
        return False
    if not path:
        return True
    # Keep the host and path structurally connected. The previous loose
    # "somewhere nearby" test treated a generic `/blog` as discovered when the
    # listing only exposed `/pl/blog/specific-article`, allowing a model to
    # downgrade a real result into an unrelated landing page.
    return f"{hostname}{path}" in haystack


def _web_read_has_substantive_content(call: dict[str, Any]) -> bool:
    """Reject a successful transport call that observed only browser chrome."""

    excerpt = str(call.get("result_excerpt", "")).strip()
    if not excerpt:
        return False
    content = excerpt
    payload: dict[str, Any] | None = None
    try:
        decoded = json.loads(excerpt)
    except (TypeError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, dict):
        payload = decoded
        if decoded.get("error"):
            return False
        content = str(decoded.get("content", "")).strip()
        if not content:
            return False
    elif excerpt.lstrip().startswith("{"):
        # Trace excerpts can end halfway through a large JSON string. Recover
        # enough of the leading content field to judge evidence quality.
        match = re.search(r'"content"\s*:\s*"(.*)', excerpt, re.DOTALL)
        if match is not None:
            content = (
                match.group(1)
                .replace(r"\n", "\n")
                .replace(r'\"', '"')
                .replace(r"\/", "/")
            )

    arguments = call.get("arguments", {})
    requested_url = (
        str(arguments.get("url", "")) if isinstance(arguments, dict) else ""
    )
    parsed = urlsplit(requested_url)
    is_github_blob = (
        (parsed.hostname or "").casefold().removeprefix("www.") == "github.com"
        and "/blob/" in parsed.path
    )
    content_url = str(payload.get("content_url", "")) if payload else ""
    raw_document_observed = (
        "raw.githubusercontent.com/" in content_url.casefold()
        or "raw.githubusercontent.com/" in excerpt.casefold()
    )
    if is_github_blob and not raw_document_observed:
        # The ordinary GitHub page front-loads its global navigation. A bounded
        # snapshot containing only that chrome is not evidence from the file.
        chrome_labels = (
            "navigation menu",
            "sign in",
            "sign up",
            "platform",
            "solutions",
            "enterprise",
        )
        folded = content.casefold()
        if sum(label in folded for label in chrome_labels) >= 4:
            return False

    meaningful = re.sub(
        r"(?:###\s+(?:Page|Snapshot)|-\s+Page\s+(?:URL|Title):[^\n]*|"
        r"-\s+Console:[^\n]*|```ya?ml|```)",
        " ",
        content,
        flags=re.IGNORECASE,
    )
    meaningful = _RAW_BROWSER_SCAFFOLD.sub(" ", meaningful)
    return len(re.sub(r"\W+", "", meaningful, flags=re.UNICODE)) >= 12


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Runtime-owned, model-independent definition of completion evidence."""

    requires_browser_navigation: bool = False
    requires_browser_snapshot: bool = False
    requires_web_discovery: bool = False
    requires_distinct_detail_page: bool = False
    minimum_detail_sources: int = 0
    requires_file_read: bool = False
    required_read_paths: tuple[str, ...] = ()
    requires_file_mutation: bool = False
    requires_command_execution: bool = False
    requires_evidence_report: bool = False
    requires_first_heading: bool = False
    requires_created_tool: bool = False
    requires_created_tool_execution: bool = False
    requires_created_skill: bool = False
    requires_created_artifact: bool = False
    allows_artifact_fallback: bool = False
    requires_runtime_review: bool = False
    requires_tor_candidate_verification: bool = False
    tor_inventory_max_pages: int = 0
    tor_inventory_max_depth: int = -1
    required_tools: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    required_public_fields: tuple[str, ...] = ()
    required_public_subject: str = ""
    required_research_facets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.tor_inventory_max_pages <= 20:
            raise ValueError("tor_inventory_max_pages must be 0 or from 1 to 20")
        if not -1 <= self.tor_inventory_max_depth <= 3:
            raise ValueError("tor_inventory_max_depth must be -1 or from 0 to 3")
        # A model-side semantic classifier may independently mark both builder
        # capabilities even when the owner's structural wording explicitly
        # offered them as alternatives. Preserve the owner's OR contract: one
        # validated artifact is enough and neither concrete kind is mandatory.
        if self.requires_created_artifact:
            object.__setattr__(self, "requires_created_tool", False)
            object.__setattr__(self, "requires_created_tool_execution", False)
            object.__setattr__(self, "requires_created_skill", False)
        if (
            (
                "exhaustive_coverage" in self.required_research_facets
                or "github_repositories" in self.required_research_facets
            )
            and self.requires_web_discovery
            and self.requires_evidence_report
            and self.minimum_detail_sources < 2
        ):
            # One page can provide a useful example, but it cannot support a
            # claim that an online catalogue is complete. Corroboration is a
            # structural evidence rule and does not depend on the request's
            # language.
            object.__setattr__(self, "minimum_detail_sources", 2)

    @staticmethod
    def needs_web_discovery(prompt: str) -> bool:
        """Return whether online routing must discover a URL from search results."""

        return extract_web_target(prompt) is None

    @staticmethod
    def requests_read_only(prompt: str) -> bool:
        return bool(_READ_ONLY.search(prompt))

    def without_mutations(self) -> "TaskContract":
        """An explicit read-only boundary overrides inferred builder work."""
        return replace(
            self, requires_file_mutation=False, requires_command_execution=False,
            requires_created_tool=False, requires_created_tool_execution=False,
            requires_created_skill=False, requires_created_artifact=False,
            allows_artifact_fallback=False,
            required_tools=(), required_capabilities=(),
        )

    @staticmethod
    def is_search_listing_url(url: str) -> bool:
        """Expose the runtime's structural search/listing classifier."""

        return _is_search_listing_url(url)

    @staticmethod
    def disables_web(prompt: str) -> bool:
        """Return a hard owner constraint that semantic routing cannot weaken."""

        return bool(_DISABLE_WEB.search(prompt))

    @staticmethod
    def has_explicit_local_file_target(prompt: str) -> bool:
        """Return whether the message names a file-like object or local path.

        Extensions and path syntax are structural across human languages. This
        lets semantic routing support multilingual file requests without letting
        a model turn an abstract word such as "plan" into an invented plan.txt.
        """

        without_urls = re.sub(
            r"https?://[^\s<>]+",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        return bool(
            _FILE_TARGET.search(without_urls)
            or _EXPLICIT_LOCAL_PATH.search(without_urls)
        )

    def without_web(self) -> "TaskContract":
        """Remove network requirements while preserving all local task evidence."""

        return replace(
            self,
            requires_browser_navigation=False,
            requires_browser_snapshot=False,
            requires_web_discovery=False,
            requires_distinct_detail_page=False,
            minimum_detail_sources=0,
            tor_inventory_max_pages=0,
            tor_inventory_max_depth=-1,
            required_tools=tuple(
                name
                for name in self.required_tools
                if name
                not in {
                    "full_tor_search",
                    "full_tor_fetch",
                    "full_tor_inventory",
                    "full_tor_browser_inventory",
                    "full_tor_browser_close",
                }
            ),
            required_capabilities=tuple(
                capability
                for capability in self.required_capabilities
                if not capability.startswith("network.tor.")
            ),
            required_public_fields=(),
            required_public_subject="",
            required_research_facets=(),
        )

    @staticmethod
    def prefers_tor(prompt: str) -> bool:
        """Return whether the requested online surface is Tor rather than clearnet."""

        if TaskContract.disables_web(prompt):
            return False
        if _EXPLICIT_TOR_TRANSPORT.search(prompt):
            return True
        if not _TOR_ACCESS_REQUEST.search(prompt):
            return False

        # Choose the surface bound to the user's actual online action instead
        # of letting any high-salience topic word win globally.  For example,
        # "search GitHub for tools for searching the darknet" names GitHub as
        # the search surface; darknet describes the tools.  Conversely,
        # "search the darknet for GitHub repositories" still selects Tor.
        # This relation is structural: target names and action spans are used,
        # so the semantic router can supply the action in languages not covered
        # by the deterministic recovery vocabulary.
        for action in _ONLINE_ACTION.finditer(prompt):
            # A capability description such as "tools *for searching* the
            # darknet" is not the user's requested action.  Skip purpose
            # clauses before binding the first actual instruction to a surface.
            prefix = prompt[max(0, action.start() - 20) : action.start()]
            if _PURPOSE_ACTION_PREFIX.search(prefix):
                continue
            clause_end = len(prompt)
            next_punctuation = re.search(r"[.!?;\n]", prompt[action.end() :])
            if next_punctuation is not None:
                clause_end = action.end() + next_punctuation.start()
            candidates: list[tuple[int, str]] = []
            for surface, pattern in (
                ("clearnet", _GITHUB_SURFACE),
                ("tor", _TOR_SURFACE),
            ):
                next_target = pattern.search(prompt, action.end(), clause_end)
                if next_target is not None:
                    gap = next_target.start() - action.end()
                    if gap <= 64:
                        candidates.append((gap, surface))
            if candidates:
                # The first actual instruction establishes the search surface;
                # later clauses describe filters, subjects, or follow-up work.
                return min(candidates)[1] == "tor"
        if _GITHUB_SURFACE.search(prompt):
            # When the deterministic vocabulary does not cover the sentence's
            # language, do not let an ambiguous topic mention force Tor over a
            # concrete named source. The semantic router still decides whether
            # browser work is requested.
            return False
        return True

    @staticmethod
    def asks_about_creation_capability(prompt: str) -> bool:
        """Return true for a pure ability question about creating tools or skills."""

        capability_spans = _capability_question_spans(prompt)
        if not capability_spans:
            return False
        discussed = any(
            start <= match.start() < end
            for match in _CREATION_TOPIC.finditer(prompt)
            for start, end in capability_spans
        )
        return bool(
            discussed
            and _search_actionable_creation(_CREATE_TOOL, prompt) is None
            and _search_actionable_creation(_CREATE_SKILL, prompt) is None
        )

    @staticmethod
    def requests_interactive_tor_browser(prompt: str) -> bool:
        """Return whether Tor work needs a persistent owner-assisted session."""

        return bool(
            TaskContract.prefers_tor(prompt)
            and _EXPLICIT_ONION_ADDRESS.search(prompt)
            and _TOR_INTERACTIVE_BROWSER.search(prompt)
        )

    @staticmethod
    def requests_tor_inventory(prompt: str) -> bool:
        """Return whether an exact onion target needs a passive site inventory."""

        return bool(
            TaskContract.prefers_tor(prompt)
            and _EXPLICIT_ONION_ADDRESS.search(prompt)
            and _TOR_SITE_INVENTORY.search(prompt)
            and not TaskContract.requests_interactive_tor_browser(prompt)
        )

    @staticmethod
    def implies_artifact_discovery(prompt: str) -> bool:
        """Recognize a search attempt preceding conditional artifact creation.

        This is a model-independent recovery path for an intent classifier that
        returns contradictory empty capabilities. Explicit file and command
        targets remain local rather than being silently converted into web work.
        """

        without_urls = re.sub(
            r"https?://[^\s<>]+",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        local_file_work = bool(_FILE_TARGET.search(without_urls))
        local_command_work = bool(
            _requests_command_execution(prompt)
        )
        return bool(_ONLINE_ACTION.search(prompt)) and not (
            local_file_work or local_command_work
        )

    @staticmethod
    def requested_public_fields(prompt: str) -> tuple[str, ...]:
        """Return concrete public-data fields expressed in any routed request."""

        fields = [
            name
            for name, pattern in _PUBLIC_FACT_FIELDS
            if pattern.search(prompt) is not None
        ]
        if (
            "count" in fields
            and _PUBLIC_PRICE_CONTEXT.search(prompt) is not None
            and _PUBLIC_EXPLICIT_COUNT_CONTEXT.search(prompt) is None
        ):
            fields.remove("count")
        # ``Address`` is overloaded across languages: it may mean a postal
        # location or a URL/onion endpoint. Public-fact address evidence is
        # intentionally limited to the former. Otherwise a request for the
        # names and addresses of online services becomes an impossible postal
        # lookup and every mechanically successful search still leaves the
        # contract open. Explicit physical-location wording wins when both
        # meanings appear in the same request.
        if (
            "address" in fields
            and _NETWORK_ADDRESS_CONTEXT.search(prompt) is not None
            and _PHYSICAL_ADDRESS_CONTEXT.search(prompt) is None
        ):
            fields.remove("address")
        # A bare ``where`` can request a physical location, but in a purchase
        # clause it asks for a seller/source instead of a postal address. Treat
        # that distinction before public-fact recovery starts; otherwise an
        # ordinary product-research task is diverted into address lookups.
        if (
            "address" not in fields
            and bool(fields)
            and _PUBLIC_WHERE_FIELD.search(prompt) is not None
            and _PURCHASE_SOURCE_CONTEXT.search(prompt) is None
        ):
            fields.append("address")
        return tuple(fields)

    @staticmethod
    def implies_public_web_lookup(prompt: str) -> bool:
        """Recover a concrete public-fact lookup misread as ordinary chat.

        The multilingual semantic router remains the primary path. This narrow
        deterministic fallback requires an action plus at least two independent
        externally verifiable fields, such as count, opening hours, address, or
        contact details. It therefore does not turn a normal question or a
        creative writing request into browser work.
        """

        return bool(_ONLINE_ACTION.search(prompt)) and len(
            TaskContract.requested_public_fields(prompt)
        ) >= 2

    @staticmethod
    def research_source_minimum(prompt: str) -> int:
        """Require corroboration for a structurally broad research request."""

        words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", prompt, re.UNICODE)
        return 2 if len(words) >= 32 else 1

    @classmethod
    def from_prompt(cls, prompt: str) -> "TaskContract":
        explicit_web_targets = extract_web_targets(prompt)
        web_disabled = bool(_DISABLE_WEB.search(prompt))
        tor_requested = cls.prefers_tor(prompt) and (
            bool(_ONLINE_ACTION.search(prompt))
            or bool(_EXPLICIT_ONION_ADDRESS.search(prompt))
        )
        interactive_tor = tor_requested and cls.requests_interactive_tor_browser(prompt)
        tor_inventory = tor_requested and cls.requests_tor_inventory(prompt)
        online = not tor_requested and not web_disabled and (
            requests_web_access(prompt)
            or bool(_ONLINE_ACTION.search(prompt) and _ONLINE_RESOURCE.search(prompt))
        )
        file_prompt = re.sub(r"https?://[^\s<>]+", "", prompt, flags=re.IGNORECASE)
        explicit_file_target = cls.has_explicit_local_file_target(file_prompt)
        file_targets = list(_FILE_TARGET.finditer(file_prompt))
        read_actions = list(_READ_FILE.finditer(file_prompt))
        mutation_actions = list(_MUTATE_FILE.finditer(file_prompt))
        # Bind a read verb to the file it actually describes. A broad prompt
        # such as "inspect the forum and save it to a file" used to combine the
        # unrelated words "inspect" and "file" and require reading an output
        # artifact before it existed. Natural action-before-target phrasing is
        # accepted, while a strong explicit read verb may also follow a target.
        file_read = bool(
            explicit_file_target
            and (
                any(
                    (read_candidates := [
                        action
                        for action in read_actions
                        if 0 <= target.start() - action.end() <= 96
                    ])
                    and (
                        not (
                            mutation_candidates := [
                                action
                                for action in mutation_actions
                                if 0 <= target.start() - action.end() <= 96
                            ]
                        )
                        or max(action.end() for action in read_candidates)
                        > max(action.end() for action in mutation_candidates)
                    )
                    for target in file_targets
                )
                or any(
                    0 <= action.start() - target.end() <= 64
                    for action in _STRONG_READ_FILE.finditer(file_prompt)
                    for target in file_targets
                )
            )
        )
        project_code_review = bool(
            not online
            and _READ_FILE.search(file_prompt)
            and _PROJECT_CODE_TARGET.search(file_prompt)
            and _REPORT_RESULT.search(file_prompt)
        )
        file_read = file_read or project_code_review
        file_mutation = bool(
            _MUTATE_FILE.search(file_prompt) and explicit_file_target
        )
        command_execution = bool(
            not online and _requests_command_execution(prompt)
        )
        tool_creation_match = _search_actionable_creation(_CREATE_TOOL, prompt)
        skill_creation_match = _search_actionable_creation(_CREATE_SKILL, prompt)
        creation_matches = [
            match
            for match in (tool_creation_match, skill_creation_match)
            if match is not None
        ]
        primary_creation = min(
            creation_matches,
            key=lambda match: match.start(),
            default=None,
        )
        conditional_artifact = False
        if primary_creation is not None:
            before_creation = prompt[
                max(0, primary_creation.start() - 280) : primary_creation.start()
            ]
            after_creation = prompt[primary_creation.end() :]
            conditional_artifact = bool(
                _CONDITIONAL_FALLBACK.search(before_creation)
                or re.match(
                    r"[\s,;:-]*(?:only\s+)?(?:if|when|unless|"
                    r"je(?:sli|śli|zeli|żeli)|gdy|wenn|falls|si|se|"
                    r"ha|amennyiben|pokud|ak|если|якщо)\b",
                    after_creation,
                    re.IGNORECASE,
                )
            )
        artifact_disjunction = False
        if (
            tool_creation_match is not None
            and skill_creation_match is not None
            and not conditional_artifact
        ):
            artifact_span = prompt[
                min(tool_creation_match.start(), skill_creation_match.start()) :
                max(tool_creation_match.end(), skill_creation_match.end())
            ]
            artifact_disjunction = bool(_ARTIFACT_DISJUNCTION.search(artifact_span))
        creates_tool = (
            bool(tool_creation_match)
            and not conditional_artifact
            and not artifact_disjunction
        )
        runtime_review = bool(_RUNTIME_REVIEW.search(prompt))
        web_discovery = online and cls.needs_web_discovery(prompt)
        evidence_report = runtime_review or (
            (online or tor_requested or file_read or command_execution)
            and bool(_REPORT_RESULT.search(prompt))
        )
        public_fields = (
            cls.requested_public_fields(prompt)
            if online and evidence_report
            else ()
        )
        distinct_detail_page = online and (
            bool(_DETAIL_PAGE.search(prompt))
            or (web_discovery and evidence_report)
            or len(explicit_web_targets) > 1
        )
        return cls(
            requires_browser_navigation=online,
            requires_browser_snapshot=online,
            # When Boss asks for online work without supplying an address, the
            # runtime must discover a real URL before visiting candidate sites.
            # This structural rule is language-independent and prevents a local
            # model from turning remembered names into invented domains.
            requires_web_discovery=web_discovery,
            # A research report based only on a search-results listing is not
            # research. Discovery must open and observe at least one real source.
            requires_distinct_detail_page=distinct_detail_page,
            minimum_detail_sources=(
                cls.research_source_minimum(prompt)
                if distinct_detail_page and web_discovery and evidence_report
                else 1 if distinct_detail_page else 0
            ),
            requires_file_read=file_read,
            requires_file_mutation=file_mutation,
            requires_command_execution=command_execution,
            requires_evidence_report=evidence_report,
            requires_first_heading=file_read and bool(_FIRST_HEADING.search(prompt)),
            requires_created_tool=creates_tool,
            requires_created_tool_execution=(
                creates_tool and bool(_USE_CREATED_TOOL.search(prompt))
            ),
            requires_created_skill=(
                bool(skill_creation_match)
                and not conditional_artifact
                and not artifact_disjunction
            ),
            requires_created_artifact=artifact_disjunction,
            allows_artifact_fallback=conditional_artifact,
            requires_runtime_review=runtime_review,
            requires_tor_candidate_verification=(
                tor_requested
                and not _EXPLICIT_ONION_ADDRESS.search(prompt)
                and evidence_report
            ),
            required_tools=(
                (
                    "full_tor_browser_inventory"
                    if interactive_tor
                    else (
                        "full_tor_inventory"
                        if tor_inventory
                        else (
                            "full_tor_fetch"
                            if _EXPLICIT_ONION_ADDRESS.search(prompt)
                            else "full_tor_search"
                        )
                    )
                ),
            )
            if tor_requested
            else (),
            required_capabilities=(
                (
                    "network.tor.browser"
                    if interactive_tor
                    else (
                        "network.tor.inventory"
                        if tor_inventory
                        else (
                            "network.tor.fetch"
                            if _EXPLICIT_ONION_ADDRESS.search(prompt)
                            else "network.tor.search"
                        )
                    )
                ),
            )
            if tor_requested
            else (),
            required_public_fields=public_fields,
            required_research_facets=(
                ("github_repositories",)
                if web_discovery
                and evidence_report
                and _GITHUB_CANDIDATE_RESEARCH.search(prompt)
                and not _DETAIL_PAGE.search(prompt)
                else ()
            ),
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "TaskContract":
        """Rebuild a contract from runtime-owned checkpoint data."""

        source = values if isinstance(values, dict) else {}
        tuple_fields = {
            "required_read_paths",
            "required_tools",
            "required_capabilities",
            "required_public_fields",
            "required_research_facets",
        }
        special_fields = {
            *tuple_fields,
            "minimum_detail_sources",
            "required_public_subject",
            "tor_inventory_max_pages",
            "tor_inventory_max_depth",
        }
        flags = {
            name: bool(source.get(name, False))
            for name in cls.__dataclass_fields__
            if name not in special_fields
        }
        raw_tools = source.get("required_tools", [])
        required_tools = (
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_tools
                    if str(item).strip()
                )
            )
            if isinstance(raw_tools, (list, tuple))
            else ()
        )
        raw_capabilities = source.get("required_capabilities", [])
        required_capabilities = (
            normalize_capabilities(raw_capabilities)
            if isinstance(raw_capabilities, (list, tuple))
            else ()
        )
        raw_public_fields = source.get("required_public_fields", [])
        required_public_fields = (
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_public_fields
                    if str(item).strip()
                    in {"count", "opening_hours", "address", "contact"}
                )
            )
            if isinstance(raw_public_fields, (list, tuple))
            else ()
        )
        raw_research_facets = source.get("required_research_facets", [])
        required_research_facets = (
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_research_facets
                    if str(item).strip() in _RESEARCH_FACET_NAMES
                )
            )
            if isinstance(raw_research_facets, (list, tuple))
            else ()
        )
        return cls(
            **flags,
            required_read_paths=tuple(dict.fromkeys(
                p for p in source.get("required_read_paths", []) if isinstance(p, str) and p
            )) if isinstance(source.get("required_read_paths", []), (list, tuple)) else (),
            minimum_detail_sources=max(
                0,
                min(8, int(source.get("minimum_detail_sources", 0) or 0)),
            ),
            tor_inventory_max_pages=max(
                0,
                min(20, int(source.get("tor_inventory_max_pages", 0) or 0)),
            ),
            tor_inventory_max_depth=max(
                -1,
                min(3, int(source.get("tor_inventory_max_depth", -1))),
            ),
            required_tools=required_tools,
            required_capabilities=required_capabilities,
            required_public_fields=required_public_fields,
            required_public_subject=str(
                source.get("required_public_subject", "")
            ).strip()[:160],
            required_research_facets=required_research_facets,
        )

    def merged(self, other: "TaskContract") -> "TaskContract":
        """Return the union of two independently detected requirements."""

        tuple_fields = {
            "required_read_paths",
            "required_tools",
            "required_capabilities",
            "required_public_fields",
            "required_research_facets",
        }
        special_fields = {
            *tuple_fields,
            "minimum_detail_sources",
            "required_public_subject",
            "tor_inventory_max_pages",
            "tor_inventory_max_depth",
        }
        flags = {
            name: bool(getattr(self, name) or getattr(other, name))
            for name in self.__dataclass_fields__
            if name not in special_fields
        }
        required_tools = tuple(
            dict.fromkeys((*self.required_tools, *other.required_tools))
        )
        required_capabilities = tuple(
            dict.fromkeys(
                (*self.required_capabilities, *other.required_capabilities)
            )
        )
        required_public_fields = tuple(
            dict.fromkeys(
                (*self.required_public_fields, *other.required_public_fields)
            )
        )
        required_research_facets = tuple(
            dict.fromkeys(
                (*self.required_research_facets, *other.required_research_facets)
            )
        )
        return type(self)(
            **flags,
            required_read_paths=tuple(dict.fromkeys((*self.required_read_paths, *other.required_read_paths))),
            minimum_detail_sources=max(
                self.minimum_detail_sources,
                other.minimum_detail_sources,
            ),
            tor_inventory_max_pages=min(
                (
                    value
                    for value in (
                        self.tor_inventory_max_pages,
                        other.tor_inventory_max_pages,
                    )
                    if value > 0
                ),
                default=0,
            ),
            tor_inventory_max_depth=min(
                (
                    value
                    for value in (
                        self.tor_inventory_max_depth,
                        other.tor_inventory_max_depth,
                    )
                    if value >= 0
                ),
                default=-1,
            ),
            required_tools=required_tools,
            required_capabilities=required_capabilities,
            required_public_fields=required_public_fields,
            required_public_subject=(
                self.required_public_subject or other.required_public_subject
            ),
            required_research_facets=required_research_facets,
        )

    def with_required_tools(self, names: list[str] | tuple[str, ...]) -> "TaskContract":
        return type(self)(
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "required_tools",
                    "required_capabilities",
                    "required_public_fields",
                    "required_research_facets",
                    "required_public_subject",
                }
            },
            required_tools=tuple(
                dict.fromkeys(
                    (*self.required_tools, *(item for item in names if item))
                )
            ),
            required_capabilities=tuple(
                self.required_capabilities
            ),
            required_public_fields=self.required_public_fields,
            required_public_subject=self.required_public_subject,
            required_research_facets=self.required_research_facets,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def unmet(self, calls: list[dict[str, Any]]) -> list[str]:
        succeeded = [call for call in calls if call.get("status", "succeeded") == "succeeded"]
        names = [str(call.get("tool", "")) for call in succeeded]
        missing: list[str] = []
        tool_builders = {
            "learning_create_tool",
            "learning_create_snapshot_extractor",
        }
        created_tool_name = ""
        created_tool_index = -1
        for index, call in enumerate(succeeded):
            if call.get("tool") not in tool_builders:
                continue
            if call.get("created_tool_name"):
                created_tool_name = str(call["created_tool_name"])
                created_tool_index = index
                continue
            try:
                payload = json.loads(str(call.get("result_excerpt", "")))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("name"):
                created_tool_name = str(payload["name"])
                created_tool_index = index
        created_tool_executed = bool(
            created_tool_name
            and any(
                index > created_tool_index and call.get("tool") == created_tool_name
                for index, call in enumerate(succeeded)
            )
        )
        artifact_fallback_completed = self.allows_artifact_fallback and any(
            name in {*tool_builders, "learning_create_skill"}
            for name in names
        )

        if self.requires_browser_navigation:
            navigation_tools = (
                {"browser_navigate", "web_search", "web_read"}
                if self.requires_web_discovery
                else {"browser_navigate", "web_read"}
            )
            if not any(name in navigation_tools for name in names):
                missing.append("browser_navigate")
        if self.requires_browser_snapshot and not any(
            call.get("tool") == "browser_snapshot"
            or call.get("tool") == "web_read"
            for call in succeeded
        ):
            missing.append("browser_snapshot")

        if self.requires_distinct_detail_page and not artifact_fallback_completed:
            verified_web_read = False
            for read_index, call in enumerate(succeeded):
                if call.get("tool") != "web_read":
                    continue
                arguments = call.get("arguments", {})
                url = str(arguments.get("url", "")) if isinstance(arguments, dict) else ""
                if not url:
                    continue
                if not _web_read_has_substantive_content(call):
                    continue
                if not self.requires_web_discovery:
                    verified_web_read = True
                    break
                if any(
                    prior.get("tool") == "web_search"
                    and search_index < read_index
                    and _snapshot_mentions_url(
                        str(prior.get("result_excerpt", "")),
                        url,
                    )
                    for search_index, prior in enumerate(succeeded)
                ):
                    verified_web_read = True
                    break

            if verified_web_read:
                detail_navigation = (0, "web_read")
            else:
                detail_navigation = None
            navigations: list[tuple[int, str]] = []
            snapshots: list[tuple[int, str]] = []
            discovery_observations: list[tuple[int, str]] = []
            for index, call in enumerate(succeeded):
                name = str(call.get("tool", ""))
                if name == "browser_navigate":
                    arguments = call.get("arguments", {})
                    url = str(arguments.get("url", "")) if isinstance(arguments, dict) else ""
                    if url and all(previous_url != url for _, previous_url in navigations):
                        navigations.append((index, url))
                elif name == "browser_snapshot":
                    snapshots.append((index, str(call.get("result_excerpt", ""))))
                    discovery_observations.append(
                        (index, str(call.get("result_excerpt", "")))
                    )
                elif name == "web_search":
                    discovery_observations.append(
                        (index, str(call.get("result_excerpt", "")))
                    )
            if detail_navigation is not None:
                pass
            elif self.requires_web_discovery:
                # Freeze discovery evidence at the moment the model first leaves
                # the search/listing stage. A guessed or malformed detail URL can
                # expose more links on an unrelated landing page; those later
                # links must not retroactively become "discovered" evidence.
                # A subsequent corrected navigation is still accepted when its
                # exact URL was present in the original search-stage snapshots.
                first_detail_attempt = next(
                    (
                        index
                        for index, url in navigations
                        if not _is_search_listing_url(url)
                    ),
                    len(succeeded),
                )
                discovery_snapshots = [
                    (index, text)
                    for index, text in discovery_observations
                    if index < first_detail_attempt
                ]
                detail_navigation = next(
                    (
                        (index, url)
                        for index, url in navigations
                        if not _is_search_listing_url(url)
                        and any(
                            snapshot_index < index
                            and _snapshot_mentions_url(snapshot_text, url)
                            for snapshot_index, snapshot_text in discovery_snapshots
                        )
                    ),
                    None,
                )
            else:
                detail_navigation = navigations[1] if len(navigations) >= 2 else None
            if detail_navigation is None:
                if self.requires_web_discovery and any(
                    not _is_search_listing_url(url) for _, url in navigations
                ):
                    missing.append("browser_navigate:detail_not_discovered")
                else:
                    missing.append("browser_navigate:distinct_detail_page")
            elif not verified_web_read and not any(
                index > detail_navigation[0] for index, _ in snapshots
            ):
                missing.append("browser_snapshot:detail_page")

            if self.minimum_detail_sources > 1 and not any(
                item.startswith("browser_navigate:")
                or item == "browser_snapshot:detail_page"
                for item in missing
            ):
                verified_sources: set[str] = set()
                for navigation_index, url in navigations:
                    if _is_search_listing_url(url):
                        continue
                    if "github_repositories" in self.required_research_facets:
                        parsed = urlsplit(url)
                        if (
                            (parsed.hostname or "")
                            .casefold()
                            .removeprefix("www.")
                            != "github.com"
                            or not _github_repository_identifier(url)
                        ):
                            continue
                    discovered = any(
                        index < navigation_index
                        and call.get("tool") == "web_search"
                        and _snapshot_mentions_url(
                            str(call.get("result_excerpt", "")),
                            url,
                        )
                        for index, call in enumerate(succeeded)
                    )
                    next_navigation = next(
                        (
                            index
                            for index, _ in navigations
                            if index > navigation_index
                        ),
                        len(succeeded),
                    )
                    observed = any(
                        navigation_index < index < next_navigation
                        for index, _ in snapshots
                    )
                    if discovered and observed:
                        verified_sources.add(url.rstrip("/"))
                for read_index, call in enumerate(succeeded):
                    if call.get("tool") != "web_read":
                        continue
                    arguments = call.get("arguments", {})
                    url = (
                        str(arguments.get("url", ""))
                        if isinstance(arguments, dict)
                        else ""
                    )
                    if (
                        url
                        and not _is_search_listing_url(url)
                        and (
                            "github_repositories"
                            not in self.required_research_facets
                            or bool(_github_repository_identifier(url))
                        )
                        and _web_read_has_substantive_content(call)
                        and any(
                            index < read_index
                            and prior.get("tool") == "web_search"
                            and _snapshot_mentions_url(
                                str(prior.get("result_excerpt", "")),
                                url,
                            )
                            for index, prior in enumerate(succeeded)
                        )
                    ):
                        verified_sources.add(url.rstrip("/"))
                if len(verified_sources) < self.minimum_detail_sources:
                    missing.append(
                        "browser_evidence:detail_sources="
                        f"{len(verified_sources)}/{self.minimum_detail_sources}"
                    )

        if self.requires_file_read and not any(
            name in {"read_file", "cat"} for name in names
        ):
            missing.append("read_file")

        for path in self.required_read_paths:
            if not any(
                call.get("tool") in {"read_file", "file_read"}
                and isinstance(call.get("arguments"), dict)
                and call["arguments"].get("path") == path
                for call in succeeded
            ):
                missing.append("read_file:" + path)

        if self.requires_file_mutation and not any(
            name in {
                "write_file",
                "edit_file",
                "move_file",
                "create_directory",
            }
            for name in names
        ):
            missing.append("filesystem_mutation")

        if self.requires_command_execution and not any(
            name in {"sandbox_execute_offline", "evm_foundry_test_offline"}
            for name in names
        ) and not (
            self.requires_created_tool_execution and created_tool_executed
        ):
            missing.append("command_execution")

        if self.requires_created_tool and not any(
            name in tool_builders for name in names
        ):
            missing.append("learning_create_tool")
        if self.requires_created_tool_execution:
            if not created_tool_executed:
                missing.append("generated_tool_execution")
        if self.requires_created_skill and "learning_create_skill" not in names:
            missing.append("learning_create_skill")
        if self.requires_created_artifact and not any(
            name in {*tool_builders, "learning_create_skill"}
            for name in names
        ):
            missing.append("learning_create_tool_or_skill")
        if self.requires_runtime_review and "runtime_review_task" not in names:
            missing.append("runtime_review_task")

        if self.requires_tor_candidate_verification:
            for search_index, call in enumerate(succeeded):
                if call.get("tool") != "full_tor_search":
                    continue
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    missing.append("full_tor_search:structured_evidence")
                    break
                if not isinstance(payload, dict):
                    missing.append("full_tor_search:structured_evidence")
                    break
                candidates = [
                    str(item)
                    for item in payload.get("onion_results", [])
                    if isinstance(item, str) and item
                ]
                if not candidates:
                    break
                candidate_keys = {
                    key for key in (_grounding_url_key(item) for item in candidates) if key
                }
                verified = any(
                    later_index > search_index
                    and later.get("tool") in {"full_tor_fetch", "full_tor_inventory"}
                    and isinstance(later.get("arguments"), dict)
                    and _grounding_url_key(str(later["arguments"].get("url", "")))
                    in candidate_keys
                    for later_index, later in enumerate(succeeded)
                )
                if not verified:
                    missing.append("full_tor_fetch:candidate_verification")
                break
        for required_tool in self.required_tools:
            required_tool_capabilities = set(capabilities_for_tool(required_tool))
            observed_capabilities = {
                str(capability)
                for call in succeeded
                for capability in (
                    *capabilities_for_tool(str(call.get("tool", ""))),
                    *tuple(call.get("capabilities", []) or []),
                )
            }
            if (
                required_tool not in names
                and not required_tool_capabilities.intersection(observed_capabilities)
            ):
                missing.append(required_tool)
        observed_capabilities = {
            str(capability)
            for call in succeeded
            for capability in (
                *capabilities_for_tool(str(call.get("tool", ""))),
                *tuple(call.get("capabilities", []) or []),
            )
        }
        tool_contract_capabilities = {
            capability
            for required_tool in self.required_tools
            for capability in capabilities_for_tool(required_tool)
        }
        for capability in self.required_capabilities:
            if (
                capability not in observed_capabilities
                and capability not in tool_contract_capabilities
            ):
                missing.append(f"capability:{capability}")

        public_sources: list[str] = []
        research_sources: list[str] = []
        research_page_urls: list[str] = []
        for call in succeeded:
            tool = call.get("tool")
            excerpt = str(call.get("result_excerpt", ""))
            if tool == "web_read":
                if _web_read_has_substantive_content(call):
                    public_sources.append(excerpt)
                    research_sources.append(excerpt)
                    arguments = call.get("arguments", {})
                    if isinstance(arguments, dict) and arguments.get("url"):
                        research_page_urls.append(str(arguments["url"]))
                continue
            if tool != "browser_snapshot":
                continue
            page_url = re.search(
                r"^- Page URL:\s*(\S+)",
                excerpt,
                re.MULTILINE,
            )
            # Search listings contain the requested words plus ranks and result
            # counts. Treating that plumbing as source evidence made a query such
            # as "address ..." followed by result_count=10 look like an address.
            # Only an opened non-search page may satisfy public fact fields.
            if page_url is not None and not _is_search_listing_url(page_url.group(1)):
                public_sources.append(excerpt)
            if page_url is not None and not page_url.group(1).casefold().startswith(
                ("https://duckduckgo.com/", "https://www.google.com/search")
            ):
                research_sources.append(excerpt)
                research_page_urls.append(page_url.group(1))
        public_evidence = "\n".join(public_sources)
        research_evidence = "\n".join(research_sources)
        if self.required_public_subject:
            subject_tokens = {
                token.casefold()
                for token in re.findall(
                    r"[^\W_]+(?:[-'][^\W_]+)*",
                    self.required_public_subject,
                    re.UNICODE,
                )
                if len(token) >= 3
            }
            evidence_folded = public_evidence.casefold()
            if subject_tokens and not all(
                token in evidence_folded for token in subject_tokens
            ):
                missing.append("public_fact:subject")
        address_found = bool(_ADDRESS_EVIDENCE.search(public_evidence))
        if "address" in self.required_public_fields and not address_found:
            missing.append("public_fact:address")
        if (
            "contact" in self.required_public_fields
            and not _CONTACT_EVIDENCE.search(public_evidence)
        ):
            missing.append("public_fact:contact")
        if (
            "opening_hours" in self.required_public_fields
            and not _OPENING_HOURS_EVIDENCE.search(public_evidence)
        ):
            missing.append("public_fact:opening_hours")
        if (
            "count" in self.required_public_fields
            and "address" not in self.required_public_fields
            and not re.search(
                r"\b\d+\s+(?:branches|locations|shops|stores|"
                r"cukierni\w*|lokal\w*|oddzia\w*|plac[oó]w\w*)\b",
                public_evidence,
                re.IGNORECASE,
            )
        ):
            missing.append("public_fact:count")
        if (
            "price" in self.required_research_facets
            and not _PRICE_EVIDENCE.search(research_evidence)
        ):
            missing.append("browser_evidence:research_price")
        commerce_source_found = any(
            _is_search_listing_url(url)
            or any(
                segment in {
                    "ad",
                    "auction",
                    "buy",
                    "item",
                    "itm",
                    "listing",
                    "offer",
                    "oferta",
                    "product",
                    "shop",
                    "store",
                }
                for segment in (
                    unquote(part).casefold()
                    for part in urlsplit(url).path.split("/")
                    if part
                )
            )
            for url in research_page_urls
        )
        if (
            "purchase_source" in self.required_research_facets
            and not commerce_source_found
        ):
            missing.append("browser_evidence:research_purchase_source")
        item_labels = _research_item_labels(research_evidence)
        if (
            "item_list" in self.required_research_facets
            and len(item_labels) < 2
        ):
            missing.append("browser_evidence:research_item_list")
        description_blocks = len(
            re.findall(
                r"(?:\bparagraph\s*:|^\s{0,3}(?:[-*]|\d+[.)])\s+.{24,}$)",
                research_evidence,
                re.IGNORECASE | re.MULTILINE,
            )
        )
        if (
            "item_descriptions" in self.required_research_facets
            and (len(item_labels) < 2 or description_blocks < 2)
        ):
            missing.append("browser_evidence:research_item_descriptions")
        if (
            "images" in self.required_research_facets
            and not _IMAGE_EVIDENCE.search(research_evidence)
        ):
            missing.append("browser_evidence:research_images")
        return missing

    def answer_issues(
        self,
        answer: str,
        calls: list[dict[str, Any]],
        *,
        request: str = "",
    ) -> list[str]:
        """Reject an empty completion claim that does not use observed evidence."""

        if not self.requires_evidence_report:
            return []
        observations = [
            str(call.get("result_excerpt", ""))
            for call in calls
            if call.get("status", "succeeded") == "succeeded"
            and call.get("tool") in {
                "browser_snapshot",
                "web_search",
                "web_read",
                "read_file",
                "cat",
                "sandbox_execute_offline",
                "evm_foundry_test_offline",
                "runtime_review_task",
                "full_tor_search",
                "full_tor_fetch",
                "full_tor_inventory",
                "full_tor_browser_inventory",
            }
            and call.get("result_excerpt")
        ]
        if not observations:
            return ["answer:evidence_observation_missing"]

        if _RAW_BROWSER_SCAFFOLD.search(answer):
            return ["answer:browser_scaffolding_is_not_a_finding"]

        if self.requires_browser_navigation or any(
            name
            in {
                "full_tor_search",
                "full_tor_fetch",
                "full_tor_inventory",
                "full_tor_browser_inventory",
            }
            for name in self.required_tools
        ):
            observed_tool_names = " ".join(
                str(call.get("tool", ""))
                for call in calls
                if call.get("status", "succeeded") == "succeeded"
            )
            grounding_text = (
                request + "\n" + "\n".join(observations) + "\n" + observed_tool_names
            ).casefold()
            observed_url_keys = {
                key
                for key in (
                    _grounding_url_key(url)
                    for observation in observations
                    for url in _HTTP_URL.findall(observation)
                )
                if key
            }
            ungrounded_urls = sorted(
                {
                    url.rstrip(".,;:!?")
                    for url in _HTTP_URL.findall(answer)
                    if _grounding_url_key(url) not in observed_url_keys
                }
            )
            if ungrounded_urls:
                return [
                    "answer:ungrounded_online_urls="
                    + "|".join(ungrounded_urls[:6])
                ]
            observed_repository_identifiers = {
                identifier
                for call in calls
                if call.get("status", "succeeded") == "succeeded"
                for value in (
                    str(call.get("arguments", {}).get("url", ""))
                    if isinstance(call.get("arguments"), dict)
                    else "",
                    str(call.get("result_excerpt", "")),
                )
                for url in _HTTP_URL.findall(value)
                if (identifier := _github_repository_identifier(url))
            }
            claimed_repository_identifiers = (
                _claimed_github_repository_identifiers(answer)
            )
            ungrounded_repository_identifiers = sorted(
                claimed_repository_identifiers
                - observed_repository_identifiers
            )
            if ungrounded_repository_identifiers:
                return [
                    "answer:ungrounded_repository_identifiers="
                    + "|".join(ungrounded_repository_identifiers[:6])
                ]
            claimed_entities: set[str] = set()
            grounded_highlight_spans: list[tuple[int, int]] = []
            for highlighted in re.finditer(
                r"\*\*([^*\n]{2,120})\*\*|`([^`\n]{2,120})`",
                answer,
            ):
                phrase = next(
                    (item for item in highlighted.groups() if item),
                    "",
                )
                phrase_tokens = [
                    token
                    for token in re.findall(
                        r"[A-Za-z][A-Za-z0-9.+_-]{2,}", phrase
                    )
                    if token.casefold() not in _GROUNDING_ENTITY_STOPWORDS
                ]
                structural_label = bool(
                    re.fullmatch(
                        r"\s*(?:step\s+\d+|conclusion|summary|result|source|"
                        r"finding|findings|next\s+steps?)\s*:?[\s]*",
                        phrase,
                        flags=re.IGNORECASE,
                    )
                )
                # Markdown emphasis often wraps a whole feature label such as
                # "Chrome Extension version". If one distinctive token grounds
                # that label, its generic descriptive words are not separate
                # product claims.
                phrase_is_grounded = structural_label or any(
                    _grounding_entity_is_present(token, grounding_text)
                    for token in phrase_tokens
                )
                if phrase_is_grounded:
                    grounded_highlight_spans.append(highlighted.span())
                else:
                    claimed_entities.update(phrase_tokens)
            for match in re.finditer(r"\b[A-Z][A-Za-z0-9.+_-]{2,}\b", answer):
                token = match.group(0)
                if any(
                    start <= match.start() < end
                    for start, end in grounded_highlight_spans
                ):
                    continue
                prefix = answer[: match.start()].rstrip()
                if not prefix or prefix[-1:] in {".", "!", "?", "\n"}:
                    continue
                if re.search(
                    r"\b(?:hey|hello|okay|ok|sorry|thanks)[,\s]+$",
                    prefix[-32:],
                    flags=re.IGNORECASE,
                ):
                    # A direct form of address is relationship prose, not an
                    # online product/entity claim that needs source grounding.
                    continue
                line_prefix = answer[: match.start()].rsplit("\n", 1)[-1]
                if re.match(r"\s{0,3}#{1,6}\s", line_prefix):
                    # Markdown headings are discourse structure. Words such as
                    # "Step" and "Capture" are not online entity claims.
                    continue
                if token.casefold() not in _GROUNDING_ENTITY_STOPWORDS:
                    claimed_entities.add(token)
            ungrounded = sorted(
                entity
                for entity in claimed_entities
                if not _grounding_entity_is_present(entity, grounding_text)
            )
            if ungrounded:
                return [
                    "answer:ungrounded_online_claims="
                    + "|".join(ungrounded[:12])
                ]

        if self.requires_first_heading:
            first_line = next(
                (line.strip() for line in observations[0].splitlines() if line.strip()),
                "",
            )
            heading_tokens = {
                token
                for token in re.findall(r"[\w.-]{2,}", first_line.casefold())
                if token not in _GROUNDING_STOPWORDS
            }
            answer_heading_tokens = set(
                re.findall(r"[\w.-]{2,}", answer.casefold())
            )
            if not first_line or not heading_tokens.issubset(answer_heading_tokens):
                return ["answer:first_heading_missing"]

        report_entries = [
            " ".join(match.group(1).split())
            for match in re.finditer(
                r"^\s{0,3}(?:[-*]|\d{1,3}[.)])\s+(.+)$",
                answer,
                re.MULTILINE,
            )
        ]
        if (
            "item_list" in self.required_research_facets
            and len(report_entries) < 2
        ):
            return ["answer:research_item_list_missing"]
        if "item_descriptions" in self.required_research_facets:
            described_entries = [
                entry
                for entry in report_entries
                if len(re.findall(r"[^\W_]+", entry, re.UNICODE)) >= 5
            ]
            if len(described_entries) < 2:
                return ["answer:research_item_descriptions_missing"]

        answer_tokens = {
            token
            for token in re.findall(r"[\w.-]{4,}", answer.casefold())
            if token not in _GROUNDING_STOPWORDS
        }
        evidence_tokens = {
            token
            for token in re.findall(
                r"[\w.-]{4,}", "\n".join(observations).casefold()
            )
            if token not in _GROUNDING_STOPWORDS
        }
        if not answer_tokens.intersection(evidence_tokens):
            return ["answer:not_grounded_in_tool_evidence"]
        return []

    def deterministic_answer(
        self,
        calls: list[dict[str, Any]],
        *,
        language: str = "English",
    ) -> str | None:
        """Produce exact results for objectives that require no model judgment."""

        if "github_repositories" in self.required_research_facets:
            repositories: list[tuple[str, str, str]] = []
            seen: set[str] = set()
            for call in calls:
                if (
                    call.get("status", "succeeded") != "succeeded"
                    or call.get("tool") != "browser_snapshot"
                ):
                    continue
                excerpt = str(call.get("result_excerpt", ""))
                url_match = re.search(
                    r"^- Page URL:\s*(https?://\S+)",
                    excerpt,
                    re.MULTILINE | re.IGNORECASE,
                )
                if url_match is None:
                    continue
                url = url_match.group(1).rstrip(".,;:!?")
                identifier = _github_repository_identifier(url)
                if not identifier or identifier in seen:
                    continue
                title_match = re.search(
                    r"^- Page Title:\s*(.+)$",
                    excerpt,
                    re.MULTILINE | re.IGNORECASE,
                )
                title = (
                    title_match.group(1).strip()
                    if title_match is not None
                    else identifier
                )
                title = re.sub(r"^GitHub\s*-\s*", "", title).strip()
                title = re.sub(r"\s*[·|-]\s*GitHub\s*$", "", title).strip()
                repositories.append((identifier, url, title[:320]))
                seen.add(identifier)
            if len(repositories) >= max(1, self.minimum_detail_sources):
                polish = "pol" in language.casefold()
                if polish:
                    lines = [
                        "Mam to, Boss. Zweryfikowane kandydatury z GitHuba:",
                    ]
                    for identifier, url, title in repositories:
                        lines.append(f"- `{identifier}` — {title}. Źródło: {url}")
                    lines.append(
                        "Niczego jeszcze nie instalowałam ani nie przyswajałam. "
                        "Wybierz repozytoria, które zatwierdzasz."
                    )
                else:
                    lines = [
                        "Got it, Boss. Verified GitHub candidates:",
                    ]
                    for identifier, url, title in repositories:
                        lines.append(f"- `{identifier}` — {title}. Source: {url}")
                    lines.append(
                        "I haven't installed or adopted anything yet. Pick the "
                        "repositories you approve."
                    )
                return "\n".join(lines)

        if "full_tor_inventory" in self.required_tools:
            for call in reversed(calls):
                if (
                    call.get("status", "succeeded") != "succeeded"
                    or call.get("tool") != "full_tor_inventory"
                ):
                    continue
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    return None
                if not isinstance(payload, dict):
                    return None
                records = payload.get("records", [])
                if not isinstance(records, list):
                    return None
                polish = "pol" in language.casefold()
                observed = int(payload.get("pages_observed", len(records)) or 0)
                succeeded = int(payload.get("pages_succeeded", 0) or 0)
                failed = int(payload.get("pages_failed", 0) or 0)
                coverage = str(payload.get("coverage", "unknown"))
                budget = bool(payload.get("budget_exhausted"))
                timed_out = bool(payload.get("timed_out"))
                if polish:
                    lines = [
                        "Gotowe. Pasywna inwentaryzacja Tor zakończona.",
                        "",
                        f"Zakres: {payload.get('seed_url', '')}",
                        (
                            f"Pokrycie: {coverage}; zaobserwowano {observed} stron, "
                            f"{succeeded} odczytano poprawnie, {failed} zakończyło się błędem."
                        ),
                        (
                            f"Limity: maksymalnie {payload.get('max_pages', '?')} stron, "
                            f"głębokość {payload.get('max_depth', '?')}."
                        ),
                        (
                            "Wykonanie: JavaScript wyłączony; formularze nie były wysyłane; "
                            "nie użyto danych logowania; przekierowania wyłączone."
                        ),
                    ]
                    if budget:
                        lines.append("Przerwano zgodnie z ustawionym limitem stron.")
                    if timed_out:
                        lines.append("Upłynął limit czasu; wynik jest częściowy.")
                    lines.extend(["", "Zaobserwowane strony:"])
                    for number, record in enumerate(records, start=1):
                        if not isinstance(record, dict):
                            continue
                        title = str(record.get("title", "")).strip() or "bez tytułu"
                        status = int(record.get("status", 0) or 0)
                        lines.append(
                            f"{number}. {title} — HTTP {status or '?'} — "
                            f"{record.get('url', '')}"
                        )
                    lines.extend(
                        [
                            "",
                            (
                                "Granica dowodu: raport potwierdza wyłącznie pasywne odczyty GET "
                                "z wymienionych adresów. Nie dowodzi bezpieczeństwa serwisu, jego "
                                "właściciela ani treści poza zaobserwowanymi stronami."
                            ),
                        ]
                    )
                    return "\n".join(lines)

                lines = [
                    "Done. The passive Tor inventory is complete.",
                    "",
                    f"Scope: {payload.get('seed_url', '')}",
                    (
                        f"Coverage: {coverage}; {observed} pages observed, "
                        f"{succeeded} succeeded, {failed} failed."
                    ),
                    (
                        f"Limits: at most {payload.get('max_pages', '?')} pages, "
                        f"depth {payload.get('max_depth', '?')}."
                    ),
                    (
                        "Execution: JavaScript disabled; no forms submitted; no credentials "
                        "used; redirects disabled."
                    ),
                ]
                if budget:
                    lines.append("Stopped at the requested page budget.")
                if timed_out:
                    lines.append("The time limit expired, so coverage is partial.")
                lines.extend(["", "Observed pages:"])
                for number, record in enumerate(records, start=1):
                    if not isinstance(record, dict):
                        continue
                    title = str(record.get("title", "")).strip() or "untitled"
                    status = int(record.get("status", 0) or 0)
                    lines.append(
                        f"{number}. {title} — HTTP {status or '?'} — "
                        f"{record.get('url', '')}"
                    )
                lines.extend(
                    [
                        "",
                        (
                            "Evidence boundary: this report confirms only passive GET observations "
                            "from the listed URLs. It does not establish site safety, ownership, or "
                            "content outside the observed pages."
                        ),
                    ]
                )
                return "\n".join(lines)

        if self.requires_created_tool_execution:
            builders = {
                "learning_create_tool",
                "learning_create_snapshot_extractor",
            }
            created_name = ""
            creation_payload: dict[str, Any] = {}
            created_index = -1
            for index, call in enumerate(calls):
                if (
                    call.get("status", "succeeded") != "succeeded"
                    or call.get("tool") not in builders
                ):
                    continue
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and payload.get("name"):
                    created_name = str(payload["name"])
                    creation_payload = payload
                    created_index = index
            if created_name:
                for index in range(len(calls) - 1, created_index, -1):
                    call = calls[index]
                    if (
                        call.get("status", "succeeded") != "succeeded"
                        or call.get("tool") != created_name
                    ):
                        continue
                    try:
                        result = json.loads(str(call.get("result_excerpt", "")))
                    except (TypeError, json.JSONDecodeError):
                        return None
                    if not isinstance(result, dict):
                        return None
                    validation = creation_payload.get("validation", {})
                    lifecycle_verified = (
                        str(creation_payload.get("status", "")).casefold()
                        == "active"
                        and isinstance(validation, dict)
                        and validation.get("passed") is True
                    )
                    lifecycle = (
                        "validated and activated"
                        if lifecycle_verified
                        else "created and activated"
                    )
                    records = result.get("records")
                    if isinstance(records, list) and records:
                        lines = [
                            f"Done. DARKLINGER {lifecycle} `{created_name}`, then "
                            "executed it on the runtime-observed page data.",
                            "",
                            "Verified first three records:",
                        ]
                        for number, record in enumerate(records[:3], start=1):
                            if not isinstance(record, dict):
                                return None
                            ordered_keys = [
                                key
                                for key in (
                                    "title",
                                    "price",
                                    "availability",
                                    "relative_product_url",
                                )
                                if key in record
                            ]
                            ordered_keys.extend(
                                key for key in record if key not in ordered_keys
                            )
                            fields = "; ".join(
                                f"{key}: {record[key]}"
                                for key in ordered_keys
                            )
                            lines.append(f"{number}. {fields}")
                        return "\n".join(lines)
                    return (
                        f"Done. DARKLINGER {lifecycle} `{created_name}`, then "
                        "executed it successfully. Verified result:\n\n"
                        + json.dumps(result, ensure_ascii=False, sort_keys=True)
                    )

        if self.requires_created_tool:
            for call in reversed(calls):
                if (
                    call.get("status", "succeeded") != "succeeded"
                    or call.get("tool")
                    not in {
                        "learning_create_tool",
                        "learning_create_snapshot_extractor",
                    }
                ):
                    continue
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict) or not payload.get("name"):
                    continue
                validation = payload.get("validation", {})
                verified = (
                    str(payload.get("status", "")).casefold() == "active"
                    and isinstance(validation, dict)
                    and validation.get("passed") is True
                )
                if not verified:
                    continue
                tests = validation.get("tests", [])
                test_names = [
                    str(item.get("name", ""))
                    for item in tests
                    if isinstance(item, dict) and item.get("passed") is True
                ]
                test_report = (
                    f" Validation: {', '.join(test_names)}."
                    if test_names
                    else " Validation passed in the offline sandbox."
                )
                if (
                    validation.get("validation_strength")
                    == "behavioral_input_sensitivity"
                ):
                    runs = int(payload.get("successful_runs", 0) or 0)
                    runtime_note = (
                        " It has not yet run on a real task input."
                        if runs == 0
                        else f" Recorded task executions: {runs}."
                    )
                    return (
                        "DARKLINGER built, sandbox-tested, and activated the experimental "
                        f"tool `{payload['name']}`. Determinism and input sensitivity "
                        "passed, but no independent semantic oracle has proven its "
                        "domain correctness yet."
                        + runtime_note
                    )
                coverage_note = (
                    " Only the supplied examples were checked; broader correctness "
                    "has not been independently established."
                    if validation.get("semantic_correctness") == "not_independently_established"
                    else ""
                )
                return (
                    f"Done. DARKLINGER built the generated tool, validated, and activated "
                    f"`{payload['name']}` from the generated source."
                    + test_report
                    + coverage_note
                )

        if not self.requires_first_heading:
            return None
        for call in reversed(calls):
            if (
                call.get("status", "succeeded") != "succeeded"
                or call.get("tool") not in {"read_file", "cat"}
            ):
                continue
            for line in str(call.get("result_excerpt", "")).splitlines():
                if line.strip():
                    return line.strip()
        return None
