"""Source-bound research values, independent of the writer's memory.

Publisher anchors are identities, never release numbers. Unknown publishers do
not become official just because a model or a page calls them official. Owners
can supply an explicit source URL; additional publisher adapters can be added
with independently checked domains and current-release entry points.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Publisher:
    subject: str
    hosts: tuple[str, ...]
    current_url: str


# Checked against the publishers' own websites, not model-generated URLs.
# These entry points contain/redirect to current releases; static archive URLs
# on the same hosts do not establish that an old release is current.
PUBLISHERS = (
    Publisher("Python", ("python.org",), "https://www.python.org/downloads/latest/"),
    Publisher("Firefox", ("firefox.com", "mozilla.org"), "https://www.firefox.com/en-US/firefox/notes/"),
)
URL = re.compile(r"https?://[^\s<>\[\](){}\"'`*]+", re.I)
VERSION = re.compile(r"(?<![\w.])\d+\.\d+(?:\.(?:\d+|[xX]))*(?:(?:a|b|rc)\d+)?(?!\w|\.\w)")
NUMBER = re.compile(r"(?<![\w.])\d+(?:[,.]\d+)*(?![\w.])")
OFFICIAL = re.compile(r"\b(?:official|oficjal\w*|ofici[aá]ln\w*|offiziell\w*|officiel\w*|oficial\w*)\b", re.I)
CURRENT_RELEASE = re.compile(
    r"\b(?:stable|stabil\w*|latest|current|aktual\w*|najnowsz\w*)\b.{0,100}"
    r"\b(?:versions?|releases?|wersj\w*|wydani\w*|verz\w*)\b|"
    r"\b(?:versions?|releases?|wersj\w*|wydani\w*|verz\w*)\b.{0,100}"
    r"\b(?:stable|stabil\w*|latest|current|aktual\w*|najnowsz\w*)\b", re.I)
MONTHS = {name: i for i, names in enumerate((
    ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
    ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
    ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
    ("december", "dec")), 1) for name in names}
DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|(?:" + "|".join(MONTHS) + r")\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:" + "|".join(MONTHS) + r")\.?\s+\d{4})\b", re.I)
UNITS = {"users": "users", "user": "users", "members": "members", "member": "members",
         "requests": "requests", "request": "requests", "visits": "visits", "visit": "visits",
         "użytkowników": "users", "członków": "members", "percent": "%", "%": "%",
         "usd": "USD", "eur": "EUR", "pln": "PLN", "gbp": "GBP"}


def normalize_date(value: str) -> str | None:
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value).isoformat()
        parts = re.findall(r"[A-Za-z]+|\d+", value.casefold())
        month = next((MONTHS[p] for p in parts if p in MONTHS), None)
        numbers = [int(p) for p in parts if p.isdigit()]
        if month and len(numbers) == 2:
            return date(numbers[1], month, numbers[0]).isoformat()
    except ValueError:
        pass
    return None


def publisher(subject: str, request: str = "") -> Publisher | None:
    known = next((p for p in PUBLISHERS if p.subject.casefold() == subject.casefold()), None)
    if known:
        return known
    # An explicit owner-designated source is a separate, recorded authority
    # anchor. Never infer ownership from a search title or substring in a host.
    for sentence in re.split(r"[\n;]|\.(?=\s)", request):
        if OFFICIAL.search(sentence) and re.search(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", sentence, re.I):
            urls = URL.findall(sentence)
            if len(urls) == 1:
                host = urlsplit(urls[0]).hostname
                if host:
                    return Publisher(subject, (host,), urls[0])
    return None


def subjects_for_request(request: str) -> tuple[str, ...]:
    # Scope to the release clause, so requested report headings are not products.
    clauses = re.split(r"[\n;]|\.(?=\s)", request)
    focus = next((c for c in clauses if CURRENT_RELEASE.search(c)), request)
    subjects = [p.subject for p in PUBLISHERS if re.search(r"\b" + p.subject + r"\b", focus, re.I)]
    stop = {"find", "check", "compare", "inspect", "search", "tell", "what", "give", "please",
            "znajdź", "sprawdź", "porównaj", "wyszukaj", "podaj", "cześć", "dobra", "to",
            "current", "latest", "stable", "official", "esr", "beta", "nightly"}
    for word in re.findall(r"\b[A-Z][\w.+-]*", URL.sub("", focus)):
        if len(word) >= 3 and word.casefold() not in stop and word.casefold() not in {s.casefold() for s in subjects}:
            subjects.append(word.rstrip("."))
    return tuple(subjects)


def official_url(subject: str, url: str, request: str = "") -> bool:
    anchor = publisher(subject, request)
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        return bool(anchor and parsed.scheme == "https" and not parsed.username and
                    any(host == h or host.endswith("." + h) for h in anchor.hosts))
    except ValueError:
        return False


def catalog_target(subjects: tuple[str, ...], url: str, request: str = "") -> bool:
    return any(p and url.rstrip("/") == p.current_url.rstrip("/")
               for p in (publisher(s, request) for s in subjects))


def _clean_lines(body: str) -> list[str]:
    lines = []
    for line in body.splitlines():
        line = re.sub(r"\s+\[(?:ref|level|cursor)=[^\]]+\]", "", line)
        line = re.sub(r'^\s*-\s*(?:text|generic|paragraph|heading|listitem|link)\s*:?\s*', '', line)
        lines.append(line.strip().rstrip(':').strip().strip('"'))
    return lines


def capture_source(tool: str, result: str, arguments: dict, subjects: tuple[str, ...], request: str, observed_at: str | None = None) -> dict | None:
    """Called only by the executor, before model-facing output is clipped."""
    if tool not in {"web_read", "browser_snapshot"}:
        return None
    body = result
    source_url = ""
    if tool == "web_read":
        try:
            envelope = json.loads(result)
        except (ValueError, TypeError):
            envelope = None
        if isinstance(envelope, dict):
            if envelope.get("error"):
                return None
            body = str(envelope.get("content", ""))
            source_url = str(envelope.get("url", ""))
    actual = re.search(r"^- Page URL:\s*(\S+)", body, re.M)
    if actual:
        source_url = actual[1].rstrip('`')
    if not URL.fullmatch(source_url) or not body or re.search(r"Client Challenge|Access denied|verify you are human|captcha", body[:800], re.I):
        return None
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    source = {"url": source_url, "requested_url": str(arguments.get("url", source_url)),
              "observed_at": observed_at, "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
              "authority": {s: ("publisher_catalog" if publisher(s) else "owner_explicit_source")
                            for s in subjects if official_url(s, source_url, request)},
              "facts": [], "numeric_evidence": []}
    lines = _clean_lines(body)
    raw_lines = body.splitlines()
    today = date.fromisoformat(observed_at[:10])
    for subject in subjects:
        anchor = publisher(subject, request)
        trusted = official_url(subject, source_url, request)
        current_pointer = bool(anchor and source['requested_url'].rstrip('/') == anchor.current_url.rstrip('/') and trusted)
        versions: list[tuple[str, str]] = []
        for i, line in enumerate(lines):
            window = '\n'.join(lines[max(0, i-2):i+5])
            if re.search(r"\b(?:pre[- ]?release|beta|nightly|ESR|release candidate)\b", line, re.I):
                continue
            if (current_pointer and re.fullmatch(re.escape(subject) + r'\s+\d+\.\d+(?:\.\d+)*', line, re.I)
                    and re.search(r'\bheading\b.*\[level=1\]', raw_lines[i])):
                versions.extend((v, '\n'.join(lines[i:i+5])) for v in VERSION.findall(line))
            elif not current_pointer and re.search(r"\b(?:latest|current)\b", line, re.I):
                if not re.search(r'(?<!\w)'+re.escape(subject)+r'(?!\w)', '\n'.join(lines[i:i+3]), re.I):
                    continue
                for value in VERSION.findall(window):
                    if re.fullmatch(r"\d+\.\d+(?:\.\d+)*", value) and not re.search(r"\b(?:beta|nightly|pre[- ]?release|ESR)\b", window, re.I):
                        versions.append((value, window))
        # A pointer can also give "Version 157.0, first offered to Release ..."
        if current_pointer:
            for i, line in enumerate(lines):
                if (re.search(r"\bversion\s+\d.*(?:release channel|first offered)", line, re.I)
                        and not re.search(r'\b(?:ESR|beta|nightly|pre[- ]?release|release candidate)\b',line,re.I)):
                    versions.extend((v, '\n'.join(lines[max(0,i-1):i+3])) for v in VERSION.findall(line))
        distinct = {v for v, _ in versions}
        # Conflicting current values require another observation, not max(version).
        if len(distinct) == 1:
            value, quote = versions[0]
            source['facts'].append({"subject": subject, "field": "stable_version", "value": value,
                                    "quote": quote, "url": source_url, "observed_at": observed_at})
            dates = {d for m in DATE.finditer(quote) if (d := normalize_date(m[0]))}
            if any(date.fromisoformat(d) > today for d in dates) and re.search(r'released|release date|first offered',quote,re.I):
                source['facts'].pop()
                continue
            if len(dates) == 1 and re.search(r"release|released|offered", quote, re.I):
                source['facts'].append({"subject": subject, "field": "release_date", "value": dates.pop(),
                                        "quote": quote, "url": source_url, "observed_at": observed_at})
    # Preserve numerical context separately from the writer's abbreviated view.
    # A users count cannot become a members count, nor move to another subject.
    for i, line in enumerate(lines):
        prose = URL.sub('', line)
        local_subjects = [s for s in subjects if re.search(r'(?<!\w)' + re.escape(s) + r'(?!\w)', '\n'.join(lines[max(0,i-2):i+1]), re.I)]
        for match in NUMBER.finditer(prose):
            tail = prose[match.end():match.end()+40].strip().casefold()
            word = re.match(r'[%\w]+', tail)
            unit = UNITS.get(word[0]) if word else None
            if unit:
                for subject in local_subjects:
                    source['numeric_evidence'].append({'subject':subject,'field':'quantity','value':match[0],
                                                       'unit':unit,'quote':line,'url':source_url,'observed_at':observed_at})
    return source


def source_records(calls: list[dict], subjects: tuple[str, ...], request: str = '') -> list[dict]:
    sources = []
    for call in calls:
        if call.get('status', 'succeeded') != 'succeeded':
            continue
        stored = call.get('research_source')
        if isinstance(stored, dict):
            sources.append(stored)
            continue
        # Legacy checkpoints still receive conservative validation. Search-only
        # results never become facts; a clipped receipt may establish less.
        captured = capture_source(str(call.get('tool','')), str(call.get('result_excerpt','')),
                                  call.get('arguments',{}) or {}, subjects, request,
                                  call.get('finished_at') or call.get('started_at'))
        if captured:
            sources.append(captured)
    return sources


def fact_missing(calls: list[dict], subjects: tuple[str,...], *, official: bool, current: bool, request: str = '') -> list[str]:
    if not subjects:
        return ['research_evidence:subjects_unresolved'] if official or current else []
    sources = source_records(calls, subjects, request)
    missing = []
    for subject in subjects:
        relevant = [s for s in sources if not official or official_url(subject, s['url'], request)]
        if official and not relevant:
            missing.append('research_evidence:official_source=' + subject)
        if current:
            values = {f['value'] for s in relevant for f in s['facts'] if f['field']=='stable_version' and f['subject'].casefold()==subject.casefold()}
            if len(values) != 1:
                missing.append('research_evidence:current_stable_version=' + subject)
    return missing


def evidence_for_model(source: dict) -> str:
    view = {k:source[k] for k in ('url','observed_at','authority','facts')}
    view['numeric_evidence'] = source['numeric_evidence'][:20]
    view['numeric_evidence_total'] = len(source['numeric_evidence'])
    return '\n[RUNTIME SOURCE FACTS — values below were extracted before clipping]\n' + json.dumps(view, ensure_ascii=False)


def answer_fact_issues(answer: str, calls: list[dict], subjects: tuple[str,...], *, official: bool, current: bool, request: str = '') -> list[str]:
    sources = source_records(calls, subjects, request)
    if not sources:
        return []  # The completion contract reports missing observations.
    issues = []
    asserted_versions: set[str] = set()
    asserted_dates: set[str] = set()
    # Establish source links per subject section before reading any values.
    sections: list[tuple[str,str]] = []
    active = ''
    chunk = []
    for line in answer.splitlines():
        label = re.sub(r'^[\s#*-]+|[\s*:]+$', '', line).strip()
        heading_subject = next((s for s in subjects if label.casefold()==s.casefold()), '')
        if heading_subject:
            if chunk: sections.append((active, '\n'.join(chunk)))
            active, chunk = heading_subject, [line]
        else:
            chunk.append(line)
    if chunk: sections.append((active,'\n'.join(chunk)))
    for subject, text in sections:
        cited = set(u.rstrip('.,;:!?') for u in URL.findall(text))
        for line in text.splitlines():
            clean = URL.sub('', line)
            # Ordered-list markers are structure, not claims.
            clean = re.sub(r'^\s*\d+[.)]\s+', '', clean)
            local = next((s for s in subjects if re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)', clean, re.I)), subject)
            line_cited = set(u.rstrip('.,;:!?') for u in URL.findall(line))
            links = line_cited or cited
            relevant = [s for s in sources if (not links or s['url'].rstrip('/') in {u.rstrip('/') for u in links}) and
                        (not official or not local or official_url(local,s['url'],request))]
            if not local and len(subjects)==1: local=subjects[0]
            if current and not local and VERSION.search(clean):
                issues.append('answer:unattributed_version')
            if current and local:
                expected = {f['value'] for s in relevant for f in s['facts'] if f['field']=='stable_version' and f['subject'].casefold()==local.casefold()}
                for version in VERSION.findall(clean):
                    if version not in expected:
                        issues.append('answer:unsupported_current_version='+local+':'+version)
                    else:
                        asserted_versions.add(local)
                        if not links:
                            issues.append('answer:missing_source='+local)
                labelled = re.search(r'\b(?:version|wersj\w*|verz\w*)\s*[*:]*\s*([0-9]+)(?![\w.])',clean,re.I)
                if labelled and not any(v==labelled[1] or v==labelled[1]+'.0' for v in expected):
                    issues.append('answer:unsupported_current_version='+local+':'+labelled[1])
                if '|' in clean:
                    cells = [c.strip().strip('*') for c in clean.split('|')]
                    for cell in cells:
                        if re.fullmatch(r'\d+', cell) and not any(v==cell or v==cell+'.0' for v in expected):
                            issues.append('answer:unsupported_current_version='+local+':'+cell)
            if (current and local) or re.search(r'\b(?:release date|released|first offered|data wydania)\b',clean,re.I):
                expected_dates = {f['value'] for s in relevant for f in s['facts'] if f['field']=='release_date' and f['subject'].casefold()==local.casefold()}
                for match in DATE.finditer(clean):
                    if normalize_date(match[0]) not in expected_dates:
                        issues.append('answer:unsupported_release_date='+local+':'+match[0])
                    else:
                        asserted_dates.add(local)
            quantity_text = DATE.sub('', VERSION.sub('',clean))
            for match in NUMBER.finditer(quantity_text):
                tail = quantity_text[match.end():match.end()+40].strip().casefold()
                word = re.match(r'[%\w]+',tail)
                unit = UNITS.get(word[0]) if word else None
                if unit and not any(f['subject'].casefold()==local.casefold() and f['value']==match[0] and f['unit']==unit
                                    for s in relevant for f in s['numeric_evidence']):
                    issues.append('answer:unsupported_quantity='+local+':'+match[0]+':'+unit)
    if current:
        for subject in subjects:
            rows = [s for s in sources if not official or official_url(subject, s['url'], request)]
            values = {f['value'] for s in rows for f in s['facts'] if f['field']=='stable_version' and f['subject'].casefold()==subject.casefold()}
            if subject not in asserted_versions:
                issues.append('answer:missing_current_version='+subject)
            dates = {f['value'] for s in rows for f in s['facts'] if f['field']=='release_date' and f['subject'].casefold()==subject.casefold()}
            if dates and re.search(r'release date|date of release|dat[ęay]\s+wydania',request,re.I) and subject not in asserted_dates:
                issues.append('answer:missing_release_date='+subject)
    return list(dict.fromkeys(issues))


def render_release_facts(calls: list[dict], subjects: tuple[str,...], request: str, official: bool) -> str:
    rows = []
    for subject in subjects:
        sources = [s for s in source_records(calls,subjects,request) if not official or official_url(subject,s['url'],request)]
        versions = [f for s in sources for f in s['facts'] if f['subject'].casefold()==subject.casefold() and f['field']=='stable_version']
        values = {f['value'] for f in versions}
        if len(values)!=1:
            return ''
        fact = versions[-1]
        dates = [f for s in sources for f in s['facts'] if f['subject'].casefold()==subject.casefold() and f['field']=='release_date' and f['url']==fact['url']]
        rows.append((subject,fact['value'],dates[-1]['value'] if dates else 'Not established from the inspected source',fact['url']))
    if not rows:
        return ''
    headings = re.search(r'(?:z nagłówkami|with headings)\s+(.{2,80}?)\s+(?:oraz|and)\s+(.{2,80}?)(?:\.|\n|$)',request,re.I)
    first, second = (headings[1].strip(),headings[2].strip()) if headings else ('Verified release facts','Unresolved gaps and limitations')
    result = ['Here are the release values verified by DARKLINGER from the inspected sources.', '', '### '+first, '',
              '| Project | Stable version | Release date | Source |', '| --- | --- | --- | --- |']
    for subject,version,day,url in rows:
        result.append(f'| {subject} | {version} | {day} | {url} |')
    result.extend(['','### '+second,'','These values concern the standard stable release, not ESR, beta, nightly or release candidates. Missing dates remain explicitly unestablished.'])
    return '\n'.join(result)
