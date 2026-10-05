"""Preserve observed text in literal-reading tasks, without guessing new facts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from .evidence import missing_media, perception_inputs, records
from ..tools.media_input import AUDIO_SUFFIXES
from ..capabilities.web_target import extract_web_targets


_TRANSCRIBE = re.compile(r'\b(?:transcrib\w*|transcript\w*|przepisz|transkry\w*)\b', re.I)
_LITERAL = re.compile(
    r'\b(?:transcrib\w*|transcript\w*|przepisz|transkry\w*|ocr|odczytaj)\b'
    r'|\b(?:read|copy|quote)\b.{0,100}\b(?:text|sign|heading|label|words)\b'
    r'|\b(?:podaj|napisz)\b.{0,100}\b(?:napis|tekst|nagłówek|naglowek)\b', re.I)
_TRANSFORM = re.compile(
    r'\b(?:translat\w*|summari\w*|compar\w*|analy[sz]\w*|explain\w*|'
    r'streść|stresc|podsumuj|przetłumacz|przetlumacz|porównaj|porownaj|wyjaśnij|wyjasnij)\b', re.I)
_QUOTED = re.compile(r'"([^"\n]{3,512})"|“([^”\n]{3,512})”|`([^`\n]{3,512})`')
_OTHER_TASK = re.compile(r'\b(?:create|generate|save|open|inspect|browse|stwórz|utwórz|zapisz|otwórz|obejrzyj)\b|https?://', re.I)


def _literal_request(request: str) -> bool:
    request = _intent_text(request)
    # A prohibition on translation is not a request to translate.
    request = re.sub(r"\b(?:do not|don't|nie)\s+" + _TRANSFORM.pattern, '', request, flags=re.I)
    return bool(_LITERAL.search(request)) and not _TRANSFORM.search(request)


def _intent_text(request: str) -> str:
    for path in perception_inputs(request):
        request = request.replace(Path(path).as_uri(), '').replace(path, '')
    return request


def _plain_transcript_request(request: str) -> bool:
    paths = perception_inputs(request)
    return bool(paths) and all(Path(p).suffix.casefold() in AUDIO_SUFFIXES for p in paths) and bool(
        _TRANSCRIBE.search(_intent_text(request))) and not _OTHER_TASK.search(_intent_text(request))


def _transcript(request: str, calls: list[dict]) -> str | None:
    paths = perception_inputs(request)
    if not _plain_transcript_request(request) or missing_media(paths, calls):
        return None
    transcripts = []
    for path in paths:
        rows = records(calls, 'audio_transcribe', path)
        if not rows or any(r.get('transcription_performed') is not True for r in rows):
            return None
        # Repeated or overlapping windows cannot be concatenated by guessing.
        hashes = [r.get('source_sha256') for r in rows]
        if any(not isinstance(h, str) or not re.fullmatch(r'[a-f0-9]{64}', h) for h in hashes) or len(set(hashes)) != 1:
            return None
        windows = {}
        for row in rows:
            key = (row.get('window_start_seconds'), row.get('window_duration_seconds'))
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in key) or key[1] == 0:
                return None
            text = row.get('text')
            if not isinstance(text, str) or not text.strip() or key in windows and windows[key] != text:
                return None
            windows[key] = text
        end = 0.0
        pieces = []
        for (start, duration), text in sorted(windows.items()):
            if abs(start - end) > 0.05:
                return None
            pieces.append(text)
            end = start + duration
        transcripts.append('\n'.join(pieces))
    return '\n\n'.join(transcripts)


def _observed_literals(calls: list[dict]) -> set[str]:
    literals = set()
    for call in calls:
        if call.get('status') != 'succeeded':
            continue
        tool, raw = call.get('tool'), call.get('result_excerpt', '')
        if not isinstance(raw, str):
            continue
        if tool == 'browser_snapshot':
            literals.update(re.findall(r'- heading "([^"\n]{3,512})"', raw))
            continue
        if tool not in {'image_analyze', 'browser_vision', 'document_read'}:
            continue
        try:
            record = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict):
            continue
        visual = record.get('visual_analysis_performed') is True
        if not visual and not (tool == 'document_read' and record.get('status') == 'read'):
            continue
        text = record.get('observations' if visual else 'text', '')
        if not isinstance(text, str):
            continue
        literals.update(next(g for g in m.groups() if g) for m in _QUOTED.finditer(text))
        if not visual or record.get('mode') == 'ocr':
            literals.update(line.strip() for line in text.splitlines() if 3 <= len(line.strip()) <= 512)
    return literals


def _plain_viewport_request(request: str) -> bool:
    targets = extract_web_targets(request)
    if len(targets) != 1 or perception_inputs(request):
        return False
    return not re.search(r'\b(?:create|generate|save|stwórz|utwórz|zapisz|why|dlaczego|if|jeśli|czy|judge|oceń)\b', _intent_text(request), re.I)


def _viewport_observation(request: str, calls: list[dict]) -> str | None:
    targets = extract_web_targets(request)
    if not _plain_viewport_request(request):
        return None
    observations = set()
    for call in calls:
        if call.get('tool') != 'browser_vision' or call.get('status') != 'succeeded':
            continue
        try:
            record = json.loads(call.get('result_excerpt', ''))
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or record.get('visual_analysis_performed') is not True:
            continue
        # Do not use another page's pixels or choose between competing captures.
        if record.get('page_url', '').rstrip('/') != targets[0].rstrip('/'):
            continue
        text = record.get('observations')
        if isinstance(text, str) and text.strip() and len(text) <= 4500:
            observations.add(text)
    return next(iter(observations)) if len(observations) == 1 else None


def _source_block(label: str, text: str) -> str:
    fence = '`' * max(3, 1 + max((len(m.group()) for m in re.finditer(r'`+', text)), default=0))
    return f'{label}:\n\n{fence}text\n{text}\n{fence}'


def preserve_literal_report(request: str, answer: str, calls: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Preserve plain transcripts/viewports, or repair unique literal copy errors.

    Only literal-reading requests qualify. All nonnumeric characters must
    match an observed string; competing observations block repair. This is
    not a general fact checker or a certification of OCR accuracy.
    """
    if not _literal_request(request):
        return answer, [], []
    transcript = _transcript(request, calls)
    if transcript is not None:
        # The source is data. A long enough fence prevents embedded Markdown
        # from escaping the transcript block.
        rendered = _source_block('Transcript', transcript)
        return rendered, [{'kind': 'transcript_from_receipts', 'model_report_accepted': False}], []
    if _plain_transcript_request(request):
        return answer, [], ['answer:transcript_receipts_not_complete']
    viewport = _viewport_observation(request, calls)
    if viewport is not None:
        return _source_block('Visual observation (unverified; current viewport)', viewport), [
            {'kind': 'viewport_from_receipts', 'model_report_accepted': False}], []
    if _plain_viewport_request(request) and any(c.get('tool') == 'browser_vision' for c in calls):
        return answer, [], ['answer:viewport_source_not_unique']
    candidates: dict[tuple[int, int], set[str]] = {}
    for literal in sorted(_observed_literals(calls)):
        letters = re.findall(r'[^\W\d_]+', literal)
        if sum(map(len, letters)) < 3 or len(letters) < 2 and not re.search(r'\d', literal):
            continue
        parts = re.split(r'(\d+|\s+)', literal)
        pattern = ''.join(r'\d+' if p.isdigit() else r'\s*' if p.isspace() else re.escape(p) for p in parts)
        for match in re.finditer(r'(?<!\w)' + pattern + r'(?!\w)', answer, re.I):
            prefix = answer[max(0, match.start() - 100):match.start()]
            if re.search(r"\b(?:not|isn't|incorrect|wrong|nie|błędny|bledny)\b[^.!?\n]{0,50}$", prefix, re.I):
                continue
            candidates.setdefault(match.span(), set()).add(literal)
    repairs, issues = [], []
    replacements = []
    for (start, end), sources in sorted(candidates.items()):
        candidate = answer[start:end]
        if any(candidate.casefold() == source.casefold() for source in sources):
            continue
        if len(sources) != 1:
            issues.append('answer:perception_literal_ambiguous')
            continue
        if replacements and start < replacements[-1][1]:
            # Nested observations need a more specific receipt, not a guess.
            issues.append('answer:perception_literal_ambiguous')
            continue
        observed = next(iter(sources))
        replacements.append((start, end, observed))
        repairs.append({'kind': 'observed_literal', 'candidate': candidate, 'observed': observed})
    for start, end, observed in reversed(replacements):
        answer = answer[:start] + observed + answer[end:]
    return answer, repairs, list(dict.fromkeys(issues))
