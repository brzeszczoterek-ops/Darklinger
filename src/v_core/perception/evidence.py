"""Media coverage comes from processor receipts, never filenames or prose."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from ..tools.media_input import media_paths, IMAGE_SUFFIXES


def perception_inputs(prompt: str) -> tuple[str, ...]:
    transform = re.search(r'\b(?:resiz\w*|crop\w*|rotat\w*|przytn\w*|obro[ctć]\w*)\b', prompt, re.I)
    return tuple(path for path in media_paths(prompt) if not transform or Path(path).suffix.casefold() not in IMAGE_SUFFIXES)


def records(calls: list[dict], tool: str, path: str) -> list[dict]:
    found = []
    for call in calls:
        if call.get('status') != 'succeeded' or call.get('tool') != tool:
            continue
        try:
            result = json.loads(call.get('result_excerpt', ''))
        except (ValueError, TypeError):
            continue
        if isinstance(result, dict) and result.get('path') == path:
            found.append(result)
    return found


def _number(value, default=0):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else default


def _covered(intervals, tolerance=0):
    end = 0
    for left, right in sorted(intervals):
        if left > end + tolerance:
            break
        if right >= left:
            end = max(end, right)
    return end


def _pdf_gap(observed):
    total = int(max((_number(r.get('page_count')) for r in observed), default=0))
    if not 1 <= total <= 100000:
        return {'page': 1}
    for page in range(1, total + 1):
        rows = [r for r in observed if r.get('page') == page]
        if any(r.get('visual_analysis_performed') is True for r in rows):
            continue
        if not rows:
            return {'page': page}
        length = max(_number(r.get('characters_on_page')) for r in rows)
        intervals = [(_number(r.get('offset')), _number(r.get('offset')) + len(r.get('text', '')))
                     for r in rows if r.get('status') == 'read' and isinstance(r.get('text'), str)]
        end = _covered(intervals)
        if not intervals or end < length:
            return {'page': page, 'offset': int(end)}
    return None


def _audio_gap(observed):
    rows = [r for r in observed if r.get('transcription_performed') is True]
    total = max((_number(r.get('total_duration_seconds')) for r in rows), default=0)
    end = _covered([(_number(r.get('window_start_seconds')),
                     _number(r.get('window_start_seconds')) + _number(r.get('window_duration_seconds')))
                    for r in rows], tolerance=0.05)
    return end if not total or end < total - 0.05 else None


def _tool(path):
    suffix = Path(path).suffix.casefold()
    return 'document_read' if suffix == '.pdf' else 'image_analyze' if suffix in IMAGE_SUFFIXES else 'audio_transcribe'


def next_media_request(paths: tuple[str, ...], calls: list[dict], failed: list[dict]) -> tuple[str, dict] | None:
    attempted = {(c.get('tool'), c.get('arguments', {}).get('path')) for c in failed
                 if isinstance(c.get('arguments'), dict)}
    for path in paths:
        tool = _tool(path)
        if (tool, path) in attempted:
            continue
        observed = records(calls, tool, path)
        if not observed:
            return tool, {'path': path}
        if any(r.get('status') == 'unavailable' for r in observed):
            continue
        if tool == 'document_read':
            gap = _pdf_gap(observed)
            if gap is not None:
                return tool, {'path': path, **gap}
        if tool == 'audio_transcribe':
            gap = _audio_gap(observed)
            if gap is not None:
                return tool, {'path': path, 'start_seconds': gap}
    return None


def missing_media(paths: tuple[str, ...], calls: list[dict]) -> list[str]:
    missing = []
    for path in paths:
        tool = _tool(path)
        observed = records(calls, tool, path)
        if not observed or any(r.get('status') == 'unavailable' for r in observed):
            missing.append('media_evidence:unread=' + path)
        elif tool == 'image_analyze' and not any(r.get('visual_analysis_performed') is True for r in observed):
            missing.append('media_evidence:pixels=' + path)
        elif tool == 'document_read' and _pdf_gap(observed) is not None:
            missing.append('media_evidence:pages=' + path)
        elif tool == 'audio_transcribe' and _audio_gap(observed) is not None:
            missing.append('media_evidence:audio_window=' + path)
    return missing
