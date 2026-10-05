"""Owner-only continuity for a bounded, visible photo investigation."""
from __future__ import annotations

import re
import unicodedata

from ..tools.media_input import image_paths as literal_image_paths


def folded(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                   if not unicodedata.combining(c)).replace('ł', 'l')


def photo_location_request(text: str) -> bool:
    for path in literal_image_paths(text):
        text = text.replace(path, ' ')
    text = re.sub(r'file://\S+', ' ', text, flags=re.I)
    return bool(re.search(r'\b(?:where|locat\w*|geolocat\w*|gdzie|lokac\w*|'
                          r'(?:z|geo)?lokaliz\w*|place|miejsce|miejsca)\b', folded(text)))


def photo_followup(text: str) -> bool:
    if text.strip().startswith('/'):
        return False
    return bool(re.search(r'\b(?:photos?|images?|pictures?|zdjec\w*|fotograf\w*|'
                          r'dzielnic\w*|lokac\w*|lokaliz\w*|google maps|openstreetmap\w*|'
                          r'(?:this|that) (?:place|district)|tego miejsca|tej dzielnicy|'
                          r'jest to|it is (?:in|near)|co jeszcze potrzebujesz)\b', folded(text)))


def owner_photo_context(prompt: str, messages: list[dict]) -> str:
    """Never use assistant claims, archived unrelated work, or inferred paths.

    New literal photos replace the earlier input. A follow-up can refer through
    adjacent owner clarifications, but an unrelated owner turn ends that chain.
    """
    if literal_image_paths(prompt) or not photo_followup(prompt):
        return prompt
    owners = [str(m.get('content', '')) for m in messages if m.get('role') == 'user']
    chain = [prompt]
    for previous in reversed(owners[-5:]):
        if literal_image_paths(previous):
            if photo_location_request(previous):
                return '\n\nOwner follow-up: '.join([previous, *reversed(chain)])
            break
        if not photo_followup(previous):
            break
        chain.append(previous)
    return prompt


def location_query(text: str) -> str:
    """Use explicit place names, never map providers as the search subject."""
    clean = re.sub(r'file://\S+|https?://\S+', ' ', text, flags=re.I)
    for path in literal_image_paths(text):
        clean = clean.replace(path, ' ')
    clean = re.sub(r'\b(?:Google\s+Maps|OpenStreetMaps?|Owner follow-up|Follow-up)\b', ' ', clean, flags=re.I)
    # Preserve owner spelling (including inflected names). Search engines can
    # resolve that spelling; the runtime must not invent a translated district.
    places = []
    for match in re.finditer(r'\b(?:in|near|around|district|border|w|we|dzielnic\w*|'
                             r'pogranicz\w*|jest to)\s+((?:[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż-]*)(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż-]*)*)', clean):
        places.append(match.group(1))
    return ' '.join(dict.fromkeys(places)) + ' street photographs landmarks' if places else ''


def relevant_image_evidence(evidence: str, subjects: str = '') -> bool:
    """An image label must describe a subject; branding and portraits do not.

    Accessibility labels prove only that a reference is present, not that its
    pixels match the supplied photo. Visual matching remains a separate stage.
    """
    labels = re.findall(r'\b(?:img|image)\s+["\']([^"\']+)["\']|!\[([^\]]+)\]\(', evidence, re.I)
    words = [folded(w)[:5] for w in re.findall(r'[^\W_]+', subjects) if len(w) >= 4]
    for pair in labels:
        label = next((part for part in pair if part), '')
        name = folded(label)
        if re.search(r'\b(?:logo|avatar|author|autork\w*|autor\w*|portrait|profile|icon|banner)\b', name):
            continue
        if words and any(word in name for word in words):
            return True
        if not words and len(re.findall(r'[^\W_]+', name)) >= 2:
            return True
    return False
