from __future__ import annotations

import re


_ARTICLE_BOUNDARY = re.compile(r"(?m)^\s*-\s*article:\s*$")
_URL = re.compile(r"(?m)^\s*-\s*/url:\s*(\S+)\s*$")
_IMAGE_TITLE = re.compile(r'(?m)^\s*-\s*img\s+"([^"]+)"\s*$')
_HEADING_TITLE = re.compile(
    r'(?m)^\s*-\s*link\s+"([^"]+)"\s*:\s*$'
)
_PRICE = re.compile(r"(?:US\$|[$£€])\s?\d[\d,.]*")
_AVAILABILITY = re.compile(
    r"\b(?:in\s+stock|out\s+of\s+stock|available|unavailable)\b",
    re.IGNORECASE,
)


def extract_accessibility_product_cards(snapshot_text: str) -> list[dict[str, str]]:
    """Extract common product-card facts from an accessibility snapshot.

    The parser deliberately consumes only literal page observations. It does
    not browse, infer missing fields, or ask a model to rewrite source data.
    """

    boundaries = list(_ARTICLE_BOUNDARY.finditer(snapshot_text))
    records: list[dict[str, str]] = []
    for index, boundary in enumerate(boundaries):
        end = (
            boundaries[index + 1].start()
            if index + 1 < len(boundaries)
            else len(snapshot_text)
        )
        block = snapshot_text[boundary.end() : end]
        url_match = _URL.search(block)
        title_match = _IMAGE_TITLE.search(block) or _HEADING_TITLE.search(block)
        price_match = _PRICE.search(block)
        availability_match = _AVAILABILITY.search(block)
        if not all((url_match, title_match, price_match, availability_match)):
            continue
        records.append(
            {
                "title": title_match.group(1),
                "price": price_match.group(0),
                "availability": availability_match.group(0),
                "relative_product_url": url_match.group(1),
            }
        )
    return records


def product_card_fixture(snapshot_text: str, *, maximum_records: int = 3) -> str:
    """Return complete leading article blocks for a bounded exact test fixture."""

    boundaries = list(_ARTICLE_BOUNDARY.finditer(snapshot_text))
    if not boundaries:
        return ""
    end_index = min(maximum_records, len(boundaries))
    end = (
        boundaries[end_index].start()
        if end_index < len(boundaries)
        else len(snapshot_text)
    )
    return snapshot_text[boundaries[0].start() : end].rstrip()


# The generated artifact is standalone because it runs in DARKLINGER's isolated
# sandbox without importing the application package. Its behavior mirrors the
# trusted builder above and is validated against a literal observed fixture.
ACCESSIBILITY_PRODUCT_CARD_SOURCE = r'''import re


ARTICLE_BOUNDARY = r"(?m)^\s*-\s*article:\s*$"
URL = r"(?m)^\s*-\s*/url:\s*(\S+)\s*$"
IMAGE_TITLE = r'(?m)^\s*-\s*img\s+"([^"]+)"\s*$'
HEADING_TITLE = r'(?m)^\s*-\s*link\s+"([^"]+)"\s*:\s*$'
PRICE = r"(?:US\$|[$£€])\s?\d[\d,.]*"
AVAILABILITY = r"\b(?:in\s+stock|out\s+of\s+stock|available|unavailable)\b"


def run(arguments):
    snapshot_text = arguments["snapshot_text"]
    boundaries = list(re.finditer(ARTICLE_BOUNDARY, snapshot_text))
    records = []
    for index, boundary in enumerate(boundaries):
        end = (
            boundaries[index + 1].start()
            if index + 1 < len(boundaries)
            else len(snapshot_text)
        )
        block = snapshot_text[boundary.end():end]
        url_match = re.search(URL, block)
        title_match = re.search(IMAGE_TITLE, block) or re.search(HEADING_TITLE, block)
        price_match = re.search(PRICE, block)
        availability_match = re.search(AVAILABILITY, block, re.IGNORECASE)
        if not all((url_match, title_match, price_match, availability_match)):
            continue
        records.append({
            "title": title_match.group(1),
            "price": price_match.group(0),
            "availability": availability_match.group(0),
            "relative_product_url": url_match.group(1),
        })
    return {"records": records}
'''
