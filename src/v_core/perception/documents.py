"""Page-addressed PDF text and raster observations, with explicit coverage."""
from __future__ import annotations

import io
import threading

from pypdf import PdfReader

_PDFIUM_LOCK = threading.Lock()


def pdf_page(data: bytes, page: int, offset: int, max_characters: int = 4000) -> dict:
    if not data.startswith(b'%PDF-'):
        raise ValueError('Input is not a PDF document')
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise ValueError('PDF is encrypted; decrypt an owner-authorized copy first')
    total = len(reader.pages)
    if not 1 <= page <= total or offset < 0:
        raise ValueError('Requested PDF page or character offset is outside the document')
    text = reader.pages[page - 1].extract_text() or ''
    if offset > len(text):
        raise ValueError('Character offset exceeds this page text')
    end = min(len(text), offset + max_characters)
    return {'page': page, 'page_count': total, 'method': 'pdf_text', 'text': text[offset:end],
            'characters_on_page': len(text), 'offset': offset,
            'next_offset': end if end < len(text) else None,
            'next_page': page + 1 if page < total else None,
            'page_complete': offset == 0 and end == len(text),
            'document_complete': page == total == 1 and offset == 0 and end == len(text),
            'has_text_layer': bool(text.strip())}


def render_pdf_page(data: bytes, page: int) -> bytes:
    import pypdfium2 as pdfium
    # PDFium is not thread-safe. Every document/render/destruction operation is
    # serialized while CPU work is kept off the async loop.
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(data)
        try:
            target = document[page - 1]
            try:
                width, height = target.get_size()
                scale = min(2.0, (4_000_000 / max(1, width * height)) ** 0.5)
                bitmap = target.render(scale=scale)
                try:
                    image = bitmap.to_pil()
                    with io.BytesIO() as buffer:
                        image.save(buffer, format='PNG')
                        return buffer.getvalue()
                finally:
                    bitmap.close()
            finally:
                target.close()
        finally:
            document.close()
