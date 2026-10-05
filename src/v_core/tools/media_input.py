"""Exact owner-selected media inputs, read without redirecting or modifying them."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
from urllib.parse import unquote, urlsplit

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
AUDIO_SUFFIXES = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | AUDIO_SUFFIXES | {'.pdf'}


def media_paths(prompt: str) -> tuple[str, ...]:
    found = []
    for match in re.finditer(r'file://[^\s<>"\']+|["\'](/[^"\']+)["\']|(?<![\w:/])/(?:[^\s<>"\']+/)*[^\s<>"\']+', prompt):
        raw = match.group(1) or match.group(0)
        if raw.startswith('file://'):
            uri = urlsplit(raw)
            if uri.netloc not in ('', 'localhost') or uri.query or uri.fragment:
                continue
            raw = unquote(uri.path)
        path = Path(raw)
        if path.is_absolute() and path.suffix.casefold() in MEDIA_SUFFIXES and '..' not in path.parts:
            found.append(str(path))
    return tuple(dict.fromkeys(found))


def read_owner_media(path: str, owner_prompt: str, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    target = Path(path)
    if str(target) not in media_paths(owner_prompt):
        raise PermissionError('Media path must be explicitly supplied by the owner')
    if target.resolve() != target:
        raise PermissionError('Symlink or redirected media targets are not accepted')
    descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError('Expected a regular, bounded media file')
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError('Media file grew beyond its read bound')
    return data


def image_paths(prompt: str) -> tuple[str, ...]:
    return tuple(path for path in media_paths(prompt) if Path(path).suffix.casefold() in IMAGE_SUFFIXES)
