"""Bounded, read-only JPEG/EXIF inspection using only the standard library.

This reports metadata, not visual recognition or the location of a depicted object.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import stat
import struct
from urllib.parse import unquote, urlsplit

MAX_IMAGE_BYTES = 16 * 1024 * 1024


def literal_image_paths(prompt: str) -> tuple[str, ...]:
    """Only explicit absolute JPEG targets; never basename substitution."""
    found = re.findall(
        r'file://[^\s<>"\']+|(?<![\w:/])/(?:[^\s<>"\']+/)*[^\s<>"\']+\.(?:jpe?g)\b',
        prompt, re.I,
    )
    paths = []
    for raw in found:
        if raw.startswith('file://'):
            uri = urlsplit(raw)
            if uri.netloc not in ('', 'localhost') or uri.query or uri.fragment:
                continue
            raw = unquote(uri.path)
        if (Path(raw).is_absolute() and Path(raw).suffix.lower() in ('.jpg', '.jpeg')
                and '..' not in Path(raw).parts):
            paths.append(str(Path(raw)))
    return tuple(dict.fromkeys(paths))


def _exif(tiff: bytes) -> dict:
    if len(tiff) < 8 or tiff[:2] not in (b'II', b'MM'):
        raise ValueError('Invalid EXIF byte order')
    endian = '<' if tiff[:2] == b'II' else '>'

    def unpack(fmt, offset):
        size = struct.calcsize(endian + fmt)
        if offset < 0 or offset + size > len(tiff):
            raise ValueError('Truncated EXIF field')
        return struct.unpack_from(endian + fmt, tiff, offset)

    if unpack('H', 2)[0] != 42:
        raise ValueError('Invalid EXIF TIFF header')

    def directory(offset):
        count = unpack('H', offset)[0]
        if count > 512:
            raise ValueError('Too many EXIF fields')
        values = {}
        for index in range(count):
            entry = offset + 2 + index * 12
            tag, kind, length = unpack('HHI', entry)
            unit = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}.get(kind)
            if not unit:
                continue
            size = unit * length
            if size > 65_536:
                raise ValueError('Oversized EXIF field')
            start = entry + 8 if size <= 4 else unpack('I', entry + 8)[0]
            if start + size > len(tiff):
                raise ValueError('Truncated EXIF data')
            data = tiff[start:start + size]
            if kind == 2:
                value = data.rstrip(b'\0').decode('ascii', errors='replace')
            elif kind == 1:
                value = list(data)
            elif kind in (3, 4):
                value = list(struct.unpack(endian + ('H' if kind == 3 else 'I') * length, data))
            else:
                pairs = struct.unpack(endian + 'II' * length, data)
                if any(pairs[i + 1] == 0 for i in range(0, len(pairs), 2)):
                    raise ValueError('Invalid EXIF rational denominator')
                value = [pairs[i] / pairs[i + 1] for i in range(0, len(pairs), 2)]
            values[tag] = value
        return values

    main = directory(unpack('I', 4)[0])
    result = {name: main[tag][:128] for tag, name in ((271, 'make'), (272, 'model'), (306, 'datetime'))
              if isinstance(main.get(tag), str)}
    if any(isinstance(main.get(tag), str) and len(main[tag]) > 128 for tag in (271, 272, 306)):
        result['metadata_text_truncated'] = True
    if isinstance(main.get(274), list) and main[274]:
        result['orientation'] = main[274][0]
    result['gps_status'] = 'absent'
    result['gps'] = None
    if 34853 not in main:
        return result
    pointer = main[34853]
    if not isinstance(pointer, list) or len(pointer) != 1:
        raise ValueError('Invalid GPS directory pointer')
    gps = directory(pointer[0])

    def coordinate(tag, ref_tag, references, maximum):
        parts, reference = gps.get(tag), gps.get(ref_tag)
        if (not isinstance(parts, list) or len(parts) != 3 or reference not in references
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in parts)
                or not (0 <= parts[0] <= maximum and 0 <= parts[1] < 60 and 0 <= parts[2] < 60)):
            raise ValueError('Invalid or incomplete GPS coordinates')
        value = parts[0] + parts[1] / 60 + parts[2] / 3600
        if value > maximum:
            raise ValueError('GPS coordinate out of range')
        return -value if reference == references[1] else value

    latitude = coordinate(2, 1, ('N', 'S'), 90)
    longitude = coordinate(4, 3, ('E', 'W'), 180)
    result.update(gps_status='present', gps={'latitude': latitude, 'longitude': longitude,
                  'meaning': 'Recorded camera position; not proof of the depicted object location.'})
    return result


def inspect_jpeg(data: bytes) -> dict:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Image exceeds 16 MiB metadata inspection limit')
    if not data.startswith(b'\xff\xd8'):
        raise ValueError('Only JPEG images are supported by this metadata reader')
    result = {'format': 'JPEG', 'width': None, 'height': None,
              'exif_present': False, 'gps_status': 'absent', 'gps': None,
              'visual_analysis_performed': False}
    position = 2
    seen_exif = False
    while position < len(data):
        if data[position] != 255:
            raise ValueError('Invalid JPEG marker')
        while position < len(data) and data[position] == 255:
            position += 1
        if position == len(data):
            raise ValueError('Truncated JPEG marker')
        marker = data[position]
        position += 1
        if marker in (0xD9, 0xDA):
            break
        if marker in range(0xD0, 0xD9) or marker == 1:
            continue
        if position + 2 > len(data):
            raise ValueError('Truncated JPEG segment')
        size = int.from_bytes(data[position:position + 2], 'big')
        if size < 2 or position + size > len(data):
            raise ValueError('Invalid JPEG segment length')
        segment = data[position + 2:position + size]
        if marker == 0xE1 and segment.startswith(b'Exif\0\0'):
            if seen_exif:
                raise ValueError('Ambiguous duplicate EXIF segments')
            seen_exif = True
            result['exif_present'] = True
            try:
                result.update(_exif(segment[6:]))
            except (ValueError, struct.error) as error:
                result.update(gps_status='unreadable', gps=None, exif_error=str(error))
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if len(segment) < 6:
                raise ValueError('Truncated JPEG dimensions')
            result['height'] = int.from_bytes(segment[1:3], 'big')
            result['width'] = int.from_bytes(segment[3:5], 'big')
        position += size
    if not result['width'] or not result['height']:
        raise ValueError('JPEG has no valid dimensions')
    return result


def read_owner_jpeg(path: str, owner_prompt: str) -> tuple[bytes, dict]:
    requested = str(Path(path))
    if requested not in literal_image_paths(owner_prompt):
        raise PermissionError('Image path must be explicitly supplied by the owner in this interaction')
    target = Path(requested)
    # No symlink, directory expansion, fallback path, or foreign substitute.
    if target.resolve() != target:
        raise PermissionError('Symlink or redirected image targets are not accepted')
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_IMAGE_BYTES:
            raise ValueError('Expected a regular JPEG file no larger than 16 MiB')
        data = handle.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError('JPEG grew beyond the image size bound')
        result = inspect_jpeg(data)
    return data, {'path': requested, 'bytes': len(data), **result}


def read_image_metadata(path: str, owner_prompt: str) -> dict:
    return read_owner_jpeg(path, owner_prompt)[1]
