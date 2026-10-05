"""File transcription uses the existing Whisper stack, with timestamp receipts."""
from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import re
import tempfile
import wave

from ..branding import env_value
from ..speech.config import _resolve_command, _resolve_under
from ..speech.preferences import read_preferences


async def run_process(command: list[str]) -> str:
    process = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f'Local media processor failed ({process.returncode}): '
                           + stderr.decode(errors='replace')[-1500:])
    return stdout.decode(errors='replace')


def speech_paths(root: Path) -> tuple[Path, Path, str, int]:
    saved = read_preferences(root)
    cli = _resolve_command(saved.get('whisper_cli', env_value('DARKLINGER_WHISPER_CLI', 'whisper-cli')))
    model = _resolve_under(root, saved.get('whisper_model', env_value('DARKLINGER_WHISPER_MODEL', 'models/ggml-base.bin')))
    language = str(saved.get('whisper_language', env_value('DARKLINGER_WHISPER_LANGUAGE', 'auto')))
    threads = int(saved.get('whisper_threads', env_value('DARKLINGER_WHISPER_THREADS', '4')))
    if not re.fullmatch(r'auto|[a-z]{2}(?:-[a-z]{2})?', language) or not 1 <= threads <= 32:
        raise ValueError('Invalid local Whisper language/threads configuration')
    return cli, model, language, threads


def audio_window(window: float, duration: float, start: float):
    if not 0 <= start < duration or not 0 < window <= 300:
        raise ValueError('Audio window must start inside the file and cover 0–300 seconds')


async def normalized_audio(data: bytes, suffix: str, directory: Path, *, start: float, seconds: float) -> tuple[Path, float]:
    import imageio_ffmpeg
    if start < 0 or not 0 < seconds <= 300:
        raise ValueError('Invalid audio observation window')
    source = directory / ('input' + suffix)
    source.write_bytes(data)
    destination = directory / 'audio.wav'
    ffmpeg = os.getenv('DARKLINGER_FFMPEG') or imageio_ffmpeg.get_ffmpeg_exe()
    # Only decoding of the copied, explicitly selected media file is permitted.
    await run_process([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-protocol_whitelist', 'file,pipe',
        '-ss', str(start), '-i', str(source), '-t', str(seconds), '-vn', '-ar', '16000', '-ac', '1',
        '-c:a', 'pcm_s16le', str(destination)])
    with wave.open(str(destination)) as recording:
        duration = recording.getnframes() / recording.getframerate()
    if duration <= 0:
        raise ValueError('No audio samples in the selected window')
    return destination, duration


async def transcribe(data: bytes, suffix: str, voice_root: Path, work_root: Path,
                     *, start: float = 0, seconds: float = 60) -> dict:
    import math
    import mutagen
    info = mutagen.File(io.BytesIO(data))
    total_duration = float(info.info.length) if info is not None else None
    if total_duration is None or not math.isfinite(total_duration) or total_duration <= 0 or not 0 <= start < total_duration:
        raise ValueError("Could not establish a valid audio duration/window")
    cli, model, language, threads = speech_paths(voice_root)
    if not cli.is_file() or not model.is_file():
        return {'status': 'unavailable', 'reason': 'missing_local_whisper_cli_or_model', 'transcription_performed': False}
    work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='audio-', dir=work_root) as temporary:
        directory = Path(temporary)
        normalized, duration = await normalized_audio(data, suffix, directory, start=start, seconds=seconds)
        output = directory / 'transcript'
        await run_process([str(cli), '--model', str(model), '--file', str(normalized), '--language', language,
            '--threads', str(threads), '--output-json', '--output-file', str(output), '--no-prints'])
        result_path = output.with_suffix('.json')
        if not result_path.is_file() or result_path.stat().st_size > 1_000_000:
            raise ValueError('Whisper returned no bounded timestamped JSON transcript')
        payload = json.loads(result_path.read_text())
        segments = []
        for segment in payload.get('transcription', []):
            offsets = segment.get('offsets', {})
            text = str(segment.get('text', '')).strip()
            left, right = offsets.get('from'), offsets.get('to')
            if not text or re.fullmatch(r'\[(?:BLANK_AUDIO|silence|music)\]', text, re.I):
                continue
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)) or not 0 <= left <= right <= duration * 1000 + 100:
                raise ValueError('Whisper returned invalid audio timestamps')
            segments.append({'start_seconds': start + left / 1000, 'end_seconds': start + right / 1000, 'text': text})
        text = ' '.join(segment['text'] for segment in segments)
        if len(text) > 4000:
            raise ValueError('Transcript is too large for one evidence window; request a shorter audio window')
        result = {'status': 'transcribed' if segments else 'no_speech_detected',
                'transcription_performed': True, 'engine': 'whisper.cpp', 'model': model.name,
                'segments': segments, 'text': text, 'window_start_seconds': start,
                'window_duration_seconds': duration, 'requested_window_seconds': seconds,
                'next_start_seconds': start + duration if start + duration < total_duration - 0.05 else None,
                'total_duration_seconds': total_duration,
                'coverage': 'only the reported window', 'non_speech_classification_performed': False}
        if len(json.dumps(result)) > 5000:
            raise ValueError('Transcript exceeds the evidence window; request a shorter audio window')
        return result
