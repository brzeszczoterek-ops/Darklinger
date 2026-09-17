from __future__ import annotations

import asyncio
from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import signal
import tempfile
import time

from .config import SpeechConfig


class SpeechRuntimeError(RuntimeError):
    """Raised when recording, transcription, synthesis, or playback fails."""


class NoSpeechDetected(SpeechRuntimeError):
    """Raised when the microphone did not receive speech before its timeout."""


@dataclass(slots=True)
class VoiceActivityDetector:
    threshold: float
    minimum_speech_seconds: float
    end_silence_seconds: float
    speech_seconds: float = 0.0
    silence_seconds: float = 0.0
    started: bool = False

    def feed(self, pcm: bytes, duration: float) -> bool:
        if not pcm or duration <= 0:
            return False
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
        if not samples:
            return False
        if array("H", [1]).tobytes() == b"\x00\x01":
            samples.byteswap()

        square_mean = sum(sample * sample for sample in samples) / len(samples)
        rms = math.sqrt(square_mean) / 32768.0
        if rms >= self.threshold:
            self.speech_seconds += duration
            self.silence_seconds = 0.0
            if self.speech_seconds >= self.minimum_speech_seconds:
                self.started = True
        elif self.started:
            self.silence_seconds += duration

        return self.started and self.silence_seconds >= self.end_silence_seconds


class SpeechRuntime:
    """Local half-duplex speech input and output for V."""

    def __init__(self, config: SpeechConfig):
        self.config = config
        self._ptt_process: asyncio.subprocess.Process | None = None
        self._ptt_directory: tempfile.TemporaryDirectory[str] | None = None
        self._ptt_recording: Path | None = None
        self._ptt_watchdog: asyncio.Task[None] | None = None
        self._kokoro_process: asyncio.subprocess.Process | None = None
        self._kokoro_lock = asyncio.Lock()
        self._kokoro_request = 0
        self.last_tts_fallback_reason: str | None = None

    @property
    def push_to_talk_recording(self) -> bool:
        return self._ptt_process is not None

    async def listen(self) -> str:
        with tempfile.TemporaryDirectory(prefix="darklinger-stt-") as directory:
            recording = Path(directory) / "utterance.wav"
            await self._record_until_silence(recording)
            return await self._transcribe(recording)

    async def start_push_to_talk(self) -> None:
        if self.push_to_talk_recording:
            raise SpeechRuntimeError("Push-to-talk recording is already active")
        directory = tempfile.TemporaryDirectory(prefix="darklinger-ptt-")
        recording = Path(directory.name) / "utterance.wav"
        try:
            process = await asyncio.create_subprocess_exec(
                *self._recorder_command(recording),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.sleep(0.08)
            if process.returncode is not None:
                detail = ""
                if process.stderr is not None:
                    detail = (
                        await process.stderr.read()
                    ).decode("utf-8", errors="replace").strip()
                raise SpeechRuntimeError(
                    "Microphone recording could not start"
                    + (f": {detail}" if detail else "")
                )
        except BaseException:
            directory.cleanup()
            raise

        self._ptt_process = process
        self._ptt_directory = directory
        self._ptt_recording = recording
        self._ptt_watchdog = asyncio.create_task(
            self._limit_push_to_talk(process)
        )

    async def stop_push_to_talk(self) -> str:
        process = self._ptt_process
        recording = self._ptt_recording
        directory = self._ptt_directory
        watchdog = self._ptt_watchdog
        if process is None or recording is None or directory is None:
            raise SpeechRuntimeError("Push-to-talk recording is not active")

        self._ptt_process = None
        self._ptt_recording = None
        self._ptt_directory = None
        self._ptt_watchdog = None
        if watchdog is not None:
            watchdog.cancel()
        try:
            await self._stop_recorder(process)
            minimum_bytes = int(
                self.config.sample_rate
                * 2
                * self.config.minimum_speech_seconds
            )
            try:
                payload_bytes = max(0, recording.stat().st_size - 44)
            except OSError as error:
                raise SpeechRuntimeError(
                    "Push-to-talk recording was not created"
                ) from error
            if payload_bytes < minimum_bytes:
                raise NoSpeechDetected("That recording was too short. Try again.")
            return await self._transcribe(recording)
        finally:
            directory.cleanup()

    async def cancel_push_to_talk(self) -> None:
        process = self._ptt_process
        directory = self._ptt_directory
        watchdog = self._ptt_watchdog
        self._ptt_process = None
        self._ptt_recording = None
        self._ptt_directory = None
        self._ptt_watchdog = None
        if watchdog is not None:
            watchdog.cancel()
        if process is not None:
            await self._stop_recorder(process)
        if directory is not None:
            directory.cleanup()

    async def close(self) -> None:
        try:
            await self.cancel_push_to_talk()
        finally:
            await self._stop_kokoro_worker()

    async def speak(self, text: str) -> None:
        spoken = self._prepare_spoken_text(text)
        if not spoken:
            return

        self.last_tts_fallback_reason = None
        if self.config.voice.engine == "kokoro":
            try:
                await self._speak_kokoro(spoken)
                return
            except SpeechRuntimeError as error:
                await self._stop_kokoro_worker()
                fallback = self.config.voice.fallback
                if fallback is None:
                    raise
                self.last_tts_fallback_reason = str(error)
                await self._speak_piper(
                    spoken,
                    model=fallback.model,
                    model_config=fallback.model_config,
                    effects=fallback.effects,
                )
                return

        model_config = self.config.voice.model_config
        if model_config is None:
            raise SpeechRuntimeError("Selected Piper voice has no model config")
        await self._speak_piper(
            spoken,
            model=self.config.voice.model,
            model_config=model_config,
            effects=self.config.voice.effects,
        )

    async def _speak_piper(
        self,
        spoken: str,
        *,
        model: Path,
        model_config: Path,
        effects: tuple[str, ...],
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="darklinger-tts-") as directory:
            clean = Path(directory) / "clean.wav"
            rendered = Path(directory) / "rendered.wav"

            await self._run(
                [
                    str(self.config.piper),
                    "--model",
                    str(model),
                    "--config",
                    str(model_config),
                    "--output-file",
                    str(clean),
                ],
                input_text=spoken,
                timeout=max(30.0, len(spoken) * 0.12),
                operation="Piper synthesis",
            )

            playback = clean
            if effects:
                await self._run(
                    [
                        str(self.config.sox),
                        str(clean),
                        str(rendered),
                        *effects,
                    ],
                    timeout=max(20.0, len(spoken) * 0.05),
                    operation="voice texturing",
                )
                playback = rendered

            await self._play(playback, spoken_length=len(spoken))

    async def _speak_kokoro(self, spoken: str) -> None:
        async with self._kokoro_lock:
            process = await self._ensure_kokoro_worker()
            if process.stdin is None or process.stdout is None:
                raise SpeechRuntimeError("Kokoro worker has no local I/O channel")

            self._kokoro_request += 1
            request_id = f"speech-{self._kokoro_request}"
            with tempfile.TemporaryDirectory(prefix="darklinger-kokoro-") as directory:
                request = {
                    "command": "synthesize",
                    "id": request_id,
                    "text": spoken,
                    "output_dir": directory,
                }
                try:
                    process.stdin.write(
                        (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
                    )
                    await process.stdin.drain()
                    chunks = 0
                    timeout = max(90.0, len(spoken) * 0.30)
                    async with asyncio.timeout(timeout):
                        while True:
                            line = await process.stdout.readline()
                            if not line:
                                raise await self._kokoro_exit_error(process)
                            message = self._decode_worker_message(line)
                            if message.get("id") != request_id:
                                raise SpeechRuntimeError(
                                    "Kokoro worker returned an out-of-order response"
                                )
                            event = message.get("event")
                            if event == "error":
                                raise SpeechRuntimeError(
                                    f"Kokoro synthesis failed: {message.get('message', 'unknown error')}"
                                )
                            if event == "done":
                                if chunks == 0:
                                    raise SpeechRuntimeError("Kokoro returned no audio chunks")
                                return
                            if event != "chunk":
                                raise SpeechRuntimeError(
                                    f"Unexpected Kokoro worker event: {event}"
                                )
                            chunk = self._validated_kokoro_chunk(
                                message.get("path"), Path(directory)
                            )
                            chunks += 1
                            await self._play(chunk, spoken_length=len(spoken))
                            chunk.unlink(missing_ok=True)
                except TimeoutError as error:
                    raise SpeechRuntimeError("Kokoro synthesis timed out") from error
                except (BrokenPipeError, ConnectionError) as error:
                    raise SpeechRuntimeError("Kokoro worker stopped unexpectedly") from error
                except BaseException:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                    raise

    async def _ensure_kokoro_worker(self) -> asyncio.subprocess.Process:
        process = self._kokoro_process
        if process is not None and process.returncode is None:
            return process

        voice = self.config.voice
        if voice.python is None or voice.voices is None:
            raise SpeechRuntimeError("Kokoro voice profile is incomplete")
        worker = Path(__file__).with_name("kokoro_worker.py")
        process = await asyncio.create_subprocess_exec(
            str(voice.python),
            str(worker),
            "--model",
            str(voice.model),
            "--voices",
            str(voice.voices),
            "--voice",
            voice.voice_id,
            "--language",
            voice.language,
            "--speed",
            str(voice.speed),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._kokoro_process = process
        if process.stdout is None:
            raise SpeechRuntimeError("Kokoro worker has no output channel")
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30.0)
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise SpeechRuntimeError("Kokoro worker startup timed out") from error
        if not line:
            raise await self._kokoro_exit_error(process)
        message = self._decode_worker_message(line)
        if message.get("event") != "ready":
            process.kill()
            await process.wait()
            raise SpeechRuntimeError("Kokoro worker did not become ready")
        return process

    async def _stop_kokoro_worker(self) -> None:
        process = self._kokoro_process
        self._kokoro_process = None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            try:
                process.stdin.write(b'{"command":"shutdown"}\n')
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionError):
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _kokoro_exit_error(
        self, process: asyncio.subprocess.Process
    ) -> SpeechRuntimeError:
        detail = ""
        if process.stderr is not None:
            try:
                detail = (
                    await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                ).decode("utf-8", errors="replace").strip()
            except TimeoutError:
                detail = ""
        return SpeechRuntimeError(
            "Kokoro worker stopped"
            + (f": {detail[-800:]}" if detail else " unexpectedly")
        )

    @staticmethod
    def _decode_worker_message(line: bytes) -> dict[str, object]:
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SpeechRuntimeError("Kokoro worker returned invalid data") from error
        if not isinstance(message, dict):
            raise SpeechRuntimeError("Kokoro worker response is not an object")
        return message

    @staticmethod
    def _validated_kokoro_chunk(value: object, directory: Path) -> Path:
        if not isinstance(value, str):
            raise SpeechRuntimeError("Kokoro worker returned no audio path")
        try:
            chunk = Path(value).resolve(strict=True)
            expected_parent = directory.resolve(strict=True)
        except OSError as error:
            raise SpeechRuntimeError("Kokoro audio chunk does not exist") from error
        if chunk.parent != expected_parent or chunk.suffix.casefold() != ".wav":
            raise SpeechRuntimeError("Kokoro worker returned an unsafe audio path")
        return chunk

    async def _play(self, playback: Path, *, spoken_length: int) -> None:

        command = [str(self.config.player)]
        if self.config.output_target:
            command.extend(["--target", self.config.output_target])
        command.append(str(playback))
        await self._run(
            command,
            timeout=max(30.0, spoken_length * 0.20),
            operation="audio playback",
        )

    async def _record_until_silence(self, destination: Path) -> None:
        process = await asyncio.create_subprocess_exec(
            *self._recorder_command(destination),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        detector = VoiceActivityDetector(
            threshold=self.config.speech_threshold,
            minimum_speech_seconds=self.config.minimum_speech_seconds,
            end_silence_seconds=self.config.end_silence_seconds,
        )
        started_at = time.monotonic()
        offset = 44
        last_sample_time = started_at

        try:
            while True:
                await asyncio.sleep(0.08)
                now = time.monotonic()
                if process.returncode is not None:
                    if process.returncode != 0:
                        detail = ""
                        if process.stderr is not None:
                            detail = (
                                await process.stderr.read()
                            ).decode("utf-8", errors="replace").strip()
                        raise SpeechRuntimeError(
                            "Microphone recording stopped unexpectedly"
                            + (f": {detail}" if detail else "")
                        )
                    break
                if destination.exists() and destination.stat().st_size > offset:
                    with destination.open("rb") as handle:
                        handle.seek(offset)
                        pcm = handle.read()
                    offset += len(pcm)
                    duration = len(pcm) / (self.config.sample_rate * 2)
                    if detector.feed(pcm, duration):
                        break
                    last_sample_time = now

                elapsed = now - started_at
                if not detector.started and elapsed >= self.config.start_timeout_seconds:
                    raise NoSpeechDetected("I didn't hear any speech. Try again.")
                if elapsed >= self.config.maximum_record_seconds:
                    break

                if now - last_sample_time > 2.0:
                    raise SpeechRuntimeError("The microphone stopped delivering audio")
        finally:
            await self._stop_recorder(process)

        if not detector.started:
            raise NoSpeechDetected("I didn't hear any speech. Try again.")

    def _recorder_command(self, destination: Path) -> list[str]:
        command = [
            str(self.config.recorder),
            "--rate",
            str(self.config.sample_rate),
            "--channels",
            "1",
            "--format",
            "s16",
        ]
        if self.config.input_target:
            command.extend(["--target", self.config.input_target])
        command.append(str(destination))
        return command

    async def _limit_push_to_talk(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        try:
            await asyncio.sleep(self.config.maximum_record_seconds)
            if self._ptt_process is process:
                await self._stop_recorder(process)
        except asyncio.CancelledError:
            return

    async def _transcribe(self, recording: Path) -> str:
        command = self._whisper_command(
            self.config.whisper_cli,
            self.config.whisper_model,
            recording,
        )
        try:
            output = await self._run(
                command,
                timeout=max(60.0, self.config.maximum_record_seconds * 2),
                operation="Whisper transcription",
            )
        except SpeechRuntimeError as primary_error:
            fallback_cli = self.config.whisper_fallback_cli
            fallback_model = self.config.whisper_fallback_model
            if fallback_cli is None or fallback_model is None:
                raise
            try:
                output = await self._run(
                    self._whisper_command(fallback_cli, fallback_model, recording),
                    timeout=max(60.0, self.config.maximum_record_seconds * 2),
                    operation="Whisper CPU fallback transcription",
                )
            except SpeechRuntimeError as fallback_error:
                raise SpeechRuntimeError(
                    f"Primary STT failed ({primary_error}); "
                    f"fallback also failed ({fallback_error})"
                ) from fallback_error

        transcript = " ".join(
            line.strip() for line in output.splitlines() if line.strip()
        )
        transcript = re.sub(
            r"\[[^]]*(?:music|silence|blank_audio)[^]]*]",
            "",
            transcript,
            flags=re.I,
        )
        transcript = transcript.strip()
        if not transcript:
            raise NoSpeechDetected("Whisper could not find speech in that recording")
        return transcript

    def _whisper_command(
        self,
        cli: Path,
        model: Path,
        recording: Path,
    ) -> list[str]:
        command = [
            str(cli),
            "--model",
            str(model),
            "--file",
            str(recording),
            "--language",
            self.config.whisper_language,
            "--threads",
            str(self.config.whisper_threads),
            "--flash-attn",
            "--no-prints",
            "--no-timestamps",
            "--suppress-nst",
        ]
        if self.config.whisper_initial_prompt:
            command.extend(["--prompt", self.config.whisper_initial_prompt])
        return command

    async def _run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        timeout: float,
        operation: str,
    ) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(
                    input_text.encode("utf-8") if input_text is not None else None
                ),
                timeout=timeout,
            )
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise SpeechRuntimeError(f"{operation} timed out") from error
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise SpeechRuntimeError(
                f"{operation} failed with exit code {process.returncode}: {detail}"
            )
        return stdout.decode("utf-8", errors="replace")

    @staticmethod
    async def _stop_recorder(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    @staticmethod
    def _prepare_spoken_text(text: str) -> str:
        text = re.sub(
            r"```.*?```",
            " I left the code block in the terminal. ",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"\[([^]]+)]\((?:[^()]|\([^)]*\))+\)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Strip paired emphasis, not arithmetic or underscores inside names.
        text = re.sub(r"(\*{1,3})(\S(?:.*?\S)?)\1", r"\2", text)
        text = re.sub(r"(?<!\w)(_{1,3})(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
        text = re.sub(r"~~(\S(?:.*?\S)?)~~", r"\1", text)
        text = re.sub(r"^[#>*+-]+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
