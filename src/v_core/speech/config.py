from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from ..branding import env_value


class SpeechConfigurationError(RuntimeError):
    """Raised when the local speech stack is missing or malformed."""


def _resolve_command(value: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        return expanded.absolute()
    discovered = shutil.which(value)
    # pw-record and pw-play are mode-selecting symlinks to pw-cat. Preserve the
    # invoked filename instead of resolving the symlink and losing that mode.
    return Path(discovered).absolute() if discovered else expanded


def _resolve_under(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_executable_under(root: Path, value: str) -> Path:
    """Keep virtual-environment launcher symlinks intact."""
    path = Path(value).expanduser()
    combined = path if path.is_absolute() else root / path
    return Path(os.path.abspath(combined))


def _float_env(name: str, default: float) -> float:
    value = env_value(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise SpeechConfigurationError(f"{name} must be a number") from error


def _int_env(name: str, default: int) -> int:
    value = env_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise SpeechConfigurationError(f"{name} must be an integer") from error


@dataclass(frozen=True, slots=True)
class PiperFallback:
    model: Path
    model_config: Path
    effects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VoiceSelection:
    display_name: str
    model: Path
    model_config: Path | None
    style: str
    effects: tuple[str, ...]
    engine: str = "piper"
    voices: Path | None = None
    python: Path | None = None
    voice_id: str = ""
    language: str = "en-gb"
    speed: float = 1.0
    fallback: PiperFallback | None = None

    @classmethod
    def load(cls, root: Path) -> VoiceSelection:
        selection_path = root / "selected_voice.json"
        try:
            raw: Any = json.loads(selection_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise SpeechConfigurationError(
                f"No selected voice profile at {selection_path}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise SpeechConfigurationError(
                f"Cannot read selected voice profile: {error}"
            ) from error

        if not isinstance(raw, dict):
            raise SpeechConfigurationError("Selected voice profile must be an object")

        engine = str(raw.get("engine", "piper")).strip().casefold()
        if engine not in {"piper", "kokoro"}:
            raise SpeechConfigurationError(
                f"Unsupported local voice engine: {engine or '<empty>'}"
            )

        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise SpeechConfigurationError("Selected voice profile has no model")

        model_config_value = raw.get("model_config")
        if engine == "piper" and (
            not isinstance(model_config_value, str) or not model_config_value.strip()
        ):
            raise SpeechConfigurationError("Selected Piper voice has no model_config")

        effects = _string_array(raw.get("effects", []), "Voice effects")
        voices_value = raw.get("voices")
        python_value = raw.get("python")
        voice_id = str(raw.get("voice", "")).strip()
        if engine == "kokoro":
            if not isinstance(voices_value, str) or not voices_value.strip():
                raise SpeechConfigurationError("Selected Kokoro voice has no voices file")
            if not isinstance(python_value, str) or not python_value.strip():
                raise SpeechConfigurationError("Selected Kokoro voice has no Python runtime")
            if not voice_id:
                raise SpeechConfigurationError("Selected Kokoro voice has no voice id")

        speed_value = raw.get("speed", 1.0)
        if isinstance(speed_value, bool):
            raise SpeechConfigurationError("Voice speed must be a number")
        try:
            speed = float(speed_value)
        except (TypeError, ValueError) as error:
            raise SpeechConfigurationError("Voice speed must be a number") from error
        if not 0.5 <= speed <= 2.0:
            raise SpeechConfigurationError("Voice speed must be between 0.5 and 2.0")

        fallback = _load_fallback(root, raw.get("fallback"))

        return cls(
            display_name=str(raw.get("display_name", "V")),
            model=_resolve_under(root, model),
            model_config=(
                _resolve_under(root, model_config_value)
                if isinstance(model_config_value, str) and model_config_value.strip()
                else None
            ),
            style=str(raw.get("style", "clean")),
            effects=effects,
            engine=engine,
            voices=(
                _resolve_under(root, voices_value)
                if isinstance(voices_value, str) and voices_value.strip()
                else None
            ),
            python=(
                _resolve_executable_under(root, python_value)
                if isinstance(python_value, str) and python_value.strip()
                else None
            ),
            voice_id=voice_id,
            language=str(raw.get("language", "en-gb")).strip() or "en-gb",
            speed=speed,
            fallback=fallback,
        )


def _string_array(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpeechConfigurationError(f"{label} must be a string array")
    return tuple(value)


def _load_fallback(root: Path, value: Any) -> PiperFallback | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SpeechConfigurationError("Voice fallback must be an object")
    if str(value.get("engine", "piper")).strip().casefold() != "piper":
        raise SpeechConfigurationError("Only a local Piper fallback is supported")
    model = value.get("model")
    model_config = value.get("model_config")
    if not isinstance(model, str) or not model.strip():
        raise SpeechConfigurationError("Piper fallback has no model")
    if not isinstance(model_config, str) or not model_config.strip():
        raise SpeechConfigurationError("Piper fallback has no model_config")
    return PiperFallback(
        model=_resolve_under(root, model),
        model_config=_resolve_under(root, model_config),
        effects=_string_array(value.get("effects", []), "Fallback effects"),
    )


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    root: Path
    recorder: Path
    player: Path
    whisper_cli: Path
    whisper_model: Path
    piper: Path
    sox: Path
    voice: VoiceSelection
    input_target: str | None
    output_target: str | None
    speech_threshold: float
    minimum_speech_seconds: float
    end_silence_seconds: float
    start_timeout_seconds: float
    maximum_record_seconds: float
    whisper_language: str = "auto"
    whisper_threads: int = 4
    whisper_initial_prompt: str = ""
    whisper_fallback_cli: Path | None = None
    whisper_fallback_model: Path | None = None
    sample_rate: int = 16_000

    @classmethod
    def load(cls, root: Path) -> SpeechConfig:
        root = root.expanduser().resolve()
        voice = VoiceSelection.load(root)
        input_target = env_value("DARKLINGER_AUDIO_INPUT_TARGET", "").strip() or None
        output_target = env_value("DARKLINGER_AUDIO_OUTPUT_TARGET", "").strip() or None

        config = cls(
            root=root,
            recorder=_resolve_command(env_value("DARKLINGER_RECORDER", "pw-record")),
            player=_resolve_command(env_value("DARKLINGER_PLAYER", "pw-play")),
            whisper_cli=_resolve_command(
                env_value("DARKLINGER_WHISPER_CLI", "whisper-cli")
            ),
            whisper_model=_resolve_under(
                root,
                env_value("DARKLINGER_WHISPER_MODEL", "models/ggml-base.bin"),
            ),
            piper=_resolve_command(env_value("DARKLINGER_PIPER", "piper")),
            sox=_resolve_command(env_value("DARKLINGER_SOX", "sox")),
            voice=voice,
            input_target=input_target,
            output_target=output_target,
            speech_threshold=_float_env("DARKLINGER_SPEECH_THRESHOLD", 0.01),
            minimum_speech_seconds=_float_env(
                "DARKLINGER_MINIMUM_SPEECH_SECONDS", 0.20
            ),
            end_silence_seconds=_float_env("DARKLINGER_END_SILENCE_SECONDS", 1.20),
            start_timeout_seconds=_float_env("DARKLINGER_SPEECH_START_TIMEOUT", 12.0),
            maximum_record_seconds=_float_env("DARKLINGER_MAXIMUM_RECORD_SECONDS", 60.0),
            whisper_language=(
                env_value("DARKLINGER_WHISPER_LANGUAGE", "auto").strip().casefold()
                or "auto"
            ),
            whisper_threads=_int_env("DARKLINGER_WHISPER_THREADS", 4),
            whisper_initial_prompt=env_value(
                "DARKLINGER_WHISPER_INITIAL_PROMPT", ""
            ).strip(),
            whisper_fallback_cli=_optional_command(
                env_value("DARKLINGER_WHISPER_FALLBACK_CLI", "")
            ),
            whisper_fallback_model=_optional_under(
                root, env_value("DARKLINGER_WHISPER_FALLBACK_MODEL", "")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        paths = {
            "PipeWire recorder": self.recorder,
            "PipeWire player": self.player,
            "Whisper CLI": self.whisper_cli,
            "Whisper model": self.whisper_model,
        }
        if (self.whisper_fallback_cli is None) != (
            self.whisper_fallback_model is None
        ):
            raise SpeechConfigurationError(
                "Whisper fallback requires both CLI and model paths"
            )
        if self.whisper_fallback_cli is not None:
            paths["Whisper fallback CLI"] = self.whisper_fallback_cli
            paths["Whisper fallback model"] = self.whisper_fallback_model
        if self.voice.engine == "piper":
            paths["Piper"] = self.piper
            paths["Piper voice model"] = self.voice.model
            paths["Piper voice config"] = self.voice.model_config
            if self.voice.effects:
                paths["SoX"] = self.sox
        else:
            paths["Kokoro Python runtime"] = self.voice.python
            paths["Kokoro model"] = self.voice.model
            paths["Kokoro voices"] = self.voice.voices
            if self.voice.effects:
                paths["SoX"] = self.sox
            if self.voice.fallback is not None:
                paths["Piper fallback engine"] = self.piper
                paths["Piper fallback model"] = self.voice.fallback.model
                paths["Piper fallback config"] = self.voice.fallback.model_config
                if self.voice.fallback.effects:
                    paths["SoX"] = self.sox

        missing = [
            f"{label}: {path}"
            for label, path in paths.items()
            if not isinstance(path, Path) or not path.is_file()
        ]
        if missing:
            raise SpeechConfigurationError(
                "Local speech stack is incomplete:\n- " + "\n- ".join(missing)
            )

        numeric = {
            "DARKLINGER_SPEECH_THRESHOLD": self.speech_threshold,
            "DARKLINGER_MINIMUM_SPEECH_SECONDS": self.minimum_speech_seconds,
            "DARKLINGER_END_SILENCE_SECONDS": self.end_silence_seconds,
            "DARKLINGER_SPEECH_START_TIMEOUT": self.start_timeout_seconds,
            "DARKLINGER_MAXIMUM_RECORD_SECONDS": self.maximum_record_seconds,
        }
        invalid = [name for name, value in numeric.items() if value <= 0]
        if invalid:
            raise SpeechConfigurationError(
                f"Speech timing and threshold values must be positive: {', '.join(invalid)}"
            )
        if self.maximum_record_seconds <= self.minimum_speech_seconds:
            raise SpeechConfigurationError(
                "DARKLINGER_MAXIMUM_RECORD_SECONDS must exceed minimum speech duration"
            )
        if not 1 <= self.whisper_threads <= 32:
            raise SpeechConfigurationError(
                "DARKLINGER_WHISPER_THREADS must be between 1 and 32"
            )
        if not re.fullmatch(r"auto|[a-z]{2}(?:-[a-z]{2})?", self.whisper_language):
            raise SpeechConfigurationError(
                "DARKLINGER_WHISPER_LANGUAGE must be auto or a language code"
            )
        if len(self.whisper_initial_prompt) > 1_000:
            raise SpeechConfigurationError(
                "DARKLINGER_WHISPER_INITIAL_PROMPT must not exceed 1000 characters"
            )


def _optional_command(value: str) -> Path | None:
    value = value.strip()
    return _resolve_command(value) if value else None


def _optional_under(root: Path, value: str) -> Path | None:
    value = value.strip()
    return _resolve_under(root, value) if value else None
