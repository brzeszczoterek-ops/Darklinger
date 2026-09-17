from __future__ import annotations

from array import array
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_core.main import _configure_push_to_talk_hotkey, _ends_voice_mode
from v_core.speech import (
    SpeechConfig,
    SpeechConfigurationError,
    SpeechRuntime,
    SpeechRuntimeError,
    VoiceSelection,
)
from v_core.speech.runtime import VoiceActivityDetector


def _pcm(value: int, samples: int = 1_600) -> bytes:
    return array("h", [value] * samples).tobytes()


def test_voice_activity_detector_stops_after_speech_and_silence() -> None:
    detector = VoiceActivityDetector(
        threshold=0.01,
        minimum_speech_seconds=0.2,
        end_silence_seconds=1.2,
    )

    assert detector.feed(_pcm(10), 0.5) is False
    assert detector.feed(_pcm(1_000), 0.1) is False
    assert detector.feed(_pcm(1_000), 0.1) is False
    assert detector.started is True
    for _ in range(11):
        assert detector.feed(_pcm(10), 0.1) is False
    assert detector.feed(_pcm(10), 0.1) is True


def test_voice_activity_detector_does_not_start_from_room_noise() -> None:
    detector = VoiceActivityDetector(
        threshold=0.01,
        minimum_speech_seconds=0.2,
        end_silence_seconds=1.2,
    )

    for _ in range(30):
        assert detector.feed(_pcm(50), 0.1) is False
    assert detector.started is False


def test_voice_selection_loads_relative_paths_and_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    paths = {}
    for name in (
        "pw-record",
        "pw-play",
        "whisper-cli",
        "whisper-fallback",
        "piper",
        "sox",
    ):
        path = commands / name
        path.touch()
        paths[name] = path

    models = tmp_path / "models"
    models.mkdir()
    for name in ("voice.onnx", "voice.onnx.json", "whisper.bin"):
        (models / name).touch()
    (tmp_path / "selected_voice.json").write_text(
        json.dumps(
            {
                "display_name": "V Test",
                "model": "models/voice.onnx",
                "model_config": "models/voice.onnx.json",
                "style": "textured",
                "effects": ["pitch", "-28"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DARKLINGER_RECORDER", str(paths["pw-record"]))
    monkeypatch.setenv("DARKLINGER_PLAYER", str(paths["pw-play"]))
    monkeypatch.setenv("DARKLINGER_WHISPER_CLI", str(paths["whisper-cli"]))
    monkeypatch.setenv("DARKLINGER_WHISPER_MODEL", "models/whisper.bin")
    monkeypatch.setenv("DARKLINGER_PIPER", str(paths["piper"]))
    monkeypatch.setenv("DARKLINGER_SOX", str(paths["sox"]))

    config = SpeechConfig.load(tmp_path)

    assert config.voice.display_name == "V Test"
    assert config.voice.model == (models / "voice.onnx").resolve()
    assert config.voice.effects == ("pitch", "-28")


def test_incomplete_speech_stack_is_reported(tmp_path: Path) -> None:
    (tmp_path / "selected_voice.json").write_text(
        json.dumps(
            {
                "model": "missing.onnx",
                "model_config": "missing.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SpeechConfigurationError, match="incomplete"):
        SpeechConfig.load(tmp_path)


def test_kokoro_voice_loads_isolated_runtime_and_piper_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    for name in (
        "pw-record",
        "pw-play",
        "whisper-cli",
        "whisper-fallback",
        "piper",
        "sox",
    ):
        (commands / name).touch()
    models = tmp_path / "models"
    models.mkdir()
    for name in (
        "kokoro.onnx",
        "voices.bin",
        "whisper.bin",
        "whisper-fallback.bin",
        "fallback.onnx",
        "fallback.json",
    ):
        (models / name).touch()
    python = tmp_path / "engine" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    (tmp_path / "selected_voice.json").write_text(
        json.dumps(
            {
                "engine": "kokoro",
                "display_name": "Emma",
                "model": "models/kokoro.onnx",
                "voices": "models/voices.bin",
                "python": "engine/.venv/bin/python",
                "voice": "bf_emma",
                "language": "en-gb",
                "speed": 1.05,
                "fallback": {
                    "engine": "piper",
                    "model": "models/fallback.onnx",
                    "model_config": "models/fallback.json",
                    "effects": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DARKLINGER_RECORDER", str(commands / "pw-record"))
    monkeypatch.setenv("DARKLINGER_PLAYER", str(commands / "pw-play"))
    monkeypatch.setenv("DARKLINGER_WHISPER_CLI", str(commands / "whisper-cli"))
    monkeypatch.setenv("DARKLINGER_WHISPER_MODEL", "models/whisper.bin")
    monkeypatch.setenv("DARKLINGER_WHISPER_LANGUAGE", "pl")
    monkeypatch.setenv("DARKLINGER_WHISPER_THREADS", "6")
    monkeypatch.setenv(
        "DARKLINGER_WHISPER_INITIAL_PROMPT", "V, DARKLINGER, Brzeszczot"
    )
    monkeypatch.setenv(
        "DARKLINGER_WHISPER_FALLBACK_CLI", str(commands / "whisper-fallback")
    )
    monkeypatch.setenv(
        "DARKLINGER_WHISPER_FALLBACK_MODEL", "models/whisper-fallback.bin"
    )
    monkeypatch.setenv("DARKLINGER_PIPER", str(commands / "piper"))
    monkeypatch.setenv("DARKLINGER_SOX", str(commands / "sox"))

    config = SpeechConfig.load(tmp_path)

    assert config.voice.engine == "kokoro"
    assert config.voice.voice_id == "bf_emma"
    assert config.voice.language == "en-gb"
    assert config.voice.speed == 1.05
    assert config.voice.python == python.absolute()
    assert config.voice.fallback is not None
    assert config.voice.fallback.model == (models / "fallback.onnx").resolve()
    assert config.whisper_language == "pl"
    assert config.whisper_threads == 6
    assert config.whisper_initial_prompt == "V, DARKLINGER, Brzeszczot"
    assert config.whisper_fallback_cli == (commands / "whisper-fallback").absolute()
    assert config.whisper_fallback_model == (
        models / "whisper-fallback.bin"
    ).resolve()

    command = SpeechRuntime(config)._whisper_command(
        config.whisper_cli,
        config.whisper_model,
        tmp_path / "speech.wav",
    )
    assert command[command.index("--language") + 1] == "pl"
    assert command[command.index("--threads") + 1] == "6"
    assert "--flash-attn" in command
    assert command[command.index("--prompt") + 1] == "V, DARKLINGER, Brzeszczot"


def test_spoken_text_removes_terminal_only_markup() -> None:
    prepared = SpeechRuntime._prepare_spoken_text(
        "## Result\nUse [`DARKLINGER`](https://example.com).\n"
        "```python\nprint('terminal only')\n```"
    )

    assert prepared == (
        "Result Use DARKLINGER. I left the code block in the terminal."
    )


@pytest.mark.asyncio
async def test_whisper_uses_cpu_fallback_after_primary_failure(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        whisper_cli=tmp_path / "whisper-cuda",
        whisper_model=tmp_path / "turbo.bin",
        whisper_fallback_cli=tmp_path / "whisper-cpu",
        whisper_fallback_model=tmp_path / "base.bin",
        whisper_language="pl",
        whisper_threads=6,
        whisper_initial_prompt="V, DARKLINGER, Brzeszczot",
        maximum_record_seconds=60,
    )
    runtime = SpeechRuntime(config)
    calls: list[tuple[str, ...]] = []

    async def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        if len(calls) == 1:
            raise SpeechRuntimeError("CUDA unavailable")
        return " Rozpoznany tekst. "

    runtime._run = fake_run

    transcript = await runtime._transcribe(tmp_path / "speech.wav")

    assert transcript == "Rozpoznany tekst."
    assert calls[0][0] == str(config.whisper_cli)
    assert calls[1][0] == str(config.whisper_fallback_cli)
    assert calls[1][calls[1].index("--language") + 1] == "pl"


@pytest.mark.parametrize(
    "phrase",
    [
        "Stop listening.",
        "WYŁĄCZ TRYB GŁOSOWY!",
        "koniec rozmowy",
    ],
)
def test_voice_stop_phrases(phrase: str) -> None:
    assert _ends_voice_mode(phrase) is True


def test_normal_prompt_does_not_end_voice_mode() -> None:
    assert _ends_voice_mode("Powiedz mi, jak działa ten kod") is False


def test_f2_binds_to_immediate_push_to_talk_command() -> None:
    bindings: list[str] = []

    key = _configure_push_to_talk_hotkey(
        environ={"DARKLINGER_PTT_KEY": "f2"},
        bind=bindings.append,
        stdin_is_tty=True,
    )

    assert key == "F2"
    assert bindings == [
        '"\\eOQ": "\\C-A\\C-K/ptt\\C-M"',
        '"\\e[12~": "\\C-A\\C-K/ptt\\C-M"',
    ]


def test_f2_is_the_default_push_to_talk_key() -> None:
    bindings: list[str] = []

    key = _configure_push_to_talk_hotkey(
        environ={},
        bind=bindings.append,
        stdin_is_tty=True,
    )

    assert key == "F2"
    assert bindings[0].startswith('"\\eOQ"')


def test_push_to_talk_hotkey_is_disabled_for_noninteractive_input() -> None:
    assert (
        _configure_push_to_talk_hotkey(
            environ={"DARKLINGER_PTT_KEY": "F2"},
            bind=lambda _: None,
            stdin_is_tty=False,
        )
        is None
    )


@pytest.mark.asyncio
async def test_push_to_talk_starts_and_stops_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SpeechConfig(
        root=tmp_path,
        recorder=Path("/usr/bin/pw-record"),
        player=Path("/usr/bin/pw-play"),
        whisper_cli=Path("/usr/bin/true"),
        whisper_model=tmp_path / "whisper.bin",
        piper=Path("/usr/bin/true"),
        sox=Path("/usr/bin/true"),
        voice=VoiceSelection(
            display_name="test",
            model=tmp_path / "voice.onnx",
            model_config=tmp_path / "voice.json",
            style="clean",
            effects=(),
        ),
        input_target="test-input",
        output_target="test-output",
        speech_threshold=0.01,
        minimum_speech_seconds=0.2,
        end_silence_seconds=1.2,
        start_timeout_seconds=12,
        maximum_record_seconds=60,
    )

    class FakeProcess:
        returncode = None
        stderr = SimpleNamespace(read=lambda: None)

        def send_signal(self, _signal: int) -> None:
            self.returncode = -2

        async def wait(self) -> int:
            return int(self.returncode or 0)

        def kill(self) -> None:
            self.returncode = -9

    commands: list[tuple[str, ...]] = []

    async def fake_subprocess(*command, **_kwargs):
        commands.append(tuple(str(item) for item in command))
        destination = Path(command[-1])
        destination.write_bytes(b"W" * (44 + 16_000))
        return FakeProcess()

    async def fake_transcribe(_recording: Path) -> str:
        return "Cześć V"

    monkeypatch.setattr(
        "v_core.speech.runtime.asyncio.create_subprocess_exec",
        fake_subprocess,
    )
    runtime = SpeechRuntime(config)
    monkeypatch.setattr(runtime, "_transcribe", fake_transcribe)

    await runtime.start_push_to_talk()
    assert runtime.push_to_talk_recording is True
    assert "--target" in commands[0]
    assert "test-input" in commands[0]

    transcript = await runtime.stop_push_to_talk()

    assert transcript == "Cześć V"
    assert runtime.push_to_talk_recording is False
