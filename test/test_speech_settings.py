from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import httpx
import pytest

from v_core.speech.config import VoiceSelection, SpeechConfig
from v_core.speech.preferences import read_preferences
from v_core.speech.settings import SpeechSettings
import v_core.speech.settings as settings


@pytest.fixture
def manager(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("a.onnx", "a.onnx.json", "ggml-base.bin", "ggml-small.bin"):
        (models / name).write_bytes(b"fixture")
    executable = tmp_path / "whisper-cli"
    executable.write_text("fixture")
    executable.chmod(0o700)
    monkeypatch.setenv("DARKLINGER_WHISPER_CLI", str(executable))
    monkeypatch.setenv("DARKLINGER_WHISPER_MODEL", str(models / "ggml-base.bin"))
    monkeypatch.setenv("DARKLINGER_WHISPER_LANGUAGE", "pl")
    (tmp_path / "selected_voice.json").write_text(json.dumps({
        "display_name": "Original", "model": "models/a.onnx", "model_config": "models/a.onnx.json",
        "effects": ["gain", "-6"],
    }))
    return SpeechSettings(tmp_path)


def test_catalog_is_read_only_and_keeps_defaults(manager):
    before = sorted(manager.root.rglob("*"))
    state = manager.status()
    assert state["selected"]["voice"] == "default"
    assert state["selected"]["language"] == "pl"
    assert len(state["models"]) == 2
    assert len(state["voices"]) == 2
    assert state["installing"] is False
    assert sorted(manager.root.rglob("*")) == before
    assert not (manager.root / "speech_settings.json").exists()


def test_selection_persists_without_editing_original(manager):
    original = (manager.root / "selected_voice.json").read_bytes()
    choice = next(v["id"] for v in manager.status()["voices"] if v["id"] != "default")
    manager.save({"voice": choice, "language": "en", "threads": 3})
    assert stat.S_IMODE((manager.root / "speech_settings.json").stat().st_mode) == 0o600
    assert VoiceSelection.load(manager.root).effects == ()
    assert SpeechSettings(manager.root).status()["selected"]["language"] == "en"
    assert (manager.root / "selected_voice.json").read_bytes() == original
    manager.save({"voice": "default"})
    assert VoiceSelection.load(manager.root).effects == ("gain", "-6")


def test_saved_speech_config_overrides_only_owner_selected_values(manager, monkeypatch):
    for key in ("RECORDER", "PLAYER", "PIPER", "SOX", "WHISPER_FALLBACK_CLI"):
        monkeypatch.setenv("DARKLINGER_" + key, str(manager.root / "whisper-cli"))
    monkeypatch.setenv("DARKLINGER_WHISPER_FALLBACK_MODEL", str(manager.root / "models/ggml-base.bin"))
    model = next(key for key, value in manager.catalog()["models"].items() if value.name == "ggml-small.bin")
    manager.save({"model": model, "language": "en", "threads": 2})
    config = SpeechConfig.load(manager.root)
    assert config.whisper_model.name == "ggml-small.bin"
    assert config.whisper_fallback_model == config.whisper_model
    assert config.whisper_threads == 2 and config.whisper_language == "en"
    assert config.voice.display_name == "Original"


@pytest.mark.parametrize("payload", [
    {"engine": "/bin/sh"}, {"model": "../../secret"}, {"voice": []},
    {"threads": True}, {"threads": 0}, {"threads": 33}, {"language": "$(evil)"},
    {"whisper_cli": "arbitrary"},
])
def test_invalid_preferences_are_atomic(manager, payload):
    with pytest.raises(ValueError):
        manager.save(payload)
    assert read_preferences(manager.root) == {}


def test_kokoro_catalog_discovers_only_supported_installed_voices(manager):
    with zipfile.ZipFile(manager.root / "voices.bin", "w") as archive:
        for name in ("bf_emma.npy", "af_heart.npy", "../evil.npy", "ff_siwis.npy"):
            archive.writestr(name, b"fixture")
    (manager.root / "selected_voice.json").write_text(json.dumps({
        "engine": "kokoro", "model": "models/a.onnx", "voices": "voices.bin",
        "python": "/usr/bin/python3", "voice": "bf_emma", "display_name": "Emma",
    }))
    voices = manager.catalog()["voices"]
    assert "kokoro-bf_emma" in voices
    assert "kokoro-af_heart" in voices
    assert "kokoro-ff_siwis" not in voices
    manager.save({"voice": "kokoro-af_heart"})
    assert VoiceSelection.load(manager.root).language == "en-us"


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_checks_digest_and_never_activates_model(manager, monkeypatch, corrupt):
    body = b"tiny fake test-only model"
    monkeypatch.setitem(settings.MODEL_DOWNLOADS, "fixture", (
        "ggml-fixture.bin", len(body), "0" * 64 if corrupt else hashlib.sha256(body).hexdigest()))
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    monkeypatch.setattr(settings.httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))

    async def scenario():
        manager.start_install("model", "fixture")
        with pytest.raises(RuntimeError):
            manager.start_install("model", "fixture")
        await manager.task
    asyncio.run(scenario())
    assert manager.job["state"] == ("error" if corrupt else "complete")
    assert (manager.root / "models/ggml-fixture.bin").exists() is not corrupt
    assert not list((manager.root / "models").glob("*.part"))
    assert read_preferences(manager.root) == {}


def test_existing_model_is_not_overwritten(manager, monkeypatch):
    target = manager.root / "models/ggml-tiny.bin"
    target.write_bytes(b"owner data")
    asyncio.run(manager._install("model", "tiny"))
    assert manager.job["state"] == "error"
    assert target.read_bytes() == b"owner data"


def test_engine_build_is_pinned_and_opt_in(manager, monkeypatch):
    commands = []
    monkeypatch.setattr(settings.shutil, "which", lambda command: "/usr/bin/" + command)

    async def command(*args, cwd):
        commands.append(args)
        if args[:2] == ("git", "clone"):
            (cwd / "source").mkdir()
        if args[:2] == ("git", "rev-parse"):
            return settings.ENGINE_REVISION
        if args[:2] == ("cmake", "--build"):
            target = cwd / "build/bin/whisper-cli"
            target.parent.mkdir(parents=True)
            target.write_text("fixture")
        return ""
    monkeypatch.setattr(manager, "_command", command)
    asyncio.run(manager._install("engine", "cpu"))
    assert manager.job["state"] == "complete"
    assert manager.engine_path.is_file()
    assert any("-DGGML_CUDA=OFF" in args for args in commands)
    assert read_preferences(manager.root) == {}


def test_cancel_install_leaves_preferences_alone(manager, monkeypatch):
    async def blocked(choice):
        await asyncio.sleep(100)
    monkeypatch.setattr(manager, "_download_model", blocked)

    async def scenario():
        manager.start_install("model", "tiny")
        await asyncio.sleep(0)
        await manager.close()
        assert manager.task.cancelled()
    asyncio.run(scenario())
    assert manager.job["state"] == "cancelled"
    assert read_preferences(manager.root) == {}
