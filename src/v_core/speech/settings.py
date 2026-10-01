"""Local speech catalog and opt-in, pinned Whisper installer. No shell or sudo."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import zipfile

import httpx

from ..branding import env_value
from .config import _resolve_command, _resolve_under
from .preferences import atomic_json, read_preferences

HF_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
ENGINE_REVISION = "2eeeba56e9edd762b4b38467bab96c2517163158"
MODEL_DOWNLOADS = {
    "tiny": ("ggml-tiny.bin", 77691713, "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"),
    "base": ("ggml-base.bin", 147951465, "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"),
    "small": ("ggml-small.bin", 487601967, "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"),
    "large-v3-turbo-q5_0": ("ggml-large-v3-turbo-q5_0.bin", 574041195, "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"),
}


def _id(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()[:20]


class SpeechSettings:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.job: dict[str, Any] = {"state": "idle"}
        self.task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def engine_path(self) -> Path:
        return self.root / "engines" / f"whisper-{ENGINE_REVISION[:12]}" / "build" / "bin" / "whisper-cli"

    def catalog(self) -> dict[str, Any]:
        saved = read_preferences(self.root)
        cli = _resolve_command(saved.get("whisper_cli", env_value("DARKLINGER_WHISPER_CLI", "whisper-cli")))
        model = _resolve_under(self.root, saved.get("whisper_model", env_value("DARKLINGER_WHISPER_MODEL", "models/ggml-base.bin")))
        voices: dict[str, dict] = {}
        try:
            default = json.loads((self.root / "selected_voice.json").read_text())
            if isinstance(default, dict):
                voices["default"] = default
            else:
                default = {}
        except (OSError, ValueError):
            default = {}
        # Only already installed voices; discovery never downloads anything.
        for path in sorted((self.root / "models").glob("*.onnx")):
            config = path.with_suffix(path.suffix + ".json")
            if config.is_file():
                voices[f"piper-{_id(path)}"] = {
                    "engine": "piper", "model": str(path), "model_config": str(config),
                    "display_name": path.stem + " · Piper", "effects": [],
                }
        if default.get("engine") == "kokoro":
            try:
                with zipfile.ZipFile(_resolve_under(self.root, default["voices"])) as archive:
                    names = archive.namelist()
                for name in sorted(names):
                    voice = name.removesuffix(".npy")
                    if re.fullmatch(r"[ab][fm]_[a-z0-9_]+", voice):
                        voices[f"kokoro-{voice}"] = {
                            **default, "voice": voice,
                            "language": "en-gb" if voice.startswith("b") else "en-us",
                            "display_name": voice + " · Kokoro",
                        }
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                pass
        paths = {model}
        for directory in {model.parent, self.root / "models"}:
            paths.update(directory.glob("ggml-*.bin"))
        models = {_id(p): p for p in sorted(paths) if p.is_file() and p.stat().st_size > 0}
        engines = {_id(p): p for p in (cli, self.engine_path) if p.is_file() and os.access(p, os.X_OK)}
        return {"voices": voices, "models": models, "engines": engines,
                "selected": {"voice": saved.get("voice_id", "default"), "model": _id(model),
                             "engine": _id(cli),
                             "language": saved.get("whisper_language", env_value("DARKLINGER_WHISPER_LANGUAGE", "auto")),
                             "threads": int(saved.get("whisper_threads", env_value("DARKLINGER_WHISPER_THREADS", "4")))}}

    def status(self) -> dict[str, Any]:
        catalog = self.catalog()
        return {
            "voices": [{"id": key, "label": raw.get("display_name", key)} for key, raw in catalog["voices"].items()],
            "models": [{"id": key, "label": path.name} for key, path in catalog["models"].items()],
            "engines": [{"id": key, "label": str(path)} for key, path in catalog["engines"].items()],
            "selected": catalog["selected"], "job": dict(self.job), "installing": self.busy,
            "downloads": [{"id": key, "label": value[0], "bytes": value[1],
                           "installed": any(p.name == value[0] for p in catalog["models"].values())}
                          for key, value in MODEL_DOWNLOADS.items()],
            "cpu_engine_installed": self.engine_path.is_file(),
        }

    def save(self, payload: dict) -> None:
        catalog = self.catalog()
        if set(payload) - {"voice", "model", "engine", "language", "threads"}:
            raise ValueError("Unknown speech setting")
        saved = read_preferences(self.root)
        for field, group, destination in (("voice", "voices", "voice_profile"),
                                          ("model", "models", "whisper_model"),
                                          ("engine", "engines", "whisper_cli")):
            if field not in payload:
                continue
            choice = payload[field]
            if not isinstance(choice, str) or choice not in catalog[group]:
                raise ValueError(f"Unavailable {field}; refresh the installed catalog")
            saved[destination] = catalog[group][choice] if field == "voice" else str(catalog[group][choice])
            if field == "voice":
                saved["voice_id"] = choice
                if choice == "default":
                    saved.pop("voice_profile", None)
        if "language" in payload:
            if payload["language"] not in {"auto", "pl", "en", "de", "fr", "es", "uk", "ru"}:
                raise ValueError("Unsupported transcription language")
            saved["whisper_language"] = payload["language"]
        if "threads" in payload:
            value = payload["threads"]
            if type(value) is not int or not 1 <= value <= 32:
                raise ValueError("Whisper threads must be between 1 and 32")
            saved["whisper_threads"] = value
        atomic_json(self.root / "speech_settings.json", saved)

    def start_install(self, kind: str, choice: str) -> None:
        if self.busy:
            raise RuntimeError("An installation is already running")
        if kind not in {"model", "engine"} or (kind == "model" and choice not in MODEL_DOWNLOADS):
            raise ValueError("Unknown installer item")
        if kind == "engine" and choice != "cpu":
            raise ValueError("Only the managed CPU engine can be installed")
        self.job = {"state": "queued", "kind": kind, "item": choice, "bytes": 0, "total": 0}
        self.task = asyncio.create_task(self._install(kind, choice))

    async def _install(self, kind: str, choice: str) -> None:
        try:
            async with asyncio.timeout(1200):
                if kind == "model":
                    await self._download_model(choice)
                else:
                    await self._build_engine()
            self.job.update(state="complete", message="Installed. Select it and save to activate.")
        except asyncio.CancelledError:
            self.job.update(state="cancelled", message="Installation stopped; existing configuration unchanged.")
            raise
        except Exception as error:
            self.job.update(state="error", message=str(error)[:1000] or type(error).__name__)

    async def _download_model(self, choice: str) -> None:
        name, size, checksum = MODEL_DOWNLOADS[choice]
        directory = self.root / "models"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / name
        if destination.exists():
            raise ValueError("That managed model file already exists; it was not overwritten")
        if shutil.disk_usage(directory).free < size + 64 * 1024 * 1024:
            raise ValueError("Not enough disk space for the model")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=directory)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        self.job.update(state="downloading", total=size)
        try:
            with os.fdopen(fd, "wb") as handle:
                async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{HF_REVISION}/{name}"
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes(256 * 1024):
                            self.job["bytes"] += len(chunk)
                            if self.job["bytes"] > size:
                                raise ValueError("Download exceeded expected size")
                            handle.write(chunk)
                            digest.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if self.job["bytes"] != size or digest.hexdigest() != checksum:
                raise ValueError("Model checksum/size verification failed")
            # link is atomic and refuses to overwrite a file created during download.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    async def _command(self, *args: str, cwd: Path) -> str:
        process = await asyncio.create_subprocess_exec(
            *args, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        tail = b""
        try:
            assert process.stdout is not None
            while chunk := await process.stdout.read(4096):
                tail = (tail + chunk)[-4096:]
                self.job["detail"] = tail.decode(errors="replace")[-1500:]
            await process.wait()
            if process.returncode:
                raise RuntimeError(f"{Path(args[0]).name} failed: {tail.decode(errors='replace')[-1000:]}")
            return tail.decode(errors="replace").strip()
        finally:
            if process.returncode is None:
                import signal
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    async def _build_engine(self) -> None:
        if self.engine_path.exists():
            return
        if not all(shutil.which(command) for command in ("git", "cmake", "c++")):
            raise ValueError("Install git, cmake and a C++ compiler using your system package manager first")
        parent = self.root / "engines"
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < 1024**3:
            raise ValueError("The CPU engine build needs at least 1 GB of free disk space")
        self.job.update(state="building", message="Building local whisper.cpp v1.8.3 (CPU); existing CUDA engine stays unchanged.")
        with tempfile.TemporaryDirectory(prefix=".whisper-build-", dir=parent) as directory:
            work = Path(directory)
            await self._command("git", "clone", "--depth", "1", "--branch", "v1.8.3",
                                "https://github.com/ggml-org/whisper.cpp.git", "source", cwd=work)
            source = work / "source"
            revision = await self._command("git", "rev-parse", "HEAD", cwd=source)
            if revision != ENGINE_REVISION:
                raise ValueError("Whisper source revision verification failed")
            await self._command("cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                                "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DWHISPER_BUILD_TESTS=OFF", cwd=source)
            await self._command("cmake", "--build", "build", "--target", "whisper-cli", "-j", "2", cwd=source)
            if not (source / "build/bin/whisper-cli").is_file():
                raise ValueError("Whisper build did not produce whisper-cli")
            source.rename(self.engine_path.parents[2])

    async def close(self) -> None:
        if self.busy:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
