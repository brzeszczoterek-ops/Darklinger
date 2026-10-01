from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import secrets
import time
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from v_core.speech import SpeechConfig, SpeechRuntime
from v_core.speech.settings import SpeechSettings
from v_core.memory.memoir import SessionMemoir, current_memoir_turn
from v_core.response_preview import response_preview
from v_core.ui.runtime_activity import runtime_activity, model_log
from v_core.ui.memory_browser import browse


_STATIC_ROOT = Path(__file__).with_name("static")
_SESSION_PLACEHOLDER = "__DARKLINGER_SESSION_TOKEN__"


@dataclass(slots=True)
class UIRuntime:
    core: Any
    config: Any
    model_session: Any | None = None
    session_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    started_at: float = field(default_factory=time.monotonic)
    started_wall_time: float = field(default_factory=time.time)
    shutdown_callback: Callable[[], None] | None = None
    chat_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    speech_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_chat_task: asyncio.Task[Any] | None = None
    closing: bool = False
    speech: SpeechRuntime | None = None
    activity_sample: dict[str, Any] = field(default_factory=dict)
    speech_settings: SpeechSettings | None = None
    memoir: SessionMemoir | None = None
    shutdown_task: asyncio.Task[Any] | None = None

    def __post_init__(self) -> None:
        self.speech_settings = SpeechSettings(Path(getattr(self.config, "voice_root", "voice")))
        memory_root = getattr(self.config, "memory_root", None)
        if memory_root is not None:
            self.memoir = SessionMemoir(Path(memory_root))

    def speech_busy(self) -> bool:
        return bool(self.closing or self.chat_lock.locked() or self.speech_lock.locked()
                    or getattr(self.speech, "push_to_talk_recording", False))

    @property
    def edition_extension(self) -> Any:
        tools = getattr(getattr(self.core, "agent", None), "tools", None)
        return getattr(tools, "edition_extension", None)

    @property
    def active_session(self) -> Any | None:
        routed = getattr(self.core, "model_runtime", None)
        return getattr(routed, "session", None) or self.model_session

    def status(self) -> dict[str, Any]:
        session = self.active_session
        profile = getattr(session, "profile", None)
        process = getattr(session, "process", None)
        process_state = "external"
        if process is not None:
            process_state = "running" if process.poll() is None else "stopped"
        model = {
            "alias": str(getattr(profile, "alias", "external model")),
            "filename": Path(str(getattr(profile, "model_path", ""))).name,
            "context_size": int(getattr(profile, "context_size", 0) or 0),
            "reasoning": str(getattr(profile, "reasoning", "unknown")),
            "cache_k": str(getattr(profile, "cache_type_k", "unknown")),
            "cache_v": str(getattr(profile, "cache_type_v", "unknown")),
            "state": process_state,
        }
        tools = getattr(getattr(self.core, "agent", None), "tools", None)
        try:
            tool_names = list(tools.local_tool_names()) if tools is not None else []
        except Exception:
            tool_names = []
        edition = getattr(getattr(self.config, "edition", None), "name", "public")
        inference = getattr(getattr(self.core, "llm", None), "inference", None)
        payload: dict[str, Any] = {
            "ready": not self.chat_lock.locked() and not self.closing,
            "closing": self.closing,
            "memoir": self.memoir.state if self.memoir else {"state": "unavailable"},
            "edition": edition,
            "uptime_seconds": max(0, int(time.monotonic() - self.started_at)),
            "model": model,
            "inference": inference.status() if inference is not None else {},
            "tools": {"count": len(tool_names), "active": tool_names},
            "voice": {
                "recording": bool(
                    self.speech is not None and self.speech.push_to_talk_recording
                ),
                "configured": self.speech is not None,
            },
            "owner": None,
            "activity": runtime_activity(
                getattr(self.config, "autonomy_root", None),
                session,
                ready=not self.chat_lock.locked(),
                session_started_at=self.started_wall_time,
                sample_state=self.activity_sample,
            ),
        }
        extension = self.edition_extension
        if extension is not None:
            manifest = extension.ui_manifest()
            details = extension.ui_status(
                {"config": self.config, "core": self.core, "session": session}
            )
            if manifest is not None and details is not None:
                payload["owner"] = {**manifest, **details}
        return payload

    def require_token(self, request: Request) -> Response | None:
        supplied = request.headers.get("x-darklinger-session", "") or request.headers.get(
            "x-" + "pala" + "dyn-session",
            "",
        )
        if not supplied or not secrets.compare_digest(supplied, self.session_token):
            return JSONResponse({"error": "invalid local UI session"}, status_code=403)
        return None

    def ensure_speech(self) -> SpeechRuntime:
        if self.speech is None:
            self.speech = SpeechRuntime(SpeechConfig.load(self.config.voice_root))
        return self.speech

    async def cancel_active_chat(self) -> None:
        self.closing = True
        task = self.active_chat_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def close(self) -> None:
        await self.cancel_active_chat()
        if self.shutdown_task and not self.shutdown_task.done():
            self.shutdown_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.shutdown_task
        if self.speech_settings is not None:
            await self.speech_settings.close()
        if self.speech is not None:
            with suppress(Exception):
                await self.speech.close()
        await self.core.close()


def _ndjson(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def create_app(runtime: UIRuntime) -> Starlette:
    async def index(_: Request) -> Response:
        source = (_STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        source = source.replace(_SESSION_PLACEHOLDER, runtime.session_token)
        asset_version = hashlib.sha256(
            (_STATIC_ROOT / "app.js").read_bytes() + (_STATIC_ROOT / "app.css").read_bytes()
        ).hexdigest()[:16]
        source = source.replace("__DARKLINGER_ASSET_VERSION__", asset_version)
        return HTMLResponse(
            source,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'none'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def status(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        return JSONResponse(runtime.status(), headers={"Cache-Control": "no-store"})

    async def memory_browser(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        try:
            payload = browse(runtime, view=request.query_params.get("view", "sessions"),
                             session_id=request.query_params.get("session", "current"))
            code = 200
        except PermissionError:
            payload, code = {"error": "Brak dostępu do archiwum w tej edycji."}, 403
        except (OSError, ValueError, UnicodeError):
            payload, code = {"error": "Nie można odczytać danych: plik niedostępny, nieprawidłowy lub zbyt duży."}, 400
        return JSONResponse(payload, status_code=code, headers={"Cache-Control": "no-store"})

    async def chat(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        if runtime.closing:
            return JSONResponse({"error": "V is shutting down"}, status_code=503)
        if runtime.chat_lock.locked():
            return JSONResponse({"error": "V is already working"}, status_code=409)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)
        prompt = str(payload.get("message", "")).strip() if isinstance(payload, dict) else ""
        if not prompt:
            return JSONResponse({"error": "message cannot be empty"}, status_code=400)
        if len(prompt) > 65_536:
            return JSONResponse({"error": "message exceeds 65536 characters"}, status_code=413)
        speak = bool(payload.get("speak", False))

        async def stream() -> Any:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            emitted = False

            def emit_token(token: str) -> None:
                nonlocal emitted
                emitted = True
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "token", "text": str(token)},
                )

            async def execute() -> None:
                def preview(kind: str, text: str) -> None:
                    queue.put_nowait({"type": kind, "text": text})

                preview_token = response_preview.set(preview)
                memoir_turn = runtime.memoir.begin(prompt) if runtime.memoir else None
                memoir_token = current_memoir_turn.set(memoir_turn)
                try:
                    answer = await runtime.core.ask(prompt, on_token=emit_token)
                    if memoir_turn is not None:
                        memoir_turn.update(assistant=answer, status="answered")
                    # on_token may arrive through call_soon_threadsafe; give the
                    # loop one turn so token events stay ahead of the done event.
                    await asyncio.sleep(0)
                    if not emitted and answer:
                        await queue.put({"type": "token", "text": answer})
                    if speak and answer:
                        await queue.put({"type": "speech", "state": "speaking"})
                        try:
                            async with runtime.speech_lock:
                                await runtime.ensure_speech().speak(answer)
                        except Exception as exc:
                            await queue.put(
                                {"type": "speech", "state": "error", "error": str(exc)}
                            )
                        else:
                            await queue.put({"type": "speech", "state": "complete"})
                    await queue.put({"type": "done", "answer": answer})
                except asyncio.CancelledError:
                    await queue.put(
                        {"type": "error", "error": "V is shutting down"}
                    )
                    raise
                except Exception as exc:
                    if memoir_turn is not None:
                        memoir_turn["status"] = "error"
                    await queue.put({"type": "error", "error": str(exc)})
                finally:
                    current_memoir_turn.reset(memoir_token)
                    response_preview.reset(preview_token)

            async with runtime.chat_lock:
                if runtime.closing:
                    yield _ndjson({"type": "error", "error": "V is shutting down"})
                    return
                task = asyncio.create_task(execute())
                runtime.active_chat_task = task
                yield _ndjson({"type": "started"})
                try:
                    while True:
                        event = await queue.get()
                        yield _ndjson(event)
                        if event["type"] in {"done", "error"}:
                            break
                finally:
                    if not task.done():
                        task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    if runtime.active_chat_task is task:
                        runtime.active_chat_task = None

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def ptt_start(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        if runtime.speech_busy():
            return JSONResponse({"error": "V is busy or shutting down"}, status_code=409)
        async with runtime.speech_lock:
            try:
                speech = runtime.ensure_speech()
                await speech.start_push_to_talk()
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=503)
        return JSONResponse({"recording": True})

    async def ptt_stop(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        if runtime.speech is None or not runtime.speech.push_to_talk_recording:
            return JSONResponse({"error": "push-to-talk is not recording"}, status_code=409)
        async with runtime.speech_lock:
            try:
                transcript = await runtime.speech.stop_push_to_talk()
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=503)
        return JSONResponse({"recording": False, "transcript": transcript})

    async def shutdown(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        if runtime.shutdown_callback is None:
            return JSONResponse({"error": "shutdown controller unavailable"}, status_code=503)
        emergency = request.query_params.get("emergency") == "1"
        if runtime.shutdown_task and not emergency:
            return JSONResponse({"status": "saving_memory", "memoir": runtime.memoir.state})
        await runtime.cancel_active_chat()
        if runtime.speech is not None:
            # Exiting stops microphone capture before spending time on prose.
            with suppress(Exception):
                await runtime.speech.cancel_push_to_talk()
        if emergency and runtime.shutdown_task:
            runtime.shutdown_task.cancel()
            with suppress(asyncio.CancelledError):
                await runtime.shutdown_task
        if not emergency and runtime.memoir and runtime.memoir.turns:
            async def remember_and_stop() -> None:
                try:
                    cancel_memory = getattr(getattr(runtime.core, "agent", None), "cancel_background_memory", None)
                    if callable(cancel_memory):
                        await cancel_memory()
                    await runtime.memoir.save(getattr(runtime.core, "llm", None))
                finally:
                    asyncio.get_running_loop().call_later(2.5, runtime.shutdown_callback)
            runtime.shutdown_task = asyncio.create_task(remember_and_stop())
            return JSONResponse({"status": "saving_memory"}, status_code=202)
        asyncio.get_running_loop().call_later(0.15, runtime.shutdown_callback)
        return JSONResponse({"status": "shutting_down"})

    async def log(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        return JSONResponse(model_log(runtime.active_session), headers={"Cache-Control": "no-store"})

    async def voice_settings(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        manager = runtime.speech_settings
        try:
            if request.method == "POST":
                if runtime.speech_busy() or manager.busy:
                    return JSONResponse({"error": "Zakończ zadanie, nagrywanie lub instalację przed zmianą ustawień."}, status_code=409)
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise ValueError("Settings must be an object")
                async with runtime.speech_lock:
                    if runtime.closing or runtime.chat_lock.locked() or getattr(runtime.speech, "push_to_talk_recording", False):
                        return JSONResponse({"error": "V is busy or recording"}, status_code=409)
                    manager.save(payload)
                    if runtime.speech is not None:
                        await runtime.speech.close()
                    runtime.speech = None
            return JSONResponse(manager.status(), headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def voice_install(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        if runtime.speech_busy() or runtime.speech_settings.busy:
            return JSONResponse({"error": "V or installer is busy"}, status_code=409)
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Installer request must be an object")
            runtime.speech_settings.start_install(str(payload.get("kind", "")), str(payload.get("id", "")))
            return JSONResponse(runtime.speech_settings.job, status_code=202)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def voice_preview(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        if runtime.speech_busy():
            return JSONResponse({"error": "V is busy"}, status_code=409)
        try:
            async with runtime.speech_lock:
                await asyncio.wait_for(runtime.ensure_speech().speak(
                    "Cześć, Boss. To próbka zapisanego głosu V. Hello Boss, this is V's saved voice."
                ), timeout=90)
            return JSONResponse({"status": "complete"})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

    async def voice_install_cancel(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        await runtime.speech_settings.close()
        return JSONResponse(runtime.speech_settings.job)

    async def decide_proposal(request: Request) -> Response:
        denied = runtime.require_token(request)
        if denied is not None:
            return denied
        if not bool(getattr(getattr(runtime.config, "edition", None), "is_full", False)):
            return JSONResponse({"error": "proposal control requires Full"}, status_code=403)
        manager = getattr(
            getattr(getattr(runtime.core, "agent", None), "memory", None),
            "manager",
            None,
        )
        if manager is None or not hasattr(manager, "decide_proposal"):
            return JSONResponse({"error": "proposal store unavailable"}, status_code=503)
        decision = str(request.path_params.get("decision", "")).casefold()
        if decision not in {"approve", "reject"}:
            return JSONResponse({"error": "invalid proposal decision"}, status_code=400)
        try:
            proposal = manager.decide_proposal(
                str(request.path_params.get("proposal_id", "")),
                approve=decision == "approve",
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse(
            {"status": proposal.get("status"), "proposal_id": proposal.get("proposal_id")}
        )

    app = Starlette(
        debug=False,
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/api/status", status, methods=["GET"]),
            Route("/api/memory/browser", memory_browser, methods=["GET"]),
            Route("/api/model/log", log, methods=["GET"]),
            Route("/api/voice/settings", voice_settings, methods=["GET", "POST"]),
            Route("/api/voice/install", voice_install, methods=["POST"]),
            Route("/api/voice/install/cancel", voice_install_cancel, methods=["POST"]),
            Route("/api/voice/preview", voice_preview, methods=["POST"]),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/voice/ptt/start", ptt_start, methods=["POST"]),
            Route("/api/voice/ptt/stop", ptt_stop, methods=["POST"]),
            Route("/api/shutdown", shutdown, methods=["POST"]),
            Route(
                "/api/proposals/{proposal_id}/{decision}",
                decide_proposal,
                methods=["POST"],
            ),
            Mount("/assets", app=StaticFiles(directory=_STATIC_ROOT), name="assets"),
        ],
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )
    return app
