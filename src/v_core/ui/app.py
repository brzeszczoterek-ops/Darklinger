from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
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
from v_core.response_preview import response_preview
from v_core.ui.runtime_activity import runtime_activity


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
            "ready": not self.chat_lock.locked(),
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
                try:
                    answer = await runtime.core.ask(prompt, on_token=emit_token)
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
                    await queue.put({"type": "error", "error": str(exc)})
                finally:
                    response_preview.reset(preview_token)

            async with runtime.chat_lock:
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
        await runtime.cancel_active_chat()
        asyncio.get_running_loop().call_later(0.15, runtime.shutdown_callback)
        return JSONResponse({"status": "shutting_down"})

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
