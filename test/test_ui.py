from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from starlette.testclient import TestClient

from v_core.edition import load_edition_extension, resolve_edition
from v_core.ui import UIRuntime, create_app
from v_core.ui.runtime_activity import runtime_activity
from v_core.response_preview import response_preview


def test_draft_events_are_separate_from_verified_speech_and_result():
    runtime = _runtime()
    spoken = []

    async def ask(prompt, on_token=None):
        emit = response_preview.get()
        emit("draft_start", "")
        emit("draft_token", "unverified draft")
        emit("draft_validating", "")
        return "verified replacement"

    async def speak(text):
        spoken.append(text)

    runtime.core.ask = ask
    runtime.speech = SimpleNamespace(speak=speak)
    response = TestClient(create_app(runtime)).post(
        "/api/chat", headers={"X-DARKLINGER-Session": runtime.session_token},
        json={"message": "review", "speak": True},
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [e["type"] for e in events[:5]] == [
        "started", "draft_start", "draft_token", "draft_validating", "token",
    ]
    assert spoken == ["verified replacement"]
    assert events[-1]["answer"] == "verified replacement"
    assert response_preview.get() is None


class _Process:
    def poll(self) -> None:
        return None


class _Tools:
    def __init__(self) -> None:
        self.edition_extension = load_edition_extension(resolve_edition("public"))

    def local_tool_names(self) -> list[str]:
        return ["web_search", "web_read", "learning_create_tool"]


class _Core:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(tools=_Tools())
        self.model_runtime = None
        self.closed = False

    async def ask(self, prompt: str, on_token=None) -> str:
        answer = f"V received: {prompt}"
        if on_token is not None:
            on_token("V received: ")
            on_token(prompt)
        return answer

    async def close(self) -> None:
        self.closed = True


def _runtime() -> UIRuntime:
    edition = resolve_edition("public")
    config = SimpleNamespace(edition=edition, voice_root="voice")
    session = SimpleNamespace(
        profile=SimpleNamespace(
            alias="test-model",
            model_path="/models/test.gguf",
            context_size=12_032,
            reasoning="off",
            cache_type_k="q8_0",
            cache_type_v="q4_0",
        ),
        process=_Process(),
    )
    return UIRuntime(core=_Core(), config=config, model_session=session)


def test_ui_is_local_session_token_protected() -> None:
    runtime = _runtime()
    client = TestClient(create_app(runtime))

    index = client.get("/")
    denied = client.get("/api/status")
    accepted = client.get(
        "/api/status",
        headers={"X-DARKLINGER-Session": runtime.session_token},
    )

    assert index.status_code == 200
    assert runtime.session_token in index.text
    assert "frame-ancestors 'none'" in index.headers["content-security-policy"]
    assert denied.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json()["edition"] == "public"
    assert accepted.json()["model"]["alias"] == "test-model"
    assert accepted.json()["owner"] is None


def test_ui_uses_darklinger_public_identity() -> None:
    index = TestClient(create_app(_runtime())).get("/")

    assert index.status_code == 200
    assert "<title>DARKLINGER // V</title>" in index.text
    assert '<div class="brand-mark">D</div>' in index.text
    assert "<h1>DARKLINGER <span>// V</span></h1>" in index.text
    assert "Hold to stop Darklinger and its model" in index.text


def test_ui_accepts_legacy_session_header_during_rename() -> None:
    runtime = _runtime()
    response = TestClient(create_app(runtime)).get(
        "/api/status",
        headers={"X-PALADYN-Session": runtime.session_token},
    )

    assert response.status_code == 200


def test_ui_uses_independent_scroll_regions_and_fixed_composer() -> None:
    runtime = _runtime()
    client = TestClient(create_app(runtime))

    index = client.get("/")
    css = client.get("/assets/app.css")
    javascript = client.get("/assets/app.js")

    assert index.status_code == 200
    assert 'id="messages" class="messages" tabindex="0"' in index.text
    assert css.status_code == 200
    assert ".conversation { display: flex; min-height: 0; flex-direction: column; overflow: hidden; }" in css.text
    assert "flex: 0 0 auto;" in css.text
    assert "scrollbar-gutter: stable;" in css.text
    assert javascript.status_code == 200
    assert "function chatIsPinnedToBottom()" in javascript.text
    assert "scrollChatToBottom(followOutput);" in javascript.text
    assert 'id="live-state"' in index.text
    assert 'id="live-context-meter"' in index.text
    assert "function renderRuntimeActivity(activity)" in javascript.text


def test_runtime_activity_combines_live_model_and_task_ledger(tmp_path: Path) -> None:
    autonomy = tmp_path / "autonomy"
    checkpoints = autonomy / "interactive" / "checkpoints"
    journals = autonomy / "interactive" / "journal"
    checkpoints.mkdir(parents=True)
    journals.mkdir(parents=True)
    task_id = "interactive-live"
    now = datetime.now(timezone.utc).isoformat()
    (checkpoints / f"{task_id}.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "status": "running",
                "objective": "Build a verified local tool",
                "updated_at": now,
                "tool_calls": [
                    {
                        "sequence": 1,
                        "tool": "learning_create_tool",
                        "status": "failed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (journals / f"{task_id}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": now,
                        "event": "task_started",
                        "data": {},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": now,
                        "event": "model_phase_routed",
                        "data": {"from_phase": "tool_use", "to_phase": "coding"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": now,
                        "event": "tool_failed",
                        "data": {"sequence": 1, "tool": "learning_create_tool"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log = tmp_path / "llama.log"
    log.write_text(
        "slot print_timing: id 0 | n_decoded = 181, tg = 4.5 t/s, "
        "tg_3s = 4.2 t/s\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(
        profile=SimpleNamespace(port=5001, context_size=12_032),
        log_path=log,
    )

    def fake_http_json(port: int, path: str):
        assert (port, path) == (5001, "/slots")
        return [
            {
                "n_ctx": 12_032,
                "is_processing": True,
                "n_prompt_tokens": 6_278,
                "n_prompt_tokens_processed": 6_278,
                "next_token": [{"n_decoded": 181, "n_remain": 2_819}],
            }
        ]

    activity = runtime_activity(
        autonomy,
        session,
        ready=False,
        http_json=fake_http_json,
    )

    assert activity["state"] == "repairing"
    assert activity["phase"] == "coding"
    assert activity["generated_tokens"] == 181
    assert activity["generation_tokens_per_second"] == 4.2
    assert activity["context_used"] == 6_459
    assert activity["last_tool"] == "learning_create_tool"
    assert any(event["label"] == "FAIL" for event in activity["events"])


def test_runtime_activity_does_not_leak_previous_ui_session_task(
    tmp_path: Path,
) -> None:
    autonomy = tmp_path / "autonomy"
    checkpoints = autonomy / "interactive" / "checkpoints"
    checkpoints.mkdir(parents=True)
    stale = checkpoints / "interactive-stale.json"
    stale.write_text(
        json.dumps(
            {
                "task_id": "interactive-stale",
                "status": "running",
                "objective": "Old task",
            }
        ),
        encoding="utf-8",
    )

    activity = runtime_activity(
        autonomy,
        None,
        ready=True,
        session_started_at=stale.stat().st_mtime + 1,
    )

    assert activity["state"] == "idle"
    assert activity["phase"] == "idle"
    assert activity["task_id"] == ""
    assert activity["objective"] == ""
    assert activity["events"] == []


def test_runtime_activity_derives_live_rate_between_slot_samples(
    tmp_path: Path,
) -> None:
    log = tmp_path / "llama.log"
    log.write_text("model generating without periodic timing yet\n", encoding="utf-8")
    session = SimpleNamespace(
        profile=SimpleNamespace(port=5001, context_size=12_032),
        process=SimpleNamespace(pid=42),
        log_path=log,
    )
    generated = iter((100, 112))

    def fake_http_json(_: int, __: str):
        return [
            {
                "id_task": 7,
                "n_ctx": 12_032,
                "is_processing": True,
                "n_prompt_tokens": 2_000,
                "n_prompt_tokens_processed": 2_000,
                "next_token": [{"n_decoded": next(generated), "n_remain": 2_000}],
            }
        ]

    sample: dict = {}
    first = runtime_activity(
        None,
        session,
        ready=False,
        sample_state=sample,
        sample_now=10.0,
        http_json=fake_http_json,
    )
    second = runtime_activity(
        None,
        session,
        ready=False,
        sample_state=sample,
        sample_now=12.0,
        http_json=fake_http_json,
    )

    assert first["generation_tokens_per_second"] is None
    assert second["generation_tokens_per_second"] == 6.0


def test_runtime_activity_keeps_completed_response_timing(tmp_path: Path) -> None:
    log = tmp_path / "llama.log"
    log.write_text(
        "slot print_timing: prompt eval time = 100.00 ms / 200 tokens "
        "(0.50 ms per token, 2000.00 tokens per second)\n"
        "slot print_timing: eval time = 1000.00 ms / 80 tokens "
        "(12.50 ms per token, 80.00 tokens per second)\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(
        profile=SimpleNamespace(port=5001, context_size=12_032),
        log_path=log,
    )

    activity = runtime_activity(
        None,
        session,
        ready=True,
        http_json=lambda _port, _path: [
            {
                "n_ctx": 12_032,
                "is_processing": False,
                "next_token": [],
            }
        ],
    )

    assert activity["generated_tokens"] == 80
    assert activity["generation_tokens_per_second"] == 80.0
    assert activity["prompt_tokens"] == 200
    assert activity["prompt_tokens_per_second"] == 2000.0


def test_ui_streams_v_response_as_ndjson() -> None:
    runtime = _runtime()
    client = TestClient(create_app(runtime))

    with client.stream(
        "POST",
        "/api/chat",
        headers={"X-DARKLINGER-Session": runtime.session_token},
        json={"message": "hello", "speak": False},
    ) as response:
        events = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert '"type": "started"' in events[0]
    assert any('"text": "V received: "' in line for line in events)
    assert any('"text": "hello"' in line for line in events)
    assert '"type": "done"' in events[-1]


def test_ui_shutdown_uses_server_callback() -> None:
    runtime = _runtime()
    called: list[bool] = []
    runtime.shutdown_callback = lambda: called.append(True)
    client = TestClient(create_app(runtime))

    response = client.post(
        "/api/shutdown",
        headers={"X-DARKLINGER-Session": runtime.session_token},
        json={},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "shutting_down"}
    assert runtime.closing is True


def test_ui_runtime_cancels_active_chat_before_shutdown() -> None:
    runtime = _runtime()

    async def scenario() -> None:
        runtime.active_chat_task = asyncio.create_task(asyncio.sleep(60))
        active = runtime.active_chat_task

        await runtime.cancel_active_chat()

        assert runtime.closing is True
        assert active.cancelled()

    asyncio.run(scenario())


def test_ui_runtime_close_stops_core_and_managed_model() -> None:
    runtime = _runtime()

    asyncio.run(runtime.close())

    assert runtime.core.closed is True
