import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from v_core.memory.memoir import SessionMemoir
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.ui import UIRuntime, create_app


SID = "20261001T120000Z-abcdef123456"


def setup_browser(tmp_path, edition="public"):
    runtime = UIRuntime(
        core=SimpleNamespace(agent=SimpleNamespace()),
        config=SimpleNamespace(edition=SimpleNamespace(name=edition),
                               memory_root=tmp_path, voice_root=tmp_path / "voice"),
    )
    client = TestClient(create_app(runtime))
    client.headers["X-DARKLINGER-Session"] = runtime.session_token
    return runtime, client


def archive(runtime, sid=SID, **extra):
    runtime.memoir.root.mkdir(exist_ok=True)
    path = runtime.memoir.root / (sid + ".json")
    data = dict(schema_version=1, kind="session_memoir", session_id=sid,
                memory="old private memory", turns=[], source="self_generated")
    data.update(extra)
    path.write_text(json.dumps(data))
    return path


def test_auth_and_read_only(tmp_path):
    runtime, client = setup_browser(tmp_path)
    assert TestClient(create_app(runtime)).get("/api/memory/browser").status_code == 403
    assert client.post("/api/memory/browser").status_code == 405
    response = client.get("/api/memory/browser")
    assert response.headers["cache-control"] == "no-store"
    assert not runtime.memoir.root.exists()


def test_public_cannot_access_previous_sessions_even_by_id(tmp_path):
    runtime, client = setup_browser(tmp_path)
    archive(runtime)
    runtime.memoir.begin("current conversation")
    assert [s["id"] for s in client.get("/api/memory/browser").json()["sessions"]] == ["current"]
    assert client.get("/api/memory/browser", params={"view": "session", "session": SID}).status_code == 403
    data = client.get("/api/memory/browser?view=session").json()
    assert data["record"]["turns"][0]["user"] == "current conversation"
    assert "old private memory" not in json.dumps(data)


@pytest.mark.parametrize("edition", ["full", "full_access"])
def test_archives_are_bounded_untrusted_and_read_only(tmp_path, edition):
    runtime, client = setup_browser(tmp_path, edition)
    path = archive(runtime, verified=True, turns=[{"id": i, "user": "x" * 13000,
                  "assistant": "<script>alert(1)</script>", "secret": "not exposed"} for i in range(60)])
    before = path.read_bytes()
    result = client.get("/api/memory/browser", params={"view": "session", "session": SID})
    data = result.json()["record"]
    assert data["verified"] is False
    assert data["omitted_turns"] == 10
    assert len(data["turns"]) == 50
    assert "truncated" in data["turns"][0]["user"]
    assert "not exposed" not in result.text
    assert path.read_bytes() == before


@pytest.mark.parametrize("sid", ["../secret", "/etc/passwd", "", "arbitrary.json"])
def test_path_traversal_rejected(tmp_path, sid):
    runtime, client = setup_browser(tmp_path, "full")
    response = client.get("/api/memory/browser", params={"view": "session", "session": sid})
    assert response.status_code == 400


def test_symlinks_corruption_oversize_and_schema(tmp_path):
    runtime, client = setup_browser(tmp_path, "full")
    path = archive(runtime)
    for content in ["{", "[]", '{"schema_version": 2}', "x" * (2 * 1024 * 1024 + 1)]:
        path.write_text(content)
        assert client.get("/api/memory/browser", params={"view": "session", "session": SID}).status_code == 400
    path.unlink()
    secret = tmp_path / "secret"
    secret.write_text("do not expose")
    path.symlink_to(secret)
    response = client.get("/api/memory/browser", params={"view": "session", "session": SID})
    assert response.status_code == 400
    assert "do not expose" not in response.text
    assert len(client.get("/api/memory/browser").json()["sessions"]) == 1


def test_saved_current_memoir_keeps_provenance(tmp_path):
    runtime, client = setup_browser(tmp_path)
    runtime.memoir.begin("hello")
    asyncio.run(runtime.memoir.save(None))
    data = client.get("/api/memory/browser?view=session").json()
    assert data["save_state"] == "saved"
    assert data["record"]["source"] == "runtime_fallback"
    assert data["record"]["fallback_reason"] == "model_unavailable"


def test_public_persona_does_not_reveal_historical_relationship(tmp_path):
    runtime, client = setup_browser(tmp_path)
    runtime.core.agent.persona = PersonaRuntime(IdentityKernel(), VoiceProfile())
    # An invalid relationship object would fail if dynamic history was accessed.
    runtime.core.agent.memory = SimpleNamespace(relationship_state=object())
    response = client.get("/api/memory/browser?view=persona")
    assert response.status_code == 200
    assert len(response.json()["sections"]) == 3
    assert response.json()["delivery_verified"] is False


def test_unknown_edition_fails_closed(tmp_path):
    runtime, client = setup_browser(tmp_path, "unknown")
    archive(runtime)
    assert not client.get("/api/memory/browser").json()["archive_allowed"]


def test_memory_ui_uses_text_content_and_has_controls(tmp_path):
    _, client = setup_browser(tmp_path)
    html = client.get("/").text
    assert 'id="memory-browser-dialog"' in html
    js = client.get("/assets/app.js").text
    memory_js = js[js.index("function memorySection"):js.index("const ui =")]
    assert "innerHTML" not in memory_js
    assert "body.textContent = text" in memory_js
