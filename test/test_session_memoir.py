from __future__ import annotations

import asyncio
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from v_core.memory.memoir import SessionMemoir, current_memoir_turn, record_memoir_execution, recalled_memoirs
from v_core.ui import UIRuntime, create_app
from starlette.testclient import TestClient
from test_ui import _runtime


def test_memoir_is_only_current_session_and_saves_once(tmp_path):
    journal = tmp_path / "conversation/dialogue.jsonl"
    journal.parent.mkdir()
    journal.write_text("older session must never appear")
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Boss poprawił mój opis.").update(assistant="Masz rację.", status="answered")

    async def ask(**kw):
        assert "older session" not in json.dumps(kw["messages"])
        assert "Boss poprawił" in kw["messages"][1]["content"]
        if kw["max_tokens"] == 32:
            return '{"supported":true}'
        return json.dumps({"memory": "Boss poprawił mój opis i przyznałam mu rację.", "turn_ids": [1]})

    async def scenario():
        result = await memoir.save(SimpleNamespace(ask=ask))
        await memoir.save(SimpleNamespace(ask=ask))
        return result
    result = asyncio.run(scenario())
    path = Path(result["file"])
    saved = json.loads(path.read_text())
    assert len(list(path.parent.glob("*.json"))) == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert saved["automatic_policy"] is False and saved["verified"] is False
    assert len(saved["turns"]) == 1
    assert saved["source"] == "self_generated"
    assert recalled_memoirs(tmp_path, "hello") == ""
    assert "fallible" in recalled_memoirs(tmp_path, "wspomnienie", recall=True)


@pytest.mark.parametrize("reply", ["garbage", '{"memory":"made up","turn_ids":[99]}', '{"memory":42,"turn_ids":[1]}'])
def test_invalid_generation_saves_honest_fallback(tmp_path, reply):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("hello")
    async def ask(**kw):
        return reply
    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert result["fallback"] is True
    assert "Nie udało" in result["memory"]
    assert json.loads(Path(result["file"]).read_text())["source"] == "runtime_fallback"


def test_model_json_wrappers_do_not_turn_a_valid_memoir_into_fallback(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Pamiętasz temat?").update(assistant="Tak, pamiętam ten temat.", status="answered")
    replies = iter([
        '<|begin of thinking|>I will follow the format.<|end of thinking|>\n'
        '```json\n{"memory":"Przypomniałam sobie temat rozmowy.","turn_ids":[1]}\n```',
        'Pewnie: {"supported":true}',
    ])

    async def ask(**_):
        return next(replies)

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert result["fallback"] is False
    assert result["memory"] == "Przypomniałam sobie temat rozmowy."


def test_verbatim_turn_echo_is_not_saved_as_a_generated_memoir(tmp_path):
    user_text = "To jest długa wiadomość Bossa o konkretnym temacie, " * 12
    memoir = SessionMemoir(tmp_path)
    memoir.begin(user_text).update(assistant="Odpowiedziałam na temat.", status="answered")

    async def ask(**kw):
        if kw["max_tokens"] == 32:
            raise AssertionError("copied draft must be rejected before the grounding call")
        return json.dumps({"memory": user_text, "turn_ids": [1]})

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    saved = json.loads(Path(result["file"]).read_text())
    assert result["fallback"] is True
    assert saved["fallback_reason"] == "generation_copied_excerpt"


def test_unsupported_inner_state_is_not_saved_as_a_generated_memoir(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Co się wydarzyło?").update(assistant="Odpowiedziałam.", status="answered")

    async def ask(**kw):
        return json.dumps({"memory": "Teraz wiem, co się wydarzyło.", "turn_ids": [1]})

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    saved = json.loads(Path(result["file"]).read_text())
    assert result["fallback"] is True
    assert saved["fallback_reason"] == "generation_unsupported_reflection"


def test_acknowledgement_does_not_claim_follow_up_work(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Nie o to mi chodziło.").update(assistant="Masz rację, skupmy się na tym.", status="answered")

    async def ask(**_):
        return json.dumps({"memory": "Zmieniłam kierunek i skupiłam się na temacie.", "turn_ids": [1]})

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    saved = json.loads(Path(result["file"]).read_text())
    assert result["fallback"] is True
    assert saved["fallback_reason"] == "generation_unsupported_action"


def test_generation_timeout_still_saves_current_session(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("interrupted task")
    async def ask(**kw):
        await asyncio.sleep(100)
    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask), timeout=0.01))
    assert result["fallback"]
    assert json.loads(Path(result["file"]).read_text())["turns"][0]["status"] == "interrupted"


def test_rejected_grounding_gets_one_repair_and_fresh_check(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Hasło to BURSZTYN-42.").update(assistant="BURSZTYN-42", status="answered")
    rejected = "Boss poprosił o wdrożenie aplikacji."
    accepted = "Boss podał hasło BURSZTYN-42. Odpowiedziałam tym hasłem."
    calls = []

    async def ask(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return json.dumps({"memory": rejected, "turn_ids": [1]})
        if len(calls) == 2:
            return '{"supported":false}'
        if len(calls) == 3:
            assert "grounding check rejected" in kw["messages"][0]["content"]
            assert rejected not in json.dumps(kw["messages"])
            return json.dumps({"memory": accepted, "turn_ids": [1]})
        assert len(calls) == 4
        assert accepted in kw["messages"][1]["content"]
        return '{"supported":true}'

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    saved = Path(result["file"]).read_text()
    assert not result["fallback"]
    assert len(calls) == 4
    assert rejected not in saved
    assert json.loads(saved)["verified"] is False


@pytest.mark.parametrize("check", ['[]', '{"supported":"true"}', '{"supported":false}'])
def test_grounding_retries_are_bounded_and_fail_closed(tmp_path, check):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("hello")
    calls = []

    async def ask(**kw):
        calls.append(kw)
        if kw["max_tokens"] == 32:
            return check
        return '{"memory":"Boss przywitał się.","turn_ids":[1]}'

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert result["fallback"]
    assert len(calls) == 4
    assert json.loads(Path(result["file"]).read_text())["fallback_reason"] == "grounding_check_rejected"


def test_empty_session_does_not_invent_memory(tmp_path):
    memoir = SessionMemoir(tmp_path)
    assert asyncio.run(memoir.save(None))["state"] == "empty"
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("narrative,reason", [
    ("Usłyszałam dźwięk w domu.", "generation_physical_autobiography"),
    ("Boss podał BURSZTYN-43.", "generation_ungrounded_identifier"),
])
def test_model_cannot_rubber_stamp_ungrounded_personal_claims_or_ids(tmp_path, narrative, reason):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("BURSZTYN-42").update(assistant="I heard a sound at home.", status="answered")

    async def ask(**kw):
        assert kw["max_tokens"] != 32, "deterministic failure must not reach model grounding"
        return json.dumps({"memory": narrative, "turn_ids": [1]})

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    saved = json.loads(Path(result["file"]).read_text())
    assert result["fallback"]
    assert saved["fallback_reason"] == reason
    assert saved["turns"][0]["assistant"] == "I heard a sound at home."


def test_memoir_identifier_must_belong_to_cited_turn(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Hasło BURSZTYN-42")
    memoir.begin("Drugi temat")

    async def ask(**kw):
        return '{"memory":"Boss podał BURSZTYN-42.","turn_ids":[2]}'

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert result["fallback"]


@pytest.mark.parametrize("draft", ["COPPER-19.", "You corrected the label from SAPPHIRE-73 to COPPER-19."])
def test_memoir_repairs_identifier_only_and_short_answer_echo(tmp_path, draft):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("The label is SAPPHIRE-73.").update(assistant="Noted.", status="answered")
    memoir.begin("Correction: COPPER-19.").update(
        assistant="You corrected the label from SAPPHIRE-73 to COPPER-19.", status="answered",
    )
    calls = 0

    async def ask(**kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps({"memory": draft, "turn_ids": [2]})
        if calls == 2:
            assert kw["max_tokens"] != 32
            return json.dumps({"memory": "Boss zmienił etykietę z SAPPHIRE-73 na COPPER-19.", "turn_ids": [1, 2]})
        assert calls == 3 and kw["max_tokens"] == 32
        return '{"supported":true}'

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert not result["fallback"]
    assert result["memory"] == "Boss zmienił etykietę z SAPPHIRE-73 na COPPER-19."


def test_repair_uses_honestly_marked_shortened_evidence_and_no_invented_template(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("x" * 400).update(assistant="Noted.", status="answered")
    calls = []

    async def ask(**kw):
        calls.append(kw)
        instruction = kw["messages"][0]["content"]
        evidence = json.loads(kw["messages"][1]["content"])
        if len(calls) == 1:
            assert evidence["partial_excerpts"] is False
            assert "One accurate sentence" in instruction
            assert "A safe shape is" not in instruction
            return "invalid"
        assert evidence["partial_excerpts"] is True
        assert len(evidence["session_turns"][0]["user"]) == 320
        if kw["max_tokens"] == 32:
            assert evidence["cited_turn_ids"] == [1]
            return '{"supported":true}'
        assert "exactly two" not in instruction
        return '{"memory":"Boss podał ciąg znaków.","turn_ids":[1]}'

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert not result["fallback"]
    assert json.loads(Path(result["file"]).read_text())["partial_excerpts"] is True


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_masculine_narrator_requires_repair_before_grounding(tmp_path, repair_succeeds):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Hasło?").update(assistant="BURSZTYN-42", status="answered")
    generations = 0
    checks = 0

    async def ask(**kw):
        nonlocal generations, checks
        if kw["max_tokens"] == 32:
            checks += 1
            assert repair_succeeds
            return '{"supported":true}'
        generations += 1
        if generations == 2:
            assert "first person feminine" in kw["messages"][0]["content"]
        verb = "Odpowiedziałam" if generations == 2 and repair_succeeds else "Odpowiedziałem"
        return json.dumps({"memory": f"Boss zapytał o hasło. {verb}: BURSZTYN-42.", "turn_ids": [1]})

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert generations == 2
    assert checks == int(repair_succeeds)
    assert result["fallback"] is not repair_succeeds
    if not repair_succeeds:
        assert json.loads(Path(result["file"]).read_text())["fallback_reason"] == "generation_wrong_narrator"


def test_quoted_user_voice_does_not_change_narrator(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Powiedziałem: hello").update(assistant="hello", status="answered")

    async def ask(**kw):
        if kw["max_tokens"] == 32:
            return '{"supported":true}'
        return json.dumps({"memory": 'Boss napisał „Powiedziałem: hello”. Odpowiedziałam powitaniem.', "turn_ids": [1]})

    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert not result["fallback"]


def test_execution_capture_omits_arguments_and_outputs():
    turn = {}
    token = current_memoir_turn.set(turn)
    try:
        record_memoir_execution({"status": "failed", "tool_calls": [
            {"tool": "web_read", "status": "failed", "arguments": "private", "output": "private"}
        ]})
    finally:
        current_memoir_turn.reset(token)
    assert turn == {"execution": {"status": "failed", "tool_calls": [{"tool": "web_read", "status": "failed"}]}}


def test_ui_shutdown_saves_after_answer_and_blocks_new_input(tmp_path):
    runtime = _runtime()
    runtime.memoir = SessionMemoir(tmp_path)
    runtime.shutdown_callback = lambda: None
    headers = {"X-DARKLINGER-Session": runtime.session_token}
    with TestClient(create_app(runtime)) as client:
        client.post("/api/chat", headers=headers, json={"message": "this session only"})
        result = client.post("/api/shutdown", headers=headers)
        assert result.status_code == 202
        assert client.post("/api/chat", headers=headers, json={"message": "too late"}).status_code == 503
        assert client.post("/api/voice/ptt/start", headers=headers).status_code == 409
        # No model stub: deterministic fallback writes synchronously in the task.
        state = client.get("/api/status", headers=headers).json()
        assert state["memoir"]["state"] == "saved"
        assert client.post("/api/shutdown", headers=headers).status_code == 200
    records = list((tmp_path / "session_memoirs").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["turns"][0]["user"] == "this session only"


def test_emergency_stop_cancels_pending_memoir(tmp_path):
    runtime = _runtime()
    runtime.memoir = SessionMemoir(tmp_path)
    runtime.memoir.begin("test")
    runtime.shutdown_callback = lambda: None
    async def ask(**kw):
        await asyncio.sleep(100)
    runtime.core.llm = SimpleNamespace(ask=ask)
    headers = {"X-DARKLINGER-Session": runtime.session_token}
    with TestClient(create_app(runtime)) as client:
        assert client.post("/api/shutdown", headers=headers).status_code == 202
        assert client.post("/api/shutdown?emergency=1", headers=headers).status_code == 200
        assert runtime.shutdown_task.cancelled()
    assert not list(tmp_path.rglob("*.json"))


def test_memory_disk_error_is_visible(tmp_path, monkeypatch):
    import v_core.memory.memoir as module
    memoir = SessionMemoir(tmp_path)
    memoir.begin("hello")
    def failed(*args):
        raise OSError("disk full")
    monkeypatch.setattr(module, "atomic_json", failed)
    assert asyncio.run(memoir.save(None))["state"] == "error"


def test_rejected_memoir_never_enters_persistent_narrative(tmp_path):
    memoir = SessionMemoir(tmp_path)
    memoir.begin("Dodaj przykład").update(assistant="GCC", status="answered")
    async def ask(**kw):
        if kw["max_tokens"] == 32:
            return '{"supported":false}'
        return '{"memory":"Boss był rozczarowany także poprawką.","turn_ids":[1]}'
    result = asyncio.run(memoir.save(SimpleNamespace(ask=ask)))
    assert result["fallback"] is True
    assert "rozczarowany" not in Path(result["file"]).read_text()
    assert json.loads(Path(result["file"]).read_text())["fallback_reason"] == "grounding_check_rejected"
