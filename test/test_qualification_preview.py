from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from v_core.llm.llm import LLMResponse, LLMToolCall
from v_core.model_loader.models import ModelProfile
from v_core.model_loader.qualification import ModelQualifier, QUALIFICATION_HARNESS_VERSION
from v_core.model_loader.qualification_trace import QualificationTrace, qualification_preview, read_qualification_run
from v_core.ui import create_app
from test_ui import _runtime


class Model:
    async def respond(self, **kwargs):
        return LLMResponse(content="DARKLINGER_READY_731", finish_reason="stop")


def profile(tmp_path):
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"GGUF fixture only")
    return ModelProfile(model_path=str(model), alias="FIXTURE_ONLY", reasoning="off")


def start_probe(qualifier, root):
    qualifier.configure_preview(root)
    qualifier.trace.start("/fixture.gguf", "FIXTURE ONLY", 14, QUALIFICATION_HARNESS_VERSION)


@pytest.mark.asyncio
async def test_question_visible_while_reply_is_pending(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    class Pending(Model):
        async def respond(self, **kwargs):
            entered.set()
            await release.wait()
            return await super().respond(**kwargs)
    qualifier = ModelQualifier(Pending())
    start_probe(qualifier, tmp_path)
    task = asyncio.create_task(qualifier._exact_instruction_probe())
    await entered.wait()
    pending = qualification_preview(tmp_path)["selected"]
    assert pending["completed"] == 0 and pending["state"] == "unfinished"
    request = next(e for e in pending["events"] if e["event"] == "request_started")
    assert "Return exactly DARKLINGER_READY_731" in request["messages"][-1]["content"]
    assert not any(e["event"] == "response_received" for e in pending["events"])
    release.set()
    result = await task
    assert result.score == 100
    complete = qualification_preview(tmp_path)["selected"]
    assert complete["completed"] == 1
    assert next(e for e in complete["events"] if e["event"] == "response_received")["response"]["content"] == "DARKLINGER_READY_731"


@pytest.mark.asyncio
async def test_preview_preserves_grades_and_completed_history(tmp_path):
    selected = profile(tmp_path)
    plain = await ModelQualifier(Model()).qualify(selected)
    output = []
    qualifier = ModelQualifier(Model())
    qualifier.configure_preview(tmp_path, output=output.append)
    observed = await qualifier.qualify(selected)
    assert observed.capabilities == plain.capabilities
    assert [(p.name, p.score, p.passed, p.detail, p.output_digest) for p in observed.probes] == [
        (p.name, p.score, p.passed, p.detail, p.output_digest) for p in plain.probes]
    run = qualification_preview(tmp_path)["selected"]
    assert run["state"] == "completed" and run["completed"] == run["total"] == 14
    assert run["overall_score"] == observed.overall_score
    assert any("Return exactly" in line for line in output)
    assert any("[model answer]" in line for line in output)
    restarted = qualification_preview(tmp_path, run["run_id"])["selected"]
    assert restarted["events"] == run["events"]
    assert len(list((tmp_path / "qualifications").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_retry_keeps_both_attempts_and_final_score(tmp_path):
    class Retry(Model):
        calls = 0
        async def respond(self, **kwargs):
            self.calls += 1
            return LLMResponse(content="partial" if self.calls == 1 else "DARKLINGER_READY_731",
                               finish_reason="length" if self.calls == 1 else "stop")
    qualifier = ModelQualifier(Retry())
    start_probe(qualifier, tmp_path)
    result = await qualifier._exact_instruction_probe()
    events = qualification_preview(tmp_path)["selected"]["events"]
    requests = [e for e in events if e["event"] == "request_started"]
    responses = [e for e in events if e["event"] == "response_received"]
    assert [e["attempt"] for e in requests] == [1, 2]
    assert [e["max_tokens"] for e in requests] == [32, 2048]
    assert [e["response"]["content"] for e in responses] == ["partial", "DARKLINGER_READY_731"]
    assert result.score == 100


@pytest.mark.asyncio
async def test_native_calls_and_multiturn_context_are_recorded_without_execution(tmp_path):
    class Calls:
        turn = 0
        async def respond(self, **kwargs):
            self.turn += 1
            target = "primary" if self.turn == 1 else "backup"
            if self.turn < 3:
                return LLMResponse(tool_calls=[LLMToolCall("call-fixture", "probe_fetch", {"target": target},
                    raw_arguments=json.dumps({"target": target}))], native_tools_enabled=True, finish_reason="tool_calls")
            return LLMResponse(content="The backup fixture was retrieved successfully.", finish_reason="stop")
    qualifier = ModelQualifier(Calls())
    start_probe(qualifier, tmp_path)
    await qualifier._failed_tool_recovery_probe()
    events = qualification_preview(tmp_path)["selected"]["events"]
    requests = [e for e in events if e["event"] == "request_started"]
    responses = [e for e in events if e["event"] == "response_received"]
    assert len(requests) == len(responses) == 3
    assert len(requests[0]["messages"]) == 2  # Saved before later turns append context.
    assert "SIMULATED RUNTIME EVIDENCE" in requests[1]["messages"][-1]["content"]
    call = responses[0]["response"]["tool_calls"][0]
    assert call["name"] == "probe_fetch" and call["arguments"] == {"target": "primary"}
    assert call["call_id"] == "call-fixture" and call["raw_arguments"] == '{"target": "primary"}'
    assert requests[0]["tools"][0]["function"]["name"] == "probe_fetch"


@pytest.mark.asyncio
async def test_preview_write_failure_does_not_change_grade(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    qualifier = ModelQualifier(Model())
    qualifier.configure_preview(blocked, output=lambda _: (_ for _ in ()).throw(RuntimeError("closed output")))
    qualifier.trace.start("/fixture.gguf", "fixture", 14, 9)
    assert (await qualifier._exact_instruction_probe()).score == 100
    assert qualifier.trace.error


@pytest.mark.asyncio
async def test_cancelled_run_has_no_final_grade(tmp_path):
    entered = asyncio.Event()
    class Pending:
        async def respond(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()
    qualifier = ModelQualifier(Pending())
    qualifier.configure_preview(tmp_path)
    task = asyncio.create_task(qualifier.qualify(profile(tmp_path)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run = qualification_preview(tmp_path)["selected"]
    assert run["state"] == "cancelled" and run["overall_score"] is None
    assert run["completed"] == 0


def test_partial_line_is_ignored_and_symlink_target_is_denied(tmp_path):
    trace = QualificationTrace(tmp_path)
    trace.start("/fixture.gguf", "fixture", 14, 9)
    with trace.path.open("a") as handle:
        handle.write('{"event":"response_received"')
    result = read_qualification_run(tmp_path, trace.run_id)
    assert result["incomplete_line"] is True and len(result["events"]) == 1
    trace.path.unlink()
    target = tmp_path / "secret.txt"
    target.write_text("not a transcript")
    trace.path.symlink_to(target)
    with pytest.raises(FileNotFoundError):
        read_qualification_run(tmp_path, trace.run_id)


@pytest.mark.parametrize("edition", ["public", "full"])
def test_ui_preview_is_token_protected_and_read_only_in_both_editions(tmp_path, edition):
    trace = QualificationTrace(tmp_path)
    trace.start("/fixture.gguf", "fixture", 14, 9)
    trace.emit("response_received", probe="example", request_id=1, attempt=1, latency_ms=1,
               response={"content": '<img src=x onerror="alert(1)">', "tool_calls": []})
    runtime = _runtime()
    runtime.config.model_runtime_root = tmp_path
    runtime.config.edition = SimpleNamespace(name=edition)
    client = TestClient(create_app(runtime))
    before = {p.name: p.read_bytes() for p in (tmp_path / "qualifications").iterdir()}
    assert client.get("/api/model/tests").status_code == 403
    headers = {"X-DARKLINGER-Session": runtime.session_token}
    response = client.get("/api/model/tests", headers=headers)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["selected"]["events"][-1]["response"]["content"].startswith("<img")
    assert client.get("/api/model/tests?run_id=../../secret", headers=headers).status_code == 400
    assert client.get("/api/model/tests?run_id=20261005T000000-aaaaaaaaaaaa", headers=headers).status_code == 404
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "qualifications").iterdir()}
    index = client.get("/").text
    assert 'id="model-tests-open"' in index and 'id="model-tests-run"' in index


def test_empty_history_does_not_create_files(tmp_path):
    assert qualification_preview(tmp_path) == {"runs": [], "selected": None}
    assert list(tmp_path.iterdir()) == []


def test_history_uses_start_time_and_explicit_selection(tmp_path):
    first, second = QualificationTrace(tmp_path), QualificationTrace(tmp_path)
    first.start("/first.gguf", "FIRST", 14, 9)
    second.start("/second.gguf", "SECOND", 14, 9)
    assert qualification_preview(tmp_path)["selected"]["run_id"] == second.run_id
    assert qualification_preview(tmp_path, first.run_id)["selected"]["alias"] == "FIRST"
