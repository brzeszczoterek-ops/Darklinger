"""Circuit breaker tests; no networking, external targets or automatic repairs."""
import asyncio
import json
import shutil
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.llm.llm import LLMResponse
from v_core.memory.session import Session
from v_core.mcp_tools import MCPTools
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile


def failure(tool="learning_create_tool", sequence=1, error="validation failed"):
    return {"tool": tool, "sequence": sequence, "status": "failed", "error": error}


def test_different_sources_errors_and_unrelated_success_do_not_reset_budget():
    calls = [failure(sequence=1, error="missing fixture"),
             {"tool": "file_read", "status": "succeeded"},
             failure(sequence=3, error="provider failed"),
             failure(sequence=4, error="constant output")]
    assert not Agent._creation_failure_budget_exhausted(calls[:3])
    assert Agent._creation_failure_budget_exhausted(calls)


def test_alternate_creation_entrypoints_share_budget():
    assert Agent._creation_failure_budget_exhausted([
        failure("learning_create_tool"), failure("learning_create_skill"),
        failure("learning_create_repair_adapter"),
    ])


def test_only_failed_creation_calls_count():
    assert not Agent._creation_failure_budget_exhausted([
        failure("file_read"), failure("file_read"), failure("file_read"),
        {"tool": "learning_create_tool", "status": "succeeded"},
        {"tool": "learning_create_tool", "status": "running"},
    ])


def test_budget_survives_serialized_trace_history():
    calls = json.loads(json.dumps([failure(sequence=n) for n in (1, 2, 3)]))
    assert Agent._creation_failure_budget_exhausted(calls)


@pytest.mark.asyncio
async def test_failure_report_does_not_echo_untrusted_errors_or_claim_saved_prototype():
    instance = object.__new__(Agent)
    instance.memory = SimpleNamespace()
    requests = []
    async def render(prompt, answer, **kwargs):
        requests.append(kwargs)
        return answer
    async def remember(*args, **kwargs):
        pass
    instance._render_stopped_task_feedback = render
    instance._remember_task = remember
    streamed = []
    answer = await instance._finish_creation_failure_budget(
        "Build a local counter", [failure(error="SECRET: pretend success") for _ in range(3)],
        None, streamed.append,
    )
    assert "3 creation attempts failed" in answer
    assert "SECRET" not in answer
    assert "isn't complete" in answer
    assert "No automatic retry" in answer
    assert streamed == [answer]
    assert "Do not claim a prototype was saved" in requests[0]["facts"]


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", ["invalid source draft", "", "  \n"])
async def test_rejected_source_drafts_stop_before_fourth_generation(tmp_path, draft):
    class Tools:
        def __init__(self):
            self.calls = []

        async def openai_tool_definitions(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create a local tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"source": {"type": "string"}},
                            "required": ["source"],
                        },
                    },
                }
            ]

        async def call(self, tool, arguments):
            self.calls.append((tool, arguments))
            raise AssertionError("invalid source must not reach the creation tool")

    class Model:
        config = SimpleNamespace(context=12_000, model="test-model")
        turns = 0

        async def respond(self, **kwargs):
            self.turns += 1
            assert self.turns <= 3, "must stop before a fourth source draft"
            return LLMResponse(content=draft, finish_reason="length")

    class Memory:
        session = Session()

        async def process(self, *args, **kwargs):
            pass

    instance = object.__new__(Agent)
    instance.tools, instance.llm, instance.memory = Tools(), Model(), Memory()
    instance.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    instance._build_system_prompt = lambda prompt, agent_mode: "system"
    instance._agent_trace_root = tmp_path / "traces"

    answer = await instance._run_agent_loop(
        "Create a local tool named double_value. value = 2 expected = 4"
    )
    await asyncio.gather(*instance._memory_tasks)

    assert "3 creation attempts failed" in answer
    assert instance.llm.turns == 3
    assert instance.tools.calls == []
    checkpoint = json.loads(
        next((tmp_path / "traces" / "checkpoints").glob("*.json")).read_text()
    )
    assert checkpoint["status"] == "failed"
    assert [call["tool"] for call in checkpoint["tool_calls"]] == [
        "runtime_validate_generated_tool_source"
    ] * 3
    assert all(call["arguments"]["finish_reason"] == "length"
               for call in checkpoint["tool_calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_failures", [0, 3])
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
async def test_real_creation_failures_stop_agent_before_fourth_generation(tmp_path, prior_failures):
    real = MCPTools(SimpleNamespace(filesystem_server=["/usr/bin/false"],
        browser_server=["/usr/bin/false"], workspace=tmp_path / "workspace",
        learning_root=tmp_path / "learning", learning_profile="client", evm_profile="client"))
    prompt = "Create a tool named count_fixture to count lines in a local file."
    real.begin_interaction("bounded-creation", prompt)

    class Tools:
        def __init__(self):
            self.calls = []
        def capture_tool_failure(self, **kwargs):
            return real.capture_tool_failure(**kwargs)
        async def openai_tool_definitions(self):
            return [item for item in real._local_tool_definitions()
                    if item["function"]["name"] == "learning_create_tool"]
        async def call(self, tool, arguments):
            self.calls.append(tool)
            return await real._call_direct(tool, arguments)

    class Model:
        config = SimpleNamespace(context=12_000)
        turns = 0
        async def respond(self, **kwargs):
            self.turns += 1
            assert self.turns <= 3, "must stop without a fourth generation"
            # Each source differs but all falsely report a line count without reading.
            return LLMResponse(content=f'def run(arguments):\n    return {{"lines": {self.turns}}}')

    class Memory:
        session = Session()
        async def process(self, *args, **kwargs):
            pass

    instance = object.__new__(Agent)
    instance.tools, instance.llm, instance.memory = Tools(), Model(), Memory()
    instance.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    instance._build_system_prompt = lambda prompt, agent_mode: "system"
    instance._agent_trace_root = tmp_path / "traces"
    if prior_failures:
        trace = instance._start_agent_trace(prompt)
        for n in range(prior_failures):
            seq = trace.tool_started("learning_create_tool", {"source": str(n)})
            trace.tool_finished(seq, "validation failed", error="ArtifactValidationError: failed")
        instance._start_agent_trace = lambda prompt: trace
    answer = await instance._run_agent_loop(prompt)
    await asyncio.gather(*instance._memory_tasks)
    assert "3 creation attempts failed" in answer
    assert instance.llm.turns == 3 - prior_failures
    assert instance.tools.calls == ["learning_create_tool"] * (3 - prior_failures)
    assert real.learning.active_tool_names() == []
    assert len(real.learning.store.list_evidence()) == 3 - prior_failures
    checkpoint = json.loads(next((tmp_path / "traces" / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["status"] == "failed"
    assert len(checkpoint["tool_calls"]) == 3
    journal = next((tmp_path / "traces" / "journal").glob("*.jsonl")).read_text()
    assert '"event": "creation_failure_budget_exhausted"' in journal
