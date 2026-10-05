import asyncio
from datetime import datetime, timedelta, timezone
import json
import shutil
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.creation_budget import evaluate, source_candidate, validation_receipt, TOTAL_ATTEMPT_LIMIT
from v_core.learning.runtime import ArtifactValidationError
from v_core.llm.llm import LLMResponse
from v_core.memory.session import Session
from v_core.mcp_tools import MCPTools
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile


OBJECTIVE = 'Create a tool named double_fixture. value = 2\nexpected = {"result": 4}'


def source(expression):
    return f'def run(arguments):\n    return {{"result": {expression}}}'


def candidate(multiplier=3, objective=OBJECTIVE):
    return source_candidate(source(f'arguments["value"] * {multiplier}'), objective)


def call(rank=2, multiplier=3, model="A", probe="ordinary_attempt", objective=OBJECTIVE):
    info = candidate(multiplier, objective)
    return {"tool": "learning_create_tool", "status": "failed",
            "creation_model": model, "creation_probe": probe,
            "creation_candidate": info,
            "creation_validation": {**info, "rank": rank, "origin": "local_executor"}}


def test_verified_stage_progress_resets_stagnation_not_total_or_history():
    calls = [call(), call(multiplier=4), call(rank=3, multiplier=5)]
    state = evaluate(json.loads(json.dumps(calls)))
    assert state.stagnant == 0
    assert state.progress_resets == 1
    assert state.failures == state.attempts == 3
    assert len(state.approaches) == 3
    assert state.admit(candidate(6), "A") == (True, "ordinary_attempt")


def test_regression_then_recovery_to_previous_highwater_is_not_new_progress():
    state = evaluate([call(rank=3), call(rank=2), call(rank=3)])
    assert state.stagnant == 3
    assert state.progress_resets == 0


def test_changed_expected_output_or_fixture_cannot_reset_counter():
    calls = [call(), call(), call(rank=4, objective=OBJECTIVE.replace('"result": 4', '"result": 6'))]
    state = evaluate(calls)
    assert state.stagnant == 3
    assert state.progress_resets == 0
    assert state.admit(candidate(objective=OBJECTIVE.replace("value = 2", "value = 3")), "B") == (
        False, "no_comparable_contract")


def test_code_probe_is_one_shot_and_does_not_reset_stagnation():
    calls = [call(multiplier=n) for n in (3, 4, 5)]
    assert evaluate(calls).admit(candidate(6), "A") == (True, "changed_code_probe")
    calls.append(call(multiplier=6, probe="changed_code_probe"))
    state = evaluate(calls)
    assert state.stagnant == 4
    assert not state.admit(candidate(7), "A")[0]
    assert state.admit(candidate(6), "B") == (True, "changed_model_probe")


def test_model_probe_is_one_shot_and_switching_back_does_not_help():
    calls = [call() for _ in range(3)]
    assert evaluate(calls).admit(candidate(), "B") == (True, "changed_model_probe")
    calls.append(call(model="B", probe="changed_model_probe"))
    assert not evaluate(calls).admit(candidate(), "A")[0]
    assert not evaluate(calls).admit(candidate(), "C")[0]


def test_total_attempt_cap_includes_successes_and_never_resets():
    calls = [call(rank=2), call(rank=3)]
    calls.extend({**call(rank=4), "status": "succeeded"} for _ in range(TOTAL_ATTEMPT_LIMIT - 2))
    assert evaluate(calls).stop_reason == "attempt_budget"
    assert not evaluate(calls).admit(candidate(9), "new model")[0]


def test_elapsed_time_survives_serialization_without_becoming_stop_reason():
    entry = call()
    now = datetime.now(timezone.utc)
    entry["started_at"] = (now - timedelta(seconds=601)).isoformat()
    state = evaluate(json.loads(json.dumps([entry])), now=now)
    assert state.stop_reason == ""
    assert state.remaining_seconds == 0


@pytest.mark.asyncio
async def test_old_creation_receipt_does_not_cancel_inflight_local_operation():
    entry = call()
    entry["started_at"] = (datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat()
    async def slow():
        await asyncio.sleep(.02)
        return "completed"
    agent = object.__new__(Agent)
    assert await agent._await_creation_step(slow(), [entry]) == "completed"


def test_formatting_comments_and_diagnostic_text_are_not_new_approaches():
    original = source('arguments["value"] * 3')
    changed = '\n# new strategy\n' + original.replace('    return', '    """changed everything"""\n    return')
    assert source_candidate(original, OBJECTIVE)["approach_sha256"] == source_candidate(changed, OBJECTIVE)["approach_sha256"]


@pytest.mark.parametrize("code,objective", [
    (source('arguments.get("value", 2) * 3'), 'expected = {"result": 4}'),
    (source('arguments["value"] * 3'), 'value = 2'),
    ('import os\n' + source('arguments["value"] * 3'), OBJECTIVE),
])
def test_self_derived_fixtures_or_dependencies_are_not_retry_evidence(code, objective):
    assert source_candidate(code, objective) is None


def test_model_written_output_cannot_supply_progress_receipt():
    entries = [{"tool": "learning_create_tool", "status": "failed",
                "result_excerpt": json.dumps({"creation_validation": call()["creation_validation"]})} for _ in range(3)]
    assert evaluate(entries).stop_reason == "no_verified_progress"


def test_receipt_requires_actual_offline_report_with_unchanged_expected_digest():
    info = candidate()
    report = {"network": "offline", "sandbox": "bubblewrap",
              "tests": [{"stage": "comparison", "expected_sha256": info["expected_sha256"]}]}
    assert validation_receipt(info, report)["rank"] == 3
    assert validation_receipt(info, {**report, "network": "online"}) is None
    report["tests"][0]["expected_sha256"] = "changed expected answer"
    assert validation_receipt(info, report) is None


def tools(root):
    result = MCPTools(SimpleNamespace(filesystem_server=["/usr/bin/false"],
        browser_server=["/usr/bin/false"], workspace=root / "workspace",
        learning_root=root / "learning", learning_profile="client", evm_profile="client"))
    result.begin_interaction("bounded-progress", OBJECTIVE)
    return result


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
async def test_real_local_validation_advances_stage_without_changing_oracle(tmp_path):
    runtime = tools(tmp_path)
    calls = []
    for expression in ('arguments["value"] / 0', 'arguments["value"] // 0', 'arguments["value"] * 3'):
        arguments = {"source": source(expression)}
        with pytest.raises(ArtifactValidationError):
            await runtime._call_direct("learning_create_tool", arguments)
        receipt = runtime.take_creation_validation("learning_create_tool", arguments)
        assert receipt is not None
        assert runtime.take_creation_validation("learning_create_tool", arguments) is None
        calls.append({"tool": "learning_create_tool", "status": "failed", "creation_validation": receipt})
    state = evaluate(calls)
    assert state.attempts == 3
    assert state.stagnant == 0
    assert state.progress_resets == 1
    assert runtime.learning.active_tool_names() == []
    # Correct arithmetic is checked against the same user-supplied expectation.
    record = json.loads(await runtime._call_direct("learning_create_tool", {"source": source('arguments["value"] * 2')}))
    assert record["status"] == "active"
    runtime.begin_interaction("new-task", OBJECTIVE)
    assert runtime.take_creation_validation("learning_create_tool", {"source": source('arguments["value"] * 2')}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_probe", [False, True])
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
async def test_agent_allows_only_one_unverified_probe_with_real_local_handler(tmp_path, model_probe):
    runtime = tools(tmp_path)
    class Tools:
        def __init__(self):
            self.calls = []
        creation_candidate = runtime.creation_candidate
        take_creation_validation = runtime.take_creation_validation
        capture_tool_failure = runtime.capture_tool_failure
        async def openai_tool_definitions(self):
            return [item for item in runtime._local_tool_definitions() if item["function"]["name"] == "learning_create_tool"]
        async def call(self, name, arguments):
            self.calls.append(name)
            return await runtime._call_direct(name, arguments)
    class Model:
        config = SimpleNamespace(context=12_000, model="A")
        turns = 0
        async def respond(self, **kwargs):
            self.turns += 1
            assert self.turns <= 5
            if model_probe and self.turns >= 4:
                self.config.model = "B" if self.turns == 4 else "A"
            number = min(self.turns + 2, 5) if model_probe else self.turns + 2
            return LLMResponse(content=source(f'arguments["value"] * {number}'))
    class Memory:
        session = Session()
        async def process(self, *args, **kwargs):
            pass
    agent = object.__new__(Agent)
    agent.tools, agent.llm, agent.memory = Tools(), Model(), Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "traces"
    answer = await agent._run_agent_loop(OBJECTIVE)
    await asyncio.gather(*agent._memory_tasks)
    assert len(agent.tools.calls) == 4
    assert "4 creation attempts failed" in answer
    checkpoint = json.loads(next((tmp_path / "traces" / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["status"] == "failed"
    state = evaluate(checkpoint["tool_calls"])
    assert state.stagnant == 4
    assert state.model_probe_used if model_probe else state.code_probe_used
    assert runtime.learning.active_tool_names() == []
