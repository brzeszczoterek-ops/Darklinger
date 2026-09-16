from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.autonomy.task_contract import TaskContract
from v_core.learning import (
    ArtifactValidationError,
    ArtifactPolicyError,
    ArtifactScope,
    LearningRuntime,
    ToolManifest,
    ToolTestCase,
)
from v_core.mcp_tools import MCPTools
from v_core.llm.llm import LLMResponse, LLMToolCall
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.sandbox import BubblewrapBackend
from v_core.tool_recovery import (
    ToolRecoveryRegistry,
    execute_with_recovery,
)


def _runtime(tmp_path: Path) -> LearningRuntime:
    workspace = tmp_path / "workspace"
    envelope = AuthorizationEnvelope(workspace=str(workspace))
    return LearningRuntime(
        tmp_path / "learning",
        AuthorizationGuard(tmp_path, envelope),
        BubblewrapBackend(),
    )


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
async def test_mcp_creation_failure_repair_and_retest_cycle(tmp_path: Path) -> None:
    tools = MCPTools(
        SimpleNamespace(
            filesystem_server=["/usr/bin/false"],
            browser_server=["/usr/bin/false"],
            workspace=tmp_path / "workspace",
            learning_root=tmp_path / "learning",
            autonomy_root=tmp_path / "autonomy",
            learning_profile="client",
            evm_profile="client",
        )
    )
    assert tools.learning is not None
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    original = await tools.learning.create_tool(
        ToolManifest(
            name="safe_divide_ten",
            version="1.0.0",
            description="Divide ten by a non-zero fixture value.",
            input_schema=schema,
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "integer"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            tests=(
                ToolTestCase(
                    name="known non-zero value",
                    arguments={"value": 2},
                    expected={"result": 5},
                ),
            ),
        ),
        'def run(arguments):\n    return {"result": 10 // arguments["value"]}\n',
    )
    assert original.status.value == "active"
    tools._register_recovery_providers()

    failed = await tools.call_with_recovery("safe_divide_ten", {"value": 0})

    assert failed.error
    assert failed.failure_details["stage"] == "provider_execution"
    assert failed.failure_details["execution_attempted"] is True
    assert len(failed.attempts) == 1
    ticket_id = failed.recovery_ticket["ticket_id"]
    assert tools.recovery.ticket(ticket_id).state == "open"

    repair_source = (
        'def run(arguments):\n'
        '    value = arguments["value"]\n'
        '    return {"result": 0 if value == 0 else 10 // value}\n'
    )
    repair_result = json.loads(
        await tools._call_direct(
            "learning_create_repair_adapter",
            {
                "ticket_id": ticket_id,
                "name": "safe_divide_ten_repair",
                "description": "Handle the captured zero-value failure.",
                "source": repair_source,
                "expected": {"result": 0},
            },
        )
    )

    assert repair_result["artifact"]["status"] == "active"
    assert repair_result["recovery"]["state"] == "active"
    repaired = await tools.call_with_recovery("safe_divide_ten", {"value": 0})
    regression = await tools.call_with_recovery("safe_divide_ten", {"value": 2})

    assert repaired.error == ""
    assert repaired.provider_tool == "safe_divide_ten_repair"
    assert json.loads(repaired.result) == {"result": 0}
    assert regression.error == ""
    assert regression.provider_tool == "safe_divide_ten_repair"
    assert json.loads(regression.result) == {"result": 5}


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
async def test_agent_completes_failure_repair_and_retest_cycle(tmp_path: Path) -> None:
    tools = MCPTools(
        SimpleNamespace(
            filesystem_server=["/usr/bin/false"],
            browser_server=["/usr/bin/false"],
            workspace=tmp_path / "workspace",
            learning_root=tmp_path / "learning",
            autonomy_root=tmp_path / "autonomy",
            learning_profile="client",
            evm_profile="client",
        )
    )
    assert tools.learning is not None
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    await tools.learning.create_tool(
        ToolManifest(
            name="safe_divide_ten",
            version="1.0.0",
            description="Divide ten by a non-zero fixture value.",
            input_schema=schema,
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "integer"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            tests=(
                ToolTestCase(
                    name="known non-zero value",
                    arguments={"value": 2},
                    expected={"result": 5},
                ),
            ),
        ),
        'def run(arguments):\n    return {"result": 10 // arguments["value"]}\n',
    )
    tools._register_recovery_providers()
    repair_source = (
        'def run(arguments):\n'
        '    value = arguments["value"]\n'
        '    return {"result": 0 if value == 0 else 10 // value}\n'
    )

    class Model:
        config = SimpleNamespace(context=12_000, model="scripted-local-model")

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            names = {
                item["function"]["name"]
                for item in (kwargs.get("tools") or [])
            }
            if self.turn == 1:
                assert "safe_divide_ten" in names
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("initial", "safe_divide_ten", {"value": 0})
                    ]
                )
            if self.turn == 2:
                assert "learning_create_repair_adapter" in names
                ticket_id = next(
                    item["ticket_id"]
                    for item in tools.recovery.list_tickets(state="open")
                    if item["capability"] == "generated.safe_divide_ten"
                )
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "bad_repair",
                            "learning_create_repair_adapter",
                            {
                                "ticket_id": ticket_id,
                                "name": "safe_divide_ten_repair",
                                "description": "Handle the captured zero-value failure.",
                                "source": (
                                    'def run(arguments):\n'
                                    '    value = arguments["value"]\n'
                                    '    return {"result": value - value + 1}\n'
                                ),
                                "expected": {"result": 0},
                            },
                        )
                    ]
                )
            if self.turn == 3:
                assert names == {"learning_create_repair_adapter"}
                ticket_id = next(
                    item["ticket_id"]
                    for item in tools.recovery.list_tickets(state="open")
                    if item["capability"] == "generated.safe_divide_ten"
                )
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "good_repair",
                            "learning_create_repair_adapter",
                            {
                                "ticket_id": ticket_id,
                                "name": "safe_divide_ten_repair",
                                "description": "Handle the captured zero-value failure.",
                                "source": repair_source,
                                "expected": {"result": 0},
                            },
                        )
                    ]
                )
            if self.turn == 4:
                # The objective keeps its original capability name. Recovery
                # selects the activated replacement behind that stable name.
                assert names == {"safe_divide_ten"}
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("retest", "safe_divide_ten", {"value": 0})
                    ]
                )
            return LLMResponse(
                content=(
                    "The original provider failed on the zero fixture. The repair "
                    "passed its replay and regression checks; the verified result is 0."
                )
            )

    class Memory:
        session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.tools = tools
    agent.llm = Model()
    agent.memory = Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"

    answer = await agent._run_agent_loop(
        "Run safe_divide_ten with value 0. If it fails, repair it, retest the "
        "same input, and report the verified result."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "verified result is 0" in answer
    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text()
    )
    calls = checkpoint["tool_calls"]
    assert [call["tool"] for call in calls] == [
        "safe_divide_ten",
        "learning_create_repair_adapter",
        "learning_create_repair_adapter",
        "safe_divide_ten",
    ]
    assert calls[0]["status"] == "failed"
    assert calls[0]["failure_details"]["execution_attempted"] is True
    assert calls[1]["status"] == "failed"
    assert calls[1]["failure_details"]["validation"]["stage"] == "comparison"
    assert calls[2]["status"] == "succeeded"
    assert calls[3]["status"] == "succeeded"
    assert calls[3]["provider_tool"] == "safe_divide_ten_repair"


@pytest.mark.asyncio
async def test_provider_failure_falls_back_and_resolves_ticket(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("primary_reader", ("filesystem.read",))
    registry.register_provider("backup_reader", ("filesystem.read",))
    calls: list[str] = []

    async def call(tool: str, arguments: dict) -> str:
        calls.append(tool)
        if tool == "primary_reader":
            raise RuntimeError("provider is down")
        return "verified content"

    outcome = await execute_with_recovery(
        registry,
        requested_tool="primary_reader",
        arguments={"path": "README.md"},
        call_provider=call,
        detect_failure=lambda _result, _tool: "",
        task_id="interactive-test",
    )

    assert outcome.result == "verified content"
    assert outcome.provider_tool == "backup_reader"
    assert outcome.capabilities == ("filesystem.read",)
    assert calls == ["primary_reader", "backup_reader"]
    assert registry.list_tickets()[0]["state"] == "resolved"


@pytest.mark.asyncio
async def test_repeated_failures_open_circuit_and_persist(tmp_path: Path) -> None:
    root = tmp_path / "recovery"
    registry = ToolRecoveryRegistry(root)
    registry.register_provider("broken_transform", ("generated.transform",))

    async def fail(_tool: str, _arguments: dict) -> str:
        return "Tool execution failed: deterministic regression"

    for _ in range(2):
        outcome = await execute_with_recovery(
            registry,
            requested_tool="broken_transform",
            arguments={"value": 3},
            call_provider=fail,
            detect_failure=lambda result, _tool: result,
        )
        assert outcome.error

    restarted = ToolRecoveryRegistry(root)
    state = next(
        item for item in restarted.provider_state() if item["tool"] == "broken_transform"
    )
    assert state["circuit_open"] is True
    assert restarted.providers_for("broken_transform") == []


@pytest.mark.asyncio
async def test_open_circuit_reports_skipped_execution_and_historical_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "recovery"
    registry = ToolRecoveryRegistry(root)
    registry.register_provider("broken_transform", ("generated.transform",))
    for _ in range(2):
        registry.record_failure(
            tool="broken_transform",
            requested_tool="broken_transform",
            arguments={"value": 3},
            error="RuntimeError: sandbox service unavailable",
        )
    calls: list[str] = []

    async def should_not_run(tool: str, _arguments: dict) -> str:
        calls.append(tool)
        return "unexpected"

    outcome = await execute_with_recovery(
        ToolRecoveryRegistry(root),
        requested_tool="broken_transform",
        arguments={"value": 3},
        call_provider=should_not_run,
        detect_failure=lambda _result, _tool: "",
    )

    assert calls == []
    assert outcome.attempts == ()
    assert outcome.exception is None
    assert outcome.failure_details["stage"] == "dispatch"
    assert outcome.failure_details["execution_attempted"] is False
    assert outcome.failure_details["reason"] == "matching_providers_circuit_open"
    assert "no provider was executed" in outcome.error
    assert "historical previous failure" in outcome.error
    assert "sandbox service unavailable" in outcome.error
    assert "all capability providers failed" not in outcome.error
    assert outcome.result.startswith("Tool execution not started:")


@pytest.mark.asyncio
async def test_provider_exception_preserves_validation_stage(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("validator", ("generated.validate",))

    async def reject(_tool: str, _arguments: dict) -> str:
        raise ArtifactValidationError(
            "fixture comparison failed",
            report={
                "stage": "comparison",
                "tests_total": 2,
                "tests_passed": 1,
                "tests_failed": 1,
                "tests_not_run": 0,
                "semantic_correctness": "not_independently_established",
                "source": "must not be copied into diagnostics",
            },
        )

    outcome = await execute_with_recovery(
        registry,
        requested_tool="validator",
        arguments={"value": 3},
        call_provider=reject,
        detect_failure=lambda _result, _tool: "",
    )

    assert outcome.failure_details["stage"] == "provider_execution"
    assert outcome.failure_details["execution_attempted"] is True
    assert outcome.failure_details["exception_type"] == "ArtifactValidationError"
    assert outcome.failure_details["validation"] == {
        "stage": "comparison",
        "tests_total": 2,
        "tests_passed": 1,
        "tests_failed": 1,
        "tests_not_run": 0,
        "semantic_correctness": "not_independently_established",
    }


@pytest.mark.asyncio
async def test_last_exception_matches_last_failed_provider(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("first", ("generated.shared",), priority=200)
    registry.register_provider("second", ("generated.shared",), priority=100)

    async def fail(tool: str, _arguments: dict) -> str:
        if tool == "first":
            raise RuntimeError("first provider exception")
        return "Tool execution failed: second provider returned an error"

    outcome = await execute_with_recovery(
        registry,
        requested_tool="first",
        arguments={"value": 3},
        call_provider=fail,
        detect_failure=lambda result, _tool: (
            result if result.startswith("Tool execution failed:") else ""
        ),
    )

    assert [attempt["provider"] for attempt in outcome.attempts] == ["first", "second"]
    assert outcome.exception is None
    assert outcome.provider_tool == "second"
    assert "second provider returned an error" in outcome.error


@pytest.mark.asyncio
async def test_invalid_call_does_not_open_provider_circuit(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("exact_fetch", ("network.tor.fetch",))

    async def reject(_tool: str, _arguments: dict) -> str:
        raise ValueError("the target must be a valid Tor v3 .onion address")

    for _ in range(3):
        outcome = await execute_with_recovery(
            registry,
            requested_tool="exact_fetch",
            arguments={"url": "broken"},
            call_provider=reject,
            detect_failure=lambda _result, _tool: "",
        )
        assert outcome.error

    state = next(
        item for item in registry.provider_state() if item["tool"] == "exact_fetch"
    )
    assert state["circuit_open"] is False
    assert state["consecutive_failures"] == 0
    assert len(registry.providers_for("exact_fetch")) == 1


def test_legacy_argument_error_circuit_is_healed_on_next_lookup(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("exact_fetch", ("network.tor.fetch",))
    for _ in range(2):
        registry.record_failure(
            tool="exact_fetch",
            requested_tool="exact_fetch",
            arguments={"url": "broken"},
            error="HostBridgeError: the target must be a valid Tor v3 .onion address.",
        )

    assert registry.provider_state()[0]["circuit_open"] is True
    assert len(registry.providers_for("exact_fetch")) == 1
    assert registry.provider_state()[0]["circuit_open"] is False


def test_failed_repair_is_rolled_back_to_previous_provider(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("original_transform", ("generated.transform",))
    ticket = registry.record_failure(
        tool="original_transform",
        requested_tool="original_transform",
        arguments={"value": 4},
        error="original regression",
    )
    registry.register_provider(
        "repair_transform",
        ("generated.transform",),
        priority=200,
        kind="generated_repair",
    )
    registry.activate_repair(ticket.ticket_id, "repair_transform")

    for _ in range(2):
        registry.record_failure(
            tool="repair_transform",
            requested_tool="original_transform",
            arguments={"value": 4},
            error="repair regression",
        )

    assert registry.ticket(ticket.ticket_id).state == "rolled_back"
    assert [
        provider.tool for provider in registry.providers_for("original_transform")
    ] == ["original_transform"]


def test_recovery_ticket_redacts_secrets(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("private_lookup", ("generated.private_lookup",))
    ticket = registry.record_failure(
        tool="private_lookup",
        requested_tool="private_lookup",
        arguments={"query": "safe", "api_token": "never-store-this"},
        error="failed",
    )

    assert ticket.arguments == {"query": "safe", "api_token": "<redacted>"}
    assert ticket.fixture_redacted is True


@pytest.mark.asyncio
async def test_generated_repair_replays_runtime_fixture_before_activation(
    tmp_path: Path,
) -> None:
    learning = _runtime(tmp_path)
    await learning.create_tool(
        ToolManifest(
            name="broken_double",
            version="1.0.0",
            description="Double one integer.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "integer"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            tests=(
                ToolTestCase(
                    name="established regression",
                    arguments={"value": 2},
                    expected={"result": 4},
                ),
            ),
        ),
        'def run(arguments):\n    return {"result": arguments["value"] * 2}',
    )
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("broken_double", ("generated.broken_double",))
    ticket = registry.record_failure(
        tool="broken_double",
        requested_tool="broken_double",
        arguments={"value": 7},
        error="wrong result",
        task_id="interactive-repair",
    )

    record = await learning.create_repair_tool(
        ticket=ticket,
        name="repaired_double",
        description="Repair integer doubling from the captured failure fixture.",
        source='def run(arguments):\n    return {"result": arguments["value"] * 2}',
        expected={"result": 14},
    )

    assert record.status.value == "active"
    assert await learning.execute_tool("repaired_double", {"value": 9}) == {
        "result": 18
    }
    manifest = next(
        item for item in learning.active_tool_manifests() if item.name == "repaired_double"
    )
    assert manifest.provides_capabilities == ("generated.broken_double",)
    assert manifest.repair_ticket_id == ticket.ticket_id


def test_generated_tool_cannot_claim_network_or_host_capability(tmp_path: Path) -> None:
    learning = _runtime(tmp_path)
    manifest = ToolManifest(
        name="fake_tor_repair",
        version="1.0.0",
        description="Pretend to replace network transport.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"results": {"type": "array"}},
            "required": ["results"],
            "additionalProperties": False,
        },
        tests=(
            ToolTestCase(
                name="fake network result",
                arguments={"query": "test"},
                expected={"results": []},
            ),
        ),
        scope=ArtifactScope.TASK,
        provides_capabilities=("network.tor.search",),
    )

    with pytest.raises(ArtifactPolicyError, match="trusted provider"):
        learning.stage_tool(
            manifest,
            'def run(arguments):\n    return {"results": []}',
        )


def test_task_contract_accepts_equivalent_capability_provider() -> None:
    contract = TaskContract(required_tools=("read_file",))
    calls = [
        {
            "tool": "repair_read_file",
            "provider_tool": "repair_read_file",
            "capabilities": ["filesystem.read"],
            "status": "succeeded",
            "result_excerpt": "content",
        }
    ]

    assert contract.unmet(calls) == []
