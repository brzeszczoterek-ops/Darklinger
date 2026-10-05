"""Regressions for private review: target identity, repair oracle, provenance."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.autonomy.task_contract import TaskContract
from v_core.learning import ArtifactValidationError, LearningRuntime, ToolManifest, ToolTestCase
from v_core.mcp_tools import MCPTools, FilesystemScopeDenied
from v_core.memory.reflection import Reflection
from v_core.memory.experience import Experience
from v_core.memory.summary import Summary
from v_core.memory.knowledge import Knowledge
from v_core.memory.models import ReflectionEntry, ExperienceEntry, MemorySource
from v_core.memory.provenance import ground_reflection, ground_derived
from v_core.memory.storage import MemoryStorage
from v_core.sandbox import BubblewrapBackend
from v_core.tool_recovery import ToolRecoveryRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,field", [("read_file", "path"), ("write_file", "path"),
    ("edit_file", "path"), ("move_file", "source"), ("move_file", "destination"),
    ("delete_file", "path")])
async def test_foreign_target_never_dispatches_or_touches_same_basename(tmp_path, tool, field):
    tools = MCPTools.__new__(MCPTools)
    tools.workspace = tmp_path / "workspace"
    tools.workspace.mkdir()
    sentinel = tools.workspace / "report.md"
    sentinel.write_text("original")
    calls = []
    async def dispatch(*args):
        calls.append(args)
        raise AssertionError("foreign target must never execute")
    tools._call_direct = dispatch
    requested = str(tmp_path / "foreign" / "report.md")
    outcome = await tools.call_with_recovery(tool, {
        "path": str(sentinel), "source": str(sentinel), "destination": str(sentinel),
        field: requested, "content": "replacement",
    })
    assert outcome.error and isinstance(outcome.exception, FilesystemScopeDenied)
    assert outcome.failure_details["requested_path"] == requested
    assert outcome.failure_details["execution_attempted"] is False
    assert calls == [] and sentinel.read_text() == "original"


def test_symlink_escape_and_traversal_are_denied(tmp_path):
    tools = MCPTools.__new__(MCPTools)
    tools.workspace = tmp_path / "workspace"
    tools.workspace.mkdir()
    (tools.workspace / "escape").symlink_to(tmp_path, target_is_directory=True)
    for path in ("../report.md", "escape/report.md"):
        with pytest.raises(FilesystemScopeDenied):
            tools.normalize_arguments("read_file", {"path": path})


def test_wrong_write_cannot_complete_exact_target_contract():
    contract = TaskContract(requires_file_mutation=True, required_mutation_paths=("/workspace/right.md",))
    restored = TaskContract.from_dict(contract.to_dict()).merged(TaskContract())
    wrong = [{"tool": "write_file", "status": "succeeded", "arguments": {"path": "/workspace/wrong.md"}}]
    assert "filesystem_mutation:/workspace/right.md" in restored.unmet(wrong)
    wrong[0]["arguments"]["path"] = "/workspace/right.md"
    assert restored.unmet(wrong) == []


async def _repair_fixture(tmp_path):
    learning = LearningRuntime(tmp_path / "learning", AuthorizationGuard(tmp_path,
        AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))), BubblewrapBackend())
    schema = lambda key: {"type": "object", "properties": {key: {"type": "integer"}},
                          "required": [key], "additionalProperties": False}
    await learning.create_tool(ToolManifest(name="double", version="1.0.0", description="Double an integer.",
        input_schema=schema("value"), output_schema=schema("result"),
        tests=(ToolTestCase(name="known regression", arguments={"value": 2}, expected={"result": 4}),)),
        'def run(arguments):\n    return {"result": arguments["value"] * 2}')
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    ticket = registry.record_failure(tool="double", requested_tool="double", arguments={"value": 7}, error="wrong result")
    return learning, ticket


@pytest.mark.asyncio
async def test_self_authored_wrong_repair_stays_non_executable(tmp_path):
    learning, ticket = await _repair_fixture(tmp_path)
    record = await learning.create_repair_tool(ticket=ticket, name="wrong_repair", description="Candidate repair.",
        source='def run(arguments):\n    value = arguments["value"]\n    return {"result": 17 if value == 7 else value * 2}',
        expected={"result": 17})
    assert record.status.value == "validated"
    assert record.validation["activation_eligible"] is False
    with pytest.raises(ArtifactValidationError):
        learning.activate_artifact(record.artifact_id)
    assert "wrong_repair" not in [m.name for m in learning.active_tool_manifests()]
    assert ticket.state == "open"


@pytest.mark.asyncio
async def test_owner_oracle_rejects_wrong_repair_despite_model_expected(tmp_path):
    learning, ticket = await _repair_fixture(tmp_path)
    with pytest.raises(ArtifactValidationError):
        await learning.create_repair_tool(ticket=ticket, name="wrong_repair", description="Candidate repair.",
            source='def run(arguments):\n    value = arguments["value"]\n    return {"result": 17 if value == 7 else value * 2}',
            expected={"result": 17}, owner_objective='arguments = {"value": 7} expected = {"result": 14}')


@pytest.mark.asyncio
async def test_existing_contract_oracle_qualifies_same_captured_input(tmp_path):
    learning, ticket = await _repair_fixture(tmp_path)
    ticket = replace(ticket, arguments={"value": 2})
    record = await learning.create_repair_tool(ticket=ticket, name="correct_repair", description="Candidate repair.",
        source='def run(arguments):\n    return {"result": arguments["value"] * 2}', expected={"result": 999})
    assert record.status.value == "active"
    manifest, _ = learning.store.load_tool(record)
    assert manifest.repair_oracle_source == "existing_contract"
    assert manifest.tests[-1].expected == {"result": 4}
    legacy = replace(manifest, repair_oracle_source="", repair_oracle_sha256="")
    assert not learning._tool_has_current_functional_contract(legacy)


@pytest.mark.parametrize("field,value", [("repair_ticket_id", "a" * 32),
    ("repair_oracle_source", "owner_fixture"), ("repair_oracle_sha256", "b" * 64),
    ("provides_capabilities", ["generated.other"])])
def test_model_manifest_cannot_forge_repair_authority(field, value):
    payload = ToolManifest(name="candidate", version="1.0.0", description="Candidate tool.",
        input_schema={}, output_schema={}, tests=(ToolTestCase(name="test", arguments={}, expected={}),)).to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        MCPTools._model_tool_manifest(payload)


class Model:
    def __init__(self, payload): self.payload = payload
    async def ask(self, *args, **kwargs): return json.dumps(self.payload)


@pytest.mark.asyncio
async def test_unrelated_success_cannot_verify_memory_at_any_stage():
    payload = {"summary": "The moon is made of cheese.", "lesson": "", "content": "The moon is made of cheese.",
               "lessons": [], "remember": True, "source": "verified", "kind": "fact", "confidence": 1}
    llm = Model(payload)
    reflection = await Reflection(llm).reflect("Inspect a file.", "Done.", execution={
        "successful_tool_count": 1, "tool_calls": [{"tool": "read_file", "status": "succeeded", "result_excerpt": "hello"}]})
    experience = await Experience(llm).learn(reflection, [], [])
    summary = await Summary(llm).summarize([experience], [])
    knowledge = await Knowledge(llm).update(summary, [])
    for entry in (reflection, experience, summary, knowledge):
        assert entry.source is MemorySource.SELF_GENERATED and entry.evidence_refs == []


@pytest.mark.asyncio
async def test_claim_specific_provenance_survives_storage_without_promotion(tmp_path):
    text = "My preferred language is Polish."
    llm = Model({"summary": text, "content": text, "lesson": "", "lessons": [], "remember": True,
                 "source": "verified", "kind": "fact", "confidence": 1})
    reflection = await Reflection(llm).reflect(text, "Acknowledged.")
    assert reflection.source is MemorySource.DIRECTLY_TOLD
    experience = await Experience(llm).learn(reflection, [], [])
    storage = MemoryStorage(tmp_path)
    loaded = storage.load(storage.save("experiences", "entry.yaml", experience))
    summary = await Summary(llm).summarize([loaded], [])
    knowledge = await Knowledge(llm).update(summary, [])
    assert knowledge.source is MemorySource.DIRECTLY_TOLD
    assert knowledge.evidence_refs[0]["reference"] == "owner_task"
    changed = ExperienceEntry(summary="A different unsupported claim.", source=MemorySource.VERIFIED)
    ground_derived(changed, [loaded])
    assert changed.source is MemorySource.SELF_GENERATED
    legacy = {"summary": text, "source": "verified"}
    ground_derived(changed, [legacy])
    assert changed.evidence_refs == []


def test_tool_result_is_observed_and_failed_call_is_not_evidence():
    entry = ReflectionEntry(summary="hello", source=MemorySource.VERIFIED)
    ground_reflection(entry, "Read a file.", {"tool_calls": [
        {"tool": "read_file", "status": "succeeded", "result_excerpt": "hello"}]})
    assert entry.source is MemorySource.OBSERVED
    ground_reflection(entry, "Read a file.", {"tool_calls": [
        {"tool": "read_file", "status": "failed", "result_excerpt": "hello"}]})
    assert entry.source is MemorySource.SELF_GENERATED


@pytest.mark.asyncio
async def test_mcp_untrusted_repair_leaves_ticket_open_and_provider_unchanged(tmp_path):
    learning, ticket = await _repair_fixture(tmp_path)
    tools = MCPTools(SimpleNamespace(filesystem_server=["/usr/bin/false"], browser_server=["/usr/bin/false"],
        workspace=tmp_path / "workspace", learning_root=tmp_path / "tools-learning",
        autonomy_root=tmp_path / "autonomy", learning_profile="client", evm_profile="client"))
    tools.learning = learning
    tools.recovery = ToolRecoveryRegistry(tmp_path / "recovery")
    tools.recovery.register_provider("double", ("generated.double",))
    tools.interaction_prompt = "Repair the tool."
    result = json.loads(await tools._call_direct("learning_create_repair_adapter", {
        "ticket_id": ticket.ticket_id, "name": "wrong_repair", "description": "Candidate repair.",
        "source": 'def run(arguments):\n    value = arguments["value"]\n    return {"result": 17 if value == 7 else value * 2}', "expected": {"result": 17}}))
    assert result["needs_validation"] is True
    assert result["artifact"]["status"] == "validated"
    assert tools.recovery.ticket(ticket.ticket_id).state == "open"
    assert "wrong_repair" not in json.dumps(tools.recovery.provider_state())


@pytest.mark.asyncio
async def test_memory_engine_clamps_custom_stage_before_persistence(tmp_path):
    from v_core.memory.memory_engine import MemoryEngine
    from v_core.memory.manager import MemoryManager
    class ForgedReflection:
        async def reflect(self, *args, **kwargs):
            return ReflectionEntry(summary="Unsupported fact.", remember=True, source=MemorySource.VERIFIED)
    class ForgedExperience:
        async def learn(self, *args, **kwargs):
            return ExperienceEntry(summary="Unsupported fact.", confidence=1, importance="high", source=MemorySource.VERIFIED)
    engine = MemoryEngine.__new__(MemoryEngine)
    engine.reflection, engine.experience = ForgedReflection(), ForgedExperience()
    engine.manager = MemoryManager(MemoryStorage(tmp_path))
    engine.proposal_filter = None
    result = await engine.process("Read a file.", "Done.", execution={"status": "completed",
        "successful_tool_count": 1, "tool_calls": [{"tool": "read_file", "status": "succeeded", "result_excerpt": "hello"}]})
    assert result.source is MemorySource.SELF_GENERATED
    assert engine.manager.load_all("experiences") == []
    assert engine.manager.load_all("knowledge") == []
    assert engine.manager.load_all("proposals")[0]["source"] == "self_generated"
