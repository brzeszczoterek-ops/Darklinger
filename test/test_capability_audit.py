from __future__ import annotations

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.learning import (
    ArtifactScope,
    CapabilityRisk,
    SkillManifest,
    SkillTestCase,
    ToolManifest,
    ToolTestCase,
    LearningRuntime,
    assess_skill,
    assess_tool,
)
from v_core.sandbox import BubblewrapBackend


def manifest(name: str = "count_values") -> ToolManifest:
    return ToolManifest(
        name=name,
        version="1.0.0",
        description="Count supplied values.",
        input_schema={
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "integer"}}
            },
            "required": ["values"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        },
        tests=(
            ToolTestCase(
                name="counts two values",
                arguments={"values": [1, 2]},
                expected={"count": 2},
            ),
        ),
        scope=ArtifactScope.TASK,
    )


def test_pure_transform_is_routine_and_does_not_notify_owner() -> None:
    assessment = assess_tool(
        manifest(),
        "def run(arguments):\n    return {'count': len(arguments['values'])}",
        privileged=False,
        creation_origin="agent_initiated",
    )

    assert assessment.risk is CapabilityRisk.ROUTINE
    assert assessment.owner_notification is False
    assert assessment.primitives == ("generated.count_values",)


def test_privileged_process_tool_is_critical_and_records_origin() -> None:
    assessment = assess_tool(
        manifest("local_process_probe"),
        "import subprocess\ndef run(arguments):\n    return {'count': subprocess.run(['true']).returncode}",
        privileged=True,
        creation_origin="agent_initiated",
    )

    assert assessment.risk is CapabilityRisk.CRITICAL
    assert assessment.owner_notification is True
    assert assessment.creation_origin == "agent_initiated"
    assert "process_control" in assessment.primitives
    assert "agent_initiated_sensitive_capability" in assessment.reasons


def test_file_write_is_review_without_treating_description_as_policy() -> None:
    assessment = assess_tool(
        manifest("write_local_note"),
        "def run(arguments):\n    open('note.txt', 'w').write('x')\n    return {'count': 1}",
        privileged=True,
        creation_origin="owner_requested",
    )

    assert assessment.risk is CapabilityRisk.REVIEW
    assert assessment.owner_notification is True
    assert assessment.reasons == (
        "filesystem_write_primitive",
        "owner_privileged_generated_code",
    )


def test_skill_orchestrating_sensitive_tool_enters_review_ledger() -> None:
    skill = SkillManifest(
        name="publish_report",
        version="1.0.0",
        description="Write a verified report.",
        triggers=("publish report",),
        steps=("Write the report.",),
        required_tools=("write_file",),
        tests=(
            SkillTestCase("publish report", True),
            SkillTestCase("read report", False),
        ),
    )

    assessment = assess_skill(skill, creation_origin="owner_requested")

    assert assessment.risk is CapabilityRisk.REVIEW
    assert assessment.primitives == ("write_file",)
    assert assessment.owner_notification is True


def test_staged_artifact_writes_runtime_owned_capability_audit(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = LearningRuntime(
        tmp_path / "learning",
        AuthorizationGuard(
            tmp_path,
            AuthorizationEnvelope(workspace=str(workspace)),
        ),
        BubblewrapBackend(),
    )
    runtime.set_creation_context(
        origin="agent_initiated",
        task_id="interactive-audit-test",
    )

    record = runtime.stage_tool(
        manifest(),
        "def run(arguments):\n    return {'count': len(arguments['values'])}",
    )
    audits = runtime.capability_audits()

    assert audits == [
        {
            **audits[0],
            "artifact_id": record.artifact_id,
            "artifact_digest": record.digest,
            "task_id": "interactive-audit-test",
            "creation_origin": "agent_initiated",
            "risk": "routine",
            "owner_notification": False,
        }
    ]
