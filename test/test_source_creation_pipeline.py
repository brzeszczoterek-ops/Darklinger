import json

import pytest

from v_core.agent import Agent
from v_core.generated_tool_contract import (
    GeneratedToolContractError, explicit_generated_tool_contract,
    extract_generated_tool_contract,
)
from v_core.learning import ArtifactValidationError
from v_core.tool_recovery import ToolRecoveryRegistry, execute_with_recovery


def contract_prompt():
    return 'Create a parser. tool_contract=' + json.dumps({
        'tests': [{'arguments': {'har_json': '{"escaped": "a\\nb"}'},
                   'expected': {'count': 1}}],
        'final_arguments': {'har_json': '{"other": 2}'},
    })


@pytest.mark.asyncio
async def test_explicit_contract_needs_no_model_and_preserves_escaped_data():
    contract = await extract_generated_tool_contract(object(), contract_prompt())
    assert contract.provenance == 'owner_structured_contract'
    assert contract.tests[0].arguments['har_json'] == '{"escaped": "a\\nb"}'
    exported = contract.to_dict()
    exported['tests'][0]['expected']['count'] = 99
    assert contract.tests[0].expected == {'count': 1}


@pytest.mark.parametrize('payload', [
    {'tests': [], 'final_arguments': {}},
    {'tests': [{'arguments': {'x': 1}, 'expected': {'y': 2}},
               {'arguments': {'x': 1}, 'expected': {'y': 3}}], 'final_arguments': {'x': 4}},
    {'tests': [{'arguments': {'x': 1}, 'expected': {'y': 2}}], 'final_arguments': {'z': 4}},
    {'tests': [{'arguments': {'x': float('nan')}, 'expected': {'y': 2}}], 'final_arguments': {'x': 4}},
])
def test_invalid_explicit_contract_is_rejected(payload):
    with pytest.raises(GeneratedToolContractError):
        explicit_generated_tool_contract('tool_contract=' + json.dumps(payload))


def test_source_context_keeps_current_failure_and_no_persona_or_chat_history():
    contract = explicit_generated_tool_contract(contract_prompt())
    messages = Agent._generated_source_messages(
        contract_prompt(), generated_contract=contract,
        failures=[{'tool': 'web_search', 'error': 'unrelated old content'},
                  {'tool': 'learning_create_tool', 'arguments': {'source': 'def run(arguments): return {}'},
                   'error': 'expected count=1 but got {}'}],
    )
    assert 'SOURCE PHASE' in messages[0]['content']
    assert 'expected count=1' in messages[-1]['content']
    assert 'def run(arguments)' in messages[-1]['content']
    assert 'unrelated old content' not in json.dumps(messages)
    assert all(message['role'] != 'assistant' for message in messages)


@pytest.mark.asyncio
async def test_rejected_source_does_not_disable_builder(tmp_path):
    registry = ToolRecoveryRegistry(tmp_path)
    calls = []
    async def provider(tool, arguments):
        calls.append(tool)
        if len(calls) <= 3:
            raise ArtifactValidationError('fixture expected count=2, got count=0')
        return '{"status":"active"}'
    for attempt in range(4):
        outcome = await execute_with_recovery(
            registry, requested_tool='learning_create_tool', arguments={'source': str(attempt)},
            call_provider=provider, detect_failure=lambda result, tool: '',
        )
        assert bool(outcome.error) == (attempt < 3)
    assert len(calls) == 4
    assert registry.providers_for('learning_create_tool')


def test_historical_artifact_rejections_do_not_keep_circuit_open(tmp_path):
    registry = ToolRecoveryRegistry(tmp_path)
    registry.register_provider('learning_create_tool', ('tool.create',))
    for _ in range(2):
        registry.record_failure(tool='learning_create_tool', requested_tool='learning_create_tool',
                                arguments={}, error='ArtifactValidationError: invalid draft')
    assert ToolRecoveryRegistry(tmp_path).providers_for('learning_create_tool')
