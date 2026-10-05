import json
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.generated_tool_contract import (
    GeneratedToolContractError, explicit_generated_tool_contract,
    extract_generated_tool_contract,
    source_generation_objective,
)
from v_core.learning import ArtifactValidationError
from v_core.tool_recovery import ToolRecoveryRegistry, execute_with_recovery
from v_core.learning.source_builder import MissingToolInputError, build_source_blueprint
from v_core.llm.llm import LLMResponse
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile


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


@pytest.mark.parametrize('objective', [
    'Zbuduj parser. Nie używaj sieci. ',
    'Build a parser. Do not access the network. ',
    '解析器を作成。ネットワークは使わない。 ',
    'أنشئ محللاً دون اتصال بالشبكة. ',
])
def test_source_context_deduplicates_only_frozen_json_preserving_objective(objective):
    prompt = objective + contract_prompt() + '\nReturn count, not visitor totals.'
    contract = explicit_generated_tool_contract(prompt)
    before = contract.to_dict()
    messages = Agent._generated_source_messages(prompt, generated_contract=contract, failures=[])

    assert messages[1]['content'].startswith(objective + 'Create a parser. ')
    assert messages[1]['content'].endswith('\nReturn count, not visitor totals.')
    assert 'tool_contract=' not in messages[1]['content']
    examples = json.loads(messages[2]['content'].split('\n', 1)[1])
    assert examples == {'tests': [{'arguments': case.arguments, 'expected': case.expected}
                                  for case in contract.tests]}
    assert 'final_arguments' not in examples
    assert 'evidence_quote' not in examples['tests'][0]
    examples['tests'][0]['expected']['count'] = 77
    assert contract.to_dict() == before


def test_source_context_keeps_all_examples_without_mutating_full_contract():
    payload = {
        'tests': [{'arguments': {'value': n}, 'expected': {'result': n * 2}} for n in range(8)],
        'final_arguments': {'value': 999},
    }
    prompt = 'Double the input. tool_contract=' + json.dumps(payload)
    contract = explicit_generated_tool_contract(prompt)
    projected = contract.source_examples()
    assert projected == {'tests': payload['tests']}
    projected['tests'][0]['arguments']['value'] = -1
    assert contract.tests[0].arguments == {'value': 0}
    assert contract.final_arguments == {'value': 999}
    assert contract.final_evidence_quote in prompt
    messages = Agent._generated_source_messages(prompt, generated_contract=contract, failures=[])
    legacy_size = len(messages[0]['content']) + len(prompt) + len(json.dumps(contract.to_dict()))
    assert sum(len(message['content']) for message in messages) < legacy_size / 2


def test_semantic_contract_keeps_natural_language_and_runtime_specification():
    contract = replace(explicit_generated_tool_contract(contract_prompt()),
                       provenance='owner_text_semantic_extraction',
                       specification='Parse records; never contact a host.')
    objective = 'Przetwórz lokalne dane; brak wyniku nie oznacza zera.'
    messages = Agent._generated_source_messages(objective, generated_contract=contract, failures=[])
    assert messages[1]['content'] == objective
    assert contract.specification in messages[0]['content']
    assert json.loads(messages[2]['content'].split('\n', 1)[1]) == contract.source_examples()


def test_source_objective_does_not_strip_unverified_or_unknown_data():
    prompt = contract_prompt()
    contract = explicit_generated_tool_contract(prompt)
    for value in ['Natural wording only.', 'tool_contract={', prompt + ' tool_contract={}']:
        assert source_generation_objective(value, contract) == value
    assert source_generation_objective(prompt, replace(contract, final_arguments={'har_json': 'new'})) == prompt
    payload = json.loads(prompt.split('tool_contract=', 1)[1])
    payload['constraints'] = 'Preserve this unknown extension.'
    extended = 'tool_contract=' + json.dumps(payload)
    assert source_generation_objective(extended, explicit_generated_tool_contract(extended)) == extended
    del payload['constraints']
    payload['tests'][0]['description'] = 'Preserve this unknown case extension.'
    extended = 'tool_contract=' + json.dumps(payload)
    assert source_generation_objective(extended, explicit_generated_tool_contract(extended)) == extended


def test_source_context_without_contract_keeps_original_prompt():
    prompt = 'Arbitrary instructions with tool_contract={malformed data}'
    messages = Agent._generated_source_messages(prompt, generated_contract=None, failures=[])
    assert messages[1]['content'] == prompt
    assert len(messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('fields', [('image',), ('text', 'coordinates')])
@pytest.mark.parametrize('route', ['source', 'compatibility', 'native'])
async def test_missing_creation_input_awaits_owner_without_call_or_retry(tmp_path, fields, route):
    class Tools:
        calls = []

        async def openai_tool_definitions(self):
            return [{'type': 'function', 'function': {
                'name': 'learning_create_tool', 'description': 'Create a local tool',
                'parameters': {'type': 'object', 'properties': {'source': {'type': 'string'}},
                               'required': ['source']},
            }}]

        async def call(self, tool, arguments):
            self.calls.append(tool)
            raise AssertionError('Missing input must pause before creating an artifact')

    source = 'def run(arguments):\n    return {' + ', '.join(
        f'{field!r}: arguments[{field!r}]' for field in fields) + '}\n'

    class Model:
        config = SimpleNamespace(context=12_000, model='test-model')
        turns = 0

        async def respond(self, **kwargs):
            self.turns += 1
            assert self.turns == 1, 'Missing data must not trigger another model request'
            if route == 'native':
                from v_core.llm.llm import LLMToolCall
                return LLMResponse(content='', tool_calls=[LLMToolCall(
                    name='learning_create_tool', arguments={'source': source}, call_id='create1')])
            if route == 'compatibility':
                return LLMResponse(content=json.dumps({'tool': 'learning_create_tool', 'arguments': {'source': source}}))
            return LLMResponse(content=source)

    class Memory:
        session = Session()

        async def process(self, *args, **kwargs):
            pass

    agent = object.__new__(Agent)
    agent.tools, agent.llm, agent.memory = Tools(), Model(), Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: 'system'
    agent._agent_trace_root = tmp_path / 'traces'
    streamed = []
    prompt = 'Create a reusable local tool.' if route == 'source' else 'Create a tool or a skill to process data.'
    answer = await agent._run_agent_loop(prompt, streamed.append)
    await asyncio.gather(*agent._memory_tasks)

    assert agent.tools.calls == []
    assert agent.llm.turns == 1
    assert 'known correct result' in answer
    assert 'not executed or activated' in answer
    assert streamed == [answer]
    checkpoint = json.loads(next((tmp_path / 'traces' / 'checkpoints').glob('*.json')).read_text())
    assert checkpoint['status'] == 'awaiting_owner'
    assert checkpoint['tool_calls'] == []
    assert agent._last_execution_context['status'] == 'awaiting_owner'
    journal = next((tmp_path / 'traces' / 'journal').glob('*.jsonl')).read_text()
    assert 'generated_tool_input_missing' in journal
    assert 'creation_failure_budget_exhausted' not in journal


def test_missing_input_has_structured_fields_and_concrete_fixture_still_works():
    source = "def run(arguments):\n    return {'width': arguments['image']['width']}"
    with pytest.raises(MissingToolInputError) as caught:
        build_source_blueprint(objective='Build an image data helper.', source=source)
    assert caught.value.fields == ('image',)
    blueprint = build_source_blueprint(
        objective='image = {"width": 20} expected = {"width": 20}', source=source)
    assert blueprint.arguments == {'image': {'width': 20}}
    assert blueprint.expected == {'width': 20}


def test_observed_snapshot_remains_valid_input_for_preflight():
    blueprint = build_source_blueprint(
        objective='Create a parser.', observed_snapshot='actual captured page',
        source="def run(arguments):\n    return {'text': arguments['snapshot_text']}",
    )
    assert blueprint.arguments == {'snapshot_text': 'actual captured page'}


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
