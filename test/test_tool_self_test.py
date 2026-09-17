import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_core.tool_self_test import test_local_tools as run_checks
from v_core.agent import Agent
from v_core.autonomy import SemanticIntent
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.mcp_tools import MCPTools
from v_core.osint_self_test import self_test_osint_tool


LOCAL_TOOLS = ('read_file', 'write_file', 'list_directory', 'create_directory',
               'edit_file', 'move_file', 'search_files', 'directory_tree', 'get_file_info')


class LocalProvider:
    def __init__(self, root):
        self.workspace = root
        self.calls = []

    def normalize_arguments(self, name, arguments):
        return arguments

    async def _call_direct(self, name, arguments):
        self.calls.append(name)
        if name == 'move_file':
            Path(arguments['source']).rename(arguments['destination'])
            return 'moved'
        path = Path(arguments['path'])
        if name == 'write_file':
            path.write_text(arguments['content'])
            return 'written'
        if name == 'read_file':
            return path.read_text()
        if name == 'create_directory':
            path.mkdir()
            return 'created'
        if name == 'edit_file':
            content = path.read_text()
            for edit in arguments['edits']:
                content = content.replace(edit['oldText'], edit['newText'])
            path.write_text(content)
            return 'edited'
        if name == 'search_files':
            return '\n'.join(str(p) for p in path.glob(arguments['pattern']))
        if name == 'get_file_info':
            return f'size: {path.stat().st_size}\nisFile: true\nisDirectory: false'
        if name == 'directory_tree':
            def tree(root):
                return [dict(name=p.name, type='directory', children=tree(p)) if p.is_dir()
                        else dict(name=p.name, type='file') for p in root.iterdir()]
            return json.dumps(tree(path))
        return '\n'.join(p.name for p in path.iterdir())


def definitions(*names):
    return [{'function': {'name': name}} for name in names]


@pytest.mark.asyncio
async def test_local_fixtures_and_unknown_tools(tmp_path):
    provider = LocalProvider(tmp_path)
    text, evidence = await run_checks(provider, definitions('read_file', 'write_file', 'list_directory', 'unknown'))
    assert evidence['counts'] == {'passed': 3, 'failed': 0, 'not_tested': 1}
    assert 'unknown' not in provider.calls
    assert list(tmp_path.iterdir()) == []
    assert 'not an OS sandbox' in text
    assert evidence['live_network_tests'] == 0


@pytest.mark.asyncio
async def test_no_false_success_for_plausible_output(tmp_path):
    provider = LocalProvider(tmp_path)
    async def fake(*args):
        return 'Everything works perfectly'
    provider._call_direct = fake
    _, evidence = await run_checks(provider, definitions('read_file', 'write_file'))
    assert evidence['counts']['failed'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['mention', 'suffix', 'extra', 'directory', 'mapping'])
async def test_listing_requires_exact_fixture_entries(tmp_path, shape):
    provider = LocalProvider(tmp_path)

    async def misleading(name, arguments):
        filename = next(Path(arguments['path']).iterdir()).name
        return {
            'mention': f'Could not list {filename}',
            'suffix': filename + '.backup',
            'extra': f'[FILE] {filename}\n[FILE] nonexistent.txt',
            'directory': f'[DIR] {filename}',
            'mapping': {'message': filename},
        }[shape]

    provider._call_direct = misleading
    _, evidence = await run_checks(provider, definitions('list_directory'))
    assert evidence['counts'] == {'passed': 0, 'failed': 1, 'not_tested': 0}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['plain', 'mcp_text', 'mcp_blocks'])
async def test_listing_accepts_supported_provider_formats(tmp_path, shape):
    provider = LocalProvider(tmp_path)

    async def listing(name, arguments):
        filename = next(Path(arguments['path']).iterdir()).name
        return {'plain': filename, 'mcp_text': f'[FILE] {filename}\n',
                'mcp_blocks': [f'[FILE] {filename}']}[shape]

    provider._call_direct = listing
    _, evidence = await run_checks(provider, definitions('list_directory'))
    assert evidence['counts']['passed'] == 1


@pytest.mark.asyncio
async def test_in_place_argument_remapping_is_not_executed(tmp_path):
    provider = LocalProvider(tmp_path)

    def mutate(name, arguments):
        arguments['path'] = str(tmp_path / 'outside-fixture.txt')
        return arguments

    provider.normalize_arguments = mutate
    _, evidence = await run_checks(provider, definitions('write_file'))
    assert evidence['counts']['not_tested'] == 1
    assert provider.calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_remapped_paths_are_never_executed(tmp_path):
    provider = LocalProvider(tmp_path)
    provider.normalize_arguments = lambda *args: {'path': '/somewhere/else'}
    _, evidence = await run_checks(provider, definitions('read_file'))
    assert evidence['counts']['not_tested'] == 1
    assert not provider.calls


@pytest.mark.asyncio
async def test_timeout_and_cleanup(tmp_path):
    provider = LocalProvider(tmp_path)
    async def slow(*args):
        await asyncio.sleep(60)
    provider._call_direct = slow
    _, evidence = await run_checks(provider, definitions('read_file'), timeout=0.01)
    assert evidence['counts']['failed'] == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_edition_fixture_is_reported_without_calling_live_provider(tmp_path):
    class Extension:
        def handles_tool(self, name):
            return name == 'edition_network_tool'

        async def self_test_tool(self, name):
            return {'tool': name, 'case': 'offline_protocol_fixture', 'status': 'passed',
                    'reason': 'Matched fixture.', 'evidence_level': 'deterministic_offline_fixture',
                    'live_network': False}

    provider = LocalProvider(tmp_path)
    provider.edition_extension = Extension()
    _, evidence = await run_checks(provider, definitions('edition_network_tool'))
    assert evidence['counts'] == {'passed': 1, 'failed': 0, 'not_tested': 0}
    assert evidence['live_network_tests'] == 0
    assert provider.calls == []


@pytest.mark.asyncio
async def test_edition_fixture_failure_is_not_reported_as_untested(tmp_path):
    class Extension:
        def handles_tool(self, name):
            return True

        async def self_test_tool(self, name):
            raise RuntimeError('fixture broke')

    provider = LocalProvider(tmp_path)
    provider.edition_extension = Extension()
    _, evidence = await run_checks(provider, definitions('edition_network_tool'))
    assert evidence['counts'] == {'passed': 0, 'failed': 1, 'not_tested': 0}
    assert evidence['results'][0]['evidence_level'] == 'deterministic_offline_fixture'


@pytest.mark.asyncio
async def test_public_web_fixtures_preserve_search_read_grounding():
    search = await self_test_osint_tool('web_search', MCPTools)
    read = await self_test_osint_tool('web_read', MCPTools)
    assert search['status'] == 'passed'
    assert read['status'] == 'passed'
    assert search['live_network'] is False
    assert read['live_network'] is False
    assert await self_test_osint_tool('unknown', MCPTools) is None


@pytest.mark.asyncio
async def test_core_osint_fixtures_integrate_without_live_provider_calls(tmp_path):
    class Provider(LocalProvider):
        async def self_test_tool(self, name):
            return await self_test_osint_tool(name, MCPTools)

    provider = Provider(tmp_path)
    _, evidence = await run_checks(provider, definitions('web_search', 'web_read'))
    assert evidence['counts'] == {'passed': 2, 'failed': 0, 'not_tested': 0}
    assert evidence['functional_tests'] == 2
    assert evidence['live_network_tests'] == 0
    assert provider.calls == []


@pytest.mark.asyncio
async def test_core_fixture_failure_is_not_reported_as_untested(tmp_path):
    class Provider(LocalProvider):
        async def self_test_tool(self, name):
            raise RuntimeError('fixture broke')

    provider = Provider(tmp_path)
    _, evidence = await run_checks(provider, definitions('web_search'))
    assert evidence['counts'] == {'passed': 0, 'failed': 1, 'not_tested': 0}
    assert evidence['results'][0]['evidence_level'] == 'deterministic_offline_fixture'


@pytest.mark.asyncio
async def test_cancellation_propagates_and_cleans(tmp_path):
    provider = LocalProvider(tmp_path)
    async def cancelled(*args):
        raise asyncio.CancelledError
    provider._call_direct = cancelled
    with pytest.raises(asyncio.CancelledError):
        await run_checks(provider, definitions('read_file'))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_no_generated_override_runs(tmp_path):
    provider = LocalProvider(tmp_path)
    provider.learning = SimpleNamespace(active_tool_names=lambda: ['read_file'])
    _, evidence = await run_checks(provider, definitions('read_file'))
    assert evidence['counts']['not_tested'] == 1
    assert not provider.calls


@pytest.mark.asyncio
async def test_scope_and_read_only(tmp_path):
    provider = LocalProvider(tmp_path)
    _, evidence = await run_checks(provider, definitions('read_file', 'write_file'), prompt='Test read_file')
    assert provider.calls == ['read_file']
    assert evidence['counts']['passed'] == 1
    provider.calls.clear()
    _, evidence = await run_checks(provider, definitions('read_file', 'write_file'), allow_write=False)
    assert provider.calls == ['read_file']
    assert evidence['counts']['not_tested'] == 1


@pytest.mark.asyncio
async def test_all_local_checks_are_isolated_and_have_case_evidence(tmp_path):
    provider = LocalProvider(tmp_path)
    _, evidence = await run_checks(provider, definitions(*LOCAL_TOOLS))
    assert evidence['counts'] == {'passed': 9, 'failed': 0, 'not_tested': 0}
    assert set(provider.calls) == set(LOCAL_TOOLS)
    assert all(row['case'] for row in evidence['results'])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_read_only_skips_every_mutating_provider(tmp_path):
    provider = LocalProvider(tmp_path)
    _, evidence = await run_checks(provider, definitions(*LOCAL_TOOLS), allow_write=False)
    assert evidence['counts'] == {'passed': 5, 'failed': 0, 'not_tested': 4}
    assert set(provider.calls) == {'read_file', 'list_directory', 'search_files', 'directory_tree', 'get_file_info'}


@pytest.mark.asyncio
@pytest.mark.parametrize('name', LOCAL_TOOLS)
async def test_every_local_check_rejects_success_without_evidence(tmp_path, name):
    provider = LocalProvider(tmp_path)
    async def fake(*args):
        return 'Done, all checks passed'
    provider._call_direct = fake
    _, evidence = await run_checks(provider, definitions(name))
    assert evidence['counts']['failed'] == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_failed_tool_does_not_hide_later_results(tmp_path):
    provider = LocalProvider(tmp_path)
    real = provider._call_direct
    async def faulty(name, arguments):
        if name == 'create_directory':
            raise OSError('fixture provider failed')
        return await real(name, arguments)
    provider._call_direct = faulty
    _, evidence = await run_checks(provider, definitions('create_directory', 'read_file'))
    assert evidence['counts'] == {'passed': 1, 'failed': 1, 'not_tested': 0}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_nested_edit_argument_mutation_does_not_change_expected_fixture(tmp_path):
    provider = LocalProvider(tmp_path)
    def mutate(name, arguments):
        arguments['edits'][0]['newText'] = 'not the prepared result'
        return arguments
    provider.normalize_arguments = mutate
    _, evidence = await run_checks(provider, definitions('edit_file'))
    assert evidence['counts']['not_tested'] == 1
    assert provider.calls == []


@pytest.mark.asyncio
async def test_fixture_directory_failure_is_reported_without_execution(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError('fixture unavailable')
    monkeypatch.setattr('v_core.tool_self_test.TemporaryDirectory', denied)
    provider = LocalProvider(tmp_path)
    _, evidence = await run_checks(provider, definitions(*LOCAL_TOOLS))
    assert evidence['counts'] == {'passed': 0, 'failed': 0, 'not_tested': 9}
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('size_kind', ['characters', 'duplicate', 'wrong_type'])
async def test_metadata_requires_byte_size_and_unambiguous_file_type(tmp_path, size_kind):
    provider = LocalProvider(tmp_path)
    async def fake(name, arguments):
        path = Path(arguments['path'])
        size = len(path.read_text()) if size_kind == 'characters' else path.stat().st_size
        data = f'size: {size}\nisFile: true\nisDirectory: false'
        if size_kind == 'duplicate':
            data += '\nsize: 1'
        if size_kind == 'wrong_type':
            data = data.replace('isDirectory: false', 'isDirectory: true')
        return data
    provider._call_direct = fake
    _, evidence = await run_checks(provider, definitions('get_file_info'))
    assert evidence['counts']['failed'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_names', [('read_file', 'write_file', 'list_directory'), LOCAL_TOOLS])
async def test_agent_records_actual_fixture_calls_without_model_report(tmp_path, tool_names):
    class Provider(LocalProvider):
        async def openai_tool_definitions(self):
            return definitions(*tool_names, 'unknown')

    class Router:
        async def classify(self, *args, **kwargs):
            return SemanticIntent(action_requested=True, capabilities=('tool_self_test',), requires_report=True)

    fixture_root = tmp_path / 'fixtures'
    fixture_root.mkdir()
    agent = object.__new__(Agent)
    agent.tools = Provider(fixture_root)
    agent.intent_router = Router()
    agent.memory = SimpleNamespace(session=Session())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path / 'traces'
    agent._last_execution_context = None
    answer = await agent._run_agent_loop('Check tool health using local fixtures.')
    assert 'Untested tools remain unverified' in answer
    journal = next((agent._agent_trace_root / 'journal').glob('*.jsonl'))
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    checks = [event for event in events if event['event'] == 'tool_self_test']
    assert checks[0]['data']['functional_tests'] == len(tool_names)
    assert len([event for event in events if event['event'] == 'tool_started']) == len(tool_names)
    assert list(fixture_root.iterdir()) == []
