"""Reproduce the owner's photos -> district -> map-search conversation."""
import asyncio
import base64
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from v_core.agent import Agent
from v_core.autonomy import TaskContract
from v_core.autonomy.photo_context import (
    owner_photo_context, photo_location_request, location_query, relevant_image_evidence,
)
from v_core.tools.image_analysis import analyze_owner_image, local_endpoint
from v_core.tools.image_metadata import read_image_metadata
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.llm.llm import LLMResponse, LLMToolCall
from v_core.learning.source_builder import build_source_blueprint
from v_core.mcp_tools import MCPTools
from test_image_metadata import jpeg


FIRST = ('file:///tmp/photo.jpg masz tu zdjecia miejsca gdzies w warszawie. '
         'stworz narzedzie badz tez umiejetnosc ktore umozliwia ci znalezienie prawdopodobnej lokacji tego miejsca')
DISTRICT = ('Nie podam Ci dokładnej lokacji. Mogę Ci tylko powiedzieć, że jest to Warszawa '
            'i że jest to gdzieś na pograniczu dzielnicy Włochy.')
MAPS = ('V, kochana moja, musisz użyć Google Maps, OpenStreetMaps, '
        'poszukać w internecie zdjęć podobnych z tej dzielnicy.')


def test_three_owner_turns_keep_files_and_area_but_not_assistant_claims():
    messages = [{'role': 'user', 'content': FIRST},
                {'role': 'assistant', 'content': 'file:///tmp/foreign.jpg it is in ImaginaryCity'},
                {'role': 'user', 'content': DISTRICT}]
    scoped = owner_photo_context(MAPS, messages)
    assert FIRST in scoped and DISTRICT in scoped and MAPS in scoped
    assert 'foreign' not in scoped and 'ImaginaryCity' not in scoped
    query = location_query(scoped)
    assert 'Warszawa' in query and 'Włochy' in query
    assert 'Maps' not in query and 'OpenStreet' not in query
    assert photo_location_request(FIRST)
    assert not photo_location_request('Resize file:///tmp/location_1.jpg')


@pytest.mark.parametrize('barrier', ['/stop', 'Find Python packages', 'What is the weather in Warsaw?'])
def test_unrelated_owner_turn_ends_photo_chain(barrier):
    messages = [{'role': 'user', 'content': FIRST}, {'role': 'user', 'content': barrier}]
    assert owner_photo_context(MAPS, messages) == MAPS
    assert owner_photo_context('Read /tmp/secret.txt', messages) == 'Read /tmp/secret.txt'


def test_new_images_replace_old_inputs_and_loaded_session_is_not_auto_resumed(tmp_path):
    new = 'Locate file:///tmp/new.jpg'
    assert owner_photo_context(new, [{'role': 'user', 'content': FIRST}]) == new
    session = Session(tmp_path)
    session.add('task', {'task': FIRST, 'result': 'awaiting image processor'})
    agent = object.__new__(Agent)
    agent.memory = SimpleNamespace(session=Session(tmp_path))
    assert agent._photo_routing_prompt(MAPS) == MAPS


def test_logos_and_author_portrait_do_not_satisfy_requested_photos():
    contract = TaskContract(required_research_facets=('images',))
    evidence = '- img "Blurify"\n- img "Jagoda"\n- img "blur logo"'
    calls = [{'tool': 'browser_snapshot', 'status': 'succeeded', 'result_excerpt': evidence}]
    assert 'browser_evidence:research_images' in contract.unmet(calls)
    assert not relevant_image_evidence('- img "3D projection of buildings in OSM"', 'Warszawa Włochy')
    assert relevant_image_evidence('- img "Warszawa Włochy railway bridge"', 'Warszawa Włochy')
    assert relevant_image_evidence('- img "Species alpha photograph"')


def test_photo_query_repairs_provider_comparison_but_allows_area_refinement():
    scoped = owner_photo_context(MAPS, [{'role': 'user', 'content': FIRST}, {'role': 'user', 'content': DISTRICT}])
    contract = TaskContract(requires_web_discovery=True)
    query = location_query(scoped)
    assert Agent._repair_web_discovery_navigation(scoped, 'web_search',
        {'query': 'Google Maps versus OpenStreetMap'}, contract, [], [], preferred_query=query)['query'] == query
    refined = {'query': 'Warszawa Włochy railway bridge photographs'}
    assert Agent._repair_web_discovery_navigation(scoped, 'web_search', refined,
        contract, [], [], preferred_query=query) == refined


def test_builder_binds_known_image_paths_without_asking_for_them_again(tmp_path):
    photo = tmp_path / 'input.jpg'; photo.write_bytes(jpeg())
    blueprint = build_source_blueprint(
        objective=photo.as_uri() + ' Create an input counting tool. expected = {"count": 1}',
        source="def run(arguments):\n    return {'count': len(arguments['image_paths'])}",
    )
    assert blueprint.arguments == {'image_paths': [str(photo)]}
    assert blueprint.expected == {'count': 1}


@pytest.mark.asyncio
async def test_mcp_image_analysis_is_wired_to_selected_local_endpoint(tmp_path, monkeypatch):
    photo = tmp_path / 'input.jpg'; photo.write_bytes(jpeg())
    tools = MCPTools(SimpleNamespace(filesystem_server=['/usr/bin/false'], browser_server=['/usr/bin/false'],
        workspace=tmp_path / 'workspace', learning_root=tmp_path / 'learning', learning_profile='client', evm_profile='client'))
    tools.begin_interaction('vision-test', photo.as_uri())
    monkeypatch.setenv('V_CORE_VISION_BASE_URL', 'http://127.0.0.1:5002/v1')
    monkeypatch.setenv('V_CORE_VISION_MODEL', 'separate-vision')
    async def observe(path, prompt, base_url, model):
        assert (path, prompt, base_url, model) == (str(photo), photo.as_uri(), 'http://127.0.0.1:5002/v1', 'separate-vision')
        return {'path': path, 'status': 'unavailable', 'visual_analysis_performed': False}
    monkeypatch.setattr('v_core.mcp_tools.analyze_owner_image', observe)
    outcome = await tools.call_with_recovery('image_analyze', {'path': str(photo)})
    result = json.loads(outcome.result)
    assert result['status'] == 'unavailable'
    assert tuple(outcome.capabilities) == ('filesystem.image_visual_observation',)


@pytest.mark.parametrize('url', ['https://external.test/v1', 'http://127.0.0.1@external.test/v1',
                               'http://127.0.0.1/v1?redirect=https://external.test'])
def test_images_cannot_be_sent_to_a_remote_processor(url):
    with pytest.raises(ValueError):
        local_endpoint(url)


@pytest.mark.asyncio
@pytest.mark.parametrize('vision', [False, True])
async def test_visual_transport_checks_capability_and_sends_exact_pixels(tmp_path, vision):
    image = tmp_path / 'photo.jpg'
    image.write_bytes(jpeg())
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.path == '/props':
            return httpx.Response(200, json={'modalities': {'vision': vision}})
        payload = json.loads(request.content)
        encoded = payload['messages'][1]['content'][1]['image_url']['url'].split(',', 1)[1]
        assert base64.b64decode(encoded) == image.read_bytes()
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': 'A railway bridge and an unreadable street sign.'}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await analyze_owner_image(str(image), image.as_uri(), 'http://127.0.0.1:5002/v1', 'vision', client=client)
    assert result['visual_analysis_performed'] is vision
    assert len(requests) == (2 if vision else 1)
    assert result['image_sha256'] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert result.get('location_verified', False) is False


@pytest.mark.asyncio
@pytest.mark.parametrize('reason,content', [('length', 'Partial description'), ('stop', ''), ('stop', 'x' * 4501)])
async def test_incomplete_visual_output_is_not_a_verified_observation(tmp_path, reason, content):
    image = tmp_path / 'photo.jpg'; image.write_bytes(jpeg())
    def respond(request):
        return httpx.Response(200, json={'modalities': {'vision': True}} if request.url.path == '/props'
            else {'choices': [{'finish_reason': reason, 'message': {'content': content}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError):
            await analyze_owner_image(str(image), image.as_uri(), 'http://127.0.0.1:5002/v1', 'vision', client=client)


def make_agent(tmp_path, tools, model):
    agent = object.__new__(Agent)
    agent.tools, agent.llm = tools, model
    class Memory:
        session = Session()
        async def process(self, *args, **kwargs): pass
    agent.memory = Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: 'system'
    agent._agent_trace_root = tmp_path / 'traces'
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("image_count", [3, 8])
async def test_polish_creation_and_map_followup_stop_at_actual_missing_vision(tmp_path, image_count):
    photos = [tmp_path / f'photo_{n}.jpg' for n in range(image_count)]
    for photo in photos: photo.write_bytes(jpeg())
    first = FIRST.replace('file:///tmp/photo.jpg', ' '.join(p.as_uri() for p in photos))
    class Tools:
        calls = []
        def begin_interaction(self, task_id, prompt): self.prompt = prompt
        async def openai_tool_definitions(self):
            return [{'type': 'function', 'function': {'name': name, 'parameters': {
                'type': 'object', 'properties': {field: {'type': 'string'}}, 'required': [field]}}}
                for name, field in [('image_metadata', 'path'), ('image_analyze', 'path'),
                                     ('learning_create_tool', 'source'), ('web_search', 'query'),
                                     ('browser_navigate', 'url'), ('browser_snapshot', 'unused')]]
        async def call(self, tool, arguments):
            self.calls.append(tool)
            if tool == 'image_metadata': return json.dumps(read_image_metadata(arguments['path'], self.prompt))
            assert tool == 'image_analyze'
            return json.dumps({'path': arguments['path'], 'status': 'unavailable', 'visual_analysis_performed': False})
    class Model:
        config = SimpleNamespace(context=12000, model='text-only')
        async def respond(self, **kwargs): raise AssertionError('No code or meaningless web search before image inspection')
    agent = make_agent(tmp_path, Tools(), Model())
    agent.memory.session.add('task', {'task': first, 'result': 'Metadata only.'})
    agent.memory.session.add('task', {'task': DISTRICT, 'result': 'Need a visual processor.'})
    answer = await agent._run_agent_loop(MAPS)
    await asyncio.gather(*agent._memory_tasks)
    assert agent.tools.calls == ['image_metadata'] * image_count + ['image_analyze']
    assert 'do not need to reveal the location' in answer
    assert 'not been created' in answer
    checkpoint = json.loads(next((tmp_path / 'traces/checkpoints').glob('*.json')).read_text())
    assert checkpoint['status'] == 'awaiting_owner'
    assert checkpoint['owner_checkpoint']['missing'] == ['visual_image_processor']
    assert 'Włochy' in checkpoint['objective'] and str(photos[0]) in checkpoint['objective']


@pytest.mark.asyncio
async def test_supported_vision_keeps_research_open_after_unrelated_blog(tmp_path):
    image = tmp_path / 'photo.jpg'; image.write_bytes(jpeg())
    prompt = f'Find where {image.as_uri()} was taken in Warszawa Włochy. Search the web for similar photos.'
    blog, reference = 'https://source.test/comparison', 'https://source.test/warszawa-wlochy'
    class Tools:
        calls = []
        page = ''
        def begin_interaction(self, task_id, prompt): self.prompt = prompt
        async def openai_tool_definitions(self):
            return [{'type': 'function', 'function': {'name': n}} for n in (
                'image_metadata', 'image_analyze', 'web_search', 'browser_navigate', 'browser_snapshot')]
        async def call(self, tool, arguments):
            self.calls.append((tool, arguments))
            if tool == 'image_metadata': return json.dumps(read_image_metadata(arguments['path'], self.prompt))
            if tool == 'image_analyze': return json.dumps({'path': arguments['path'], 'status': 'observed',
                'visual_analysis_performed': True, 'observations': 'A railway bridge.', 'location_verified': False})
            if tool == 'web_search': return json.dumps({'results': [
                {'title': 'Warszawa Włochy map comparison', 'url': blog},
                {'title': 'Warszawa Włochy railway bridge photos', 'url': reference}]})
            if tool == 'browser_navigate':
                self.page = arguments['url']; return f'- Page URL: {self.page}'
            assert tool == 'browser_snapshot'
            if self.page == blog:
                return f'- Page URL: {blog}\n- Page Title: Map comparison\n- img "Brand logo"\n- img "Jagoda"'
            return (f'- Page URL: {reference}\n- Page Title: Warszawa Włochy railway bridge\n'
                    '- img "Warszawa Włochy railway bridge photograph"\n'
                    '- paragraph: A railway bridge is pictured in Warszawa Włochy.')
    class Model:
        config = SimpleNamespace(context=12000, model='fake-vision')
        turns = 0
        async def respond(self, **kwargs):
            self.turns += 1
            requests = [
                ('web_search', {'query': 'Google Maps versus OpenStreetMap'}),
                ('browser_navigate', {'url': blog}),
                ('web_search', {'query': 'Warszawa Włochy railway bridge photos'}),
                ('browser_navigate', {'url': reference}),
            ]
            if self.turns <= len(requests):
                name, args = requests[self.turns - 1]
                return LLMResponse(tool_calls=[LLMToolCall(str(self.turns), name, args)])
            return LLMResponse(content=self.candidate())
        def candidate(self):
            return (f'Boss, the supplied image shows a railway bridge. I found a reference photograph of a railway bridge in Warszawa Włochy. '
                    f'This is a candidate reference, not a confirmed location match. The visible similarity alone does not establish an exact address. Source: {reference}')
        async def ask(self, **kwargs):
            return self.candidate()
    agent = make_agent(tmp_path, Tools(), Model())
    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)
    names = [name for name, _ in agent.tools.calls]
    assert names[:2] == ['image_metadata', 'image_analyze']
    assert names.count('web_search') == 2 and names.count('browser_snapshot') == 2
    assert agent.tools.calls[2][1]['query'] == location_query(prompt)
    assert 'not a confirmed location match' in answer
    journal = next((tmp_path / 'traces/journal').glob('*.jsonl')).read_text()
    assert 'post_contract_tool_call_rejected' not in journal
    checkpoint = json.loads(next((tmp_path / 'traces/checkpoints').glob('*.json')).read_text())
    assert checkpoint['status'] == 'completed'
