import asyncio
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from v_core.tools.image_metadata import inspect_jpeg, read_image_metadata, literal_image_paths
from v_core.agent import Agent
from v_core.mcp_tools import MCPTools
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile


def jpeg(exif=b''):
    def segment(marker, payload):
        return b'\xff' + bytes([marker]) + struct.pack('>H', len(payload) + 2) + payload
    return (b'\xff\xd8' + (segment(0xE1, b'Exif\0\0' + exif) if exif else b'')
            + segment(0xC0, b'\x08' + struct.pack('>HH', 480, 640) + b'\x01\x01\x11\x00') + b'\xff\xd9')


def gps_exif(endian='<', references=('N', 'E'), denominator=1):
    def pack(fmt, *values):
        return struct.pack(endian + fmt, *values)
    def entry(tag, kind, count, data):
        return pack('HHI', tag, kind, count) + data
    header = (b'II' if endian == '<' else b'MM') + pack('HI', 42, 8)
    main = pack('H', 1) + entry(34853, 4, 1, pack('I', 26)) + pack('I', 0)
    gps = (pack('H', 4)
           + entry(1, 2, 2, references[0].encode() + b'\0\0\0')
           + entry(2, 5, 3, pack('I', 80))
           + entry(3, 2, 2, references[1].encode() + b'\0\0\0')
           + entry(4, 5, 3, pack('I', 104)) + pack('I', 0))
    return header + main + gps + pack('IIIIII', 52, denominator, 15, 1, 0, 1) + pack('IIIIII', 21, 1, 30, 1, 0, 1)


@pytest.mark.parametrize('endian', ['<', '>'])
@pytest.mark.parametrize('references,signs', [(('N', 'E'), (1, 1)), (('S', 'W'), (-1, -1))])
def test_exif_gps_rationals_decode_in_both_byte_orders(endian, references, signs):
    result = inspect_jpeg(jpeg(gps_exif(endian, references)))
    assert result['gps_status'] == 'present'
    assert result['gps']['latitude'] == signs[0] * 52.25
    assert result['gps']['longitude'] == signs[1] * 21.5
    assert result['width'] == 640 and result['height'] == 480
    assert result['visual_analysis_performed'] is False


@pytest.mark.parametrize('exif', [b'', b'II*\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00'])
def test_absent_gps_is_not_zero_or_a_guessed_location(exif):
    result = inspect_jpeg(jpeg(exif))
    assert result['gps_status'] == 'absent'
    assert result['gps'] is None


@pytest.mark.parametrize('exif', [b'bad', gps_exif(denominator=0), gps_exif(references=('X', 'E')), gps_exif()[:-4]])
def test_bad_exif_is_unreadable_not_absent_or_invented(exif):
    result = inspect_jpeg(jpeg(exif))
    assert result['gps_status'] == 'unreadable'
    assert result['gps'] is None


@pytest.mark.parametrize('data', [b'not a jpeg', b'\xff\xd8\xff', b'\xff\xd8\xff\xe1\xff\xff'])
def test_malformed_jpeg_is_rejected(data):
    with pytest.raises(ValueError):
        inspect_jpeg(data)


def test_uri_exact_scope_read_only_and_symlink_denial(tmp_path):
    photo = tmp_path / 'photo space.jpg'
    photo.write_bytes(jpeg(gps_exif()))
    before = photo.read_bytes()
    prompt = photo.as_uri()
    assert literal_image_paths(prompt) == (str(photo),)
    result = read_image_metadata(str(photo), prompt)
    assert result['gps_status'] == 'present'
    assert photo.read_bytes() == before
    foreign = tmp_path / 'other.jpg'
    foreign.write_bytes(before)
    with pytest.raises(PermissionError):
        read_image_metadata(str(foreign), prompt)
    link = tmp_path / 'link.jpg'
    link.symlink_to(photo)
    with pytest.raises(PermissionError):
        read_image_metadata(str(link), link.as_uri())
    assert literal_image_paths('file://foreignhost/tmp/a.jpg /tmp/../a.jpg') == ()


def test_large_metadata_text_is_bounded_and_marked():
    text = b'A' * 4000 + b'\0'
    exif = b'II' + struct.pack('<HIH', 42, 8, 1)
    exif += struct.pack('<HHII', 271, 2, len(text), 26) + struct.pack('<I', 0) + text
    result = inspect_jpeg(jpeg(exif))
    assert len(result['make']) == 128
    assert result['metadata_text_truncated'] is True
    assert len(json.dumps(result)) < 1000


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['locate', 'resize'])
async def test_photo_location_creation_checks_real_metadata_before_model_code(tmp_path, operation):
    photos = [tmp_path / f'location_{n}.jpg' for n in range(3)]
    for photo in photos:
        photo.write_bytes(jpeg())
    original = [p.read_bytes() for p in photos]
    objective = ('Create a tool or a skill to locate where these photos were taken.'
                 if operation == 'locate' else 'Create a tool to resize these photos.')
    prompt = ' '.join(p.as_uri() for p in photos) + ' ' + objective
    real = MCPTools(SimpleNamespace(filesystem_server=['/usr/bin/false'], browser_server=['/usr/bin/false'],
        workspace=tmp_path / 'workspace', learning_root=tmp_path / 'learning', learning_profile='client', evm_profile='client'))
    real.begin_interaction('photos', prompt)

    class Tools:
        calls = []
        async def openai_tool_definitions(self):
            return [d for d in real._local_tool_definitions() if d['function']['name'] in ('image_metadata', 'learning_create_tool')]
        async def call(self, tool, arguments):
            self.calls.append(tool)
            assert tool == 'image_metadata'
            return await real.call(tool, arguments)

    class Model:
        config = SimpleNamespace(context=12_000, model='text-only')
        async def respond(self, **kwargs):
            assert operation == 'resize', 'Missing visual processing must not generate substitute code'
            from v_core.llm.llm import LLMResponse
            return LLMResponse(content="def run(arguments):\n    return {'width': arguments['width']}")

    class Memory:
        session = Session()
        async def process(self, *args, **kwargs):
            pass

    agent = object.__new__(Agent)
    agent.tools, agent.llm, agent.memory = Tools(), Model(), Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: 'system'
    agent._agent_trace_root = tmp_path / 'traces'
    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)
    assert agent.tools.calls == ['image_metadata'] * 3
    if operation == 'locate':
        assert answer.count('no GPS coordinates stored') == 3
        assert 'not been created' in answer and 'not visual recognition' in answer
    else:
        assert 'known correct result' in answer
        assert 'width' in answer
        assert 'photo-matching' not in answer
    assert [p.read_bytes() for p in photos] == original
    checkpoint = json.loads(next((tmp_path / 'traces' / 'checkpoints').glob('*.json')).read_text())
    assert checkpoint['status'] == 'awaiting_owner'
    assert len(checkpoint['tool_calls']) == 3
    assert real.learning.active_tool_names() == []
