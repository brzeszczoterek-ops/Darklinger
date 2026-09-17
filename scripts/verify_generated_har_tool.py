"""Opt-in LOCAL model trial; synthetic HAR only, isolated persistent artifacts.

Run from the repo: .venv/bin/python scripts/verify_generated_har_tool.py
No implementation is supplied to the model. Expected answers below are fixed
before generation. The holdout is never included in its prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from urllib.request import urlopen

from v_core.agent import Agent
from v_core.llm.llm import LLM
from v_core.mcp_tools import MCPTools
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile


def har(rows):
    return {'har_json': json.dumps({'log': {'version': '1.2', 'entries': [
        {'request': {'url': url, 'method': 'GET'}, 'response': {'status': status}, 'time': duration}
        for url, status, duration in rows
    ]}})}


def expected(count, domains, errors, duration):
    return dict(requests_count=count, domains=domains, error_count=errors, total_time_ms=duration)


CASES = [
    (har([('https://shop.example/a', 200, 12.5), ('https://cdn.example/a', 404, 7.5),
          ('https://shop.example/b', 500, -1)]), expected(3, ['cdn.example', 'shop.example'], 2, 20.0)),
    (har([]), expected(0, [], 0, 0.0)),
    (har([('https://news.example/x', 302, 40.0)]), expected(1, ['news.example'], 0, 40.0)),
]
FINAL = har([('https://media.example/x', 503, 5.5), ('https://media.example/y', 200, 4.5)])
FINAL_EXPECTED = expected(2, ['media.example'], 1, 10.0)
HOLDOUT = har([('https://unseen.example:8443/a', 401, 0.25),
               ('https://other.example/a?q=x', 204, 1.75), ('https://unseen.example/b', 301, -1)])
HOLDOUT_EXPECTED = expected(3, ['other.example', 'unseen.example'], 1, 2.0)


def prompt():
    return (
        'Create a reusable local tool named har_summary. Parse the HAR JSON string '
        'in arguments["har_json"]. Read log.entries. Return requests_count (all entries), '
        'domains (sorted unique lowercase hostname from request.url, excluding ports), '
        'error_count (response.status >= 400), total_time_ms (sum nonnegative time in ms; '
        'ignore negative time). This analyzes supplied data only, never accesses the network. '
        'Implement the general calculation, not fixture matching. Test against these fixed '
        'examples, activate only if they pass, then execute with final_arguments and report. '
        'tool_contract=' + json.dumps({
            'tests': [{'arguments': args, 'expected': result} for args, result in CASES],
            'final_arguments': FINAL,
        })
    )


class Memory:
    def __init__(self):
        self.session = Session()

    async def process(self, *args, **kwargs):
        pass


def measure_completions(llm, measurements, *, cold_context=False):
    """Record provider counts/timings, not reasoning text or additional requests."""
    create = llm.client.chat.completions.create

    async def measured(**request):
        if cold_context:
            request['extra_body'] = {**request.get('extra_body', {}), 'cache_prompt': False}
        started = time.monotonic()
        metric = {
            'phase': 'source' if any('SOURCE PHASE' in str(m.get('content', ''))
                                   for m in request.get('messages', [])) else 'other',
            'input_chars': sum(len(str(m.get('content', '')))
                               for m in request.get('messages', [])),
            'sampling': {key: request[key] for key in ('temperature', 'top_p', 'max_tokens')
                         if key in request},
        }
        measurements.append(metric)
        try:
            response = await create(**request)
            if response.usage is not None:
                metric['usage'] = response.usage.model_dump()
            timings = getattr(response, 'timings', None)
            if isinstance(timings, dict):
                metric['provider_timings'] = timings
            metric['finish_reason'] = response.choices[0].finish_reason
            return response
        finally:
            metric['elapsed_seconds'] = round(time.monotonic() - started, 3)
            print('Completion timing: ' + json.dumps(metric), flush=True)

    llm.client.chat.completions.create = measured


async def main(*, cold_context=False):
    # A fixed loopback address deliberately prevents accidental remote API usage.
    with urlopen('http://127.0.0.1:5001/slots', timeout=5) as response:
        slots = json.load(response)
    if any(slot.get('is_processing') for slot in slots):
        raise RuntimeError('Local model is busy; no test was started.')
    with urlopen('http://127.0.0.1:5001/v1/models', timeout=5) as response:
        model = json.load(response)['data'][0]['id']
    os.environ.update(V_CORE_BASE_URL='http://127.0.0.1:5001/v1', V_CORE_MODEL=model,
                      V_CORE_CONTEXT='16000', V_CORE_TEMPERATURE='0.0', V_CORE_TIMEOUT='120')
    root = Path(tempfile.mkdtemp(prefix='paladyn-har-trial-'))
    print(f'Trial artifacts: {root}', flush=True)
    config = SimpleNamespace(
        filesystem_server=['/usr/bin/false'], browser_server=['/usr/bin/false'],
        workspace=root / 'workspace', learning_root=root / 'learning',
        autonomy_root=root / 'autonomy', learning_profile='client', evm_profile='client',
    )
    tools = MCPTools(config)
    agent = object.__new__(Agent)
    agent.tools, agent.llm, agent.memory = tools, LLM(), Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: 'Report only actual execution results.'
    agent._agent_trace_root = root / 'traces'
    report = {'model': model, 'root': str(root), 'passed': False}
    report['completions'] = []
    report['cold_context'] = cold_context
    measure_completions(agent.llm, report['completions'], cold_context=cold_context)
    # Snapshot the server settings so runs with different reasoning/slot/cache
    # settings cannot silently be presented as a prompt-only speed comparison.
    with urlopen('http://127.0.0.1:5001/props', timeout=5) as response:
        report['server_settings'] = json.load(response).get('default_generation_settings', {})
    report['server_slots'] = len(slots)
    started = time.monotonic()
    try:
        # The outer trial also covers final reporting. The executor separately
        # bounds source generation and the shared creation budget.
        answer = await asyncio.wait_for(agent._run_agent_loop(prompt()), timeout=900)
        report['answer'] = answer
        print(answer, flush=True)
        checkpoints = list((root / 'traces' / 'checkpoints').glob('*.json'))
        assert len(checkpoints) == 1, checkpoints
        checkpoint = json.loads(checkpoints[0].read_text(encoding='utf-8'))
        report['task_status'] = checkpoint['status']
        report['executed_tools'] = [call['tool'] for call in checkpoint['tool_calls']
                                    if call['status'] == 'succeeded']
        assert report['task_status'] == 'completed', report['task_status']
        assert 'learning_create_tool' in report['executed_tools'], report['executed_tools']
        assert 'har_summary' in report['executed_tools'], report['executed_tools']
        assert tools.learning is not None
        report['final'] = await tools.learning.execute_tool('har_summary', FINAL)
        assert report['final'] == FINAL_EXPECTED, report['final']
        report['holdout'] = await tools.learning.execute_tool('har_summary', HOLDOUT)
        assert report['holdout'] == HOLDOUT_EXPECTED, report['holdout']
        # New provider instance proves reuse from disk, not a one-off closure.
        reloaded = MCPTools(config)
        report['after_reload'] = await reloaded.learning.execute_tool('har_summary', HOLDOUT)
        assert report['after_reload'] == HOLDOUT_EXPECTED
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        await asyncio.gather(*getattr(agent, '_memory_tasks', []), return_exceptions=True)
        await agent.llm.client.close()
        (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({key: value for key, value in report.items()
                          if key != 'server_settings'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cold-context', action='store_true',
                        help='Disable llama.cpp prompt-cache reuse for a comparable timing trial.')
    asyncio.run(main(cold_context=parser.parse_args().cold_context))
