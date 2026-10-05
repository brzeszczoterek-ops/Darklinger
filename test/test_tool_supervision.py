from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from v_core.agent import Agent
from v_core.mcp_tools import MCPTools
from v_core.tool_recovery import ToolCallOutcome, ToolRecoveryRegistry
from v_core.tool_supervision import ToolSupervisor, report_progress


def outcome(result='{"ok":true}', error=''):
    return ToolCallOutcome(result=result, requested_tool='full_tor_fetch',
                           provider_tool='full_tor_fetch', capabilities=('network.tor.fetch',), error=error)


@pytest.mark.asyncio
@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('failed', [False, True])
async def test_completed_job_duration_stops_advancing(tmp_path, monkeypatch, observed, failed):
    manager = ToolSupervisor(tmp_path / 'monitor')
    async def operation():
        return outcome(error='controlled failure' if failed else '')
    try:
        if observed:
            await manager.observe_call('test', {}, 'turn', operation)
            job = next(iter(manager.jobs.values()))
        else:
            job = manager.start('test', {}, 'turn', operation)
            await job.task
        before = manager.status(job.id)
        assert job.finished is not None
        with monkeypatch.context() as patch:
            patch.setattr('v_core.tool_supervision.time.monotonic', lambda: job.finished + 100)
            after = manager.status(job.id)
        assert before['elapsed_seconds'] == after['elapsed_seconds']
        assert after['state'] == ('failed' if failed else 'succeeded')
        assert after['last_signal_age_seconds'] >= 100
    finally:
        await manager.close()


def test_unstarted_monitor_does_not_claim_the_application_is_hung(tmp_path):
    manager = ToolSupervisor(tmp_path / 'monitor')
    manager.last_heartbeat -= 86400
    snapshot = manager.snapshot()
    assert snapshot['event_loop_state'] == 'not_started'
    assert snapshot['heartbeat_age_seconds'] is None


@pytest.mark.asyncio
async def test_application_heartbeat_works_before_first_tool(tmp_path):
    manager = ToolSupervisor(tmp_path / 'monitor')
    manager.last_heartbeat -= 86400
    manager.start_monitoring()
    heartbeat = manager.heartbeat
    manager.start_monitoring()
    assert manager.heartbeat is heartbeat
    try:
        await asyncio.sleep(.7)
        snapshot = json.loads((tmp_path / 'monitor/latest.json').read_text())
        assert snapshot['event_loop_state'] == 'responding'
        assert snapshot['jobs'] == []
        assert snapshot['heartbeat_age_seconds'] < 1
        assert not snapshot['automatic_cancellation']
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_old_running_job_survives_observation_and_model_failure(tmp_path):
    manager = ToolSupervisor(tmp_path / 'monitor')
    release = asyncio.Event()

    async def operation():
        report_progress('waiting_network')
        await release.wait()
        report_progress('receiving_data', bytes_received=512)
        return outcome()

    job = manager.start('full_tor_fetch', {'secret': 'must-not-be-persisted'}, 'turn', operation)
    try:
        await asyncio.sleep(0)
        job.started -= 86400
        snapshot = await manager.wait_status(job.id)
        assert snapshot['state'] == 'running'
        assert not snapshot['completion_verified']
        assert not snapshot['automatic_deadline']
        assert not job.task.cancelled()
        # A failed inference task does not own or cancel the tool task.
        async def broken_model():
            raise RuntimeError('model process failed')
        with pytest.raises(RuntimeError):
            await broken_model()
        assert job.state == 'running'
        await asyncio.sleep(.6)
        persisted = (tmp_path / 'monitor/latest.json').read_text()
        assert 'must-not-be-persisted' not in persisted
        release.set()
        await job.task
        assert manager.status(job.id)['completion_verified']
        assert manager.status(job.id)['signals']['bytes_received'] == 512
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_independent_monitor_reports_blocked_loop_without_killing_work(tmp_path):
    manager = ToolSupervisor(tmp_path / 'monitor')
    release = asyncio.Event()

    async def operation():
        await release.wait()
        return outcome()

    job = manager.start('full_tor_fetch', {}, 'turn', operation)
    try:
        await asyncio.sleep(.6)
        time.sleep(3.7)  # Deliberately freeze this isolated TEST loop.
        persisted = json.loads((tmp_path / 'monitor/latest.json').read_text())
        assert persisted['event_loop_state'] == 'unresponsive_suspected'
        assert persisted['automatic_cancellation'] is False
        assert job.state == 'running'
        release.set()
        await job.task
        assert job.state == 'succeeded'
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_explicit_stop_and_real_failure_are_distinct_from_success(tmp_path):
    manager = ToolSupervisor(tmp_path / 'monitor')

    async def operation():
        await asyncio.Event().wait()

    job = manager.start('full_tor_fetch', {}, 'turn', operation)
    await asyncio.sleep(0)
    manager.cancel(job.id)
    await job.task
    assert manager.status(job.id)['state'] == 'cancelled'
    assert not manager.status(job.id)['completion_verified']

    async def failure():
        return outcome(error='provider failed')

    failed = manager.start('full_tor_fetch', {}, 'turn', failure)
    await failed.task
    assert manager.status(failed.id)['state'] == 'failed'
    assert not manager.status(failed.id)['completion_verified']
    await manager.close()


@pytest.mark.asyncio
async def test_v_gets_pending_receipt_then_collects_trusted_result_without_duplicate_launch(tmp_path):
    tools = object.__new__(MCPTools)
    tools.supervisor = ToolSupervisor(tmp_path / 'monitor')
    tools.recovery = ToolRecoveryRegistry(tmp_path / 'recovery')
    tools.interaction_id = 'turn'
    tools.edition_extension = SimpleNamespace(tool_names=lambda: ['full_tor_fetch'])
    tools._register_recovery_providers = lambda: tools.recovery.register_provider('full_tor_fetch', ('network.tor.fetch',))
    release = asyncio.Event()
    calls = []

    async def call_provider(tool, arguments):
        calls.append((tool, arguments))
        await release.wait()
        return '{"ok":true,"content":"actual observation"}'

    tools._call_direct = call_provider
    arguments = {'url': 'http://' + 'a' * 56 + '.onion/'}
    try:
        first = await tools.call_with_recovery('full_tor_fetch', arguments)
        second = await tools.call_with_recovery('full_tor_fetch', arguments)
        job_id = json.loads(first.result)['job']['job_id']
        assert json.loads(second.result)['job']['job_id'] == job_id
        assert tools.supervised_receipt('full_tor_fetch', arguments, first.result)['pending']
        assert not json.loads(first.result)['job']['completion_verified']
        release.set()
        await tools.supervisor.get(job_id).task
        receipt = tools.supervised_receipt('runtime_tool_status', {'job_id': job_id}, '{}')
        assert receipt['tool'] == 'full_tor_fetch'
        assert receipt['arguments'] == arguments
        assert receipt['outcome'].result == '{"ok":true,"content":"actual observation"}'
        assert not receipt['pending']
        assert len(calls) == 1
        assert tools.supervised_receipt('runtime_tool_status', {'job_id': job_id}, '{}') is None
    finally:
        await tools.close_supervisor()


@pytest.mark.asyncio
async def test_commands_can_inspect_and_stop_without_running_inference(tmp_path):
    manager = ToolSupervisor(tmp_path / 'monitor')
    agent = object.__new__(Agent)
    agent.tools = SimpleNamespace(supervisor=manager)

    async def operation():
        await asyncio.Event().wait()

    job = manager.start('full_tor_fetch', {}, 'turn', operation)
    await asyncio.sleep(0)
    payload = json.loads(await agent._run_turn('/jobs'))
    assert payload['jobs'][0]['job_id'] == job.id
    await agent._run_turn('/stop-job ' + job.id)
    await job.task
    assert job.state == 'cancelled'
    await manager.close()


@pytest.mark.asyncio
async def test_ordinary_tool_is_monitored_without_moving_mcp_context_between_tasks(tmp_path):
    manager = ToolSupervisor(tmp_path / 'monitor')
    owner = asyncio.current_task()

    async def ordinary():
        assert asyncio.current_task() is owner
        snapshot = manager.snapshot()
        assert snapshot['jobs'][0]['state'] == 'running'
        return outcome()

    receipt = await manager.observe_call('browser_snapshot', {}, 'turn', ordinary)
    assert not receipt.error
    assert manager.snapshot()['jobs'][0]['state'] == 'succeeded'
    await manager.close()  # Must not await or cancel its own caller task.


@pytest.mark.asyncio
async def test_supervised_real_sandbox_waits_past_old_deadline_and_keeps_output_guard(tmp_path):
    import shutil
    from v_core.sandbox import BubblewrapBackend, SandboxSpec, SandboxLimits
    if shutil.which('bwrap') is None:
        pytest.skip('bubblewrap required')
    manager = ToolSupervisor(tmp_path / 'monitor')
    backend = BubblewrapBackend()
    limits = SandboxLimits(timeout_seconds=.01, max_output_bytes=128)
    async def execute(command):
        result = await backend.run(SandboxSpec(command=command, workspace=tmp_path / 'sandbox', limits=limits))
        return outcome(result=result.stdout, error='' if result.succeeded else 'sandbox resource failure')
    try:
        job = manager.start('learning_execute_tool', {}, 'turn', lambda: execute(
            ('/usr/bin/python3', '-c', 'import time; time.sleep(.15); print("completed")')))
        await job.task
        assert job.state == 'succeeded'
        assert job.outcome.result.strip() == 'completed'
        assert not job.measurements['process_alive']
        big = manager.start('learning_execute_tool', {}, 'turn', lambda: execute(
            ('/usr/bin/python3', '-c', 'print("x"*2000)')))
        await big.task
        assert big.state == 'failed'
        assert not manager.status(big.id)['completion_verified']
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [False, True])
async def test_agent_status_polling_never_counts_pending_as_verified_or_restarts_tool(tmp_path, complete):
    from v_core.llm.llm import LLMResponse, LLMToolCall
    from v_core.memory.session import Session
    from v_core.persona.runtime import PersonaRuntime
    from v_core.persona.kernel import IdentityKernel
    from v_core.persona.voice import VoiceProfile
    manager = ToolSupervisor(tmp_path / 'monitor')
    underlying = object.__new__(MCPTools)
    underlying.supervisor = manager
    underlying.recovery = ToolRecoveryRegistry(tmp_path / 'recovery')
    underlying.interaction_id = 'turn'
    underlying.edition_extension = SimpleNamespace(tool_names=lambda: ['full_tor_fetch'])
    underlying._register_recovery_providers = lambda: underlying.recovery.register_provider('full_tor_fetch', ('network.tor.fetch',))
    calls = []
    release = asyncio.Event()
    async def provider(name, arguments):
        if name.startswith('runtime_'):
            return await MCPTools._call_direct(underlying, name, arguments)
        calls.append(name)
        await release.wait()
        return json.dumps({"ok": True, "content": "Observed page content", "url": arguments["url"], "status_code": 200})
    underlying._call_direct = provider
    class Tools:
        supervisor = manager
        interaction_id = 'turn'
        call_with_recovery = underlying.call_with_recovery
        supervised_receipt = underlying.supervised_receipt
        async def openai_tool_definitions(self):
            return [{'type':'function', 'function':{'name':name, 'parameters':{'type':'object'}}}
                    for name in ('full_tor_fetch', 'runtime_tool_status', 'runtime_tool_cancel')]
    class Model:
        config = SimpleNamespace(context=12000)
        turns = 0
        async def respond(self, **kwargs):
            self.turns += 1
            if self.turns == 1:
                return LLMResponse(tool_calls=[LLMToolCall('start', 'full_tor_fetch', {'url':'http://'+'a'*56+'.onion/'})])
            if self.turns <= 5:
                return LLMResponse(tool_calls=[LLMToolCall('poll'+str(self.turns), 'runtime_tool_status', {'job_id':next(iter(manager.jobs))})])
            if complete and self.turns == 6:
                release.set()
                await next(iter(manager.jobs.values())).task
                return LLMResponse(tool_calls=[LLMToolCall('finished', 'runtime_tool_status', {'job_id':next(iter(manager.jobs))})])
            return LLMResponse(content='Observed page content.' if complete else 'Task completed successfully.')
    class Memory:
        session = Session()
        async def process(self, *args, **kwargs):
            pass
    agent = object.__new__(Agent)
    agent.tools, agent.llm, agent.memory = Tools(), Model(), Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: 'system'
    agent._agent_trace_root = tmp_path / 'traces'
    agent.MAX_AGENT_STEPS = 8
    try:
        answer = await agent._run_agent_loop('Fetch http://'+'a'*56+'.onion/ and report its content.')
        assert 'Task completed successfully' not in answer
        assert calls == ['full_tor_fetch']
        if not complete:
            assert 'still running in the background' in answer
            assert agent.llm.turns == 6
            assert next(iter(manager.jobs.values())).state == 'running'
        checkpoint = json.loads(next((tmp_path/'traces/checkpoints').glob('*.json')).read_text())
        if not complete:
            assert all(c['status'] != 'succeeded' for c in checkpoint['tool_calls'])
            assert checkpoint['status'] != 'completed'
        else:
            receipts = [c for c in checkpoint['tool_calls'] if c['status'] == 'succeeded' and c['tool'] == 'full_tor_fetch']
            assert len(receipts) == 1
            assert receipts[0]['arguments']['url'].endswith('.onion/')
            assert 'Observed page content' in receipts[0]['result_excerpt']
            assert next(iter(manager.jobs.values())).state == 'succeeded'
    finally:
        await manager.close()
