"""Benchmark instrumentation must observe the real call, not replay its answer."""
import asyncio
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


measure_completions = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / 'scripts' / 'verify_generated_har_tool.py'
))['measure_completions']


@pytest.mark.asyncio
@pytest.mark.parametrize('cold', [False, True])
async def test_measure_completion_preserves_response_and_request(cold):
    calls = []
    response = SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {'prompt_tokens': 100}),
                               timings={'cache_n': 0}, choices=[SimpleNamespace(finish_reason='stop')])

    async def create(**request):
        calls.append(request)
        return response

    completions = SimpleNamespace(create=create)
    llm = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    metrics = []
    measure_completions(llm, metrics, cold_context=cold)
    original = {'messages': [{'role': 'system', 'content': 'SOURCE PHASE'}],
                'extra_body': {'custom': 'preserved'}, 'max_tokens': 1536}
    assert await completions.create(**original) is response
    assert len(calls) == 1
    assert calls[0]['max_tokens'] == 1536
    assert calls[0]['extra_body']['custom'] == 'preserved'
    assert ('cache_prompt' in calls[0]['extra_body']) is cold
    assert original['extra_body'] == {'custom': 'preserved'}
    assert metrics[0]['phase'] == 'source'
    assert metrics[0]['usage']['prompt_tokens'] == 100
    assert metrics[0]['elapsed_seconds'] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeError('provider failed'), asyncio.CancelledError()])
async def test_measure_completion_records_failure_and_propagates_it(error):
    async def create(**request):
        raise error

    completions = SimpleNamespace(create=create)
    llm = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    metrics = []
    measure_completions(llm, metrics)
    with pytest.raises(type(error)):
        await completions.create(messages=[])
    assert len(metrics) == 1
    assert metrics[0]['elapsed_seconds'] >= 0
    assert 'finish_reason' not in metrics[0]
