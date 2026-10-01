from types import SimpleNamespace

import pytest

from v_core.llm.llm import LLM, IncompleteGenerationError


def make_llm(create):
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="test", temperature=0.2, top_p=0.95)
    llm._native_tools_supported = None
    llm.client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    return llm


@pytest.mark.asyncio
async def test_text_api_does_not_present_token_limit_as_complete_answer():
    async def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Unfinished sentence", tool_calls=[]),
            finish_reason="length")])

    with pytest.raises(IncompleteGenerationError, match="incomplete"):
        await make_llm(create).ask(messages=[{"role": "user", "content": "Explain"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "length"])
async def test_stream_distinguishes_success_from_truncation_and_closes(reason):
    class Stream:
        closed = False

        def __aiter__(self):
            return self.chunks()

        async def chunks(self):
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Some output"), finish_reason=None)])
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None), finish_reason=reason)])

        async def close(self):
            self.closed = True

    stream = Stream()

    async def create(**kwargs):
        return stream

    async def consume():
        return "".join([part async for part in make_llm(create).stream(
            messages=[{"role": "user", "content": "Explain"}])])

    if reason == "length":
        with pytest.raises(IncompleteGenerationError):
            await consume()
    else:
        assert await consume() == "Some output"
    assert stream.closed
