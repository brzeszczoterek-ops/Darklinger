import json
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy.intent import MultilingualIntentRouter
from v_core.llm.llm import IncompleteGenerationError
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.relationship import RelationshipState


@pytest.mark.asyncio
async def test_truncated_classification_retries_same_request_with_more_room():
    calls = []

    class LLMStub:
        async def ask(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise IncompleteGenerationError()
            return json.dumps({
                "message_clear": True, "action_requested": False,
                "continue_previous": False, "capabilities": [],
            })

    intent = await MultilingualIntentRouter(LLMStub()).classify("Hey V.")
    assert intent is not None and not intent.action_requested
    assert [c["max_tokens"] for c in calls] == [2048, 4096]
    assert calls[0]["messages"] == calls[1]["messages"]
    assert calls[0]["response_format"] == calls[1]["response_format"]


@pytest.mark.asyncio
async def test_repeated_truncation_remains_a_failure_after_bounded_retry():
    budgets = []

    class LLMStub:
        async def ask(self, **kwargs):
            budgets.append(kwargs["max_tokens"])
            raise IncompleteGenerationError()

    router = MultilingualIntentRouter(LLMStub())
    with pytest.raises(IncompleteGenerationError):
        await router.classify("Pick a test example.")
    assert budgets == [2048, 4096]
    assert router.last_failure_reason == "output_token_limit"


@pytest.mark.asyncio
async def test_invalid_json_repair_uses_larger_budget_without_unbounded_retry():
    budgets = []

    class LLMStub:
        async def ask(self, **kwargs):
            budgets.append(kwargs["max_tokens"])
            if len(budgets) == 1:
                return "invalid JSON"
            raise IncompleteGenerationError()

    with pytest.raises(IncompleteGenerationError):
        await MultilingualIntentRouter(LLMStub()).classify("Hey V.")
    assert budgets == [2048, 4096]


@pytest.mark.asyncio
async def test_truncated_chat_draft_is_never_displayed_or_remembered():
    budgets = []
    final = "There it is, Boss. One complete answer."

    class LLMStub:
        async def stream(self, **kwargs):
            budgets.append(kwargs["max_tokens"])
            if len(budgets) == 1:
                yield "Rejected unfinished draft."
                raise IncompleteGenerationError()
            yield final

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.memory = SimpleNamespace(
        session=Session(), relationship_state=RelationshipState(),
    )
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    emitted = []
    answer = await agent._run_light_chat(
        "Give me a brief reply, no tools.", emitted.append, process_memory=False,
    )
    assert budgets == [1024, 2048]
    assert answer == final
    assert emitted == [final]
    assert "Rejected unfinished draft" not in str(agent.memory.session.messages())
