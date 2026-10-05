from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.memory.session import Session
from v_core.persona.grounding import claims_physical_experience
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.persona.language import looks_non_english
from v_core.relationship import RelationshipState


@pytest.mark.parametrize("answer", ["BURSZTYN-42 — gotowy.", "COPPER-19 — gotowa.", "Gotowe."])
def test_short_polish_acknowledgement_is_not_language_neutral(answer):
    assert looks_non_english(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt,expected", [
    ("Reply only with the current label.", "COPPER-19"),
    ("Odpowiedz tylko tym hasłem.", "COPPER-19"),
    ("What is the label?", "COPPER-19."),
])
async def test_literal_only_reply_obeys_format_without_changing_the_identifier(prompt, expected):
    class LLMStub:
        async def ask(self, **kw):
            return "COPPER-19."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.memory = SimpleNamespace(session=Session(), relationship_state=RelationshipState())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    assert await agent._run_light_chat(prompt, None, remember=False) == expected


@pytest.mark.parametrize("text", [
    "The password makes me think I heard a sound at home today.",
    "I woke up in my bedroom.", "I smelled coffee.", "I was at home.",
    "Hasło przypomina mi, że zasłyszałem dziś dźwięk w domu.",
    "Usłyszałam dźwięk w pokoju.", "Spałam całą noc.",
])
def test_detects_literal_physical_autobiography(text):
    assert claims_physical_experience(text)


@pytest.mark.parametrize("text", [
    "I heard you, Boss. The marker is BURSZTYN-42.",
    "If I heard a sound, I would check its source.",
    'Boss wrote: "I heard a sound at home".',
    "Boss napisał: „Usłyszałam dźwięk w pokoju”.",
    "I haven't heard any audio in this text chat.",
    "My code is running hot; that bug is ridiculous.",
    "You said you heard a sound at home.",
])
def test_does_not_flatten_idioms_quotes_or_hypotheticals(text):
    assert not claims_physical_experience(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["BURSZTYN-42", "I heard a sound at home.", "I ran the tests."])
async def test_repair_is_bounded_and_rejected_draft_never_streams_or_persists(repair):
    rejected = "I heard a sound at home today."
    asks = []

    class LLMStub:
        async def stream(self, **kw):
            yield rejected

        async def ask(self, **kw):
            asks.append(kw)
            assert rejected not in str(kw["messages"])
            assert "BURSZTYN-42" in str(kw["messages"])
            return repair

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.memory = SimpleNamespace(session=Session(), relationship_state=RelationshipState())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    emitted = []
    result = await agent._run_light_chat(
        "The marker is BURSZTYN-42.", emitted.append, process_memory=False,
    )
    assert len(asks) == 1
    assert emitted == [result]
    assert rejected not in str(agent.memory.session.messages())
    if repair == "BURSZTYN-42":
        assert result == repair
    else:
        assert "couldn't produce a grounded answer" in result


@pytest.mark.asyncio
async def test_requested_fiction_keeps_physical_first_person_narration():
    story = "I woke up in my bedroom. Rain hammered the windows."

    class LLMStub:
        async def ask(self, **kw):
            return story

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.memory = SimpleNamespace(session=Session(), relationship_state=RelationshipState())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    result = await agent._run_light_chat(
        "Write a first-person fictional scene.", None, creative_response=True, remember=False,
    )
    assert result == story


@pytest.mark.asyncio
@pytest.mark.parametrize("creative", [False, True])
async def test_identifier_data_takes_priority_over_banter_except_in_requested_fiction(creative):
    class LLMStub:
        async def ask(self, **kw):
            policy = any("Task fidelity comes before persona banter" in m["content"] for m in kw["messages"])
            assert policy is not creative
            return "CODE-77"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.memory = SimpleNamespace(session=Session(), relationship_state=RelationshipState())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    assert await agent._run_light_chat(
        "Use CODE-77.", None, remember=False, creative_response=creative,
    ) == "CODE-77"


@pytest.mark.asyncio
async def test_factual_reply_is_not_rewritten_only_to_add_personality():
    class LLMStub:
        async def ask(self, **kw):
            raise AssertionError("no style-only rewrite of a factual acknowledgement")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    answer = "Certainly, Boss. CODE-77 noted."
    assert await agent._enforce_english(
        [{"role": "user", "content": "Use CODE-77."}], answer,
        preserve_factual_style=True,
    ) == answer


@pytest.mark.parametrize("claim", [
    "I'm live. The secure API is responding. One second. 🚀",
    "I did. The API is live, Boss. One sec—let me pull the output. 🚀",
    "The script is running in the background.",
    "I tested the API.",
    "Zrobiłam to. API już działa.",
    "Wykonałam to.",
])
@pytest.mark.asyncio
async def test_unexecuted_runtime_status_never_streams_or_becomes_visible_memory(claim):
    class Model:
        async def ask(self, **kwargs): return claim
        async def stream(self, **kwargs): yield claim
    agent = object.__new__(Agent)
    agent.llm = Model()
    agent.memory = SimpleNamespace(session=Session(), relationship_state=RelationshipState())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    emitted = []
    answer = await agent._run_light_chat("Did you do it?", emitted.append, process_memory=False)
    assert "No tools ran" in answer or "language pass mangled" in answer
    from v_core.persona.grounding import claims_unobserved_runtime_status
    assert claims_unobserved_runtime_status(claim, prompt="Did you do it?")
    assert emitted == [answer]
    assert claim not in str(agent.memory.session.messages())


@pytest.mark.parametrize("text", [
    "Want me to test the API?", "I would test the API after you provide a target.",
    "The API is not live.", "No API is running.", "Is the API live?",
    'She claimed: "The API is live".', "If the API is live, we can inspect it.",
    "I'm alive and kicking, Boss.", "Gotowe.", "I did not run anything.", "I did.",
])
def test_runtime_status_guard_preserves_plans_negatives_quotes_and_banter(text):
    from v_core.persona.grounding import claims_unobserved_runtime_status
    assert not claims_unobserved_runtime_status(text)


@pytest.mark.asyncio
async def test_runtime_status_in_explicit_fiction_remains_fiction():
    class Model:
        async def ask(self, **kwargs): return "The API is live. I did it."
    agent = object.__new__(Agent)
    agent.llm = Model()
    agent.memory = SimpleNamespace(session=Session(), relationship_state=RelationshipState())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    assert await agent._run_light_chat("Write a fictional hacker scene.", None,
        creative_response=True, remember=False) == "The API is live. I did it."


@pytest.mark.asyncio
async def test_understanding_acknowledgement_does_not_claim_tool_execution():
    class Model:
        async def ask(self, **kwargs): return "I did."
    agent = object.__new__(Agent)
    agent.llm = Model()
    agent.memory = SimpleNamespace(session=Session(), relationship_state=RelationshipState())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    assert await agent._run_light_chat("Did you understand?", None, remember=False) == "I did."
