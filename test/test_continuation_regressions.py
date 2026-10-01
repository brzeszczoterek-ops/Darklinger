"""Regressions from the owner's explanation -> continuation failure."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy import SemanticIntent, TaskContract
from v_core.autonomy.agent_trace import AgentTaskTrace
from v_core.autonomy.task_contract import (
    _claimed_github_repository_identifiers, _grounding_entity_is_present,
)
from v_core.llm.llm import LLMResponse, LLMToolCall
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile


URL = "https://example.test/explanation"
OBSERVATION = (
    f"- Page URL: {URL}\n- Page Title: Payment explanation\n"
    "- paragraph: The report describes buying/selling tokens. "
    "The displayed balance is not proof that a transfer settled."
)
CALLS = [{
    "tool": "browser_snapshot", "status": "succeeded",
    "arguments": {}, "result_excerpt": OBSERVATION,
}]
DRAFT = (
    "Certainly, Boss. The report describes buying/selling tokens.\n"
    "- The displayed balance is not proof that a transfer settled.\n"
    f"Source: {URL}\n\nWould you like me to explain more?"
)


@pytest.mark.parametrize("phrase", ["buying/selling", "sender/receiver", "input/output"])
def test_bullet_prose_is_not_a_github_repository(phrase):
    assert _claimed_github_repository_identifiers(f"- The report describes {phrase}.") == set()
    assert _claimed_github_repository_identifiers(f"1. The report describes {phrase}.") == set()


def test_repo_grounding_still_recognizes_explicit_and_contextual_identifiers():
    assert _claimed_github_repository_identifiers("- GitHub: owner/project") == {"owner/project"}
    assert _claimed_github_repository_identifiers(
        "- `owner/project`: a tool", repository_context=True,
    ) == {"owner/project"}
    assert _claimed_github_repository_identifiers("https://github.com/owner/project") == {"owner/project"}


def test_explanation_with_slash_prose_passes_grounding():
    request = f"Inspect {URL} and explain the mechanism."
    assert TaskContract.from_prompt(request).answer_issues(
        "The report describes buying/selling tokens.\n"
        "- A displayed balance is not proof that a transfer settled.",
        CALLS, request=request,
    ) == []


@pytest.mark.parametrize("heading", ["Answer", "Final Answer", "Summary"])
def test_report_labels_and_bullet_sentence_starts_are_not_product_claims(heading):
    request = f"Inspect {URL} and explain the mechanism."
    answer = (
        f"**{heading}:**\nThe report describes buying/selling tokens.\n"
        "- Always check the displayed balance.\n"
        "- Excerpt: The displayed balance is not proof that a transfer settled.\n"
        f"**Source evidence:**\n{URL}\n"
        "**Unverified:**\nThe report does not prove that a transfer settled."
    )
    assert TaskContract.from_prompt(request).answer_issues(answer, CALLS, request=request) == []
    assert TaskContract.from_prompt(request).answer_issues(
        "- **ImaginaryPlatform** can settle the transfer.", CALLS, request=request,
    ) == ["answer:ungrounded_online_claims=ImaginaryPlatform"]


def test_merged_source_words_require_adjacent_evidence():
    assert _grounding_entity_is_present("ProtectYour", "how to protect your assets")
    assert not _grounding_entity_is_present("ProtectYour", "protect assets, your choice")
    assert not _grounding_entity_is_present("InventedPlatform", "protect your assets")


def test_web_read_prioritizes_article_instead_of_clipped_json_navigation():
    article = "The mechanism uses a misleading balance, not a settled transfer."
    snapshot = (
        f"- Page URL: {URL}\n- Page Title: Payment mechanism\n### Snapshot\n"
        + ('- link "Navigation" [ref=e1]\n' * 400)
        + ('- link "Unrelated market ticker $85,000" [ref=e2]\n' * 50)
        + '- heading "Payment mechanism" [level=2]\n'
        + f'- paragraph: {article}\n'
        + '- list:\n  - listitem: Check whether the transfer actually settled.\n'
        + '- paragraph: An apparent balance does not settle a payment.\n'
    )
    raw = json.dumps({"content": snapshot})
    evidence = Agent._browser_evidence_text("web_read", raw)
    assert evidence == snapshot
    compact = Agent._fit_browser_snapshot_output(
        evidence, max_characters=6000, prioritize_prose=True,
    )
    assert article in compact
    assert "Check whether the transfer actually settled" in compact
    assert "Unrelated market ticker" not in compact
    assert len(compact) <= 6000
    report = Agent._owner_verified_final_report(
        None, [{"tool": "web_read", "status": "succeeded",
                "arguments": {"url": URL}, "result_excerpt": compact}],
        TaskContract.from_prompt(f"Inspect {URL} and explain the mechanism."),
    )
    assert article.rstrip(".") in report
    assert "[ref=" not in report
    assert Agent._browser_evidence_text("web_read", '{"content":"cut') is None
    assert Agent._browser_evidence_text("other_tool", raw) is None


def test_observed_url_label_repairs_mismatched_target_on_ambiguous_host():
    observed = "https://example.test/30057895/"
    wrong = "https://example.test/31057895/"
    calls = [{"tool": "web_read", "status": "succeeded",
              "result_excerpt": f"{observed}\nhttps://example.test/other-page"}]
    result, repairs = Agent._repair_unambiguous_observed_urls(
        f"The source explains the mechanism: [{observed}]({wrong})", calls,
    )
    assert result == f"The source explains the mechanism: [{observed}]({observed})"
    assert repairs == [{"from": wrong, "to": observed}]


@pytest.mark.parametrize("label,target", [
    ("https://example.test/not-observed", "https://example.test/mistyped"),
    ("https://example.test/observed", "https://other.test/mistyped"),
    ("https://example.test/observed", "https://example.test/also-observed"),
])
def test_ambiguous_links_are_not_guessed(label, target):
    calls = [{"tool": "web_read", "status": "succeeded", "result_excerpt":
              "https://example.test/observed\nhttps://example.test/also-observed"}]
    original = f"[{label}]({target})"
    result, repairs = Agent._repair_unambiguous_observed_urls(original, calls)
    assert result == original
    assert repairs == []


@pytest.mark.asyncio
async def test_tool_backed_answer_keeps_content_without_style_generation():
    class NoRewrite:
        async def ask(self, **kwargs):
            raise AssertionError("Style must not trigger another model generation")

    agent = object.__new__(Agent)
    agent.llm = NoRewrite()
    result = await agent._enforce_english(
        [{"role": "user", "content": "Explain the mechanism."}], DRAFT,
        verified_calls=CALLS, allow_verified_tool_fallback=False,
    )
    assert "displayed balance is not proof" in result
    assert "buying/selling" in result
    assert URL in result
    assert "Certainly" not in result
    assert "Would you like" not in result
    assert "voice pass" not in result


@pytest.mark.asyncio
async def test_failed_report_translation_cannot_become_a_source_ledger():
    class FailedTranslation:
        async def ask(self, **kwargs):
            assert kwargs["max_tokens"] >= 512
            return "Nadal odpowiadam po polsku, nie udało się przetłumaczyć."

    agent = object.__new__(Agent)
    agent.llm = FailedTranslation()
    result = await agent._enforce_english(
        [{"role": "user", "content": "Explain the mechanism."}],
        "To jest wyjaśnienie, które opisuje działanie tego mechanizmu po polsku.",
        verified_calls=CALLS, allow_verified_tool_fallback=False,
    )
    assert result == ""  # The caller must reject/retry, not mark sources completed.


class RecordedTools:
    def __init__(self):
        self.calls = []

    async def openai_tool_definitions(self):
        return [{"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")]

    async def call(self, tool, arguments):
        self.calls.append(tool)
        return OBSERVATION if tool == "browser_snapshot" else f"- Page URL: {URL}"


class RecordedModel:
    config = SimpleNamespace(context=8192)

    def __init__(self, candidate):
        self.candidate = candidate
        self.turn = 0
        self.messages = []

    async def respond(self, **kwargs):
        self.turn += 1
        self.messages = kwargs["messages"]
        if self.turn == 1:
            return LLMResponse(tool_calls=[LLMToolCall("nav", "browser_navigate", {"url": URL})])
        if self.turn == 2:
            return LLMResponse(tool_calls=[LLMToolCall("snap", "browser_snapshot", {})])
        if self.candidate == "tool_loop":
            return LLMResponse(tool_calls=[LLMToolCall("again", "browser_snapshot", {})])
        return LLMResponse(content=self.candidate)

    async def ask(self, **kwargs):
        raise AssertionError("No extra voice generation is needed for this report")


class MemoryStub:
    def __init__(self):
        self.session = Session()

    async def process(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate,expected_status", [
    (DRAFT, "completed"),
    ("Source: https://invented.test/not-observed", "blocked"),
    ("", "blocked"),
    ("tool_loop", "blocked"),
])
async def test_continuation_preserves_explanation_or_records_incomplete(
    tmp_path, candidate, expected_status,
):
    objective = f"Inspect {URL} and explain the mechanism."
    previous = AgentTaskTrace(tmp_path, objective)
    previous.set_requirements(TaskContract.from_prompt(objective).to_dict())
    previous.complete("Previous answer was incomplete.")

    class IntentRouter:
        async def classify(self, *args, **kwargs):
            return SemanticIntent(message_clear=True, action_requested=True,
                                  references_previous=True, continue_previous=True)

    agent = object.__new__(Agent)
    agent.llm = RecordedModel(candidate)
    agent.tools = RecordedTools()
    agent.intent_router = IntentRouter()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = AgentTaskTrace.latest_context(tmp_path)
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    result = await agent._run_agent_loop("kontynuj")
    await asyncio.gather(*agent._memory_tasks)

    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint["status"] == expected_status
    assert agent.tools.calls == ["browser_navigate", "browser_snapshot"]
    assert agent.llm.turn <= 4
    assert URL in result
    if expected_status == "completed":
        assert "displayed balance is not proof" in result
        assert "buying/selling" in result
        assert "voice pass" not in result
        assert objective in agent.llm.messages[-1]["content"]
        assert "not evidence" in agent.llm.messages[-1]["content"]
    else:
        assert "not complete" in result
    if candidate.startswith("Source:"):
        repair = agent.llm.messages[-1]["content"]
        assert "Tool execution is closed" in repair
        assert "Navigate to an actual source" not in repair
