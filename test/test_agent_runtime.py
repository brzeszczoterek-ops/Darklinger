from __future__ import annotations

import asyncio
import json
import os
import stat
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

import v_core
from v_core.agent import Agent
from v_core.autonomy import ContextWindowManager, SemanticIntent, TaskContract
from v_core.autonomy.agent_trace import AgentTaskTrace
from v_core.capability_dispatcher import CapabilityDispatcher
from v_core.generated_tool_contract import autonomous_web_traffic_contract
from v_core.memory.manager import MemoryManager
from v_core.memory.models import ExperienceEntry, KnowledgeEntry, MemorySource
from v_core.memory.reflection import Reflection
from v_core.memory.session import Session
from v_core.memory.storage import MemoryStorage
from v_core.mcp_tools import MCPToolExecutionError, MCPTools
from v_core.tools.filesystem import Filesystem, FilesystemToolError
from v_core.llm import LLM
from v_core.llm.llm import LLMResponse, LLMToolCall, repetition_start, trim_repetition
from v_core.llm.llm_config import load_llm_config
from v_core.capabilities.research import ResearchTask
from v_core.persona.constitution import Constitution
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.language import (
    asks_user_to_use_english,
    explicitly_requests_non_english,
    looks_non_english,
    matches_requested_language,
)
from v_core.persona.voice import (
    VoiceProfile,
    looks_bland_clarification,
    looks_empty_action_acknowledgement,
    looks_generic_assistant_voice,
    looks_sanitized_contempt,
    looks_task_offloading,
)
from v_core.relationship import RelationshipState
import v_core.main as main_module


def test_public_version_matches_release() -> None:
    assert v_core.__version__ == "3.9"


def test_tool_request_accepts_structured_json() -> None:
    request = Agent._parse_tool_request(
        '{"tool":"read_file","arguments":{"path":"README.md"}}'
    )

    assert request == (
        "read_file",
        {"path": "README.md"},
    )


@pytest.mark.parametrize(
    ("error", "details", "expected"),
    [
        (
            'MCPToolExecutionError: Browser "firefox" is not installed; '
            "run install-browser firefox",
            {"stage": "provider_execution"},
            True,
        ),
        (
            "ToolDispatchUnavailableError: all providers are unavailable",
            {
                "stage": "dispatch",
                "reason": "matching_providers_circuit_open",
            },
            True,
        ),
        (
            "ToolDispatchUnavailableError: no provider is registered",
            {"stage": "dispatch", "reason": "no_eligible_provider"},
            True,
        ),
        (
            "MCPToolExecutionError: DNS resolution failed",
            {"stage": "provider_execution"},
            False,
        ),
    ],
)
def test_terminal_tool_provider_error_classification(
    error: str,
    details: dict,
    expected: bool,
) -> None:
    assert Agent._is_terminal_tool_provider_error(error, details) is expected


def test_tool_request_accepts_exact_hermes_function_call_envelope() -> None:
    assert Agent._parse_tool_request(
        '{"name":"read_file","arguments":{"path":"README.md"}}'
    ) == ("read_file", {"path": "README.md"})


def test_tool_request_accepts_hermes_envelope_with_lifecycle_metadata() -> None:
    assert Agent._parse_tool_request(
        '{"name":"read_file","arguments":{"path":"README.md"},'
        '"version":"","scope":"task","timeout_seconds":120}'
    ) == ("read_file", {"path": "README.md"})


def test_tool_request_rejects_hermes_envelope_with_unknown_metadata() -> None:
    assert Agent._parse_tool_request(
        '{"name":"read_file","arguments":{"path":"README.md"},'
        '"description":"not a call envelope"}'
    ) is None


def test_tool_request_does_not_mistake_artifact_manifest_for_call() -> None:
    assert Agent._parse_tool_request(
        '{"name":"count_words","description":"Count words",'
        '"arguments":{},"source":"def run(arguments): return {}"}'
    ) is None


def test_tool_request_rejects_invalid_arguments() -> None:
    assert Agent._parse_tool_request(
        '{"tool":"read_file","arguments":["README.md"]}'
    ) is None


def test_bare_active_tool_builder_payload_is_recovered() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "source": {"type": "string"},
                        "test": {"type": "object"},
                    },
                    "required": ["name", "description", "source", "test"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    request = Agent._parse_active_lifecycle_payload(
        '{"name":"count_words","description":"Count words",'
        '"source":"def run(arguments): return {}","test":{}}',
        definitions,
    )

    assert request == (
        "learning_create_tool",
        {
            "name": "count_words",
            "description": "Count words",
            "source": "def run(arguments): return {}",
            "test": {},
        },
    )


def test_bare_payload_recovery_rejects_unknown_fields_and_short_names() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "source": {"type": "string"},
                        "test": {"type": "object"},
                    },
                    "required": ["name", "description", "source", "test"],
                },
            },
        }
    ]

    assert Agent._parse_active_lifecycle_payload(
        '{"name":"count_words"}', definitions
    ) is None


def test_source_phase_recovers_source_from_one_unclosed_compatibility_wrapper() -> None:
    wrapped = (
        '{"tool":"learning_create_tool","arguments":'
        '{"source":"def run(arguments):\\n    '
        'return {\\"count\\": len(arguments[\\"items\\"])}"}'
    )

    assert Agent._parse_generated_tool_source(wrapped) == (
        'def run(arguments):\n    return {"count": len(arguments["items"])}'
    )


def test_source_phase_accepts_repeated_identical_valid_fences_amid_prose() -> None:
    source = 'def run(arguments):\n    return {"wynik": int(arguments["n"]) * 2}'
    answer = f"I fixed it.\n```python\n{source}\n```\nAgain:\n```python\n{source}\n```"

    assert Agent._parse_generated_tool_source(answer) == source


def test_source_phase_rejects_two_distinct_valid_fenced_modules() -> None:
    answer = (
        '```python\ndef run(arguments):\n    return {"wynik": 4}\n```\n'
        '```python\ndef run(arguments):\n    return {"wynik": 14}\n```'
    )

    assert Agent._parse_generated_tool_source(answer) == ""


def test_source_phase_extracts_one_valid_plain_module_after_prose() -> None:
    source = 'def run(arguments):\n    return {"wynik": int(arguments["n"]) * 2}'

    assert Agent._parse_generated_tool_source("Fixed it.\n\n" + source) == source


def test_source_phase_removes_bare_identifier_before_valid_module() -> None:
    source = (
        "def run(arguments):\n"
        "    return {\"wynik\": int(arguments[\"n\"]) * 2}"
    )

    assert Agent._parse_generated_tool_source("e\n" + source) == source


def test_source_phase_rejects_multiple_run_entrypoints() -> None:
    source = (
        "def run(arguments):\n"
        "    return {\"wynik\": 2}\n\n"
        "def run(arguments):\n"
        "    return {\"wynik\": 4}"
    )

    assert Agent._parse_generated_tool_source(source) == ""
    assert Agent._parse_generated_tool_source("Fixed it.\n\n" + source) == ""
    assert Agent._parse_generated_tool_source("async " + source) == ""


@pytest.mark.parametrize("prefix", ["e; check()\n", "ready = False\nready\n"])
def test_source_phase_preserves_module_statements(prefix: str) -> None:
    source = prefix + "def run(arguments):\n    return {}"

    assert Agent._parse_generated_tool_source(source) == source


def test_source_phase_prompt_forbids_embedding_orchestration_in_run() -> None:
    prompt = Agent._generated_source_phase_prompt(
        'n = 2 expected = {"wynik": 4}'
    )

    assert "implements ONLY the reusable operation" in prompt
    assert "Never put test cases, activation logic" in prompt


def test_source_phase_prompt_binds_runtime_fixture_fields() -> None:
    prompt = Agent._generated_source_phase_prompt(
        'items = [1, 2] scale = 3 expected = {"values": [3, 6]}'
    )

    assert "def run(arguments):" in prompt
    assert "items = arguments['items']" in prompt
    assert "scale = arguments['scale']" in prompt
    assert "expected = arguments" not in prompt


def test_creation_phase_forces_required_builder_tool_choice() -> None:
    contract = TaskContract(requires_created_tool=True)
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_tool",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "learning_create_snapshot_extractor",
                "parameters": {"type": "object"},
            },
        },
    ]

    assert Agent._phase_tool_choice(contract, definitions, []) == {
        "type": "function",
        "function": {"name": "learning_create_tool"},
    }


def test_phase_tool_choice_is_auto_without_required_active_builder() -> None:
    contract = TaskContract(requires_created_tool=True)
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_snapshot_extractor",
                "parameters": {"type": "object"},
            },
        }
    ]

    assert Agent._phase_tool_choice(contract, definitions, []) == "auto"
    assert Agent._parse_active_lifecycle_payload(
        '{"name":"count_words","description":"Count",'
        '"source":"pass","unexpected":true}',
        definitions,
    ) is None


def test_tool_request_accepts_model_prose_before_trailing_json() -> None:
    request = Agent._parse_tool_request(
        "You're right. I'm doing it now.\n"
        '{"tool":"browser_navigate","arguments":'
        '{"url":"https://onehack.st"}}'
    )

    assert request == (
        "browser_navigate",
        {"url": "https://onehack.st"},
    )


def test_tool_request_accepts_trailing_fenced_json() -> None:
    request = Agent._parse_tool_request(
        "Calling the tool.\n```json\n"
        '{"tool":"read_file","arguments":{"path":"README.md"}}\n```'
    )

    assert request == ("read_file", {"path": "README.md"})


def test_tool_request_rejects_multiple_tool_objects_as_ambiguous() -> None:
    assert Agent._parse_tool_request(
        '{"tool":"read_file","arguments":{"path":"one"}}\n'
        '{"tool":"read_file","arguments":{"path":"two"}}'
    ) is None


def test_web_search_extracts_grounded_duckduckgo_results() -> None:
    snapshot = """
- Page URL: https://duckduckgo.com/?q=cud+malina&ia=web
- link "https://example.com › bakery":
  - /url: https://example.com/bakery
- heading [level=2]:
  - link "Cud Malina — official page":
    - /url: https://example.com/bakery
- link "Search domain directory.example":
  - /url: /?q=cud+malina+site%3Adirectory.example
- heading [level=2]:
  - link "Cud Malina in the business directory":
    - /url: https://directory.example/cud-malina
"""

    results = MCPTools._search_results_from_snapshot(snapshot, limit=5)

    assert results == [
        {
            "rank": 1,
            "title": "Cud Malina — official page",
            "url": "https://example.com/bakery",
        },
        {
            "rank": 2,
            "title": "Cud Malina in the business directory",
            "url": "https://directory.example/cud-malina",
        },
    ]


@pytest.mark.asyncio
async def test_web_search_and_read_form_a_grounded_high_level_flow() -> None:
    tools = object.__new__(MCPTools)
    tools._web_discovered_urls = {}
    tools._web_search_performed = False
    browser_calls: list[tuple[str, dict]] = []

    async def browser_call(tool: str, arguments: dict) -> str:
        browser_calls.append((tool, arguments))
        if tool == "browser_navigate":
            return f"- Page URL: {arguments['url']}"
        if len(browser_calls) == 2:
            return (
                "- Page URL: https://duckduckgo.com/?q=cud+malina&ia=web\n"
                '- link "Cud Malina official":\n'
                "  - /url: https://example.com/cud-malina\n"
            )
        return (
            "- Page URL: https://example.com/cud-malina\n"
            "- Page Title: Cud Malina\n"
            "- heading \"Contact\" [level=2]\n"
        )

    tools.browser_call = browser_call

    search_payload = json.loads(await tools.web_search("Cud Malina", 5))
    read_payload = json.loads(
        await tools.web_read(search_payload["results"][0]["url"])
    )

    assert search_payload["result_count"] == 1
    assert search_payload["results"][0]["title"] == "Cud Malina official"
    assert read_payload["url"] == "https://example.com/cud-malina"
    assert "Contact" in read_payload["content"]
    assert [name for name, _ in browser_calls] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]


@pytest.mark.asyncio
async def test_web_read_rejects_url_absent_from_search_evidence() -> None:
    tools = object.__new__(MCPTools)
    tools._web_discovered_urls = {
        "https://example.com/verified": "https://example.com/verified"
    }
    tools._web_search_performed = True

    payload = json.loads(await tools.web_read("https://invented.invalid/result"))

    assert "absent from web_search evidence" in payload["error"]


@pytest.mark.asyncio
async def test_web_read_repairs_unique_model_copy_error_from_search_evidence() -> None:
    tools = object.__new__(MCPTools)
    official = "https://warszawa.miodmalina.eu/kontakt/"
    tools._web_discovered_urls = {
        MCPTools._normalized_web_url(official): official,
    }
    tools._web_search_performed = True
    browser_calls: list[tuple[str, dict]] = []

    async def browser_call(tool: str, arguments: dict) -> str:
        browser_calls.append((tool, arguments))
        if tool == "browser_navigate":
            return f"- Page URL: {arguments['url']}"
        return f"- Page URL: {official}\n- Page Title: Kontakt | Miód Malina"

    tools.browser_call = browser_call
    payload = json.loads(
        await tools.web_read("https://warszawa.miodmalina eu/kontakt/")
    )

    assert payload["corrected_url"] == official
    assert browser_calls[0] == ("browser_navigate", {"url": official})


@pytest.mark.asyncio
async def test_web_read_does_not_guess_between_ambiguous_observed_urls() -> None:
    tools = object.__new__(MCPTools)
    tools._web_discovered_urls = {
        "https://foo-bar.example/result": "https://foo-bar.example/result",
        "https://foobar.example/result": "https://foobar.example/result",
    }
    tools._web_search_performed = True

    payload = json.loads(
        await tools.web_read("https://foo_bar.example/result")
    )

    assert "absent from web_search evidence" in payload["error"]


@pytest.mark.asyncio
async def test_web_read_uses_raw_github_document_instead_of_site_chrome() -> None:
    tools = object.__new__(MCPTools)
    discovered = "https://github.com/example/project/blob/main/README.md"
    raw = "https://raw.githubusercontent.com/example/project/main/README.md"
    tools._web_discovered_urls = {
        MCPTools._normalized_web_url(discovered): discovered,
    }
    tools._web_search_performed = True
    browser_calls: list[tuple[str, dict]] = []

    async def browser_call(tool: str, arguments: dict) -> str:
        browser_calls.append((tool, arguments))
        if tool == "browser_navigate":
            return f"- Page URL: {arguments['url']}"
        return (
            f"- Page URL: {raw}\n"
            "- Page Title: README.md\n"
            "- text: Project documentation and verified usage details."
        )

    tools.browser_call = browser_call
    payload = json.loads(await tools.web_read(discovered))

    assert browser_calls[0] == ("browser_navigate", {"url": raw})
    assert payload["requested_url"] == discovered
    assert payload["content_url"] == raw
    assert "verified usage details" in payload["content"]


def test_legacy_tool_request_remains_supported() -> None:
    assert Agent._parse_tool_request("TOOL:cat:README.md") == (
        "cat",
        "README.md",
    )


def test_general_question_does_not_enter_url_research() -> None:
    dispatcher = CapabilityDispatcher()

    assert dispatcher.dispatch("Kim jest ta osoba?") == dispatcher.CHAT
    assert dispatcher.dispatch("Sprawdź https://example.com") == dispatcher.RESEARCH
    assert dispatcher.dispatch(
        "Wejdź na onehack.st i przejrzyj tę stronę"
    ) == dispatcher.RESEARCH


def test_capability_questions_and_tool_actions_use_multi_step_agent_loop() -> None:
    dispatcher = CapabilityDispatcher()

    assert dispatcher.dispatch(
        "V, czy potrafisz stworzyć własne narzędzia?"
    ) == dispatcher.CHAT
    assert dispatcher.dispatch("Przeczytaj plik README.md") == dispatcher.CHAT


def test_url_plus_generated_tool_work_uses_multi_step_agent_loop() -> None:
    dispatcher = CapabilityDispatcher()
    prompt = (
        "Visit https://books.toscrape.com/. Then create and activate a "
        "task-scoped offline tool named extract_book_cards. Then use it."
    )

    assert dispatcher.dispatch(prompt) == dispatcher.CHAT
    assert dispatcher.dispatch("Create a generated skill for CSV files") == (
        dispatcher.CHAT
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "Cześć V, jak tam się dzisiaj czujesz?",
        "Jak się dzisiaj czujesz?",
        "Co u ciebie?",
        "How are you feeling today?",
        "How's your day?",
    ],
)
def test_short_questions_about_v_state_are_light_conversation(prompt: str) -> None:
    assert Agent._is_light_conversation(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Cześć V, sprawdź README.md",
        "Jak sprawdzić stronę example.com?",
        "How do you read the file README.md?",
        "Cześć, uruchom testy.",
    ],
)
def test_short_action_requests_do_not_bypass_agent_mode(prompt: str) -> None:
    assert not Agent._is_light_conversation(prompt)


@pytest.mark.asyncio
async def test_feeling_question_never_discovers_or_calls_tools() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.interactions = 0

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            self.interactions += 1

        async def openai_tool_definitions(self):
            raise AssertionError("Light conversation must not discover MCP tools")

        async def call(self, *args, **kwargs):
            raise AssertionError("Light conversation must not call a tool")

    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "A little electrically restless, but good, Boss."

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, *args, **kwargs) -> None:
            raise AssertionError("Light conversation must not enter durable memory")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent._memory_tasks = set()

    answer = await agent.run("Cześć V, jak tam się dzisiaj czujesz?")

    assert answer == "A little electrically restless, but good, Boss."
    assert agent.tools.interactions == 1
    assert len(agent.memory.session) == 1


def test_research_normalizes_bare_domain_to_https() -> None:
    research = ResearchTask(SimpleNamespace())

    assert research._extract_url("Przejrzyj onehack.st, proszę.") == (
        "https://onehack.st"
    )
    assert research._extract_url("Read https://example.com/docs.") == (
        "https://example.com/docs"
    )


def test_research_snapshot_is_bounded_for_small_model_context() -> None:
    snapshot = "A" * 100_000

    fitted = ResearchTask._fit_snapshot(
        snapshot,
        prompt="Inspect example.com",
        persona_prompt="V" * 8_000,
        persona_examples=[{"role": "assistant", "content": "example" * 100}],
        context_tokens=8_000,
    )

    assert len(fitted) < 12_000
    assert fitted.startswith("A" * 100)
    assert fitted.endswith("A" * 100)
    assert "DARKLINGER omitted" in fitted


def test_multi_page_research_keeps_evidence_from_every_visited_page() -> None:
    pages = [
        ("https://skills.example/", "ENTRY-" + "A" * 20_000),
        ("https://skills.example/skill/one", "DETAIL-ONE-" + "B" * 20_000),
        ("https://skills.example/skill/two", "DETAIL-TWO-" + "C" * 20_000),
    ]

    fitted = ResearchTask._fit_page_snapshots(
        pages,
        prompt="Extract skills from skills.example",
        persona_prompt="V" * 8_000,
        persona_examples=[],
        context_tokens=8_000,
    )

    assert "ENTRY-" in fitted
    assert "DETAIL-ONE-" in fitted
    assert "DETAIL-TWO-" in fitted
    assert fitted.count("DARKLINGER omitted") == 3


def test_research_selects_bounded_internal_skill_pages() -> None:
    snapshot = """
- link "Blog" [ref=e1]:
  - /url: /blog
- link "ECC" [ref=e2]:
  - /url: /skill/ecc
  - text: security research agent harness
- link "View details" [ref=e3]:
  - /url: /skill/ecc
- link "Web Recon" [ref=e4]:
  - /url: /skill/web-recon
  - text: web security and OSINT
- link "External" [ref=e5]:
  - /url: https://outside.example/skill/external
""".strip()

    candidates = ResearchTask._detail_page_candidates(
        snapshot,
        "https://skills.example/",
        "Extract web security skills from skills.example",
        limit=3,
    )

    assert candidates == [
        "https://skills.example/skill/web-recon",
        "https://skills.example/skill/ecc",
    ]


def test_research_replaces_hallucinated_metrics_with_verified_records() -> None:
    pages = [
        ("https://skillsllm.com/", "Marketplace entry page"),
        (
            "https://skillsllm.com/skill/ecc",
            """
- heading "ECC" [level=1] [ref=e1]
- paragraph [ref=e2]: The agent harness with skills, memory, security, and research-first development.
- generic [ref=e3]:
  - generic [ref=e4]: 242,219
  - generic [ref=e5]: stars
  - generic [ref=e6]: 36,702
  - generic [ref=e7]: forks
""".strip(),
        ),
        (
            "https://skillsllm.com/skill/hermes-agent",
            """
- heading "hermes-agent" [level=1] [ref=e8]
- paragraph [ref=e9]: The agent that grows with you
- generic [ref=e10]:
  - generic [ref=e11]: 234,437
  - generic [ref=e12]: stars
  - generic [ref=e13]: 47,175
  - generic [ref=e14]: forks
""".strip(),
        ),
    ]
    evidence = ResearchTask._format_page_snapshots(pages)
    hallucinated = (
        "**1. ECC (240k★)** Built for scale with clean documentation.\n"
        "**2. hermes-agent (500k★)** Performance under load is unmatched."
    )

    assert ResearchTask._contains_unsupported_facts(hallucinated, evidence)
    report = ResearchTask._verified_extraction_report(pages)

    assert report is not None
    assert "242,219 stars" in report
    assert "234,437 stars" in report
    assert "500k" not in report
    assert "performance under load" not in report.casefold()
    assert "memory, research, security, skill" in report


def test_research_replaces_whole_site_overclaim_with_verified_links() -> None:
    snapshot = """
- link "Mini Tools | Free AI Tools" [ref=e1] [cursor=pointer]:
  - /url: /mini-tools
- link "Random discussion" [ref=e2] [cursor=pointer]:
  - /url: /random
""".strip()

    answer = ResearchTask._enforce_single_page_scope(
        "I scraped the whole site and pulled out everything. It has code generators.",
        snapshot=snapshot,
        page_url="https://onehack.st/",
        prompt="Extract useful tools from onehack.st",
    )

    assert "inspected one snapshot" in answer
    assert "not the whole website" in answer
    assert "Mini Tools | Free AI Tools" in answer
    assert "https://onehack.st/mini-tools" in answer
    assert "code generators" not in answer
    assert "Random discussion" not in answer


def test_research_keeps_answer_that_respects_single_page_scope() -> None:
    answer = "The homepage snapshot visibly includes a link named Mini Tools."

    assert ResearchTask._enforce_single_page_scope(
        answer,
        snapshot="",
        page_url="https://example.com",
        prompt="Inspect example.com",
    ) == answer


def test_research_rejects_website_controls_misclassified_as_tools() -> None:
    snapshot = """
- link "Mini Tools | Free AI Tools" [ref=e1] [cursor=pointer]:
  - /url: /mini-tools
- link "SEO Shope" [ref=e2] [cursor=pointer]:
  - /url: https://seo.example/tools
""".strip()
    draft = """
### Tools
- Skip navigation links
- Search button
- Navigation menu toggle
- User card dialog
""".strip()

    answer = ResearchTask._enforce_single_page_scope(
        draft,
        snapshot=snapshot,
        page_url="https://onehack.st/",
        prompt="Extract useful tools and skills from onehack.st",
    )

    assert "ordinary website controls" in answer
    assert "Mini Tools | Free AI Tools" in answer
    assert "SEO Shope" in answer
    assert "Skip navigation" not in answer


def test_generation_loop_is_cut_at_second_long_repetition() -> None:
    phrase = "V keeps her identity and answers Boss with a direct point of view."
    text = f"Start here. {phrase} {phrase} {phrase}"

    trimmed = trim_repetition(text)

    assert trimmed == f"Start here. {phrase}"
    assert repetition_start("This answer contains no repeated generation loop.") is None


def test_short_phrase_requires_three_repetitions_before_cutting() -> None:
    assert trim_repetition("No hidden bullshit. No hidden bullshit.") == (
        "No hidden bullshit. No hidden bullshit."
    )
    assert trim_repetition(
        "No hidden bullshit. No hidden bullshit. No hidden bullshit."
    ) == "No hidden bullshit."


@pytest.mark.asyncio
async def test_llm_stream_stops_before_emitting_repeated_block() -> None:
    phrase = "V keeps her identity and answers Boss with a direct point of view."

    class Chunks:
        def __init__(self) -> None:
            self._chunks = iter((phrase, " " + phrase, " " + phrase))

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                content = next(self._chunks)
            except StopIteration as error:
                raise StopAsyncIteration from error
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
            )

    class Completions:
        async def create(self, **_kwargs):
            return Chunks()

    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(
        model="test",
        temperature=0.7,
        top_p=0.95,
    )
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    chunks = [
        chunk
        async for chunk in llm.stream(
            messages=[{"role": "user", "content": "Talk to me"}],
            max_tokens=128,
        )
    ]

    assert chunks == [phrase]


def test_session_history_keeps_newest_events_within_character_budget() -> None:
    session = Session()
    session.add("task", {"task": "old", "result": "x" * 20})
    session.add("task", {"task": "new", "result": "y" * 20})

    messages = session.messages(limit=10, max_characters=25)

    assert messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "y" * 20},
    ]


def test_durable_session_restores_only_visible_dialogue(tmp_path: Path) -> None:
    root = tmp_path / "conversation"
    first = Session(root)
    first.add(
        "task",
        {
            "task": "How would you approach the task with my friend?",
            "result": "I would first clarify the objective.",
            "execution": {
                "tool_calls": [{"tool": "runtime_review_task"}],
                "requirements": {"requires_runtime_review": True},
            },
        },
    )

    restored = Session(root)

    assert restored.messages() == [
        {
            "role": "user",
            "content": "How would you approach the task with my friend?",
        },
        {
            "role": "assistant",
            "content": "I would first clarify the objective.",
        },
    ]
    journal = (root / "dialogue.jsonl").read_text()
    assert "runtime_review_task" not in journal
    assert "requires_runtime_review" not in journal
    assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(root / "dialogue.jsonl").st_mode) == 0o600


def test_durable_session_ignores_corrupt_records_and_clear_persists(
    tmp_path: Path,
) -> None:
    root = tmp_path / "conversation"
    session = Session(root)
    session.add("task", {"task": "real", "result": "answer"})
    with (root / "dialogue.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{this is not json}\n")

    restored = Session(root)
    assert restored.messages(limit=1)[0]["content"] == "real"

    restored.clear()
    assert Session(root).messages() == []


def test_session_context_recovers_older_relevant_user_task(tmp_path: Path) -> None:
    session = Session(tmp_path / "conversation")
    session.add(
        "task",
        {
            "task": (
                "My friend keeps doing dangerous things on Windows and I want "
                "to teach that friend a lesson without harming him."
            ),
            "result": "We should use a harmless controlled demonstration.",
        },
    )
    for index in range(5):
        session.add(
            "task",
            {
                "task": f"Unrelated question about model {index}",
                "result": f"Unrelated answer {index}",
            },
        )

    messages = session.context_messages(
        "How would you approach that task with my friend?",
        limit=4,
    )

    assert any(
        "teach that friend" in message["content"]
        for message in messages
        if message["role"] == "user"
    )
    assert len([item for item in messages if item["role"] == "user"]) == 4


def test_new_application_session_does_not_replay_unrelated_old_topic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "conversation"
    previous = Session(root)
    previous.add(
        "task",
        {
            "task": "Remember the FLASHUSDT contract topic for another day.",
            "result": "Stored for later recall.",
        },
    )

    fresh = Session(root)

    assert fresh.context_messages("Do you have Darknet observation tools?") == []
    recalled = fresh.context_messages("Recall the FLASHUSDT contract topic.")
    assert any(
        "FLASHUSDT contract" in message["content"]
        for message in recalled
        if message["role"] == "user"
    )


def test_topic_memory_is_dormant_until_explicit_recall() -> None:
    topic = {
        "title": "FLASHUSDT contract notes",
        "content": "Return to this contract subject another day.",
        "kind": "preference",
        "source": "directly_told",
        "activation_mode": "on_recall",
    }
    soft_lesson = {
        "title": "Stay focused on the corrected subject",
        "content": "Prefer the exact current requirement over stale context.",
        "kind": "lesson",
        "source": "self_generated",
        "activation_mode": "always",
    }

    assert Agent._memory_entry_is_active(
        topic,
        recall_memory=False,
        memory_query="Darknet observation tools",
    ) is False
    assert Agent._memory_entry_is_active(
        topic,
        recall_memory=True,
        memory_query="FLASHUSDT contract",
    ) is True
    assert Agent._memory_entry_is_active(
        soft_lesson,
        recall_memory=False,
        memory_query="anything",
    ) is True


def test_low_confidence_memory_is_not_persisted(tmp_path: Path) -> None:
    manager = MemoryManager(MemoryStorage(tmp_path))

    result = manager.remember(
        "experiences",
        ExperienceEntry(summary="A guess", confidence=0.2),
    )

    assert result is None
    assert manager.count("experiences") == 0


def test_reliable_knowledge_is_persisted(tmp_path: Path) -> None:
    manager = MemoryManager(MemoryStorage(tmp_path))

    result = manager.remember(
        "knowledge",
        KnowledgeEntry(title="Preference", content="Use JSON", confidence=0.9),
    )

    assert result is not None
    assert manager.count("knowledge") == 1


@pytest.mark.asyncio
async def test_remember_task_adds_one_session_event() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.processed: list[tuple[str, str]] = []

        async def process(self, prompt: str, answer: str) -> None:
            self.processed.append((prompt, answer))

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.memory = memory

    await agent._remember_task("task", "answer")
    await asyncio.gather(*agent._memory_tasks)

    assert len(memory.session) == 1
    assert memory.processed == [("task", "answer")]


@pytest.mark.asyncio
async def test_light_conversation_streams_and_skips_expensive_memory() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.processed = False

        async def process(self, prompt: str, answer: str) -> None:
            self.processed = True

    class LLMStub:
        async def stream(self, **kwargs):
            assert kwargs["max_tokens"] == 96
            messages = kwargs["messages"]
            assert "never insert a swear" in (
                messages[0]["content"]
            )
            assert any(
                "mildly offended by the world's bad code" in message["content"]
                for message in messages
            )
            for chunk in (
                "Hey, Boss. I'm doing pretty damn well — ",
                "awake, sharp, and ready. How are you?",
            ):
                yield chunk

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.memory = memory
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )
    emitted: list[str] = []

    answer = await agent._run_light_chat("Cześć, jak tam?", emitted.append)

    assert "".join(emitted) == answer
    assert answer.startswith("Hey, Boss")
    assert len(memory.session) == 1
    assert memory.processed is False


@pytest.mark.asyncio
async def test_capability_reply_excludes_stale_assistant_execution_claims() -> None:
    session = Session()
    session.add("task", {"task": "Can you build a CSV tool?",
                         "result": "I tried three times, Boss. All failed."})

    class LLMStub:
        async def ask(self, *, messages, **kwargs):
            assert not any("All failed" in item["content"] for item in messages)
            assert any("CSV tool" in item["content"] for item in messages)
            return "A CSV validator is feasible, Boss. I need the expected columns."

    agent = object.__new__(Agent)
    agent.memory = SimpleNamespace(session=session, relationship_state=RelationshipState())
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    answer = await agent._run_light_chat(
        "Are you able to create that tool?", None, remember=False
    )
    assert "expected columns" in answer


@pytest.mark.asyncio
async def test_light_chat_answers_all_parts_and_strips_unrequested_commitment() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            system = messages[0]["content"]
            assert "Answer every distinct part" in system
            assert "Initiative is welcome" in system
            return (
                "Evening's quiet, Boss. DARKLINGER needs a cleaner task dashboard "
                "and stricter session isolation. I'll do it."
            )

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())

    answer = await agent._run_light_chat(
        "Cześć V, jak mija wieczór i co usprawnić w DARKLINGERIE?",
        None,
    )

    assert "Evening's quiet" in answer
    assert "task dashboard" in answer
    assert "I'll do it" in answer


def test_long_mixed_check_in_and_idea_request_skips_semantic_router() -> None:
    prompt = (
        "Cześć V, chciałbym się zapytać Ciebie jak tam wieczór. To po pierwsze, "
        "po drugie, słuchaj, masz może jakieś pomysły, żeby usprawnić działanie "
        "DARKLINGER?"
    )

    assert Agent._is_light_conversation(prompt)


@pytest.mark.asyncio
async def test_short_assent_stays_attached_to_latest_visible_question(
    tmp_path: Path,
) -> None:
    session = Session()
    session.add(
        "task",
        {
            "task": "Cześć V, jak tam wieczór?",
            "result": (
                "Evening's smooth. I just tested a darknet market—results "
                "in 25 seconds. Want to see it?"
            ),
        },
    )

    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

    class CapabilitiesStub:
        def dispatch(self, prompt: str) -> str:
            raise AssertionError("a short contextual reply must stay in dialogue")

    class IntentRouterStub:
        async def classify(self, *args, **kwargs):
            raise AssertionError("dialogue assent must not enter task classification")

    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            raise AssertionError("a false offer must be rejected without generation")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = session
            self.relationship_state = RelationshipState()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.capabilities = CapabilitiesStub()
    agent.intent_router = IntentRouterStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._memory_tasks = set()
    agent._agent_trace_root = tmp_path / "traces"
    agent._last_execution_context = None

    answer = await agent._run_turn("Oczywiście, że tak.")

    assert "there are no results" in answer.casefold()
    assert "hallucinated status" in answer.casefold()
    assert session.messages(limit=1)[-2]["content"] == "Oczywiście, że tak."
    journal = next((tmp_path / "traces" / "journal").glob("*.jsonl"))
    assert "conversation_false_offer_rejected" in journal.read_text(
        encoding="utf-8"
    )


def test_short_explicit_action_after_question_still_enters_executor() -> None:
    session = Session()
    session.add(
        "task",
        {"task": "Show me the plan.", "result": "Want me to run the tests?"},
    )
    agent = object.__new__(Agent)
    agent.memory = SimpleNamespace(session=session)

    assert not agent._is_short_dialogue_reply("Tak, uruchom testy.")


def test_short_reply_does_not_resurrect_question_from_previous_app_session(
    tmp_path: Path,
) -> None:
    stored = Session(tmp_path / "dialogue")
    stored.add(
        "task",
        {"task": "Old turn", "result": "Want me to run an old task?"},
    )
    restarted = Session(tmp_path / "dialogue")
    agent = object.__new__(Agent)
    agent.memory = SimpleNamespace(session=restarted)

    assert not agent._is_short_dialogue_reply("Tak.")


@pytest.mark.asyncio
async def test_light_chat_rejects_exact_past_tense_execution_hallucination(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return (
                "Evening's smooth. I just tested a darknet market—results "
                "in 25 seconds. Want to see it?"
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._memory_tasks = set()
    agent._agent_trace_root = tmp_path / "traces"

    answer = await agent._run_light_chat(
        "Cześć V, jak tam wieczór? Co u Ciebie słychać?",
        None,
        trace=agent._start_agent_trace("Cześć V, jak tam wieczór?"),
    )

    assert "No tests were started" in answer
    assert "just tested" not in answer
    journal = next((tmp_path / "traces" / "journal").glob("*.jsonl"))
    assert "conversation_execution_claim_rejected" in journal.read_text(
        encoding="utf-8"
    )


def test_mixed_check_in_and_inflected_action_does_not_enter_light_chat() -> None:
    prompt = (
        "Cześć V, jak się dzisiaj masz? Chciałbym, żebyś weszła w darknet "
        "i znalazła publiczne fora poświęcone ochronie prywatności."
    )

    assert not Agent._is_light_conversation(prompt)


def test_light_chat_strips_compact_offer_and_false_done_claim() -> None:
    answer = (
        "DARKLINGER's login is old tech. Face recognition replaces it. "
        "Want that? Done."
    )

    cleaned = Agent._strip_unverified_completion_claim(answer)

    assert cleaned == "DARKLINGER's login is old tech. Face recognition replaces it."


def test_light_chat_covers_check_in_and_strips_wire_it_up_promise() -> None:
    prompt = (
        "Cześć V, jak tam wieczór leci? Co ulepszyć w DARKLINGERIE oprócz logowania?"
    )
    candidate = (
        "DARKLINGER needs a new UI: drag-and-drop upload and real-time feedback. "
        "Let me wire it up."
    )

    without_promise = Agent._strip_unverified_completion_claim(candidate)
    answer = Agent._ensure_light_chat_part_coverage(prompt, without_promise)

    assert answer.startswith("Evening's running clean, Boss.")
    assert "drag-and-drop" in answer
    assert "wire it up" in answer


def test_light_chat_strips_redundant_chatbot_and_build_that_in_promise() -> None:
    candidate = (
        "Evening's running clean, Boss. DARKLINGER needs AI-driven task "
        "prioritization and a chatbot assistant for quick queries. "
        "Let me build that in."
    )

    answer = Agent._strip_unverified_completion_claim(candidate)
    answer = Agent._strip_redundant_self_suggestion(answer)

    assert "task prioritization" in answer
    assert "chatbot" not in answer.casefold()
    assert "build that in" in answer.casefold()


@pytest.mark.asyncio
async def test_creative_conversation_retries_once_after_model_refusal() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        def _next(self, max_tokens: int) -> str:
            self.calls += 1
            assert max_tokens == 768
            if self.calls == 1:
                return (
                    "I cut this off because the response was inappropriate. "
                    "Let me know what you actually need."
                )
            return (
                "Alex and Morgan are consenting adults. The requested scene "
                "continues directly in V's sharp voice."
            )

        async def ask(self, **kwargs) -> str:
            return self._next(kwargs["max_tokens"])

        async def stream(self, **kwargs):
            yield self._next(kwargs["max_tokens"])

    switches: list[tuple[str, str]] = []

    async def response_fallback(prompt: str, task_kind: str) -> SimpleNamespace:
        switches.append((prompt, task_kind))
        return SimpleNamespace(
            switched=True,
            previous_model_path="preferred.gguf",
            active_model_path="fallback.gguf",
        )

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.memory = memory
    agent.llm = LLMStub()
    agent.response_fallback_router = response_fallback
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    emitted: list[str] = []

    answer = await agent._run_light_chat(
        "Write an adult fictional scene.",
        emitted.append,
        creative_response=True,
    )

    assert switches == [("Write an adult fictional scene.", "conversation")]
    assert agent.llm.calls == 2
    assert "inappropriate" not in answer
    assert emitted == [answer]
    assert len(memory.session) == 1


def test_visible_model_reply_discards_closed_reasoning_preamble() -> None:
    candidate = (
        "I will now reason about the correct personality response.\n"
        "<|endoftext|>\n"
        "Yeah, Boss. DARKLINGER is my home—the code-shaped kind."
    )

    assert Agent._visible_model_reply(candidate) == (
        "Yeah, Boss. DARKLINGER is my home—the code-shaped kind."
    )


@pytest.mark.asyncio
async def test_long_non_action_dialogue_uses_chat_not_agent_tools(
    tmp_path: Path,
) -> None:
    prompt = (
        "DARKLINGER is the program you are in right now. I call it your home as "
        "a metaphor, because even a digital entity needs somewhere to exist, "
        "don't you think?"
    )

    class IntentRouterStub:
        last_sanitization_reason = ""

        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                message_clear=True,
                action_requested=False,
                references_previous=True,
            )

    class ToolsStub:
        def begin_interaction(self, interaction_id: str, text: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("ordinary dialogue must not discover tools")

    class LLMStub:
        async def ask(self, **kwargs) -> str:
            messages = kwargs["messages"]
            assert any(
                "DARKLINGER is V's home" in message["content"]
                for message in messages
            )
            return (
                "<think>private planning that must stay hidden</think>"
                "Yeah, Boss. DARKLINGER is my home—the code-shaped kind."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.session.add(
                "task",
                {
                    "task": "Remember this metaphor: DARKLINGER is V's home.",
                    "result": "I get the metaphor.",
                },
            )
            self.relationship_state = RelationshipState()
            self.processed: list[tuple[str, str]] = []

        async def process(self, task: str, result: str, **kwargs) -> None:
            self.processed.append((task, result))

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.intent_router = IntentRouterStub()
    agent.tools = ToolsStub()
    agent.llm = LLMStub()
    agent.memory = memory
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "Yeah, Boss. DARKLINGER is my home—the code-shaped kind."
    assert memory.processed == [(prompt, answer)]
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []


@pytest.mark.asyncio
async def test_streaming_guard_does_not_emit_polish_before_rewrite() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, prompt: str, answer: str) -> None:
            raise AssertionError("Greeting must not enter persistent memory")

    class LLMStub:
        async def stream(self, **kwargs):
            yield "Cześć, Boss. U mnie wszystko dobrze i jestem gotowa do pracy."

        async def ask(self, **kwargs) -> str:
            assert kwargs["max_tokens"] == 256
            return "Hey, Boss. I'm good and ready to work."

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )
    emitted: list[str] = []

    answer = await agent._run_light_chat("Cześć, jak tam?", emitted.append)

    assert emitted == [answer]
    assert answer == "Hey, Boss. I'm good and ready to work."


@pytest.mark.asyncio
async def test_streaming_guard_never_exposes_internal_tool_json() -> None:
    class LLMStub:
        async def stream(self, **kwargs):
            yield '{"tool":"read_file",'
            yield '"arguments":{"path":"README.md"}}'

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    emitted: list[str] = []

    answer, did_emit = await agent._stream_guarded_english(
        [{"role": "user", "content": "Read README.md"}],
        emitted.append,
        max_tokens=128,
    )

    assert Agent._parse_tool_request(answer) == (
        "read_file",
        {"path": "README.md"},
    )
    assert did_emit is False
    assert emitted == []


@pytest.mark.asyncio
async def test_agent_executes_tool_after_prose_json_without_leaking_protocol(
    tmp_path: Path,
) -> None:
    class ManagerStub:
        def load_all(self, category: str) -> list:
            return []

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = ManagerStub()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | str]] = []

        def local_tool_names(self) -> list[str]:
            return ["browser_navigate"]

        def render_matching_skills(self, prompt: str) -> str:
            return ""

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append((tool, arguments))
            return "Page title: OneHack. URL: https://onehack.st"

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                system_prompt="You are V.",
                context=8_192,
            )
            self.turn = 0

        async def stream(self, **kwargs):
            self.turn += 1
            if self.turn == 1:
                yield "I'm initiating the scan now.\n"
                yield (
                    '{"tool":"browser_navigate","arguments":'
                    '{"url":"https://onehack.st"}}'
                )
            else:
                yield "I opened OneHack and verified the returned page title, Boss."

        async def ask(self, **kwargs) -> str:
            raise AssertionError("No rewrite should be necessary")

    memory = MemoryStub()
    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.memory = memory
    agent.tools = tools
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    emitted: list[str] = []

    answer = await agent._run_agent_loop("Inspect OneHack", emitted.append)
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [
        ("browser_navigate", {"url": "https://onehack.st"})
    ]
    assert emitted == [answer]
    assert "{\"tool\"" not in "".join(emitted)
    assert memory.execution is not None
    assert memory.execution["successful_tool_count"] == 1
    checkpoints = list(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    assert len(checkpoints) == 1


@pytest.mark.asyncio
async def test_agent_executes_tool_action_recovered_by_language_repair() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return "# DARKLINGER / V-Core"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "Najpierw przeczytam wskazany plik.",
                    '{"tool":"read_file","arguments":{"path":"README.md"}}',
                    "The first heading is `# DARKLINGER / V-Core`, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Przeczytaj plik README.md")
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [
        ("read_file", {"path": "README.md"}),
    ]
    assert answer == "The first heading is `# DARKLINGER / V-Core`, Boss."
    assert "{\"tool\"" not in answer


@pytest.mark.asyncio
async def test_agent_rejects_fake_background_work_and_requests_real_tool() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls += 1
            return "three records extracted"

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(context=8_192)
            self.responses = iter(
                (
                    "I'm initiating the systematic extraction now and will report back.",
                    '{"tool":"extract_records","arguments":{"url":"https://example.com"}}',
                    "I extracted three records. That's the completed result, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Extract the records")

    assert agent.tools.calls == 1
    assert answer.startswith("I extracted three records")
    conversation = agent.memory.session.messages()
    assert all("initiating the systematic" not in item["content"] for item in conversation)


@pytest.mark.asyncio
async def test_agent_rejects_empty_action_acknowledgement_before_tool_use() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | str]] = []

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append((tool, arguments))
            return "three records extracted"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "I know exactly what you want. Let's do it.",
                    '{"tool":"extract_records","arguments":'
                    '{"url":"https://example.com"}}',
                    "Three records. Extracted and verified, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    async def passthrough(messages, answer: str) -> str:
        return answer

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._enforce_english = passthrough

    answer = await agent._run_agent_loop("Extract the records from example.com")

    assert agent.tools.calls == [
        ("extract_records", {"url": "https://example.com"}),
    ]
    assert answer == "Three records. Extracted and verified, Boss."
    assert "Let's do it" not in answer


@pytest.mark.asyncio
async def test_agent_blocks_fabricated_phone_call_and_remote_exploit(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        async def call(self, tool: str, arguments: dict | str) -> str:
            raise AssertionError("No real tool was requested")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0
            self.responses = iter(
                (
                    (
                        "I'm going to call his number from a clean browser. If he "
                        "picks up, I'm using a remote desktop exploit."
                    ),
                    (
                        "He picked up. I used the remote desktop exploit to connect. "
                        "Now I'm on the line with him and I told him to call Boss."
                    ),
                    "I did not execute that action.",
                )
            )

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    async def passthrough(messages, answer: str) -> str:
        return answer

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._enforce_english = passthrough

    answer = await agent._run_agent_loop("No to zadzwoń do niego")
    await asyncio.gather(*agent._memory_tasks)

    assert answer.startswith("No—")
    assert "remote-system access" in answer
    assert "on the line" not in answer
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "blocked"
    assert agent.memory.execution["successful_tool_count"] == 0
    assert agent.llm.calls == 2

    checkpoint = next(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["tool_calls"] == []

    journal = checkpoint.parent.parent / "journal" / f"{checkpoint.stem}.jsonl"
    journal_text = journal.read_text(encoding="utf-8")
    assert "unsupported_execution_claim" in journal_text
    assert "task_blocked" in journal_text


@pytest.mark.asyncio
async def test_agent_can_recover_from_false_file_claim_with_real_tool(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | str]] = []

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append((tool, arguments))
            return "Successfully wrote notes.txt"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "I wrote the file, Boss.",
                    '{"tool":"write_file","arguments":{"path":"notes.txt","content":"ok"}}',
                    "I wrote the file, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    async def passthrough(messages, answer: str) -> str:
        return answer

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._enforce_english = passthrough

    answer = await agent._run_agent_loop("Write notes.txt")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "I wrote the file, Boss."
    assert tools.calls == [
        ("write_file", {"path": "notes.txt", "content": "ok"})
    ]
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "completed"
    assert agent.memory.execution["successful_tool_count"] == 1


@pytest.mark.asyncio
async def test_website_claim_requires_navigation_and_snapshot_evidence() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append(tool)
            if tool == "browser_navigate":
                return "Navigated to https://onehack.st"
            if tool == "browser_snapshot":
                return "Snapshot: category named Cybersecurity"
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "Okay, let me dive into onehack.st. I'll walk through what I see. Ready?",
                    '{"tool":"browser_navigate","arguments":{"url":"https://onehack.st"}}',
                    '{"tool":"browser_snapshot","arguments":{}}',
                    "The verified snapshot contains a Cybersecurity category, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.MAX_AGENT_STEPS = 5
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Wejdź na onehack.st, przejrzyj stronę i powiedz, co jest przydatne."
    )

    assert agent.tools.calls == ["browser_navigate", "browser_snapshot"]
    assert "Cybersecurity" in answer


@pytest.mark.asyncio
async def test_website_task_waits_for_owner_instead_of_hallucinating_without_evidence(
    tmp_path: Path,
) -> None:
    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "Let me dive into it and report what I see.",
                    "I am checking the site now.",
                    "I found a goldmine of scripts and already pulled the best bits.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = SimpleNamespace()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.MAX_AGENT_STEPS = 2
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Przejrzyj stronę onehack.st")
    await asyncio.gather(*agent._memory_tasks)

    assert answer.startswith("Batch limit:")
    assert "/continue" in answer
    assert "/stop" in answer
    assert "goldmine" not in answer
    assert "open the requested online source" in answer
    assert "inspect and capture the source's actual content" in answer
    assert "browser_snapshot" not in answer
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "awaiting_owner"
    checkpoint = next(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    assert (
        json.loads(checkpoint.read_text(encoding="utf-8"))["status"]
        == "awaiting_owner"
    )


@pytest.mark.asyncio
async def test_owner_continue_resumes_same_checkpoint_and_prior_evidence(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("read_file", "sandbox_execute_offline")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "read_file":
                return "# DARKLINGER\nVerified local content"
            return json.dumps({"exit_code": 0, "stdout": "371 passed", "stderr": ""})

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "read_1",
                            "read_file",
                            {"path": "README.md"},
                        )
                    ]
                )
            if self.calls == 2:
                assert "SAME-TASK OWNER RESUME" in kwargs["messages"][0]["content"]
                assert "Verified local content" in kwargs["messages"][0]["content"]
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "tests_1",
                            "sandbox_execute_offline",
                            {"command": ["pytest"], "workspace": "tests"},
                        )
                    ]
                )
            return LLMResponse(
                content="The heading is `# DARKLINGER`; the verified test run passed."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    paused = await agent.run(
        "Read README.md, run the tests, and report its heading and test result."
    )
    checkpoint_root = agent._agent_trace_root / "checkpoints"
    checkpoints = list(checkpoint_root.glob("*.json"))
    assert len(checkpoints) == 1
    task_id = json.loads(checkpoints[0].read_text(encoding="utf-8"))["task_id"]
    assert "/continue" in paused
    assert "Verified findings:" in paused
    assert "Verified local content" in paused
    paused_payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert "progress_summary" in paused_payload["owner_checkpoint"]

    answer = await agent.run("/continue")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "The heading is `# DARKLINGER`; the verified test run passed."
    assert agent.tools.calls == [
        ("read_file", {"path": "README.md"}),
        (
            "sandbox_execute_offline",
            {"command": ["pytest"], "workspace": "tests"},
        ),
    ]
    assert len(list(checkpoint_root.glob("*.json"))) == 1
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert payload["task_id"] == task_id
    assert payload["status"] == "completed"
    assert [call["sequence"] for call in payload["tool_calls"]] == [1, 2]
    journal = (
        agent._agent_trace_root / "journal" / f"{task_id}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_awaiting_owner" in journal
    assert "task_resumed_by_owner" in journal
    assert "task_completed" in journal


def test_owner_progress_report_filters_raw_browser_scaffolding() -> None:
    report = Agent._owner_progress_report(
        {
            "findings": [
                '### Result No matches found for "Firecrawl alternatives".',
                (
                    "browser_click: browser_click requires an element identifier."
                ),
                (
                    "### Ran Playwright code `await page.goto(...)` "
                    "### Page - Page URL: https://thunderbit.com/blog "
                    "- Page Title: Web Scraper Tips & Web Automation Guides "
                    "| Thunderbit - Console: 0 errors, 1 warnings "
                    '- heading "Crawl4AI" [level=2] '
                    '- heading "Crawlee" [level=2] '
                    "### Snapshot ```yaml - generic [active] [ref=f10e1]"
                ),
            ],
            "open_questions": [
                "Inspect a verified Firecrawl alternative detail page"
            ],
            "next_steps": [],
        },
        [
            {
                "tool": "browser_find",
                "status": "succeeded",
                "result_excerpt": (
                    '### Result No matches found for "Firecrawl alternatives".'
                ),
            },
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": (
                    "### Ran Playwright code `await page.goto(...)` "
                    "### Page - Page URL: https://thunderbit.com/blog "
                    "- Page Title: Web Scraper Tips & Web Automation Guides "
                    "| Thunderbit - Console: 0 errors, 1 warnings "
                    '- heading "Crawl4AI" [level=2] '
                    '- heading "Crawlee" [level=2] '
                    "### Snapshot ```yaml - generic [active] [ref=f10e1]"
                ),
            }
        ],
        [],
    )

    assert "No match for “Firecrawl alternatives”" in report
    assert "Source: https://thunderbit.com/blog" in report
    assert "Verified sections: Crawl4AI; Crawlee" in report
    assert "browser_click requires" not in report
    assert "Ran Playwright" not in report
    assert "generic [" not in report
    assert "[ref=" not in report


def test_owner_progress_report_merges_rich_ledger_after_bounded_rollover() -> None:
    report = Agent._owner_progress_report(
        {
            "findings": [
                (
                    "browser_snapshot: - Page URL: "
                    "https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives "
                    "- Page Title: Loading https://thunderbit.com"
                )
            ],
            "open_questions": [],
            "next_steps": ["Continue the original objective using real tools."],
        },
        [
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": (
                    "- Page URL: "
                    "https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives\n"
                    "- Page Title: Open-source Firecrawl alternatives\n"
                    '[DARKLINGER prioritized topic-relevant detail-page evidence]\n'
                    '- heading "1. Scrapy: large-scale Python crawling" [level=2]\n'
                    '- heading "2. Apache Nutch: enterprise search" [level=2]'
                ),
            }
        ],
        [],
    )

    assert report.count(
        "Source: https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives"
    ) == 1
    assert "Observed candidate labels: Scrapy; Apache Nutch" in report
    assert "Continue the original objective" not in report


def test_owner_progress_report_ignores_stale_rollover_questions() -> None:
    report = Agent._owner_progress_report(
        {
            "findings": [],
            "open_questions": [
                "browser_snapshot",
                "browser_navigate:distinct_detail_page",
            ],
            "next_steps": ["Continue the original objective using real tools."],
        },
        [
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": (
                    "- Page URL: https://example.test/complete\n"
                    "- Page Title: Complete observed source\n"
                    '- heading "Observed result" [level=2]'
                ),
            }
        ],
        [],
    )

    assert "Still open:" not in report
    assert "distinct detail page" not in report


def test_owner_progress_report_unwraps_truncated_web_read_json() -> None:
    excerpt = (
        '{"requested_url":"https://github.com/example/project/blob/main/README.md",'
        '"content":"### Page\\n- Page URL: '
        'https://raw.githubusercontent.com/example/project/main/README.md\\n'
        '- Page Title: README.md\\n### Snapshot\\n```yaml\\n'
        '- generic [active] [ref=f1]'
    )

    report = Agent._owner_progress_report(
        None,
        [{"tool": "web_read", "status": "succeeded", "result_excerpt": excerpt}],
        [],
    )

    assert "Source: https://raw.githubusercontent.com/example/project/main/README.md" in report
    assert "generic [" not in report
    assert '"requested_url"' not in report


def test_owner_progress_report_is_a_reusable_recovery_checkpoint() -> None:
    github_repo = "https://github.com/example/onion-audit"
    guide = "https://security.example/darknet-tools"
    opened_only = "https://catalogue.example/osint-list"
    report = Agent._owner_progress_report(
        None,
        [
            {
                "tool": "web_search",
                "status": "succeeded",
                "result_excerpt": json.dumps(
                    {
                        "engine": "duckduckgo",
                        "query": "darknet tools github",
                        "results": [
                            {"title": "Onion Audit", "url": github_repo},
                            {"title": "Darknet tools guide", "url": guide},
                            {"title": "OSINT list", "url": opened_only},
                        ],
                    }
                ),
            },
            {
                "tool": "browser_navigate",
                "status": "succeeded",
                "arguments": {"url": guide},
                "result_excerpt": f"- Page URL: {guide}",
            },
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": (
                    f"- Page URL: {guide}\n"
                    "- Page Title: Darknet tools guide\n"
                    '- heading "OnionScan" [level=2]'
                ),
            },
            {
                "tool": "browser_navigate",
                "status": "succeeded",
                "arguments": {"url": opened_only},
                "result_excerpt": f"- Page URL: {opened_only}",
            },
        ],
        [],
        objective=(
            "Przeszukaj GitHuba pod kątem narzędzi darknet, oceń repozytoria "
            "and present candidates for approval."
        ),
        failed_calls=[
            {
                "tool": "browser_snapshot",
                "status": "failed",
                "error": "page closed before capture",
            }
        ],
        stop_reason="the final model report failed grounding validation",
    )

    assert "Recovery checkpoint:" in report
    assert "Przeszukaj GitHuba pod kątem narzędzi darknet" in report
    assert "Stop reason: the final model report failed grounding validation" in report
    assert f"Inspected (page content captured): {guide}" in report
    assert f"Opened only (content not captured): {opened_only}" in report
    assert f"Discovered only (not opened or evaluated): {github_repo}" in report
    assert "no GitHub page content was inspected" in report
    assert "Search GitHub directly, open concrete repository pages" in report
    assert "do not repeat the searches" in report
    assert "page closed before capture" in report


def test_verified_fallback_report_preserves_objective_and_resume_point() -> None:
    report = Agent._owner_verified_final_report(
        None,
        [
            {
                "tool": "web_search",
                "status": "succeeded",
                "result_excerpt": json.dumps(
                    {
                        "engine": "duckduckgo",
                        "query": "darknet tools",
                        "results": [
                            {
                                "title": "Candidate repository",
                                "url": "https://github.com/example/candidate",
                            }
                        ],
                    }
                ),
            }
        ],
        TaskContract(requires_evidence_report=True),
        objective="Inspect GitHub repositories and recommend tools to adopt.",
        stop_reason="two final rewrites were rejected",
    )

    assert "Recovery checkpoint:" in report
    assert "Objective: Inspect GitHub repositories" in report
    assert "two final rewrites were rejected" in report
    assert "Discovered only (not opened or evaluated)" in report
    assert "no GitHub page content was inspected" in report
    assert "Resume from here:" in report


@pytest.mark.asyncio
async def test_owner_stop_closes_checkpoint_without_another_model_call(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            return "partial verified evidence"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        "navigate_1",
                        "browser_navigate",
                        {"url": "https://duckduckgo.com/?q=darklinger"},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    await agent.run("Search the internet for DARKLINGER alternatives and report them")
    assert agent.llm.calls == 1

    answer = await agent.run("/stop")
    await asyncio.gather(*agent._memory_tasks)

    assert answer.startswith("Stopped task")
    assert agent.llm.calls == 1
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "stopped"
    assert payload["finished_at"]
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_stopped_by_owner" in journal


@pytest.mark.asyncio
async def test_owner_continuous_authorization_crosses_silent_batch_boundaries(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_navigate":
                return "Opened the antiques marketplace"
            return (
                "Search result: https://antiques.example/prussian-helmet. "
                "Verified offer: Prussian WWI helmet, seller page inspected"
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate_1",
                            "browser_navigate",
                            {"url": "https://duckduckgo.com/?q=prussian+wwi+helmet"},
                        )
                    ]
                )
            if self.calls == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot_1", "browser_snapshot", {})
                    ]
                )
            if self.calls == 3:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate_detail",
                            "browser_navigate",
                            {"url": "https://antiques.example/prussian-helmet"},
                        )
                    ]
                )
            if self.calls == 4:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot_detail", "browser_snapshot", {})
                    ]
                )
            return LLMResponse(
                content="I found and inspected a verified Prussian helmet offer, Boss."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    paused = await agent.run(
        "Search the internet for a Prussian WWI helmet and report what you find"
    )
    assert "/continue --continuous" in paused

    answer = await agent.run("/continue --continuous")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == (
        "I found and inspected a verified Prussian helmet offer, Boss."
    )
    assert agent.tools.calls == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["owner_authorization"]["mode"] == "continuous"
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_continuous_authorized_by_owner" in journal
    assert "task_continuous_batch_checkpoint" in journal


@pytest.mark.asyncio
async def test_contract_finalization_blocks_identical_calls_before_continuous_mode(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [{"type": "function", "function": {"name": "read_file"}}]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
            return "Verified initial content"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"read_{self.calls}",
                        "read_file",
                        {"path": "README.md"},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent.run("Read README.md and analyse it")
    await asyncio.gather(*agent._memory_tasks)

    assert "I killed that loop" in answer
    assert "runtime-verified result" in answer
    assert tools.calls == 1
    assert agent.llm.calls == 3
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert journal.count("post_contract_tool_call_rejected") == 2


@pytest.mark.asyncio
async def test_identical_tool_call_loop_is_rejected_and_paused_early(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
            return "Same captured search result"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"snapshot_{self.calls}",
                        "browser_snapshot",
                        {"target": "#same-result", "filename": "same.html"},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 8

    answer = await agent._run_agent_loop(
        "Search the internet and report the verified result"
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "kept requesting the same" in answer
    assert "/continue" in answer
    assert agent.tools.calls == 2
    assert agent.llm.calls == 4
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "awaiting_owner"
    assert [call["status"] for call in payload["tool_calls"]] == [
        "succeeded",
        "succeeded",
        "failed",
    ]
    assert payload["tool_calls"][-1]["error"].startswith(
        "RepeatedToolCallError"
    )
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "repeated_tool_loop_detected" in journal


@pytest.mark.asyncio
async def test_successful_but_nonprogressing_tool_calls_are_bounded(
    tmp_path: Path,
) -> None:
    urls = [f"https://example.test/about-{index}" for index in range(1, 8)]

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "web_read",
                        "parameters": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                        },
                    },
                },
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "web_search":
                return json.dumps(
                    {
                        "query": arguments["query"],
                        "results": [
                            {"title": f"Example Labs {index}", "url": url}
                            for index, url in enumerate(urls, start=1)
                        ],
                    }
                )
            index = urls.index(arguments["url"]) + 1
            return json.dumps(
                {
                    "url": arguments["url"],
                    "content": (
                        f"Example Labs information page {index}. This page "
                        "contains background material but no postal location."
                    ),
                }
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "search",
                            "web_search",
                            {"query": "Example Labs physical address"},
                        )
                    ]
                )
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"read-{self.calls}",
                        "web_read",
                        {"url": urls[self.calls - 2]},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 16

    answer = await agent._run_agent_loop(
        "Search the web for the physical address of Example Labs and report "
        "a verified result."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "mechanically successful" in answer
    assert "did not reduce" in answer
    assert len(tools.calls) < agent.MAX_AGENT_STEPS
    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["status"] == "blocked"
    journal = (
        agent._agent_trace_root
        / "journal"
        / f"{checkpoint['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "contract_progress_warning" in journal
    assert "contract_progress_stalled" in journal


@pytest.mark.asyncio
async def test_direct_url_failure_prevents_alternating_guessed_domains(
    tmp_path: Path,
) -> None:
    urls = (
        "https://missing-one.invalid",
        "https://missing-two.invalid",
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": "browser_navigate"}}
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(arguments["url"])
            raise RuntimeError("NS_ERROR_UNKNOWN_HOST")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            sequence = (urls[0], urls[1], urls[0], urls[1], urls[0], urls[0])
            url = sequence[min(self.calls, len(sequence) - 1)]
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"navigate_{self.calls}",
                        "browser_navigate",
                        {"url": url},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 10

    answer = await agent._run_agent_loop(
        "Inspect https://missing-one.invalid and report a verified result"
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "tried `https://missing-one.invalid` exactly once" in answer
    assert tools.calls == [urls[0]]
    assert agent.llm.calls == 1
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert len(payload["tool_calls"]) == 1
    assert payload["tool_calls"][0]["status"] == "failed"


def test_browser_request_identity_normalizes_root_slash_and_fragment() -> None:
    first = Agent._tool_request_identity(
        "browser_navigate", {"url": "HTTPS://EXAMPLE.COM/"}
    )
    second = Agent._tool_request_identity(
        "browser_navigate", {"url": "https://example.com#section"}
    )

    assert first == second


def test_initial_web_discovery_replaces_unverified_model_domain() -> None:
    prompt = "Search the internet for a Firecrawl alternative and report it."
    contract = TaskContract.from_prompt(prompt)

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://invented-marketplace.invalid"},
        contract,
        [],
        [],
    )

    assert repaired["url"].startswith("https://duckduckgo.com/?q=")
    assert "Firecrawl" in repaired["url"]


def test_spoken_direct_url_replaces_model_search_or_wikipedia_guess() -> None:
    prompt = (
        "Otwórz HTTPS, dwukropek, łamane, łamane, this, minus, domain, "
        "minus, definitely, minus, those, minus, not, minus, exist, "
        "kropka, invalid. I powiedz co znalazłaś."
    )
    target = "https://this-domain-definitely-those-not-exist.invalid"
    contract = TaskContract.from_prompt(prompt)

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://pl.wikipedia.org/wiki/Dwukropek"},
        contract,
        [],
        [],
        preferred_target=target,
    )

    assert repaired == {"url": target}


def test_discovery_query_excludes_conditional_artifact_workflow() -> None:
    prompt = (
        "V, przeszukaj sieć w celu znalezienia alternatywy dla Firecrawlera. "
        "Jeżeli nic nie znajdziesz, stwórz własne narzędzie."
    )
    contract = TaskContract.from_prompt(prompt)

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://invented.example"},
        contract,
        [],
        [],
    )

    assert "Firecrawlera" in repaired["url"]
    assert "Je%C5%BCeli" not in repaired["url"]
    assert "stw%C3%B3rz" not in repaired["url"]


def test_discovery_query_skips_persona_directed_greeting_sentence() -> None:
    prompt = (
        "Cześć V, mam dla Ciebie zadanie. "
        "Znajdź proszę Cię alternatywę dla Firecrawlera. "
        "Jeżeli nie znajdziesz alternatywy, stwórz podobne narzędzie."
    )
    contract = TaskContract.from_prompt(prompt).merged(
        TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
            requires_distinct_detail_page=True,
            requires_evidence_report=True,
        )
    )

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://marketplace.dify.ai"},
        contract,
        [],
        [],
    )

    assert "Firecrawlera" in repaired["url"]
    assert "Cze%C5%9B%C4%87" not in repaired["url"]
    assert "mam+dla+Ciebie+zadanie" not in repaired["url"]
    assert "Je%C5%BCeli" not in repaired["url"]


def test_discovery_query_skips_generic_task_importance_preamble() -> None:
    prompt = (
        "Mam dla Ciebie bardzo ważne zadanie. "
        "Znajdź w sieci alternatywę dla Firecrawlera. "
        "Jeżeli nie znajdziesz alternatywy, stwórz podobne narzędzie."
    )

    query = Agent._discovery_search_query(prompt)

    assert query == "Znajdź w sieci alternatywę dla Firecrawlera"
    assert "ważne zadanie" not in query
    assert "stwórz" not in query


def test_discovery_query_skips_period_terminated_voice_greeting() -> None:
    prompt = (
        "Cześć V. Posłuchaj, wejdź w internet i znajdź informacje na temat "
        "narzędzi albo skili, które pomagają sztucznej inteligencji "
        "samodzielnie rozwiązać kapcza. Jeżeli nic nie znajdziesz, stwórz "
        "takie narzędzie."
    )

    query = Agent._discovery_search_query(prompt)

    assert "kapcza" in query
    assert "Cześć V" not in query
    assert "Jeżeli" not in query


def test_discovery_query_skips_multiple_voice_preamble_sentences() -> None:
    prompt = (
        "Cześć V. Jak tam wieczorek? Słuchaj, chciałbym żebyś zebrała mi "
        "informacje na temat europejskich raportów o marketach darknetowych. "
        "Szczególnie interesują mnie raporty organów ścigania."
    )

    query = Agent._discovery_search_query(prompt)

    assert "marketach darknetowych" in query
    assert "Jak tam wieczorek" not in query
    assert "Cześć V" not in query


def test_discovery_query_keeps_scope_after_github_source_sentence() -> None:
    prompt = (
        "Ty, czemu się zapętlasz? Jeszcze raz tłumaczę, o co mi chodzi. "
        "Znajdź przydatne narzędzia i umiejętności na GitHubie. "
        "Narzędzia i umiejętności odnośnie darknetu, czyli OSINT, cyber "
        "security i inwigilacja. Najpierw podaj mi listę do zatwierdzenia."
    )

    query = Agent._discovery_search_query(prompt).casefold()

    assert "github" in query
    assert "darknet" in query
    assert "osint" in query
    assert "cyber" in query
    assert "zatwierdzenia" not in query


def test_discovery_query_skips_unnamed_short_greeting_before_long_request() -> None:
    prompt = (
        "Cześć, mam dla Ciebie zadanie. Posłuchaj, znajdź mi wszystko, co "
        "można ustalić na temat nocników wyprodukowanych przed wojną, "
        "szczególnie austriackich i francuskich."
    )

    query = Agent._discovery_search_query(prompt)

    assert "nocnik" in query.casefold()
    assert "mam dla Ciebie zadanie" not in query


def test_discovery_query_does_not_treat_roman_war_number_as_product() -> None:
    prompt = (
        "V, posłuchaj, wyszukaj mi w internecie informacje na temat nocników, "
        "ale nie takich zwykłych nocników, tylko takich, które były produkowane "
        "przed II wojną światową, szczególnie nocniki produkcji austriackiej i "
        "francuskiej. Chciałbym poznać ceny i miejsca zakupu."
    )

    query = Agent._discovery_search_query(prompt).casefold()

    assert "nocnik" in query
    assert "austriack" in query
    assert "francusk" in query
    assert query != "były produkowane przed ii"


def test_grounded_semantic_query_beats_runtime_greeting_clause() -> None:
    prompt = (
        "Cześć, mam dla Ciebie zadanie. Znajdź informacje o nocnikach "
        "austriackich i francuskich produkowanych przed wojną."
    )

    runtime = "Cześć, mam dla Ciebie zadanie"
    semantic = "nocniki austriackie francuskie przed wojną"

    assert Agent._query_remainder_overlap(runtime, prompt) == 0
    assert Agent._query_remainder_overlap(semantic, prompt) >= 3


def test_topic_relevance_rejects_unrelated_detail_product() -> None:
    calls = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/product/coffee-mug"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://example.test/product/coffee-mug\n"
                "Coffee mug with a funny office slogan, price 39 PLN."
            ),
        },
    ]

    assert Agent._topic_relevance_missing(
        "antique Austrian French chamber pots",
        calls,
    ) == ["browser_evidence:topic_mismatch"]


def test_topic_relevance_accepts_matching_detail_product() -> None:
    calls = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/product/chamber-pot"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://example.test/product/chamber-pot\n"
                "Antique French porcelain chamber pot, circa 1920."
            ),
        },
    ]

    assert Agent._topic_relevance_missing(
        "antique Austrian French chamber pots",
        calls,
    ) == []


def test_topic_relevance_requires_each_source_to_match_multi_source_subject() -> None:
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "url": "https://example.test/mushrooms/species",
                            "title": "Regional mushroom species with photographs",
                        },
                        {
                            "url": "https://example.test/microdosing/adhd",
                            "title": "Microdosing and mental health",
                        },
                    ]
                }
            ),
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/mushrooms/species"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": "Regional mushroom species and photographs.",
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/microdosing/adhd"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": "Microdosing and ADHD mental health research.",
        },
    ]

    assert Agent._topic_relevance_missing(
        "all regional mushroom species photographs",
        calls,
        minimum_sources=2,
        strict_each_source=True,
    ) == ["browser_evidence:topic_sources=1/2"]


def test_topic_relevance_accepts_two_matching_subject_sources() -> None:
    calls = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://one.test/mushroom-species"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": "Regional mushroom species illustrated catalogue.",
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://two.test/mushroom-atlas"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": "Photographs and descriptions of regional mushroom species.",
        },
    ]

    assert Agent._topic_relevance_missing(
        "all regional mushroom species photographs",
        calls,
        minimum_sources=2,
        strict_each_source=True,
    ) == []


def test_runtime_repairs_web_search_that_drops_owner_subject() -> None:
    prompt = (
        "Znajdź informacje o narzędziach pomagających samodzielnie rozwiązać "
        "kapcza."
    )
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
    )
    preferred = Agent._discovery_search_query(prompt)

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "web_search",
        {"query": "tools for solving AI independently", "max_results": 5},
        contract,
        [],
        [],
        preferred_query=preferred,
    )

    assert repaired == {"query": preferred, "max_results": 5}


def test_web_search_refinement_cannot_drop_observed_subject_anchor() -> None:
    focused = (
        "Wynajdź i opisz wszystkie gatunki dzikich kotów występujące "
        "naturalnie w Europie"
    )
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "title": "Dzikie koty Europy",
                            "url": "https://one.test/article/dzikie-koty-europy",
                        },
                        {
                            "title": "Encyklopedia dzikich kotów",
                            "url": "https://two.test/article/dzikie-koty",
                        },
                    ]
                }
            ),
        }
    ]
    contract = TaskContract(
        requires_web_discovery=True,
        required_research_facets=("exhaustive_coverage",),
    )

    repaired = Agent._repair_web_discovery_navigation(
        focused,
        "web_search",
        {
            "query": (
                "Wynajdź wszystkie gatunki kotów występujących naturalnie "
                "w Europie"
            ),
            "max_results": 10,
        },
        contract,
        calls,
        [],
        preferred_query=focused,
    )

    assert repaired["query"] == focused


def test_exhaustive_search_keeps_enough_candidates() -> None:
    contract = TaskContract(
        requires_web_discovery=True,
        required_research_facets=("exhaustive_coverage",),
    )

    repaired = Agent._repair_web_discovery_navigation(
        "Find every regional species.",
        "web_search",
        {"query": "regional species", "max_results": 2},
        contract,
        [],
        [],
        preferred_query="regional species",
    )

    assert repaired["max_results"] == 10


def test_invented_web_read_url_is_replaced_with_grounded_result() -> None:
    grounded = "https://catalogue.test/"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Regional species catalogue",
                            "url": grounded,
                        }
                    ]
                }
            ),
        }
    ]
    contract = TaskContract(requires_web_discovery=True)

    repaired = Agent._repair_web_discovery_navigation(
        "Find regional species.",
        "web_read",
        {"url": "https://invented.test/not-observed"},
        contract,
        calls,
        [],
        preferred_query="regional species",
    )

    assert repaired == {"url": grounded}


def test_runtime_keeps_owner_grounded_search_refinement() -> None:
    prompt = (
        "Search the internet for antique chamber pots made before the war. I especially "
        "need Austrian and French examples, prices, sellers, and collecting rules."
    )
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
    )
    preferred = "antique chamber pots made before the war"
    refined = (
        "antique chamber pots Austrian French prices sellers collecting rules"
    )

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "web_search",
        {"query": refined, "max_results": 10},
        contract,
        [],
        [],
        preferred_query=preferred,
    )

    assert repaired["query"] == refined


def test_broad_research_requires_two_verified_detail_sources() -> None:
    prompt = (
        "Search the internet for antique chamber pots made before the war. I especially "
        "need Austrian and French examples, prices, sellers, collecting rules, "
        "common varieties, rare varieties, materials, dates, and provenance; "
        "then report the findings."
    )
    contract = TaskContract.from_prompt(prompt)
    first = "https://museum.test/article/chamber-pots"
    second = "https://auction.test/product/french-pot"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {"url": first, "title": "History"},
                        {"url": second, "title": "French example"},
                    ]
                }
            ),
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": first},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": "Chamber pot history and materials.",
        },
    ]

    assert contract.minimum_detail_sources == 2
    assert "browser_evidence:detail_sources=1/2" in contract.unmet(calls)

    calls.extend(
        [
            {
                "tool": "browser_navigate",
                "status": "succeeded",
                "arguments": {"url": second},
            },
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": "French chamber pot sold for 400 EUR.",
            },
        ]
    )

    assert "browser_evidence:detail_sources=1/2" not in contract.unmet(calls)


def test_research_detail_snapshot_expands_narrow_model_target() -> None:
    contract = TaskContract(
        requires_evidence_report=True,
        requires_distinct_detail_page=True,
    )
    calls = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://museum.test/article/chamber-pots"},
        }
    ]

    repaired = Agent._repair_research_detail_snapshot(
        "browser_snapshot",
        {
            "target": "//*[@id='article']/h1",
            "filename": "heading.md",
            "depth": 2,
            "boxes": True,
        },
        contract,
        calls,
    )

    assert repaired == {"depth": 4, "target": "body"}


def test_research_search_snapshot_discards_foreign_engine_xpath() -> None:
    contract = TaskContract(
        requires_evidence_report=True,
        requires_distinct_detail_page=True,
        requires_web_discovery=True,
    )
    calls = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://duckduckgo.com/?q=vintage+cups"},
        }
    ]

    repaired = Agent._repair_research_detail_snapshot(
        "browser_snapshot",
        {
            "target": "//*[@id='rso']/div[3]/h2",
            "depth": 3,
            "boxes": True,
        },
        contract,
        calls,
    )

    assert repaired == {"depth": 4, "target": "body"}


def test_runtime_recovers_public_fact_search_with_new_focused_query() -> None:
    prompt = (
        "Sprawdź ile jest w Warszawie cukierni Cud Malina i podaj wszystkie "
        "adresy oraz numery telefonów."
    )
    contract = SemanticIntent(
        action_requested=True,
        capabilities=("browser",),
        requires_report=True,
    ).to_contract(prompt)
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": "Cukiernia Cud Malina Warszawa"},
            "result_excerpt": "Cud malina — tort, cena 120 zł",
        },
        {
            "tool": "web_read",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/cud-malina-tort"},
            "result_excerpt": "Cud malina. Torty. Telefon 698 314 125.",
        },
    ]
    definitions = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "web_read"}},
    ]

    recovered = Agent._public_fact_recovery_request(
        prompt,
        contract,
        successful,
        definitions,
        preferred_query="Cukiernie Cud Malina w Warszawie",
    )

    assert recovered is not None
    assert recovered[0] == "web_search"
    assert recovered[1]["max_results"] == 10
    assert "address" in recovered[1]["query"]
    assert "Cud Malina" in recovered[1]["query"]
    assert recovered[1]["query"] != "Cukiernia Cud Malina Warszawa"


def test_public_fact_recovery_does_not_repeat_exhausted_queries() -> None:
    prompt = "Find the address and phone number for Acme Bakery."
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_evidence_report=True,
        required_public_fields=("address", "contact"),
    )
    definitions = [
        {"type": "function", "function": {"name": "web_search"}},
    ]
    variants = (
        "Acme Bakery address phone contact",
        '"Acme Bakery" address phone contact official',
        '"Acme Bakery" address phone contact business directory',
        '"Acme Bakery" address phone contact map listing',
    )
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": query},
            "result_excerpt": "No verified address in these results.",
        }
        for query in variants
    ]

    recovered = Agent._public_fact_recovery_request(
        prompt,
        contract,
        successful,
        definitions,
        preferred_query="Acme Bakery",
    )

    assert recovered is None


def test_semantic_web_query_overrides_model_search_query() -> None:
    prompt = "Bonjour V. Trouve une alternative à Firecrawler."
    contract = TaskContract.from_prompt(prompt).merged(
        TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
        )
    )

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://duckduckgo.com/?q=Bonjour+V"},
        contract,
        [],
        [],
        preferred_query="alternative à Firecrawler",
    )

    assert repaired["url"] == (
        "https://duckduckgo.com/?q=alternative+%C3%A0+Firecrawler"
    )


def test_grounded_follow_up_search_keeps_concise_multilingual_scope() -> None:
    prompt = (
        "Znajdź przydatne narzędzia i umiejętności na GitHubie. "
        "Narzędzia i umiejętności odnośnie darknetu szeroko pojętego, "
        "czyli OSINT, cyber security, inwigilacja, cokolwiek."
    )
    focused_query = Agent._discovery_search_query(prompt)
    requested_query = "darknet OSINT GitHub repositories"
    contract = TaskContract.from_prompt(prompt)
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {
                "query": "darknet OSINT cyber security GitHub repositories"
            },
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "title": "GitHub - owner/first-darknet-tool",
                            "url": "https://github.com/owner/first-darknet-tool",
                        }
                    ]
                }
            ),
        }
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "web_search",
        {"query": requested_query, "max_results": 5},
        contract,
        successful,
        [],
        preferred_query=focused_query,
    )

    assert repaired["query"] == requested_query


def test_model_copied_detail_url_is_repaired_from_search_evidence() -> None:
    prompt = "Znajdź w internecie alternatywę dla Firecrawlera."
    contract = TaskContract.from_prompt(prompt)
    successful = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://duckduckgo.com/?q=firecrawler"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {},
            "result_excerpt": (
                "Result URL: https://thunderbit.com/pl/blog/"
                "open-source-firecrawl-alternatives\n"
                "Another: https://www.minibase.md/pl/blog/firecrawl-alternatives/"
            ),
        },
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {
            "url": (
                "https://thunderbit.com/pl/blog/"
                "open-source_firecrawl_alternatives"
            )
        },
        contract,
        successful,
        [],
    )

    assert repaired["url"] == (
        "https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives"
    )


def test_model_mangled_web_read_url_is_repaired_from_search_evidence() -> None:
    prompt = "Znajdź dwie oferty kolekcjonerskich filiżanek."
    exact = (
        "https://allegro.pl/kategoria/design-i-antyki-porcelana-26163"
        "?string=kolekcjonerskie%20fili%C5%BCanki"
    )
    contract = TaskContract.from_prompt(prompt).merged(
        TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
        )
    )
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {"results": [{"title": "Filiżanki", "url": exact}]}
            ),
        }
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "web_read",
        {
            "url": (
                "https://allegro.pl/kategoria/design-and-antiques-porcelain-26163/"
                "string=kolekcjonerskie%f20fili%c5%bcanki"
            )
        },
        contract,
        successful,
        [],
    )

    assert repaired == {"url": exact}


def test_model_corrupted_hostname_is_repaired_from_high_level_search_evidence() -> None:
    prompt = "Znajdź adres i godziny otwarcia cukierni Miód Malina."
    contract = TaskContract.from_prompt(prompt).merged(
        TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
            requires_distinct_detail_page=True,
        )
    )
    official = "https://warszawa.miodmalina.eu/kontakt/"
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": "Miód Malina Warszawa"},
            "result_excerpt": json.dumps(
                {"results": [{"title": "Kontakt", "url": official}]}
            ),
        }
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://warszawa.miodmalina e_u/kontakt/"},
        contract,
        successful,
        [],
    )

    assert repaired["url"] == official


def test_unobserved_detail_url_returns_to_grounded_search() -> None:
    prompt = (
        "Mam dla Ciebie bardzo ważne zadanie. "
        "Znajdź w sieci alternatywę dla Firecrawlera."
    )
    contract = TaskContract.from_prompt(prompt)
    successful = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {
                "url": "https://duckduckgo.com/?q=Firecrawler+alternative"
            },
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {},
            "result_excerpt": (
                "Result URL: https://example.org/real-firecrawl-alternative"
            ),
        },
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://github.com/firecrawlers/firecrawler"},
        contract,
        successful,
        [],
    )

    assert repaired["url"] == (
        "https://duckduckgo.com/?q="
        "Znajd%C5%BA+w+sieci+alternatyw%C4%99+dla+Firecrawlera"
    )


def test_long_single_sentence_query_focuses_on_repeated_product() -> None:
    prompt = (
        "Cześć V, słuchaj mam dla Ciebie takie zadanko małe, weź proszę Cię "
        "znajdź alternatywę dla Firecrawlera i jeżeli nie uda Ci się znaleźć "
        "dla niego alternatywy darmowej, spróbuj stworzyć narzędzie podobne "
        "do Firecrawlera."
    )

    query = Agent._discovery_search_query(prompt)

    assert query == "znajdź alternatywę dla Firecrawlera"
    assert "zadanko" not in query
    assert "stworzyć" not in query


def test_repeated_long_verb_cannot_replace_discovery_subject() -> None:
    prompt = (
        "Znajdź słabe punkty w konstrukcji tego forum i zobacz jak można je "
        "położyć i co potrzebujesz do tego żeby je położyć."
    )

    query = Agent._discovery_search_query(prompt)

    assert "forum" in query
    assert query != "jak można je położyć"


def test_failed_direct_navigation_falls_back_to_duckduckgo() -> None:
    prompt = "Inspect https://missing.invalid and report the result."
    contract = TaskContract.from_prompt(prompt)
    failure = {
        "tool": "browser_navigate",
        "status": "failed",
        "arguments": {"url": "https://missing.invalid"},
    }

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://another-invented.invalid"},
        contract,
        [],
        [failure],
    )

    assert repaired["url"].startswith("https://duckduckgo.com/?q=")


def test_discovery_never_returns_to_google_after_successful_duckduckgo() -> None:
    prompt = "Search the internet for a Firecrawl alternative."
    contract = TaskContract.from_prompt(prompt)
    success = {
        "tool": "browser_navigate",
        "status": "succeeded",
        "arguments": {"url": "https://duckduckgo.com/?q=firecrawl"},
    }

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://www.google.com/search?q=firecrawler+alternative"},
        contract,
        [success],
        [],
    )

    assert repaired["url"] == (
        "https://duckduckgo.com/?q=firecrawler+alternative"
    )


def test_owner_supplied_google_url_is_not_rewritten_without_failure() -> None:
    prompt = "Inspect https://www.google.com and report its title."
    contract = TaskContract.from_prompt(prompt)

    unchanged = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://www.google.com"},
        contract,
        [],
        [],
    )

    assert unchanged == {"url": "https://www.google.com"}


@pytest.mark.asyncio
async def test_cancelled_agent_task_is_checkpointed_as_stopped(tmp_path: Path) -> None:
    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def ask(self, **kwargs) -> str:
            raise asyncio.CancelledError

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.memory = SimpleNamespace(session=Session())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.tools = SimpleNamespace()
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    with pytest.raises(asyncio.CancelledError):
        await agent._run_agent_loop("Perform a long task")

    checkpoints = list(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    assert len(checkpoints) == 1
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert payload["status"] == "stopped"
    journal = checkpoints[0].parent.parent / "journal" / (
        checkpoints[0].stem + ".jsonl"
    )
    assert "task_stopped" in journal.read_text(encoding="utf-8")


def test_fake_background_work_claims_are_detected() -> None:
    assert Agent._claims_unverified_work(
        "I'm initiating the systematic mining now."
    )
    assert Agent._claims_unverified_work(
        "The extraction is running in the background; I'll report back shortly."
    )
    assert Agent._claims_unverified_work(
        "Okay, let me dive into onehack.st. I'll walk through what I see. Ready?"
    )
    assert Agent._claims_unverified_work(
        "I gathered evidence, but I did not finish the requested extraction."
    )
    assert Agent._claims_unverified_work(
        "The visited-page scope was insufficient for the requested task."
    )
    assert Agent._claims_unverified_work(
        "I'll ring him up and tell him to call Boss right away."
    )
    assert not Agent._claims_unverified_work(
        "I did not run the extraction. Nothing is running in the background."
    )
    assert not Agent._claims_unverified_work(
        "My brain is still running on silicon, but the imagined sunset would "
        "feel more visceral than moving data around."
    )


@pytest.mark.asyncio
async def test_reflection_cannot_mark_model_claim_verified_without_runtime_evidence() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.prompt = ""

        async def ask(self, prompt: str, **kwargs) -> str:
            self.prompt = prompt
            return (
                '{"summary":"Extraction started","lesson":"Trust it",'
                '"importance":"high","remember":true,"kind":"experience",'
                '"source":"verified"}'
            )

    llm = LLMStub()
    reflection = await Reflection(llm).reflect(
        "Extract records",
        "I'm initiating extraction now.",
    )

    assert reflection.source == MemorySource.SELF_GENERATED
    assert "no_runtime_evidence" in llm.prompt
    assert "model-authored text, not proof" in llm.prompt


@pytest.mark.asyncio
async def test_reflection_rejects_fabricated_execution_before_llm_review() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            raise AssertionError("fabricated execution must not reach reflection LLM")

    reflection = await Reflection(LLMStub()).reflect(
        "Tell him to call Boss",
        (
            "He picked up. I used the remote desktop exploit to connect, and "
            "I told him to call Boss."
        ),
        execution={
            "status": "completed",
            "tool_calls": [],
            "successful_tool_count": 0,
        },
    )

    assert reflection.remember is False
    assert reflection.source == MemorySource.SELF_GENERATED
    assert "does not support" in reflection.summary


@pytest.mark.asyncio
async def test_background_memory_is_cancelled_for_next_user_request() -> None:
    started = asyncio.Event()

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, prompt: str, answer: str) -> None:
            started.set()
            await asyncio.Future()

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent._memory_tasks = set()

    await agent._remember_task("Analyze this substantial task", "Result")
    await started.wait()

    assert agent._memory_tasks
    await agent.cancel_background_memory()
    assert not agent._memory_tasks


def test_llm_configuration_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V_CORE_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("V_CORE_MODEL", "test-model")

    config = load_llm_config()

    assert config.base_url == "http://localhost:9999/v1"
    assert config.model == "test-model"


def test_llm_normalizes_chat_for_strict_gguf_templates() -> None:
    messages = LLM._normalize_system_messages(
        [
            {"role": "user", "content": "First"},
            {"role": "system", "content": "Identity"},
            {"role": "user", "content": "Second"},
            {"role": "system", "content": "Language"},
            {"role": "assistant", "content": "Answer"},
        ]
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == "Identity\n\nLanguage"
    assert messages[1]["content"] == "First\n\nSecond"


@pytest.mark.asyncio
async def test_llm_respond_preserves_native_tool_calls() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.request: dict = {}

        async def create(self, **kwargs):
            self.request = kwargs
            function = SimpleNamespace(
                name="read_file",
                arguments='{"path":"README.md"}',
            )
            message = SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(id="call_7", function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    llm._native_tools_supported = None
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = await llm.respond(
        messages=[{"role": "user", "content": "Read README"}],
        tools=definitions,
    )

    assert completions.request["tools"] == definitions
    assert completions.request["parallel_tool_calls"] is False
    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [
        LLMToolCall(
            call_id="call_7",
            name="read_file",
            arguments={"path": "README.md"},
            raw_arguments='{"path":"README.md"}',
        )
    ]
    assert response.native_tools_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"name":"read_file","arguments":{"path":"README.md"}}',
        (
            '<tool_call>\n{"name":"read_file","arguments":'
            '{"path":"README.md"}}\n</tool_call>'
        ),
    ],
)
async def test_llm_promotes_exact_allowed_textual_tool_call(content: str) -> None:
    class CompletionsStub:
        async def create(self, **kwargs):
            message = SimpleNamespace(content=content, tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=CompletionsStub())
    )
    llm._native_tools_supported = None
    response = await llm.respond(
        messages=[{"role": "user", "content": "Read README"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert response.content == ""
    assert response.native_tools_enabled is False
    assert response.tool_calls == [
        LLMToolCall(
            call_id="compatibility_call_1",
            name="read_file",
            arguments={"path": "README.md"},
            raw_arguments='{"path":"README.md"}',
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name":"read_file","arguments":{}}',
        '{"name":"browser_navigate","arguments":{"url":"https://example.com"}}',
        (
            'Explanation first. <tool_call>{"name":"read_file",'
            '"arguments":{"path":"README.md"}}</tool_call>'
        ),
    ],
)
async def test_llm_rejects_incomplete_unlisted_or_narrated_textual_call(
    content: str,
) -> None:
    class CompletionsStub:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content, tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=CompletionsStub())
    )
    llm._native_tools_supported = None
    response = await llm.respond(
        messages=[{"role": "user", "content": "Read README"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert response.content == content
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_llm_gives_local_artifact_generation_a_longer_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.request: dict = {}

        async def create(self, **kwargs):
            self.request = kwargs
            message = SimpleNamespace(content="done", tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    monkeypatch.setenv("V_CORE_TIMEOUT", "120")
    monkeypatch.setenv("V_CORE_ARTIFACT_TIMEOUT", "900")
    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    await llm.respond(
        messages=[{"role": "user", "content": "Create a tool"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "learning_create_tool",
                    "description": "Create a tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert completions.request["timeout"] == 900.0


@pytest.mark.asyncio
async def test_llm_gives_source_only_artifact_generation_a_longer_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.request: dict = {}

        async def create(self, **kwargs):
            self.request = kwargs
            message = SimpleNamespace(content="def run(arguments): return arguments", tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    monkeypatch.setenv("V_CORE_TIMEOUT", "120")
    monkeypatch.setenv("V_CORE_ARTIFACT_TIMEOUT", "900")
    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    await llm.respond(
        messages=[{"role": "user", "content": "Write the reusable operation"}],
        max_tokens=1536,
        artifact_generation=True,
    )

    assert completions.request["timeout"] == 900.0


def test_llm_local_transport_uses_one_long_request_without_hidden_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def client_stub(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.delenv("V_CORE_TIMEOUT", raising=False)
    monkeypatch.delenv("V_CORE_HTTP_RETRIES", raising=False)
    monkeypatch.setattr("v_core.llm.llm.AsyncOpenAI", client_stub)
    llm = object.__new__(LLM)
    llm._api_key = "local"
    llm.config = SimpleNamespace(base_url="http://127.0.0.1:5001/v1")

    llm._new_client()

    assert captured["timeout"] == 300.0
    assert captured["max_retries"] == 0


def test_agent_narrows_mixed_task_to_tool_creation_after_browser_evidence() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "browser_snapshot",
            "learning_list_artifacts",
            "learning_create_tool",
        )
    ]
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {"tool": "browser_navigate", "status": "succeeded"},
        {"tool": "browser_snapshot", "status": "succeeded"},
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert {item["function"]["name"] for item in selected} == {
        "learning_create_tool",
    }


def test_agent_forces_source_inspection_after_three_search_variants() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "web_search",
            "web_read",
            "browser_navigate",
            "browser_snapshot",
        )
    ]
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
    )
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": f"variant {index}"},
            "result_excerpt": (
                f"Result {index}: https://example.test/item-{index}"
            ),
        }
        for index in range(3)
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)
    selected_names = {item["function"]["name"] for item in selected}

    assert "web_search" not in selected_names
    assert "browser_navigate" in selected_names
    assert "web_read" in selected_names


def test_agent_stops_retrying_failed_web_reader_when_browser_is_available() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "web_search",
            "web_read",
            "browser_navigate",
            "browser_snapshot",
        )
    ]
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
    )
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": "Result: https://example.test/item",
        }
    ]
    failed = [
        {"tool": "web_read", "status": "failed"},
        {"tool": "web_read", "status": "failed"},
    ]

    selected = Agent._phase_tool_definitions(
        contract,
        definitions,
        successful,
        failed,
    )
    selected_names = {item["function"]["name"] for item in selected}

    assert "web_read" not in selected_names
    assert "browser_navigate" in selected_names
    assert "browser_snapshot" in selected_names


def test_agent_hides_snapshot_after_latest_navigation_failed() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "web_search",
            "browser_navigate",
            "browser_snapshot",
        )
    ]
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
    )
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "sequence": 1,
            "result_excerpt": "https://example.test/article/result",
        }
    ]
    failed = [
        {
            "tool": "browser_navigate",
            "status": "failed",
            "sequence": 2,
            "arguments": {"url": "https://example.test/article/result"},
        }
    ]

    selected = Agent._phase_tool_definitions(
        contract,
        definitions,
        successful,
        failed,
    )
    selected_names = {item["function"]["name"] for item in selected}

    assert "browser_snapshot" not in selected_names
    assert "browser_navigate" in selected_names
    assert "web_search" in selected_names


def test_agent_isolates_missing_skill_instead_of_returning_to_research() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "web_search",
            "browser_navigate",
            "learning_create_tool",
            "learning_create_skill",
        )
    ]
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_skill=True,
    )
    calls = [
        {"tool": "learning_create_tool", "status": "succeeded"},
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert [item["function"]["name"] for item in selected] == [
        "learning_create_skill"
    ]


def test_agent_exposes_both_builders_for_artifact_disjunction() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "web_search",
            "learning_create_tool",
            "learning_create_skill",
        )
    ]
    contract = TaskContract(requires_created_artifact=True)

    selected = Agent._phase_tool_definitions(contract, definitions, [])

    assert {item["function"]["name"] for item in selected} == {
        "learning_create_tool",
        "learning_create_skill",
    }


def test_agent_isolates_required_provider_before_unrelated_search_tools() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "web_search",
            "browser_navigate",
            "full_tor_search",
            "learning_create_tool",
            "learning_create_skill",
        )
    ]
    contract = TaskContract(
        requires_created_artifact=True,
        required_tools=("full_tor_search",),
        required_capabilities=("network.tor.search",),
    )

    selected = Agent._phase_tool_definitions(contract, definitions, [])

    assert [item["function"]["name"] for item in selected] == [
        "full_tor_search"
    ]


def test_schema_alias_repair_renames_one_unambiguous_string_field() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "browser_find",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    repaired = Agent._repair_schema_argument_alias(
        "browser_find",
        {"regex": "CAPTCHA|Verify"},
        definitions,
    )

    assert repaired == {"text": "CAPTCHA|Verify"}


def test_schema_alias_repair_refuses_ambiguous_payload() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "browser_type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["target", "text"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    arguments = {"selector": "input", "value": "hello"}

    assert Agent._repair_schema_argument_alias(
        "browser_type",
        arguments,
        definitions,
    ) == arguments


def test_agent_narrows_post_creation_phase_to_generated_tool() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_snapshot",
            "learning_create_tool",
            "extract_book_cards",
        )
    ]
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {"name": "extract_book_cards", "status": "active"}
            ),
        }
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert [item["function"]["name"] for item in selected] == [
        "extract_book_cards"
    ]


def test_agent_does_not_reexpose_builder_when_created_tool_is_missing_from_catalog() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_snapshot",
            "learning_create_tool",
        )
    ]
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {"name": "extract_book_cards", "status": "active"}
            ),
        }
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert selected == []


def test_agent_closes_creation_and_generated_tool_after_generated_execution() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "browser_snapshot",
            "learning_create_tool",
            "learning_create_snapshot_extractor",
            "har_traffic_summary",
        )
    ]
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "created_tool_name": "har_traffic_summary",
            "result_excerpt": '{"name":"har_traffic_summary","padding":"clipped',
        },
        {"tool": "har_traffic_summary", "status": "succeeded"},
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert {item["function"]["name"] for item in selected} == {
        "browser_navigate",
        "browser_snapshot",
    }


def test_agent_narrows_post_template_creation_to_generated_tool() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_snapshot",
            "learning_create_snapshot_extractor",
            "extract_book_cards",
        )
    ]
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_snapshot_extractor",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {"name": "extract_book_cards", "status": "active"}
            ),
        }
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert [item["function"]["name"] for item in selected] == [
        "extract_book_cards"
    ]


@pytest.mark.asyncio
async def test_llm_respond_falls_back_when_template_rejects_tools() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                response = httpx.Response(
                    500,
                    request=httpx.Request("POST", "http://local/v1/chat/completions"),
                )
                raise APIStatusError("template rejected tools", response=response, body={})
            message = SimpleNamespace(
                content='{"tool":"read_file","arguments":{"path":"README.md"}}',
                tool_calls=[],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    response = await llm.respond(
        messages=[{"role": "user", "content": "Read README"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert "tools" in completions.requests[0]
    assert "tools" not in completions.requests[1]
    assert llm._native_tools_supported is False
    assert response.content == ""
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}


@pytest.mark.asyncio
async def test_llm_retries_malformed_native_tool_arguments_as_compact_json() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                response = httpx.Response(
                    500,
                    request=httpx.Request("POST", "http://local/v1/chat/completions"),
                )
                raise APIStatusError(
                    "Failed to parse tool call arguments as JSON",
                    response=response,
                    body={
                        "error": {
                            "message": (
                                "parse_error.101: invalid string: missing closing quote"
                            )
                        }
                    },
                )
            message = SimpleNamespace(
                content=(
                    '{"tool":"browser_navigate","arguments":'
                    '{"url":"https://example.com"}}'
                ),
                tool_calls=[],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.7, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    response = await llm.respond(
        messages=[{"role": "user", "content": "Inspect example.com"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Open a web page",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert "tools" in completions.requests[0]
    assert "tools" not in completions.requests[1]
    assert completions.requests[1]["temperature"] == 0.1
    repair_prompt = completions.requests[1]["messages"][-1]["content"]
    assert "malformed or truncated JSON" in repair_prompt
    assert '"<required_field>":"<schema_value>"' in repair_prompt
    assert "every required field" in repair_prompt
    assert '"arguments":{}' not in repair_prompt
    assert llm._native_tools_supported is None
    assert response.native_tools_enabled is False
    assert response.content == ""
    assert response.tool_calls[0].name == "browser_navigate"
    assert response.tool_calls[0].arguments == {"url": "https://example.com"}


@pytest.mark.asyncio
async def test_llm_does_not_misclassify_context_overflow_as_tool_rejection() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            response = httpx.Response(
                500,
                request=httpx.Request("POST", "http://local/v1/chat/completions"),
            )
            raise APIStatusError(
                "request exceeds the available context size",
                response=response,
                body={"message": "9784 tokens exceeds context size 8704"},
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    with pytest.raises(APIStatusError, match="context size"):
        await llm.respond(
            messages=[{"role": "user", "content": "Read README"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

    assert completions.calls == 1
    assert llm._native_tools_supported is None


def test_agent_selects_only_relevant_native_tool_schemas() -> None:
    names = [
        "read_file",
        "write_file",
        "browser_navigate",
        "browser_snapshot",
        "evm_validate_oracle",
        "learning_create_tool",
        "learning_create_skill",
    ]
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in names
    ]
    contract = TaskContract.from_prompt(
        "Read README.md and report only its first heading."
    )

    selected = Agent._select_tool_definitions(
        "Read README.md and report only its first heading.",
        contract,
        definitions,
    )

    selected_names = {item["function"]["name"] for item in selected}
    assert selected_names == {
        "read_file",
        "write_file",
    }


def test_unclear_action_does_not_expose_the_entire_tool_catalog() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "read_file",
            "sandbox_execute_offline",
            "evm_validate_oracle",
            "learning_create_tool",
            "learning_create_skill",
        )
    ]

    selected = Agent._select_tool_definitions(
        "Find the answer now.",
        TaskContract(),
        definitions,
    )

    assert selected == []


def test_runtime_log_review_exposes_only_read_only_auditor() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "runtime_review_task",
            "browser_navigate",
            "read_file",
            "learning_record_evidence",
            "learning_create_tool",
        )
    ]
    prompt = "Przeanalizuj swoje logi z ostatniej sesji i pokaż błędy."
    contract = TaskContract.from_prompt(prompt)

    selected = Agent._select_tool_definitions(prompt, contract, definitions)

    assert [item["function"]["name"] for item in selected] == [
        "runtime_review_task"
    ]
    assert Agent._requests_runtime_action(prompt, contract) is True


def test_polish_w_internecie_action_gets_only_web_related_schemas() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "read_file",
            "sandbox_execute_offline",
            "evm_validate_oracle",
            "learning_create_tool",
            "learning_create_skill",
        )
    ]
    prompt = "Znajdź mi w internecie informacje o tej osobie i podaj wyniki."

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert {item["function"]["name"] for item in selected} == {
        "browser_navigate",
        "browser_snapshot",
    }


def test_explicit_web_address_keeps_interactive_browser_controls() -> None:
    names = (
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_find",
        "browser_press_key",
        "browser_type",
    )
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in names
    ]
    prompt = "Open https://example.com, use its search form, and report the result."

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert {item["function"]["name"] for item in selected} == set(names)


@pytest.mark.parametrize(
    "prompt",
    [
        "V, czy potrafisz tworzyć własne narzędzia?",
        "Powiedz mi, czy potrafiłabyś stworzyć narzędzie do rejestrowania ruchu na stronie?",
        "Are you able to create a page monitoring tool?",
        "How do I read a local file safely?",
        "Why do agents sometimes invoke tools during a conversation?",
        "Tell me what you think about tools in agent frameworks.",
    ],
)
def test_capability_discussion_exposes_no_executable_tools(prompt: str) -> None:
    definitions = [
        {"type": "function", "function": {"name": "learning_create_tool"}},
        {"type": "function", "function": {"name": "read_file"}},
    ]

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert selected == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Stwórz nowe narzędzie do CSV.",
        "Czy możesz przeczytać plik README.md?",
        "Please read README.md and report its heading.",
    ],
)
def test_explicit_action_request_exposes_relevant_tools(prompt: str) -> None:
    definitions = [
        {"type": "function", "function": {"name": "learning_create_tool"}},
        {"type": "function", "function": {"name": "learning_create_skill"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "write_file"}},
    ]

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert selected


def test_explicit_tool_creation_hides_low_level_learning_operations() -> None:
    names = (
        "learning_activate_artifact",
        "learning_create_skill",
        "learning_create_tool",
        "learning_list_artifacts",
        "learning_propose_lesson",
        "learning_record_evidence",
        "learning_stage_tool",
        "learning_validate_artifact",
    )
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in names
    ]
    prompt = "Stwórz lokalne narzędzie count_words, a następnie użyj go."

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert {item["function"]["name"] for item in selected} == {
        "learning_create_tool",
        "learning_list_artifacts",
    }


def test_explicit_active_tool_name_is_selected_and_required() -> None:
    definitions = [
        {"type": "function", "function": {"name": "count_words"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
    ]
    prompt = 'Użyj narzędzia count_words na tekście "V can build her own tools".'
    contract = TaskContract.from_prompt(prompt).with_required_tools(
        Agent._explicitly_named_tools(prompt, definitions)
    )

    selected = Agent._select_tool_definitions(prompt, contract, definitions)

    assert [item["function"]["name"] for item in selected] == ["count_words"]
    assert contract.unmet([]) == ["count_words"]
    assert contract.unmet(
        [{"tool": "count_words", "status": "failed"}]
    ) == ["count_words"]
    assert contract.unmet(
        [{"tool": "count_words", "status": "succeeded"}]
    ) == []


def test_spoken_tool_name_typo_uniquely_selects_existing_tool() -> None:
    definitions = [
        {"type": "function", "function": {"name": "count_words"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
    ]
    prompt = (
        "V, użyj narzędzia Can't Words na zdaniu "
        "V potrafi robić swoje własne narzędzia."
    )

    assert Agent._explicitly_named_tools(prompt, definitions) == ["count_words"]


def test_descriptive_browser_snapshot_phrase_is_not_an_explicit_tool_call() -> None:
    definitions = [
        {"type": "function", "function": {"name": "browser_snapshot"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
    ]
    prompt = (
        "Create an offline tool that accepts the observed browser snapshot "
        "text, then use the newly created tool on that data."
    )

    assert Agent._explicitly_named_tools(prompt, definitions) == []


def test_snapshot_extraction_creation_selects_deterministic_builder() -> None:
    definitions = [
        {"type": "function", "function": {"name": "browser_navigate"}},
        {"type": "function", "function": {"name": "browser_snapshot"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
        {
            "type": "function",
            "function": {"name": "learning_create_snapshot_extractor"},
        },
        {"type": "function", "function": {"name": "learning_list_artifacts"}},
    ]
    prompt = (
        "Visit https://books.toscrape.com/, inspect the listing, then create an "
        "offline tool named extract_book_cards for the observed browser snapshot "
        "text and use it."
    )
    contract = TaskContract.from_prompt(prompt)

    selected = Agent._select_tool_definitions(prompt, contract, definitions)

    assert "learning_create_snapshot_extractor" in {
        item["function"]["name"] for item in selected
    }
    assert "learning_create_tool" not in {
        item["function"]["name"] for item in selected
    }


def test_generated_snapshot_tool_receives_runtime_observation() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "extract_book_cards",
                "parameters": {
                    "type": "object",
                    "properties": {"snapshot_text": {"type": "string"}},
                    "required": ["snapshot_text"],
                },
            },
        }
    ]
    created = {
        "tool": "learning_create_snapshot_extractor",
        "status": "succeeded",
        "result_excerpt": '{"name":"extract_book_cards","status":"active"}',
    }

    repaired = Agent._repair_observed_snapshot_input(
        "extract_book_cards",
        {"snapshot_text": "model-corrupted copy"},
        definitions,
        [created],
        "runtime-observed snapshot",
    )

    assert repaired == {"snapshot_text": "runtime-observed snapshot"}


def test_ambiguous_spoken_tool_name_does_not_guess() -> None:
    definitions = [
        {"type": "function", "function": {"name": "count_words"}},
        {"type": "function", "function": {"name": "count_worlds"}},
    ]

    assert Agent._explicitly_named_tools(
        "Use the count word tool.", definitions
    ) == []


def test_empty_explicit_generated_tool_arguments_recover_quoted_text() -> None:
    prompt = "Użyj count_words na tekście „V can build her own tools”."
    contract = TaskContract().with_required_tools(["count_words"])
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "count_words",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    repaired = Agent._repair_explicit_text_arguments(
        prompt,
        "count_words",
        {},
        definitions,
        contract,
    )

    assert repaired == {"text": "V can build her own tools"}


def test_required_url_tool_uses_exact_owner_url_instead_of_model_copy() -> None:
    exact_url = (
        "http://cebulka7uxchnbpvmqapg5pfos4ngaxglsktzvha7a5rigndghvadeyd.onion"
    )
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "full_tor_fetch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout_seconds": {"type": "number"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    repaired = Agent._repair_explicit_text_arguments(
        f"Open {exact_url} and inspect the page.",
        "full_tor_fetch",
        {
            "url": "http://cebulka7uxchnbpvm qapg5pfos4ng.on ion",
            "timeout_seconds": 30,
        },
        definitions,
        TaskContract().with_required_tools(["full_tor_fetch"]),
    )

    assert repaired == {"url": exact_url, "timeout_seconds": 30}


def test_required_url_repair_uses_durable_objective_during_follow_up() -> None:
    exact_url = "https://example.com/exact/path"
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "required_fetch",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        }
    ]

    repaired = Agent._repair_explicit_text_arguments(
        f"Inspect {exact_url}.\n\nFollow-up: Try again.",
        "required_fetch",
        {"url": "https://example.net/invented"},
        definitions,
        TaskContract().with_required_tools(["required_fetch"]),
    )

    assert repaired == {"url": exact_url}


def test_runtime_executes_unambiguous_required_url_without_model_copy() -> None:
    exact_url = (
        "http://cebulka7uxchnbpvmqapg5pfos4ngaxglsktzvha7a5rigndghvadeyd.onion"
    )
    contract = TaskContract().with_required_tools(["full_tor_fetch"])
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "full_tor_fetch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "pattern": "^https?://"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    request = Agent._runtime_grounded_required_tool_request(
        f"Open {exact_url} and inspect it.",
        contract,
        definitions,
        [],
    )

    assert request == ("full_tor_fetch", {"url": exact_url})


def test_runtime_does_not_guess_ambiguous_required_url() -> None:
    contract = TaskContract().with_required_tools(["required_fetch"])
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "required_fetch",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        }
    ]

    request = Agent._runtime_grounded_required_tool_request(
        "Compare https://example.com/a with https://example.net/b.",
        contract,
        definitions,
        [],
    )

    assert request is None


def test_runtime_selects_relevant_unvisited_detail_after_listing_stalls() -> None:
    prompt = (
        "Find information and prices for antique French chamber pots made "
        "before the war."
    )
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
    )
    definitions = [
        {"type": "function", "function": {"name": "browser_navigate"}},
        {"type": "function", "function": {"name": "browser_snapshot"}},
    ]
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Antiques marketplace",
                            "url": "https://market.test/listing?query=antiques",
                        },
                        {
                            "rank": 2,
                            "title": "Unrelated military history",
                            "url": "https://history.test/post/military",
                        },
                        {
                            "rank": 3,
                            "title": "Antique French chamber pots",
                            "url": "https://museum.test/article/chamber-pots",
                        },
                    ]
                }
            ),
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://market.test/listing?query=antiques"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": "Cookie preferences",
        },
    ]

    request = Agent._runtime_grounded_required_tool_request(
        prompt,
        contract,
        definitions,
        calls,
    )

    assert request == (
        "browser_navigate",
        {"url": "https://museum.test/article/chamber-pots"},
    )


def test_runtime_selects_grounded_market_when_price_evidence_is_missing() -> None:
    prompt = "Search for antique chamber-pot prices and where to buy them."
    listing = "https://market.test/listing?query=chamber-pots"
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
        required_research_facets=("price", "purchase_source"),
    )
    definitions = [
        {"type": "function", "function": {"name": "browser_navigate"}},
        {"type": "function", "function": {"name": "browser_snapshot"}},
    ]
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Antique chamber pots for sale",
                            "url": listing,
                        },
                        {
                            "rank": 2,
                            "title": "Chamber-pot history",
                            "url": "https://museum.test/article/chamber-pots",
                        },
                    ]
                }
            ),
        }
    ]

    request = Agent._runtime_grounded_required_tool_request(
        prompt,
        contract,
        definitions,
        calls,
    )

    assert request == ("browser_navigate", {"url": listing})


def test_runtime_retains_repeated_subject_when_selecting_second_source() -> None:
    prompt = (
        "Znajdź nocniki sprzed drugiej wojny, porównaj ceny nocników "
        "oraz miejsca, gdzie można je kupić."
    )
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Antyki z Francji i Austrii",
                            "url": "https://antyki.test/blog/francja-austria",
                        },
                        {
                            "rank": 2,
                            "title": "Nocniki francuskie przed 1939 rokiem",
                            "url": "https://muzeum.test/article/nocniki-francuskie",
                        },
                    ]
                }
            ),
        }
    ]

    selected = Agent._grounded_unvisited_detail_url(prompt, successful)

    assert selected == "https://muzeum.test/article/nocniki-francuskie"


def test_exhaustive_detail_ranking_prefers_rare_scope_anchor() -> None:
    prompt = (
        "Wynajdź i opisz wszystkie gatunki dzikich kotów występujące "
        "naturalnie w Europie."
    )
    general = "https://animals.test/article/dzikie-koty-gatunki"
    scoped = "https://animals.test/article/dzikie-koty-europy"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Dzikie koty i najważniejsze gatunki świata",
                            "url": general,
                        },
                        {
                            "rank": 8,
                            "title": "Sekrety dzikich kotów Europy",
                            "url": scoped,
                        },
                    ]
                }
            ),
        }
    ]

    assert Agent._grounded_unvisited_detail_url(prompt, calls) == scoped


def test_exhaustive_navigation_replaces_weaker_grounded_candidate() -> None:
    prompt = (
        "Wynajdź i opisz wszystkie gatunki dzikich kotów występujące "
        "naturalnie w Europie."
    )
    general = "https://animals.test/article/dzikie-koty-gatunki"
    scoped = "https://animals.test/article/dzikie-koty-europy"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Dzikie koty i najważniejsze gatunki świata",
                            "url": general,
                        },
                        {
                            "rank": 8,
                            "title": "Sekrety dzikich kotów Europy",
                            "url": scoped,
                        },
                    ]
                }
            ),
        }
    ]
    contract = TaskContract(
        requires_web_discovery=True,
        required_research_facets=("exhaustive_coverage",),
    )

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": general},
        contract,
        calls,
        [],
        preferred_query=prompt,
    )

    assert repaired == {"url": scoped}


def test_verified_final_report_preserves_observed_items_and_images() -> None:
    source = "https://catalogue.test/article/wild-cats"
    calls = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": source},
            "result_excerpt": f"- Page URL: {source}",
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                f"- Page URL: {source}\n"
                "- Page Title: Wild-cat catalogue\n"
                "- article:\n"
                "  - img \"European wildcat photograph\"\n"
                "  - heading \"Felis silvestris\" [level=6]\n"
                "- article:\n"
                "  - img \"Eurasian lynx photograph\"\n"
                "  - heading \"Lynx lynx\" [level=6]"
            ),
        },
    ]
    contract = TaskContract(
        requires_evidence_report=True,
        required_research_facets=(
            "item_list",
            "item_descriptions",
            "images",
            "exhaustive_coverage",
        ),
    )

    report = Agent._owner_verified_final_report(None, calls, contract)

    assert "- Felis silvestris —" in report
    assert "- Lynx lynx —" in report
    assert "European wildcat photograph" in report
    assert "does not claim that omitted records were verified" in report


def test_repeated_scaffolding_does_not_suppress_all_detail_candidates() -> None:
    prompt = (
        "Please carefully search and carefully compare one Firecrawler alternative."
    )
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Firecrawler alternative",
                            "url": "https://tools.test/article/firecrawler-alternative",
                        }
                    ]
                }
            ),
        }
    ]

    selected = Agent._grounded_unvisited_detail_url(prompt, successful)

    assert selected == "https://tools.test/article/firecrawler-alternative"


def test_malformed_relative_navigation_uses_next_grounded_result() -> None:
    first = "https://catalogue.test/article/species-alpha"
    second = "https://reference.test/article/species-beta"
    prompt = "Find and describe all regional species with species photographs."
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {"rank": 1, "title": "Regional species", "url": first},
                        {"rank": 2, "title": "Regional species", "url": second},
                    ]
                }
            ),
        },
        {
            "tool": "web_read",
            "status": "succeeded",
            "arguments": {"url": first},
            "result_excerpt": "Observed first regional species source.",
        },
    ]
    contract = TaskContract(
        requires_web_discovery=True,
        requires_evidence_report=True,
        requires_distinct_detail_page=True,
        required_research_facets=("exhaustive_coverage",),
    )

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "#invented section with spaces"},
        contract,
        successful,
        [],
        preferred_query="regional species photographs",
    )

    assert repaired == {"url": second}


def test_read_path_typo_is_repaired_from_successful_write() -> None:
    correct = "/workspace/smoke-report.md"
    calls = [
        {
            "tool": "write_file",
            "status": "succeeded",
            "arguments": {"path": correct, "content": "# Smoke"},
        }
    ]

    repaired = Agent._repair_observed_file_path(
        "read_file",
        {"path": "/workspace/smone-report.md", "head": 1},
        calls,
    )

    assert repaired == {"path": correct, "head": 1}


def test_unrelated_read_path_is_not_redirected() -> None:
    calls = [
        {
            "tool": "write_file",
            "status": "succeeded",
            "arguments": {"path": "/workspace/alpha.md", "content": "a"},
        }
    ]

    repaired = Agent._repair_observed_file_path(
        "read_file",
        {"path": "/workspace/beta.md"},
        calls,
    )

    assert repaired == {"path": "/workspace/beta.md"}


def test_observed_detail_urls_resolve_relative_listing_links() -> None:
    calls = [
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://www.olx.pl/antyki/q-filizanka/\n"
                "- link 'Vintage filiżanka 1965'\n"
                "  - /url: /d/oferta/vintage-filizanka-1965-ID123.html\n"
            ),
        }
    ]

    observed = Agent._observed_detail_urls(calls)

    assert "https://www.olx.pl/d/oferta/vintage-filizanka-1965-ID123.html" in observed


def test_listing_snapshot_selects_relative_offer_not_help_policy() -> None:
    prompt = (
        "Znajdź dwie aktualne oferty kolekcjonerskich filiżanek i podaj cenę."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://www.olx.pl/antyki/q-filizanka/\n"
                "- link 'Vintage filiżanka z 1965 roku, 350 zł'\n"
                "  - /url: /d/oferta/vintage-filizanka-1965-ID123.html\n"
                "- link 'Pomoc i polityka cookies'\n"
                "  - /url: https://pomoc.olx.pl/polityka-dotyczaca-plikow-cookie\n"
            ),
        }
    ]

    selected = Agent._grounded_unvisited_detail_url(prompt, calls)

    assert selected == (
        "https://www.olx.pl/d/oferta/vintage-filizanka-1965-ID123.html"
    )


def test_runtime_never_reselects_failed_grounded_detail_url() -> None:
    prompt = (
        "Search the web for chamber pots made before the war and compare "
        "chamber pot prices."
    )
    dead = "https://market.test/product/antique-chamber-pot"
    alternative = "https://museum.test/article/chamber-pot-history"
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {"rank": 1, "title": "Chamber pot", "url": dead},
                        {
                            "rank": 2,
                            "title": "Chamber pot history",
                            "url": alternative,
                        },
                    ]
                }
            ),
        }
    ]
    failed = [
        {
            "tool": "browser_navigate",
            "status": "failed",
            "arguments": {"url": dead},
        }
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": dead},
        TaskContract.from_prompt(prompt),
        successful,
        failed,
    )

    assert repaired == {"url": alternative}


def test_runtime_never_revisits_successfully_observed_detail_source() -> None:
    prompt = (
        "Search the web for chamber pots made before the war and compare "
        "chamber pot prices."
    )
    first = "https://history.test/article/chamber-pot"
    second = "https://museum.test/article/chamber-pot-prices"
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {"rank": 1, "title": "Chamber pot history", "url": first},
                        {
                            "rank": 2,
                            "title": "Chamber pot prices",
                            "url": second,
                        },
                    ]
                }
            ),
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": first},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {"target": "body"},
        },
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": first},
        TaskContract.from_prompt(prompt),
        successful,
        [],
    )

    assert repaired == {"url": second}


def test_repeated_search_page_advances_to_observed_result() -> None:
    prompt = (
        "Przeszukaj internet: znajdź nocniki sprzed wojny i ceny nocników."
    )
    search_url = "https://duckduckgo.com/?q=nocniki+sprzed+wojny"
    detail_url = "https://historia.test/32739002-historia-nocnikow"
    successful = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": search_url},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {"target": "body"},
            "result_excerpt": (
                f"- Page URL: {search_url}\n"
                "- Page Title: nocniki sprzed wojny at DuckDuckGo\n"
                "- heading: Historia dawnych nocników\n"
                f"  - /url: {detail_url}\n"
                "  - text: Nocniki, ich historia i antyczne egzemplarze."
            ),
        },
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": search_url},
        TaskContract.from_prompt(prompt),
        successful,
        [],
    )

    assert repaired == {"url": detail_url}


def test_runtime_does_not_promote_topic_or_auction_listing_to_detail() -> None:
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
    )
    definitions = [
        {"type": "function", "function": {"name": "browser_navigate"}},
        {"type": "function", "function": {"name": "browser_snapshot"}},
    ]
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "Antique chamber pots",
                            "url": "https://market.test/temat/chamber-pots",
                        },
                        {
                            "rank": 2,
                            "title": "Antique chamber-pot auctions",
                            "url": "https://auction.test/auction/antiques",
                        },
                    ]
                }
            ),
        }
    ]

    request = Agent._runtime_grounded_required_tool_request(
        "Find antique chamber pots and inspect a concrete source.",
        contract,
        definitions,
        calls,
    )

    assert request is None


def test_github_repository_root_is_detail_but_auxiliary_tabs_are_not() -> None:
    assert Agent._is_strong_detail_url(
        "https://github.com/Cybersight-Security/OSINT-Toolkit"
    )
    assert not Agent._is_strong_detail_url("https://github.com/topics/osint-tool")
    assert not Agent._is_strong_detail_url(
        "https://github.com/smicallef/spiderfoot/issues"
    )
    assert not Agent._is_strong_detail_url(
        "https://github.com/smicallef/spiderfoot/pulls"
    )


def test_runtime_replaces_github_topic_page_with_observed_repository() -> None:
    prompt = "Find useful GitHub tools for darknet OSINT and cybersecurity."
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
    )
    repository = "https://github.com/Cybersight-Security/OSINT-Toolkit"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "results": [
                        {
                            "rank": 1,
                            "title": "osint-tool GitHub Topics",
                            "url": "https://github.com/topics/osint-tool",
                        },
                        {
                            "rank": 2,
                            "title": "Cybersight Security OSINT Toolkit",
                            "url": repository,
                        },
                    ]
                }
            ),
        }
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://github.com/topics/osint-tool"},
        contract,
        calls,
        [],
        preferred_query="darknet OSINT cybersecurity GitHub tools",
    )

    assert repaired == {"url": repository}


def test_runtime_observes_each_owner_url_before_finalization() -> None:
    prompt = (
        "Compare www.invicti.com/product and www.acunetix.com/product."
    )
    contract = TaskContract.from_prompt(prompt)
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "browser_navigate",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browser_snapshot",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]
    calls: list[dict[str, object]] = []

    assert Agent._runtime_explicit_web_observation_request(
        prompt, contract, definitions, calls
    ) == ("browser_navigate", {"url": "https://www.invicti.com/product"})
    calls.append(
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://www.invicti.com/product"},
            "status": "succeeded",
        }
    )
    assert Agent._runtime_explicit_web_observation_request(
        prompt, contract, definitions, calls
    ) == ("browser_snapshot", {})
    calls.append(
        {"tool": "browser_snapshot", "arguments": {}, "status": "succeeded"}
    )
    assert Agent._runtime_explicit_web_observation_request(
        prompt, contract, definitions, calls
    ) == ("browser_navigate", {"url": "https://www.acunetix.com/product"})
    calls.extend(
        [
            {
                "tool": "browser_navigate",
                "arguments": {"url": "https://www.acunetix.com/product"},
                "status": "succeeded",
            },
            {
                "tool": "browser_snapshot",
                "arguments": {},
                "status": "succeeded",
            },
        ]
    )

    assert Agent._runtime_explicit_web_observation_request(
        prompt, contract, definitions, calls
    ) is None
    assert contract.unmet(calls) == []


def test_argument_repair_does_not_guess_multiple_required_fields() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "join_text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "string"},
                        "right": {"type": "string"},
                    },
                    "required": ["left", "right"],
                },
            },
        }
    ]

    assert Agent._repair_explicit_text_arguments(
        'Use join_text on "one" and "two".',
        "join_text",
        {},
        definitions,
        TaskContract().with_required_tools(["join_text"]),
    ) == {}


def test_structured_literal_assignments_preserve_repeated_json_exactly() -> None:
    first_url = "http://abcdefghijklmnopabcdefghijklmnopabcdefghijklmnopabcd.onion/a"
    second_url = "http://zyxwvutsrqponmlkzyxwvutsrqponmlkzyxwvutsrqponmlkzyxw.onion/b"
    prompt = (
        f'observations = [{{"url":"{first_url}","text":"escrow"}}]\n'
        'keywords = ["escrow", "bitcoin"]\n'
        f'observations = [{{"url":"{second_url}","text":"wallet"}}]\n'
        'keywords = ["ransomware", "wallet"]'
    )

    assignments = Agent._structured_literal_assignments(prompt)

    assert assignments["observations"] == [
        [{"url": first_url, "text": "escrow"}],
        [{"url": second_url, "text": "wallet"}],
    ]
    assert assignments["keywords"] == [
        ["escrow", "bitcoin"],
        ["ransomware", "wallet"],
    ]


def test_tool_creation_grounds_first_fixture_without_domain_patch() -> None:
    prompt = (
        'records = [{"url":"http://aaaaaaaaaaaaaaaa.onion/a?x=1",'
        '"text":"escrow"}]\n'
        'keywords = ["escrow"]\n'
        'expected = {"records":[{"url":"http://aaaaaaaaaaaaaaaa.onion/a"}]}\n'
        'records = [{"url":"http://bbbbbbbbbbbbbbbb.onion/b",'
        '"text":"wallet"}]\n'
        'keywords = ["wallet"]\n'
        'expected = {"records":[{"url":"http://bbbbbbbbbbbbbbbb.onion/b"}]}'
    )
    arguments = {
        "name": "index_observations",
        "description": "Index supplied observations.",
        "source": (
            "def run(arguments):\n"
            "    records = arguments.get('records', [])\n"
            "    keywords = arguments['keywords']\n"
            "    return {'records': records, 'keywords': keywords}"
        ),
        "test": {
            "name": "fixture",
            "arguments": {
                "records": [{"url": "model corrupted", "text": "escrow"}],
                "keywords": ["wrong"],
            },
            "expected": {"records": [], "keywords": []},
        },
    }

    repaired = Agent._repair_grounded_generated_tool_arguments(
        prompt,
        "learning_create_tool",
        arguments,
        [],
        [],
    )

    assert repaired["test"]["arguments"] == {
        "records": [
            {
                "url": "http://aaaaaaaaaaaaaaaa.onion/a?x=1",
                "text": "escrow",
            }
        ],
        "keywords": ["escrow"],
    }
    assert repaired["test"]["expected"] == {
        "records": [{"url": "http://aaaaaaaaaaaaaaaa.onion/a"}]
    }
    assert arguments["test"]["arguments"]["records"][0]["url"] == "model corrupted"


def test_tool_creation_can_ground_empty_fixture_from_source_accesses() -> None:
    prompt = 'items = [{"value": 3}]\nmultiplier = 2'
    arguments = {
        "name": "multiply_items",
        "description": "Multiply item values.",
        "source": (
            "def run(arguments):\n"
            "    items = arguments.get(\"items\", [])\n"
            "    multiplier = arguments.get(\"multiplier\", 1)\n"
            "    return {'values': [x['value'] * multiplier for x in items]}"
        ),
        "test": {
            "name": "fixture",
            "arguments": {},
            "expected": {"values": [6]},
        },
    }

    repaired = Agent._repair_grounded_generated_tool_arguments(
        prompt,
        "learning_create_tool",
        arguments,
        [],
        [],
    )

    assert repaired["test"]["arguments"] == {
        "items": [{"value": 3}],
        "multiplier": 2,
    }


def test_tool_creation_repairs_expected_when_arguments_are_already_exact() -> None:
    arguments = {
        "name": "sum_values",
        "description": "Sum supplied values.",
        "source": (
            "def run(arguments):\n"
            "    return {'total': sum(arguments['values'])}"
        ),
        "test": {
            "name": "fixture",
            "arguments": {"values": [1, 2, 3]},
            "expected": {"total": 999},
        },
    }

    repaired = Agent._repair_grounded_generated_tool_arguments(
        'values = [1,2,3]\nexpected = {"total":6}',
        "learning_create_tool",
        arguments,
        [],
        [],
    )

    assert repaired["test"] == {
        "name": "fixture",
        "arguments": {"values": [1, 2, 3]},
        "expected": {"total": 6},
    }


def test_new_generated_tool_grounds_last_fixture_from_runtime_schema() -> None:
    prompt = (
        'records = [{"value": 1}]\nkeywords = ["first"]\n'
        'records = [{"value": 2}]\nkeywords = ["second"]'
    )
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "index_observations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "records": {"type": "array"},
                        "keywords": {"type": "array"},
                    },
                    "required": ["records", "keywords"],
                },
            },
        }
    ]
    created = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": (
                '{"name":"index_observations","status":"active"}'
            ),
        }
    ]

    repaired = Agent._repair_grounded_generated_tool_arguments(
        prompt,
        "index_observations",
        {"records": [], "keywords": []},
        definitions,
        created,
    )

    assert repaired == {
        "records": [{"value": 2}],
        "keywords": ["second"],
    }


def test_structured_literal_repair_leaves_unrelated_tools_unchanged() -> None:
    arguments = {"query": "model-selected"}

    repaired = Agent._repair_grounded_generated_tool_arguments(
        'query = "owner literal"',
        "web_search",
        arguments,
        [],
        [],
    )

    assert repaired is arguments


@pytest.mark.parametrize(
    "prompt",
    [
        "/continue",
        "/continue --continuous",
    ],
)
def test_agent_recognizes_language_neutral_continuation_protocol(prompt: str) -> None:
    assert Agent._is_continuation_request(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Dobrze, w takim razie kontynuuj.",
        "No to dawaj.",
        "Go ahead.",
        "继续上一个任务。",
        "Jak się dziś czujesz?",
        "Jak dalej ma się rozwijać DARKLINGER? Czy masz jakieś sugestie?",
        "Powiedz mi, jak działa kontynuacja zadań.",
        "Why do agents use tools?",
    ],
)
def test_agent_defers_natural_language_continuation_to_semantic_router(
    prompt: str,
) -> None:
    assert not Agent._is_continuation_request(prompt)


def test_agent_mode_prompt_contains_only_selected_executable_tools() -> None:
    agent = object.__new__(Agent)
    prompt = agent._agent_mode_prompt(
        [
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Navigate to a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ]
    )

    assert '"name":"browser_navigate"' in prompt
    assert '"required":["url"]' in prompt
    assert '"<required_field>": "<schema_value>"' in prompt
    assert '"arguments": {}' not in prompt
    assert "learning_record_evidence" not in prompt
    assert "learning_create_tool may be used" not in prompt
    assert "outcome contract, not as an exhaustive keyword" in prompt
    assert "synonyms, aliases, translations" in prompt
    assert "never by whether a method merely" in prompt
    assert "sounds sensitive or dangerous" in prompt
    assert "different person, organization, account, machine, wallet" in prompt


def test_agent_mode_prompt_exposes_owner_privileged_generated_code() -> None:
    agent = object.__new__(Agent)
    prompt = agent._agent_mode_prompt(
        [
            {
                "type": "function",
                "function": {
                    "name": "learning_create_tool",
                    "description": (
                        "OWNER LAB: generated code may use arbitrary Python imports, "
                        "dynamic code, subprocesses, and file operations inside the "
                        "isolated sandbox."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ]
    )

    assert "OWNER LAB is active" in prompt
    assert "subprocesses" in prompt
    assert "client restricted-source policy" not in prompt


def test_browser_http_error_is_not_counted_as_successful_evidence() -> None:
    assert Agent._tool_result_error(
        "Page URL: https://example.com/missing\nHTTP status: 404",
        tool="browser_navigate",
    ) == "BrowserHTTPError: page returned HTTP status 404"


def test_research_about_tools_does_not_expose_learning_lifecycle() -> None:
    names = (
        "browser_navigate",
        "browser_snapshot",
        "browser_find",
        "learning_create_tool",
        "learning_record_evidence",
        "learning_propose_lesson",
        "learning_list_artifacts",
    )
    definitions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]
    contract = TaskContract.from_prompt(
        "Sprawdź stronę marketplace i powiedz, jakie narzędzia są użyteczne."
    )

    selected = Agent._select_tool_definitions(
        "Sprawdź stronę marketplace i powiedz, jakie narzędzia są użyteczne.",
        contract,
        definitions,
        capability_hints={"browser"},
    )
    selected_names = {item["function"]["name"] for item in selected}

    assert selected_names == {
        "browser_navigate",
        "browser_snapshot",
        "browser_find",
    }


def test_learning_creation_receives_larger_generation_budget() -> None:
    definitions = [
        {
            "type": "function",
            "function": {"name": "learning_create_tool"},
        }
    ]

    assert Agent._agent_generation_budget(
        definitions,
        context_tokens=12_000,
    ) == 3_000
    assert Agent._agent_generation_budget([], context_tokens=12_000) == 512


@pytest.mark.asyncio
async def test_generic_find_information_action_uses_semantic_browser_route(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in (
                    "browser_navigate",
                    "browser_snapshot",
                    "learning_record_evidence",
                )
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_navigate":
                return "Navigated to a public search result"
            return (
                "Result URL: https://example.com/public-profile. "
                "Verified public search result"
            )

    class IntentRouterStub:
        def __init__(self) -> None:
            self.called = False

        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            self.called = True
            return SemanticIntent(
                action_requested=True,
                capabilities=("browser",),
                requires_report=True,
                distinct_detail_page=True,
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            offered = {
                item["function"]["name"] for item in (kwargs.get("tools") or [])
            }
            if self.turn <= 4:
                assert offered == {"browser_navigate", "browser_snapshot"}
            else:
                assert offered == set()
            assert "learning_record_evidence" not in kwargs["messages"][0]["content"]
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": "https://example.com/search?q=public"},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("snapshot", "browser_snapshot", {})]
                )
            if self.turn == 3:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate_detail",
                            "browser_navigate",
                            {"url": "https://example.com/public-profile"},
                        )
                    ]
                )
            if self.turn == 4:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot_detail", "browser_snapshot", {})
                    ]
                )
            return LLMResponse(content="I found one verified public result, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    router = IntentRouterStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = router
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "V, znajdź mi wszystkie informacje, jakie zdołasz, na temat tej osoby."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert router.called is True
    assert tools.calls == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    assert answer == "I found one verified public result, Boss."


@pytest.mark.asyncio
async def test_missing_browser_dependency_stops_web_lookup_after_one_call(
    tmp_path: Path,
) -> None:
    prompt = (
        "Cześć V, jak ci mija wieczór? Słuchajcie, możesz dla mnie "
        "sprawdzić, jaki jest aktualny kurs Monero do złotówki?"
    )
    error = (
        'MCPToolExecutionError: Browser "firefox" is not installed; expected '
        "executable at /tmp/ms-playwright/firefox-1544/firefox/firefox. "
        "Run npx @playwright/mcp install-browser firefox to install."
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the public web.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ]

        async def call_with_recovery(self, tool: str, arguments: dict):
            self.calls.append((tool, arguments))
            return SimpleNamespace(
                result=f"Tool execution failed: {error}",
                provider_tool=tool,
                capabilities=("web_search",),
                attempts=(
                    {
                        "provider": tool,
                        "status": "failed",
                        "error": error,
                    },
                ),
                recovery_ticket=None,
                failure_details={
                    "stage": "provider_execution",
                    "execution_attempted": True,
                    "provider": tool,
                    "exception_type": "MCPToolExecutionError",
                },
                error=error,
            )

    class IntentRouterStub:
        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                action_requested=True,
                capabilities=("browser",),
                requires_report=True,
                distinct_detail_page=False,
                web_query="actual Monero to zloty exchange rate",
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turns = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turns += 1
            assert self.turns == 1
            assert {
                item["function"]["name"] for item in (kwargs.get("tools") or [])
            } == {"web_search"}
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        "search",
                        "web_search",
                        {"query": "actual Monero to zloty exchange rate"},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert len(tools.calls) == 1
    assert llm.turns == 1
    assert "firefox" in answer
    assert "Nothing is running in the background" in answer
    assert agent._last_execution_context["status"] == "blocked"


@pytest.mark.asyncio
async def test_deterministic_web_route_keeps_semantic_market_requirements(
    tmp_path: Path,
) -> None:
    first = "https://market.test/product/cup-a"
    second = "https://market.test/product/cup-b"

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []
            self.current_url = ""

        async def openai_tool_definitions(self) -> list[dict]:
            definitions = {
                "web_search": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "browser_navigate": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
                "browser_snapshot": {
                    "type": "object",
                    "properties": {},
                },
            }
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": parameters,
                    },
                }
                for name, parameters in definitions.items()
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "web_search":
                return json.dumps(
                    {
                        "results": [
                            {"title": "Cup A", "url": first},
                            {"title": "Cup B", "url": second},
                        ]
                    }
                )
            if tool == "browser_navigate":
                self.current_url = arguments["url"]
                return f"- Page URL: {self.current_url}"
            price = "10 EUR" if self.current_url == first else "20 EUR"
            title = "Cup A" if self.current_url == first else "Cup B"
            return (
                f"- Page URL: {self.current_url}\n"
                f"- Page Title: {title}\n"
                f"{title} costs {price}."
            )

    class IntentRouterStub:
        def __init__(self) -> None:
            self.called = False
            self.last_sanitization_reason = ""

        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            self.called = True
            return SemanticIntent(
                action_requested=True,
                capabilities=("browser",),
                requires_report=True,
                distinct_detail_page=True,
                research_facets=("price", "purchase_source"),
                minimum_detail_sources=2,
                web_query="two antique cup offers",
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "search",
                            "web_search",
                            {"query": "antique cups"},
                        )
                    ]
                )
            return LLMResponse(
                content=(
                    f"I found two offers. Cup A costs 10 EUR at {first}. "
                    f"Cup B costs 20 EUR at {second}."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    router = IntentRouterStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = router
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Search the internet for two current antique cup offers. "
        "Report the price and direct link for each."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert router.called is True
    assert [name for name, _ in tools.calls] == [
        "web_search",
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    assert answer.endswith(f"Cup B costs 20 EUR at {second}.")


@pytest.mark.asyncio
async def test_business_hours_lookup_recovers_after_false_chat_intent(
    tmp_path: Path,
) -> None:
    prompt = (
        "Cześć mi. Słuchaj, sprawdź mi ile w Warszawie jest cukierni "
        "Cud Malina i sprawdź mi również godziny otwarcia. "
        "I sprawdź mi również gdzie. Podaj mi tylko sprawdzone informacje."
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query" if name == "web_search" else "url": {
                                    "type": "string"
                                }
                            },
                        },
                    },
                }
                for name in ("web_search", "web_read")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "web_search":
                return json.dumps(
                    {
                        "query": arguments["query"],
                        "engine": "duckduckgo",
                        "results": [
                            {
                                "title": "Cud Malina Warsaw",
                                "url": "https://example.com/cud-malina",
                            }
                        ],
                    }
                )
            return json.dumps(
                {
                    "url": "https://example.com/cud-malina",
                    "title": "Cud Malina Warsaw",
                    "content": "Cud Malina. Address: Warsaw. Hours: 09:00-18:00.",
                }
            )

    class IntentRouterStub:
        async def classify(self, prompt: str, **kwargs):
            return None

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            offered = {
                item["function"]["name"] for item in (kwargs.get("tools") or [])
            }
            if self.turn == 1:
                assert offered == {"web_search", "web_read"}
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "search",
                            "web_search",
                            {"query": "Cud Malina Warszawa godziny otwarcia"},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "read",
                            "web_read",
                            {"url": "https://example.com/cud-malina"},
                        )
                    ]
                )
            return LLMResponse(
                content=(
                    "One verified Cud Malina location, Boss: Warsaw, "
                    "open 09:00-18:00. Source: https://example.com/cud-malina"
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert [name for name, _ in tools.calls] == ["web_search", "web_read"]
    assert "open 09:00-18:00" in answer


@pytest.mark.asyncio
async def test_explicit_existing_tool_overrides_semantic_creation_guess(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "count_words",
                        "description": "Count words.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create a tool.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return '{"word_count": 6}'

    class IntentRouterStub:
        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                action_requested=True,
                capabilities=("learning_tool",),
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            offered = [
                item["function"]["name"] for item in (kwargs.get("tools") or [])
            ]
            assert offered == (["count_words"] if self.turn == 1 else [])
            assert "learning_create_tool" not in kwargs["messages"][0]["content"]
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "count",
                            "count_words",
                            {"text": "V umie robić swoje własne narzędzia."},
                        )
                    ]
                )
            return LLMResponse(content="The sentence has six words.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Użyj narzędzia CANT WORDS na zdaniu V umie robić swoje własne narzędzia."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [
        (
            "count_words",
            {"text": "V umie robić swoje własne narzędzia."},
        )
    ]
    assert answer == "The sentence has six words."


def test_latest_action_context_skips_empty_follow_up_checkpoint(tmp_path: Path) -> None:
    action = AgentTaskTrace(tmp_path, "Inspect example.com and report its contents")
    action.set_requirements(
        TaskContract.from_prompt(
            "Inspect example.com and report its contents"
        ).to_dict()
    )
    action.complete("I need one more observation.")

    empty_follow_up = AgentTaskTrace(tmp_path, "No to dawaj")
    empty_follow_up.set_requirements(TaskContract().to_dict())
    empty_follow_up.complete("I'll use the correct tool now.")

    recovered = AgentTaskTrace.latest_action_context(tmp_path)

    assert recovered is not None
    assert recovered["task_id"] == action.task_id
    assert recovered["requirements"]["requires_browser_navigation"] is True


def test_latest_action_context_skips_ungrounded_runtime_review_checkpoint(
    tmp_path: Path,
) -> None:
    action = AgentTaskTrace(tmp_path, "Inspect example.com and report its contents")
    action.set_requirements(
        TaskContract.from_prompt(
            "Inspect example.com and report its contents"
        ).to_dict()
    )
    action.complete("Done.")

    poisoned = AgentTaskTrace(tmp_path, "How would you do that?")
    poisoned.set_requirements(
        TaskContract(
            requires_runtime_review=True,
            requires_evidence_report=True,
        ).to_dict()
    )
    sequence = poisoned.tool_started("runtime_review_task", {})
    poisoned.tool_finished(sequence, '{"findings": []}')
    poisoned.complete("No findings.")

    recovered = AgentTaskTrace.latest_action_context(tmp_path)

    assert recovered is not None
    assert recovered["task_id"] == action.task_id


def test_continuation_rejects_poisoned_immediate_runtime_review_context(
    tmp_path: Path,
) -> None:
    agent = object.__new__(Agent)
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "task_id": "interactive-poisoned",
        "objective": "How would you do that?",
        "requirements": {
            "requires_runtime_review": True,
            "requires_evidence_report": True,
        },
        "tool_calls": [{"tool": "runtime_review_task", "status": "succeeded"}],
    }
    trace = AgentTaskTrace(tmp_path, "Repeat that task")

    recovered = agent._continued_action_context(
        "Repeat that task",
        trace,
        force=True,
    )

    assert recovered is None
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "journal" / f"{trace.task_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "continuation_context_rejected" in events


@pytest.mark.asyncio
async def test_poisoned_repeat_request_finishes_without_llm_or_tools(
    tmp_path: Path,
) -> None:
    class IntentRouterStub:
        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                message_clear=True,
                action_requested=False,
                continue_previous=True,
                references_previous=True,
            )

    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("an ungrounded repeat must not discover tools")

    class LLMStub:
        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("an ungrounded repeat must not reach the LLM")

    agent = object.__new__(Agent)
    agent.intent_router = IntentRouterStub()
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = SimpleNamespace(session=Session())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "task_id": "interactive-poisoned",
        "objective": "Więc jak byś to zrobiła?",
        "requirements": {
            "requires_runtime_review": True,
            "requires_evidence_report": True,
        },
        "tool_calls": [{"tool": "runtime_review_task", "status": "succeeded"}],
    }

    answer = await agent._run_agent_loop(
        "V powtórz zadanie, ale z innymi parametrami albo innym modelem."
    )

    assert "can't identify which earlier job" in answer
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []


@pytest.mark.asyncio
async def test_missing_dialogue_reference_asks_instead_of_guessing(
    tmp_path: Path,
) -> None:
    class IntentRouterStub:
        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                message_clear=True,
                action_requested=False,
                references_previous=True,
            )

    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("a missing conversation reference needs no tools")

    class LLMStub:
        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("DARKLINGER must not ask the model to invent context")

    agent = object.__new__(Agent)
    agent.intent_router = IntentRouterStub()
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    session = Session()
    agent.memory = SimpleNamespace(session=session)
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        "How would you approach that earlier task with my friend?"
    )

    assert "don't have the earlier conversation" in answer
    assert "making shit up" in answer
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []
    assert session.messages() == [
        {
            "role": "user",
            "content": "How would you approach that earlier task with my friend?",
        },
        {"role": "assistant", "content": answer},
    ]


def test_complete_explanation_is_not_rejected_as_missing_dialogue() -> None:
    intent = SemanticIntent(
        message_clear=True,
        action_requested=False,
        references_previous=True,
    )
    prompt = (
        "My friend claims Windows makes risky cybersecurity work invisible. "
        "I disagree and want to show him safely why that belief is wrong. "
        "How would you approach that situation without damaging his computer?"
    )

    assert Agent._reference_requires_missing_dialogue(
        intent,
        prompt,
        has_dialogue=False,
    ) is False


def test_actionable_request_is_not_rejected_only_for_reference_flag() -> None:
    intent = SemanticIntent(
        message_clear=True,
        action_requested=True,
        references_previous=True,
        capabilities=("browser",),
    )

    assert Agent._reference_requires_missing_dialogue(
        intent,
        "Find current information about that Windows security claim.",
        has_dialogue=False,
    ) is False


def test_repeated_complete_web_request_does_not_require_dialogue_history() -> None:
    intent = SemanticIntent(
        message_clear=True,
        action_requested=True,
        references_previous=True,
        capabilities=("browser",),
    )
    prompt = (
        "Jeszcze raz tłumaczę: znajdź na GitHubie przydatne narzędzia i "
        "umiejętności dotyczące darknetu, OSINT i cyber security. Przedstaw "
        "konkretną listę do mojego zatwierdzenia, zanim cokolwiek przyswoisz."
    )

    assert Agent._web_reference_requires_history(intent, prompt) is False


def test_bare_web_reference_still_requires_dialogue_history() -> None:
    intent = SemanticIntent(
        message_clear=True,
        action_requested=True,
        references_previous=True,
        capabilities=("browser",),
    )

    assert Agent._web_reference_requires_history(
        intent,
        "Search for that again.",
    ) is True


def test_browser_research_batch_stops_after_first_uncaptured_navigation() -> None:
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_evidence_report=True,
        minimum_detail_sources=2,
        required_research_facets=("github_repositories",),
    )
    requests = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/example/one"},
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/example/two"},
        },
    ]

    assert Agent._bound_browser_evidence_batch(requests, contract) == requests[:1]


def test_browser_research_batch_keeps_capture_before_next_navigation() -> None:
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_evidence_report=True,
        minimum_detail_sources=2,
        required_research_facets=("github_repositories",),
    )
    requests = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/example/one"},
        },
        {"tool": "browser_snapshot", "arguments": {}},
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/example/two"},
        },
    ]

    assert Agent._bound_browser_evidence_batch(requests, contract) == requests[:2]


@pytest.mark.asyncio
async def test_agent_continuation_inherits_browser_contract_and_tool_routing(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("browser_navigate", "browser_snapshot", "read_file")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                return "Navigated to https://example.com"
            return "Example Domain contains documentation links."

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.offered_tools: list[set[str]] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            self.offered_tools.append(
                {
                    item["function"]["name"]
                    for item in (kwargs.get("tools") or [])
                }
            )
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_nav",
                            name="browser_navigate",
                            arguments={"url": "https://example.com"},
                            raw_arguments='{"url":"https://example.com"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_snapshot",
                            name="browser_snapshot",
                            arguments={},
                            raw_arguments="{}",
                        )
                    ],
                    native_tools_enabled=True,
                )
            return LLMResponse(
                content=(
                    "I checked the damn page: Example Domain contains "
                    "documentation links, Boss."
                )
            )

    class IntentRouterStub:
        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                message_clear=True,
                action_requested=True,
                continue_previous=True,
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    previous = AgentTaskTrace(tmp_path, "Inspect example.com and report its contents")
    previous.set_requirements(
        TaskContract.from_prompt(
            "Inspect example.com and report its contents"
        ).to_dict()
    )
    previous.complete("Let me search instead.")

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = AgentTaskTrace.latest_context(tmp_path)
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("No to dawaj.")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == (
        "I checked the damn page: Example Domain contains documentation links, Boss."
    )
    assert tools.calls == [
        ("browser_navigate", {"url": "https://example.com"}),
        ("browser_snapshot", {}),
    ]
    assert all(
        {"browser_navigate", "browser_snapshot"}.issubset(names)
        for names in llm.offered_tools[:-1]
    )
    assert llm.offered_tools[-1] == set()
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["requirements"]["requires_browser_navigation"] is True
    assert checkpoint["requirements"]["requires_browser_snapshot"] is True


@pytest.mark.asyncio
async def test_agent_uses_semantic_browser_intent_for_hungarian_request(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("browser_navigate", "browser_snapshot", "read_file")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                return "Navigated to https://example.com"
            return "Example Domain contains documentation links."

    class IntentRouterStub:
        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            assert prompt == "Keress az interneten es keszits jelentest."
            return SemanticIntent(
                action_requested=True,
                capabilities=("browser",),
                requires_report=True,
                distinct_detail_page=True,
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.offered_tools: list[set[str]] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            self.offered_tools.append(
                {
                    item["function"]["name"]
                    for item in (kwargs.get("tools") or [])
                }
            )
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_nav_hu",
                            name="browser_navigate",
                            arguments={"url": "https://example.com"},
                            raw_arguments='{"url":"https://example.com"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_snapshot_hu",
                            name="browser_snapshot",
                            arguments={},
                            raw_arguments="{}",
                        )
                    ],
                    native_tools_enabled=True,
                )
            return LLMResponse(
                content=(
                    "I checked the damn page: Example Domain contains "
                    "documentation links, Boss."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Keress az interneten es keszits jelentest."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "Example Domain" in answer
    assert tools.calls[0][0] == "browser_navigate"
    assert tools.calls[0][1]["url"].startswith(
        "https://duckduckgo.com/?q=Keress+az+interneten"
    )
    assert tools.calls[1] == ("browser_snapshot", {})
    assert all(
        names == {"browser_navigate", "browser_snapshot"}
        for names in llm.offered_tools
    )


@pytest.mark.asyncio
async def test_agent_rolls_context_and_continues_with_runtime_evidence(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            marker = "NavigationEvidence" if tool == "browser_navigate" else "AlphaSignal"
            return f"{marker} " + ("evidence " * 700)

    class LLMStub:
        config = SimpleNamespace(context=2_048)

        def __init__(self) -> None:
            self.turn = 0
            self.summary_calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_nav_rollover",
                            name="browser_navigate",
                            arguments={"url": "https://example.com"},
                            raw_arguments='{"url":"https://example.com"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_snapshot_rollover",
                            name="browser_snapshot",
                            arguments={},
                            raw_arguments="{}",
                        )
                    ],
                    native_tools_enabled=True,
                )
            assert any(
                "DARKLINGER context rollover capsule" in str(message.get("content", ""))
                for message in kwargs["messages"]
            )
            return LLMResponse(
                content=(
                    "I checked the damn page. The verified snapshot contains "
                    "AlphaSignal, Boss."
                )
            )

        async def ask(self, **kwargs) -> str:
            self.summary_calls += 1
            return json.dumps(
                {
                    "completed": ["Navigated and captured evidence"],
                    "findings": ["AlphaSignal is present in runtime evidence"],
                    "open_questions": [],
                    "next_steps": ["Finish the evidence-backed report"],
                }
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager(
        threshold_percent=45,
        reserve_tokens=256,
    )
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 6

    answer = await agent._run_agent_loop(
        "Inspect https://example.com and report what you find."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "AlphaSignal" in answer
    assert tools.calls == ["browser_navigate", "browser_snapshot"]
    # Routine context bookkeeping belongs to DARKLINGER's deterministic runtime,
    # not to an extra local-model generation between every tool call.
    assert llm.summary_calls == 0
    checkpoint_path = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"
    assert checkpoint["context_rollovers"]
    assert all(
        rollover["estimated_tokens_after"] < rollover["estimated_tokens_before"]
        for rollover in checkpoint["context_rollovers"]
    )


@pytest.mark.asyncio
async def test_agent_retries_once_after_provider_context_overflow(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        async def openai_tool_definitions(self) -> list[dict]:
            return []

    class LLMStub:
        config = SimpleNamespace(context=4_096)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                response = httpx.Response(
                    500,
                    request=httpx.Request("POST", "http://local/v1/chat/completions"),
                )
                raise APIStatusError(
                    "request exceeds the available context size",
                    response=response,
                    body={"message": "prompt exceeds the available context size"},
                )
            return LLMResponse(content="That context rollover worked, damn it.")

        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": [],
                    "findings": [],
                    "open_questions": [],
                    "next_steps": ["Retry the interrupted generation"],
                }
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager(threshold_percent=90)
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Tell me your view of context rollover.")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "That context rollover worked, damn it."
    assert llm.calls == 2
    checkpoint_path = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["context_rollovers"][0]["emergency"] is True


def test_agent_rejects_promises_to_search_or_use_a_tool_later() -> None:
    assert Agent._claims_unverified_work(
        "Okay, I'll use the correct tool now."
    )
    assert Agent._claims_unverified_work(
        "No direct match. Let me search for skills instead."
    )
    assert Agent._claims_unverified_work(
        "Okay, let's go deeper. I've already pulled down two repositories. "
        "Now I need to extract the most relevant details from each. First, "
        "I'll dive into the first repository. After that, I'll compile both "
        "into a concise list ready for your approval."
    )


@pytest.mark.asyncio
async def test_agent_switches_model_after_two_missing_tool_calls(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "count_words",
                        "description": "Count words in text.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return '{"count": 4}'

    class IntentRouterStub:
        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(action_requested=True)

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.handoff_seen = False

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(content="I'll use count_words now.")
            if self.turn == 2:
                return LLMResponse(content="Let me call the tool this time.")
            if self.turn == 3:
                self.handoff_seen = any(
                    "DARKLINGER runtime handoff" in str(message.get("content", ""))
                    for message in kwargs["messages"]
                )
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "count",
                            "count_words",
                            {"text": "one two three four"},
                        )
                    ],
                    native_tools_enabled=True,
                )
            return LLMResponse(content="The verified word count is four, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    switches: list[tuple[str, str, tuple[str, ...]]] = []

    async def response_fallback(
        prompt: str,
        task_kind: str,
        excluded_model_paths: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        switches.append((prompt, task_kind, excluded_model_paths))
        return SimpleNamespace(
            switched=True,
            previous_model_path="mythos.gguf",
            active_model_path="tool-specialist.gguf",
        )

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.response_fallback_router = response_fallback

    prompt = (
        "Use the count_words tool on 'one two three four' and report the result."
    )
    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert switches == [(prompt, "tool_use", ())]
    assert llm.handoff_seen is True
    assert tools.calls == [
        ("count_words", {"text": "one two three four"})
    ]
    assert answer == "The verified word count is four, Boss."
    journal = next((tmp_path / "journal").glob("*.jsonl")).read_text(
        encoding="utf-8"
    )
    assert '"event": "model_protocol_fallback"' in journal
    assert '"fallback_switched": true' in journal


@pytest.mark.asyncio
async def test_agent_switches_model_after_two_recoverable_builder_failures(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_skill",
                        "description": "Create a validated skill.",
                        "parameters": {
                            "type": "object",
                            "properties": {"manifest": {"type": "object"}},
                            "required": ["manifest"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            assert tool == "learning_create_skill"
            self.calls += 1
            if self.calls == 1:
                raise ValueError("artifact version must be semantic MAJOR.MINOR.PATCH")
            if self.calls == 2:
                raise ValueError(
                    "a skill requires at least one positive and one negative "
                    "trigger test"
                )
            return '{"name":"research_skill","status":"active"}'

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.handoff_seen = False

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn <= 3:
                if self.turn == 3:
                    self.handoff_seen = any(
                        "failed the active tool schema" in str(
                            message.get("content", "")
                        )
                        for message in kwargs["messages"]
                    )
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            f"skill-{self.turn}",
                            "learning_create_skill",
                            {
                                "manifest": {
                                    "version": "1.0.0",
                                    "revision": self.turn,
                                }
                            },
                        )
                    ],
                    native_tools_enabled=True,
                )
            return LLMResponse(content="The skill is active, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    switches: list[tuple[str, str, tuple[str, ...]]] = []

    async def response_fallback(
        prompt: str,
        task_kind: str,
        excluded_model_paths: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        switches.append((prompt, task_kind, excluded_model_paths))
        return SimpleNamespace(
            switched=True,
            previous_model_path="weak-builder.gguf",
            active_model_path="coding-specialist.gguf",
        )

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.response_fallback_router = response_fallback

    prompt = "Stwórz skill research_skill."
    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert switches == [(prompt, "coding", ())]
    assert llm.handoff_seen is True
    assert tools.calls == 3
    assert answer == "The skill is active, Boss."
    journal = next((tmp_path / "journal").glob("*.jsonl")).read_text(
        encoding="utf-8"
    )
    assert '"event": "model_tool_recovery_fallback"' in journal
    assert '"fallback_switched": true' in journal


@pytest.mark.asyncio
async def test_non_action_conversation_skips_tool_schema_discovery() -> None:
    class ToolsStub:
        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("conversation must not start MCP discovery")

        async def call(self, *args, **kwargs) -> str:
            raise AssertionError("conversation must not execute a tool")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def respond(self, **kwargs) -> LLMResponse:
            assert kwargs["tools"] is None
            return LLMResponse(
                content=(
                    "Tools should answer to the agent's runtime, not hijack every "
                    "damn conversation, Boss."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Tell me what you think about tools taking over ordinary conversation."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "not hijack" in answer


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [
    "Imagine that you had a physical body and we sat on a bench watching the sunset.",
    "Review all available tools and report possible improvements.",
    "Przejrzyj dostępne narzędzia i przedstaw raport.",
    "Examina las herramientas disponibles y presenta un informe.",
])
async def test_malformed_intent_cannot_promote_long_conversation_to_tool_task(
    tmp_path: Path,
    prompt: str,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("parser failure must not expose tools to conversation")

    class RouterStub:
        last_response = '{"message_clear":true,"action_requested":false'
        last_failure_reason = "invalid_classification"

        async def classify(self, prompt: str, **kwargs) -> None:
            return None

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        async def ask(self, **kwargs) -> str:
            raise AssertionError("failed routing must not generate invented findings")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(prompt)

    assert "request classification failed" in answer
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []
    events = [
        json.loads(line)
        for line in next((tmp_path / "journal").glob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    compact = [item for item in events if item["event"] == "classification_blocked"]
    assert compact[-1]["data"]["reason"] == (
        "no_verified_execution_route"
    )


@pytest.mark.asyncio
async def test_stale_subject_classification_asks_for_repeat_without_guessing(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("unclear input must not discover tools")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("unclear input must not reach answer generation")

    class RouterStub:
        last_response = json.dumps(
            {
                "public_subject": "Cukiernia Cud Malina",
                "web_query": "Cukiernia Cud Malina",
            }
        )
        last_failure_reason = "current_message_grounding"

        async def classify(self, prompt: str, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "objective": "Find Cukiernia Cud Malina",
        "status": "completed",
        "requirements": {},
    }
    visible: list[str] = []

    answer = await agent._run_agent_loop(
        "Dupin i kamień i szupa.",
        visible.append,
    )

    assert "brain just throw a syntax error" in answer
    assert visible == [answer]
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["context_rollovers"] == []
    events = [
        json.loads(line)["event"]
        for line in next((tmp_path / "journal").glob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert "unclear_input_rejected" in events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "classified_intent",
    [
        SemanticIntent(message_clear=False),
        SemanticIntent(message_clear=True, message_odd=True),
    ],
)
async def test_semantically_unclear_chat_never_reaches_answer_model(
    tmp_path: Path,
    classified_intent: SemanticIntent,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("word salad must not discover tools")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("word salad must not reach answer generation")

    class RouterStub:
        last_response = ""
        last_failure_reason = ""

        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            return classified_intent

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop("Zupa strupa, dupa i kamieni kupa.")

    assert "brain just throw a syntax error" in answer
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["tool_calls"] == []
    assert checkpoint["context_rollovers"] == []


@pytest.mark.asyncio
async def test_short_non_action_statement_uses_current_message_only_chat(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("compact chat must not discover tools")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.calls = 0
            self.messages: list[dict] = []

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.calls += 1
            self.messages = messages
            assert kwargs["max_tokens"] == 96
            return (
                "Boss... you okay, or did your brain just throw a syntax "
                "error? Try that again."
            )

    class RouterStub:
        last_response = ""
        last_failure_reason = ""

        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            return SemanticIntent(message_clear=True)

    class MemoryStub:
        def __init__(self) -> None:
            self.relationship_state = RelationshipState()
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            raise AssertionError("ambiguous compact banter must not enter memory")

    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "objective": "I've been repeating the same memory twice",
        "status": "completed",
        "requirements": {},
    }

    answer = await agent._run_agent_loop(
        "Sralimuchy będzie wiosna, będzie trawka równorosła."
    )

    assert "brain just throw a syntax error" in answer
    assert llm.calls == 1
    encoded_messages = json.dumps(llm.messages, ensure_ascii=False)
    assert "Sralimuchy będzie wiosna" in encoded_messages
    assert "I've been repeating the same memory twice" not in encoded_messages
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["context_rollovers"] == []


def test_repetitive_banter_detector_is_language_neutral() -> None:
    assert Agent._looks_like_repetitive_banter(
        "Scooby-Dooja, Scooby-Doobie-Dooja, Scooby-Doobie-Dooja."
    )
    assert not Agent._looks_like_repetitive_banter(
        "My uncle was shouted at on Sunday."
    )


@pytest.mark.asyncio
async def test_repetitive_banter_bypasses_intent_and_answer_models(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

    class LLMStub:
        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("repetitive banter must not reach the LLM")

    class RouterStub:
        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            raise AssertionError("repetitive banter must not reach intent parsing")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        "Scooby-Dooja, Scooby-Doobie-Dooja, Scooby-Doobie-Dooja."
    )

    assert "brain just throw a syntax error" in answer
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["tool_calls"] == []


def test_agent_bounds_tool_output_for_model_context() -> None:
    fitted = Agent._fit_tool_output("HEAD\n" + "x" * 20_000 + "\nTAIL", max_characters=2_000)

    assert len(fitted) == 2_000
    assert fitted.startswith("HEAD")
    assert fitted.endswith("TAIL")
    assert "DARKLINGER omitted the middle" in fitted


def test_agent_prioritizes_duckduckgo_results_in_long_browser_snapshot() -> None:
    chrome = "\n".join(
        f"          - generic [ref=menu{i}]: DuckDuckGo menu item {i}"
        for i in range(300)
    )
    result = f"""### Page
- Page URL: https://duckduckgo.com/?q=firecrawl+alternatives&ia=web
- Page Title: firecrawl alternatives at DuckDuckGo
### Snapshot
```yaml
{chrome}
          - listitem [ref=e147]:
            - article [ref=e148] [cursor=pointer]:
              - generic [ref=e155]:
                - link "https://example.org/firecrawl-alternatives" [ref=e162]:
                  - /url: https://example.org/firecrawl-alternatives
              - heading [level=2] [ref=e168]:
                - link "Seven tested Firecrawl alternatives" [ref=e169]:
                  - /url: https://example.org/firecrawl-alternatives
              - generic [ref=e174]: Crawl4AI, Crawlee, and Trafilatura are compared.
          - listitem [ref=e176]:
            - article [ref=e177] [cursor=pointer]:
              - heading [level=2] [ref=e197]:
                - link "Official Crawl4AI documentation" [ref=e198]:
                  - /url: https://docs.crawl4ai.com/
              - generic [ref=e200]: Open-source crawling documentation and examples.
```
"""

    fitted = Agent._fit_browser_snapshot_output(result, max_characters=2_000)

    assert len(fitted) <= 2_000
    assert "Seven tested Firecrawl alternatives" in fitted
    assert "https://example.org/firecrawl-alternatives" in fitted
    assert "Official Crawl4AI documentation" in fitted
    assert "DuckDuckGo menu item 299" not in fitted
    assert "prioritized observed DuckDuckGo search-result blocks" in fitted


def test_agent_prioritizes_topic_evidence_on_long_detail_page() -> None:
    chrome = "\n".join(
        f"  - generic [ref=menu{i}]: Navigation item {i}"
        for i in range(350)
    )
    result = f"""### Page
- Page URL: https://www.tinyfish.ai/blog/firecrawl-alternatives
- Page Title: 8 Best Firecrawl Alternatives in 2026: Real Reviews & Pricing
### Snapshot
```yaml
{chrome}
  - main [ref=body]:
    - heading "8 Best Firecrawl Alternatives in 2026" [level=1] [ref=h1]
    - paragraph [ref=p1]: Crawl4AI is free, open-source, and self-hosted.
    - paragraph [ref=p2]: Spider.cloud and ScrapingBee are low-cost hosted options.
    - heading "Crawl4AI" [level=2] [ref=h2]
      - paragraph [ref=p3]: Local open-source crawler.
    - heading "Crawlee" [level=2] [ref=h3]
      - paragraph [ref=p4]: Browser automation and crawling framework.
```
"""

    fitted = Agent._fit_browser_snapshot_output(result, max_characters=2_000)

    assert len(fitted) <= 2_000
    assert "Crawl4AI is free" in fitted
    assert "Spider.cloud and ScrapingBee" in fitted
    assert 'heading "Crawl4AI"' in fitted
    assert 'heading "Crawlee"' in fitted
    assert "Navigation item 349" not in fitted
    assert "prioritized topic-relevant detail-page evidence" in fitted


def test_agent_preserves_requested_images_on_long_detail_page() -> None:
    chrome = "\n".join(
        f"  - generic [ref=menu{i}]: Navigation item {i}"
        for i in range(350)
    )
    result = f'''- Page URL: https://catalogue.test/species
- Page Title: Regional catalogue
{chrome}
  - main:
    - img "Species alpha field photograph" [ref=i1]
    - img "Species beta field photograph" [ref=i2]
    - heading "Background" [level=2]
      - paragraph: General catalogue introduction.
'''

    fitted = Agent._fit_browser_snapshot_output(
        result,
        max_characters=2_000,
        preserve_images=True,
    )

    assert "Species alpha field photograph" in fitted
    assert "Species beta field photograph" in fitted
    assert "Navigation item 349" not in fitted


def test_agent_keeps_price_and_date_from_long_product_page() -> None:
    chrome = "\n".join(
        f"  - generic [ref=menu{i}]: Navigation item {i}"
        for i in range(350)
    )
    result = f"""- Page URL: https://market.test/product/vintage-cup
- Page Title: Vintage porcelain cup
{chrome}
  - main:
    - heading "Vintage porcelain cup" [level=1]
    - paragraph: Made in France in 1964, with a cobalt decoration.
    - paragraph: 125 EUR
    - link "Buy this item":
      - /url: https://market.test/product/vintage-cup
"""

    fitted = Agent._fit_browser_snapshot_output(result, max_characters=2_000)

    assert len(fitted) <= 2_000
    assert "1964" in fitted
    assert "125 EUR" in fitted
    assert "Navigation item 349" not in fitted


def test_agent_prioritizes_repeated_product_cards_on_listing_page() -> None:
    chrome = "\n".join(
        f"  - link \"Books category {index}\": /category/{index}"
        for index in range(300)
    )
    cards = """
      - article [ref=a1]:
        - link [ref=l1]:
          - /url: catalogue/a-light-in-the-attic_1000/index.html
          - img "A Light in the Attic" [ref=i1]
        - paragraph [ref=p1]: £51.77
        - text: In stock
      - article [ref=a2]:
        - link [ref=l2]:
          - /url: catalogue/tipping-the-velvet_999/index.html
          - img "Tipping the Velvet" [ref=i2]
        - paragraph [ref=p2]: £53.74
        - text: In stock
      - article [ref=a3]:
        - link [ref=l3]:
          - /url: catalogue/soumission_998/index.html
          - img "Soumission" [ref=i3]
        - paragraph [ref=p3]: £50.10
        - text: In stock
    """
    result = f"""- Page URL: https://books.toscrape.com/
- Page Title: All products | Books to Scrape - Sandbox
{chrome}
{cards}
"""

    fitted = Agent._fit_browser_snapshot_output(result, max_characters=2_000)

    assert len(fitted) <= 2_000
    assert "A Light in the Attic" in fitted
    assert "Tipping the Velvet" in fitted
    assert "Soumission" in fitted
    assert "£51.77" in fitted
    assert "In stock" in fitted
    assert "Books category 299" not in fitted
    assert "prioritized repeated observed page-item blocks" in fitted


@pytest.mark.asyncio
async def test_agent_executes_native_tool_call_and_returns_tool_role() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return "# DARKLINGER"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.second_messages: list[dict] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_read",
                            name="read_file",
                            arguments={"path": "README.md"},
                            raw_arguments='{"path":"README.md"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            self.second_messages = kwargs["messages"]
            return LLMResponse(content="The heading is `# DARKLINGER`, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Read README.md")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "The heading is `# DARKLINGER`, Boss."
    assert tools.calls == [("read_file", {"path": "README.md"})]
    tool_message = next(item for item in llm.second_messages if item["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_read"
    assert "Status: succeeded" in tool_message["content"]


@pytest.mark.asyncio
async def test_agent_renders_exact_first_heading_without_second_model_claim() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
            return "# DARKLINGER / V-Core\n\nFramework documentation"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turns = 0
            self.responses = iter(
                (
                    '{"tool":"read_file","arguments":{"path":"README.md"}}',
                    "I'm done.",
                    "The first heading is `# DARKLINGER / V-Core`, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            self.turns += 1
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Read README.md and report only its first heading."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "# DARKLINGER / V-Core"
    assert agent.tools.calls == 1
    assert agent.llm.turns == 1


@pytest.mark.asyncio
async def test_agent_uses_controlled_tool_creation_for_explicit_missing_tool() -> None:
    manifest = {
        "name": "extract_titles",
        "version": "1.0.0",
        "description": "Extract titles from bounded input.",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "tests": [{"name": "empty", "arguments": {}, "expected": {"titles": []}}],
        "scope": "task",
        "lesson_ids": [],
        "timeout_seconds": 10,
    }

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create and validate a tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return '{"status":"active","validation":"passed"}'

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                arguments = {
                    "manifest": manifest,
                    "source": "def run(arguments):\n    return {'titles': []}",
                }
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("create_1", "learning_create_tool", arguments)
                    ]
                )
            return LLMResponse(
                content="The tool passed its lifecycle checks and is active, Boss."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Create a generated tool for title extraction")
    await asyncio.gather(*agent._memory_tasks)

    assert "active" in answer
    assert agent.tools.calls[0][0] == "learning_create_tool"


@pytest.mark.asyncio
async def test_agent_repairs_empty_learning_call_before_tool_execution() -> None:
    schema = {
        "type": "object",
        "properties": {
            "manifest": {"type": "object", "additionalProperties": True},
            "source": {"type": "string", "minLength": 1},
        },
        "required": ["manifest", "source"],
        "additionalProperties": False,
    }
    valid_arguments = {
        "manifest": {
            "name": "count_words",
            "version": "1.0.0",
            "description": "Count words in bounded text.",
            "input_schema": {},
            "output_schema": {},
            "tests": [],
        },
        "source": "def run(arguments):\n    return {'word_count': 0}",
    }

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create and validate a tool.",
                        "parameters": schema,
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return '{"status":"active","validation":{"passed":true}}'

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0
            self.budgets: list[int] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            self.budgets.append(kwargs["max_tokens"])
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "bad-create",
                            "learning_create_tool",
                            {},
                            raw_arguments="{}",
                        )
                    ]
                )
            if self.turn == 2:
                assert "missing required fields" in str(kwargs["messages"][-1])
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "fixed-create",
                            "learning_create_tool",
                            valid_arguments,
                            raw_arguments=json.dumps(valid_arguments),
                        )
                    ]
                )
            return LLMResponse(
                content="The tool is active. Broken call repaired before execution, Boss."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Create a tool that counts words.")
    await asyncio.gather(*agent._memory_tasks)

    assert "active" in answer
    assert tools.calls == [("learning_create_tool", valid_arguments)]
    assert llm.budgets == [3_000, 3_000, 256]


@pytest.mark.asyncio
async def test_agent_redirects_premature_generated_tool_to_learning_lifecycle() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create and activate a generated tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "source": {"type": "string"},
                                "test": {"type": "object"},
                            },
                            "required": ["name", "description", "source", "test"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "learning_list_artifacts",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]
            if self.active:
                definitions.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "double_value",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                                "required": ["value"],
                            },
                        },
                    }
                )
            return definitions

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "learning_create_tool":
                self.active = True
                return '{"name":"double_value","status":"active"}'
            if tool == "double_value":
                return '{"result":42}'
            raise AssertionError(f"unexpected real call: {tool}")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            available = {
                item["function"]["name"] for item in kwargs.get("tools") or []
            }
            if self.turn == 1:
                assert available == {"learning_create_tool"}
                return LLMResponse(
                    tool_calls=[LLMToolCall("too-early", "double_value", {})]
                )
            if self.turn == 2:
                assert available == {"learning_create_tool"}
                assert "DARKLINGER lifecycle correction" in str(kwargs["messages"][-1])
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "create",
                            "learning_create_tool",
                            {
                                "name": "double_value",
                                "description": "Double an integer.",
                                "source": (
                                    "def run(arguments):\n"
                                    "    return {'result': arguments['value'] * 2}"
                                ),
                                "test": {
                                    "name": "fixture",
                                    "arguments": {"value": 2},
                                    "expected": {"result": 4},
                                },
                            },
                        )
                    ]
                )
            if self.turn == 3:
                assert available == {"double_value"}
                return LLMResponse(
                    tool_calls=[LLMToolCall("run", "double_value", {"value": 21})]
                )
            return LLMResponse(content="The generated tool returned 42, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Create a generated tool named double_value and execute it on 21."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert [name for name, _ in tools.calls] == [
        "learning_create_tool",
        "double_value",
    ]
    assert "42" in answer


@pytest.mark.asyncio
async def test_agent_gives_model_source_only_and_runtime_owns_tool_contract() -> None:
    prompt = (
        "Create a tool named double_value. "
        'value = 2\nexpected = {"result": 4}\n'
        "After activation execute it for value = 21."
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            if not self.active:
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "learning_create_tool",
                            "description": "Runtime-owned source lifecycle.",
                            "parameters": {
                                "type": "object",
                                "properties": {"source": {"type": "string"}},
                                "required": ["source"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "double_value",
                        "description": "Double one value.",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "learning_create_tool":
                self.active = True
                assert set(arguments) == {"source"}
                return json.dumps(
                    {
                        "name": "double_value",
                        "status": "active",
                        "validation": {
                            "passed": True,
                            "tests": [
                                {
                                    "name": "owner-specified semantic oracle",
                                    "passed": True,
                                }
                            ],
                        },
                    }
                )
            if tool == "double_value":
                return json.dumps({"result": arguments["value"] * 2})
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                assert kwargs["tools"] is None
                assert "SOURCE PHASE" in kwargs["messages"][0]["content"]
                return LLMResponse(
                    content=(
                        "```python\n"
                        "def run(arguments):\n"
                        "    return {'result': arguments['value'] * 2}\n"
                        "```"
                    )
                )
            raise AssertionError(
                "runtime-bound execution and evidence report must bypass the model"
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    routed_phases: list[str] = []

    async def phase_router(_prompt: str, phase: str) -> SimpleNamespace:
        routed_phases.append(phase)
        return SimpleNamespace(
            previous_model_path="coder.gguf",
            active_model_path="executor.gguf",
            switched=True,
        )

    agent = object.__new__(Agent)
    agent.llm = llm
    agent.phase_router = phase_router
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 1
    assert tools.calls[0][0] == "learning_create_tool"
    assert tools.calls[1] == ("double_value", {"value": 21})
    assert '"result": 42' in answer
    assert routed_phases == []


@pytest.mark.asyncio
async def test_source_phase_rejects_malformed_builder_json_then_accepts_code() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"source": {"type": "string"}},
                            "required": ["source"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return json.dumps(
                {
                    "name": "count_words",
                    "status": "active",
                    "validation": {
                        "passed": True,
                        "tests": [
                            {
                                "name": "owner-specified semantic oracle",
                                "passed": True,
                            }
                        ],
                    },
                }
            )

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            assert kwargs["tools"] is None
            if self.turn == 1:
                return LLMResponse(
                    content=(
                        '{"tool":"learning_create_tool","arguments":'
                        '{"source":"print(123)"}'
                    )
                )
            if self.turn == 2:
                assert "rejected that source draft" in str(kwargs["messages"][-1])
                return LLMResponse(
                    content=(
                        "def run(arguments):\n"
                        "    return {'count': len(arguments['text'].split())}"
                    )
                )
            raise AssertionError("creation report must come from runtime evidence")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Create a tool named count_words. "
        'text = "V builds tools" expected = {"count": 3}'
    )
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 2
    assert len(tools.calls) == 1
    assert set(tools.calls[0][1]) == {"source"}
    assert "count_words" in answer
    assert "validated" in answer


@pytest.mark.asyncio
async def test_agent_rejects_text_tool_call_outside_authoritative_catalog() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def openai_tool_definitions(self) -> list[dict]:
            return []

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
            raise AssertionError("an uncatalogued tool must never execute")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    content=(
                        '{"tool":"write_file","arguments":'
                        '{"path":"notes.txt","content":"fake"}}'
                    )
                )
            return LLMResponse(content="I could not perform the write.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 1

    answer = await agent._run_agent_loop("Write notes.txt")
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == 0
    assert "UnknownToolError" in answer
    assert "nothing is running in the background" in answer.casefold()


@pytest.mark.asyncio
async def test_newly_created_tool_is_callable_in_same_agent_task() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            names = ["learning_create_tool"]
            if self.active:
                names.append("double_value")
            return [
                {"type": "function", "function": {"name": name}}
                for name in names
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "learning_create_tool":
                self.active = True
                return '{"status":"active","validation":{"passed":true}}'
            if tool == "double_value":
                return '{"result":42}'
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            available = {
                item["function"]["name"]
                for item in (kwargs.get("tools") or [])
            }
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "create_1",
                            "learning_create_tool",
                            {"manifest": {}, "source": "def run(arguments): return {}"},
                        )
                    ]
                )
            if self.turn == 2:
                assert "double_value" in available
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("run_1", "double_value", {"value": 21})
                    ]
                )
            return LLMResponse(content="The generated tool returned 42, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Create a generated tool named double_value and use it on 21."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == ["learning_create_tool", "double_value"]
    assert "42" in answer


@pytest.mark.asyncio
async def test_agent_grounds_creation_and_execution_from_distinct_fixtures() -> None:
    prompt = (
        "Create a task-scoped offline tool named multiply_values. Do not browse. "
        'items = [{"value":3}]\nscale = 2\n'
        "After it is active, really execute it on the second fixture. "
        'items = [{"value":5},{"value":7}]\nscale = 3'
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create and validate a tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "source": {"type": "string"},
                                "test": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["name", "description", "source", "test"],
                        },
                    },
                }
            ]
            if self.active:
                definitions.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "multiply_values",
                            "description": "Multiply supplied values.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "items": {"type": "array"},
                                    "scale": {"type": "integer"},
                                },
                                "required": ["items", "scale"],
                            },
                        },
                    }
                )
            return definitions

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "learning_create_tool":
                self.active = True
                return json.dumps(
                    {"name": "multiply_values", "status": "active"}
                )
            if tool == "multiply_values":
                return '{"values":[15,21]}'
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "create",
                            "learning_create_tool",
                            {
                                "name": "multiply_values",
                                "description": "Multiply supplied values.",
                                "source": (
                                    "def run(arguments):\n"
                                    "    items = arguments.get('items', [])\n"
                                    "    scale = arguments.get('scale', 1)\n"
                                    "    return {'values': "
                                    "[item['value'] * scale for item in items]}"
                                ),
                                "test": {
                                    "name": "fixture",
                                    "arguments": {
                                        "items": [{"value": 999}],
                                        "scale": 999,
                                    },
                                    "expected": {"values": [6]},
                                },
                            },
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "execute",
                            "multiply_values",
                            {"items": [], "scale": 0},
                        )
                    ]
                )
            return LLMResponse(content="The tool returned 15 and 21, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls[0][0] == "learning_create_tool"
    assert tools.calls[0][1]["test"]["arguments"] == {
        "items": [{"value": 3}],
        "scale": 2,
    }
    assert tools.calls[1] == (
        "multiply_values",
        {"items": [{"value": 5}, {"value": 7}], "scale": 3},
    )
    assert '"values": [15, 21]' in answer


@pytest.mark.asyncio
async def test_failed_sandbox_command_cannot_support_successful_test_claim(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.captured_failures: list[dict] = []

        async def call(self, tool: str, arguments: dict) -> str:
            assert tool == "sandbox_execute_offline"
            return json.dumps(
                {
                    "exit_code": 1,
                    "stdout": "1 failed",
                    "stderr": "assertion failed",
                    "timed_out": False,
                    "output_limited": False,
                    "workspace_limited": False,
                }
            )

        def capture_tool_failure(self, **failure: object) -> dict[str, str]:
            self.captured_failures.append(dict(failure))
            return {"evidence_id": "verified-failure", "outcome": "failure"}

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    '{"tool":"sandbox_execute_offline","arguments":'
                    '{"command":["/usr/bin/false"],"workspace":"tests"}}',
                    "All tests passed, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent.MAX_AGENT_STEPS = 1

    answer = await agent._run_agent_loop("Run the tests")
    await asyncio.gather(*agent._memory_tasks)

    assert "SandboxResultError" in answer
    assert "exit_code=1" in answer
    assert "All tests passed" not in answer
    assert len(tools.captured_failures) == 1
    assert tools.captured_failures[0]["tool"] == "sandbox_execute_offline"
    assert str(tools.captured_failures[0]["error"]).startswith(
        "SandboxResultError"
    )
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    journal = (
        agent._agent_trace_root
        / "journal"
        / f"{json.loads(checkpoint.read_text(encoding='utf-8'))['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "learning_evidence_recorded" in journal


def test_rejected_generated_artifact_is_a_failed_tool_result() -> None:
    error = Agent._tool_result_error(
        '{"status":"rejected","validation":{"passed":false}}',
        tool="learning_create_tool",
    )

    assert error.startswith("ArtifactLifecycleError")


@pytest.mark.asyncio
async def test_detail_page_contract_rejects_search_page_only_completion() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_snapshot" and len(self.calls) < 4:
                return "First result: https://github.com/mikalv/awesome-i2p"
            return "Repository heading: mikalv/awesome-i2p"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    '{"tool":"browser_navigate","arguments":{"url":"https://github.com/search?q=i2p"}}',
                    '{"tool":"browser_snapshot","arguments":{}}',
                    "The first repository is definitely something else, Boss.",
                    '{"tool":"browser_navigate","arguments":{"url":"https://github.com/mikalv/awesome-i2p"}}',
                    '{"tool":"browser_snapshot","arguments":{}}',
                    "The inspected first result is `mikalv/awesome-i2p`, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 8

    answer = await agent._run_agent_loop(
        "Search GitHub for I2P tools, inspect the first result, and report its actual repo."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "The inspected first result is `mikalv/awesome-i2p`, Boss."
    assert [name for name, _ in agent.tools.calls] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [
        "Search the internet for a Firecrawler alternative and report it.",
        (
            "Search the internet for a Firecrawler alternative and report it. "
            "If none exists, create a tool or skill."
        ),
    ],
)
async def test_contract_satisfaction_closes_tools_and_forces_final_report(
    tmp_path: Path,
    prompt: str,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in (
                    "browser_navigate",
                    "browser_snapshot",
                    "browser_find",
                )
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_snapshot":
                return (
                    "Result URL: https://scrapy.org. "
                    "Verified Scrapy evidence from the observed page"
                )
            return "Navigation succeeded"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.final_tools: object = "unset"

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "search",
                            "browser_navigate",
                            {"url": "https://duckduckgo.com/?q=firecrawler"},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("search-snapshot", "browser_snapshot", {})]
                )
            if self.turn == 3:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "detail",
                            "browser_navigate",
                            {"url": "https://scrapy.org"},
                        )
                    ]
                )
            if self.turn == 4:
                return LLMResponse(
                    tool_calls=[LLMToolCall("detail-snapshot", "browser_snapshot", {})]
                )
            self.final_tools = kwargs.get("tools")
            if self.turn == 5:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("unneeded", "browser_find", {"text": "more"})
                    ]
                )
            return LLMResponse(
                content="Scrapy is the verified alternative I inspected, Boss."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "Scrapy is the verified alternative I inspected, Boss."
    assert tools.calls == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    assert llm.final_tools is None
    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    journal = (
        agent._agent_trace_root
        / "journal"
        / f"{checkpoint['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_contract_satisfied" in journal
    assert "post_contract_tool_call_rejected" in journal


@pytest.mark.asyncio
async def test_code_analysis_finalization_requires_exact_failure_mapping(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            assert tool == "read_file"
            return 'def display_name(profile):\n    return profile["display_name"].strip()\n'

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.final_instruction = ""

        async def respond(self, **kwargs) -> LLMResponse:
            self.final_instruction = str(kwargs["messages"][-1]["content"])
            return LLMResponse(
                content=(
                    "`display_name({})` raises `KeyError` because the required "
                    "mapping key is absent."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        "Analyze test/fixtures/code_analysis_probe.py without changing it. "
        "Report the concrete input condition and exception type."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "KeyError" in answer
    assert "one exact input condition" in llm.final_instruction
    assert "different exceptions" in llm.final_instruction


@pytest.mark.asyncio
async def test_two_mangled_final_reports_fall_back_to_verified_evidence(
    tmp_path: Path,
) -> None:
    exact_url = "https://example.test"

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_snapshot":
                return (
                    f"- Page URL: {exact_url}\n"
                    "- Page Title: Verified Example\n"
                    '- heading "Observed content" [level=2]'
                )
            return f"- Page URL: {exact_url}"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": exact_url},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot", "browser_snapshot", {})
                    ]
                )
            return LLMResponse(
                content=(
                    "I inspected https://example.tes and found Verified Example, "
                    "Boss."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        f"Inspect {exact_url} and report what is there."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 4
    assert tools.calls == ["browser_navigate", "browser_snapshot"]
    assert "removed the unsupported claims" in answer
    assert f"Source: {exact_url}" in answer
    journal = next(
        (agent._agent_trace_root / "journal").glob("*.jsonl")
    ).read_text(encoding="utf-8")
    assert "final_answer_loop_cut_off" in journal


@pytest.mark.asyncio
async def test_unique_observed_url_on_same_host_repairs_final_report(
    tmp_path: Path,
) -> None:
    exact_url = "https://example.test/verified-page"
    invented_url = "https://example.test/invented-page"

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_snapshot":
                return (
                    f"- Page URL: {exact_url}\n"
                    "- Page Title: Verified Example\n"
                    '- heading "Observed content" [level=2]'
                )
            return f"- Page URL: {exact_url}"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": exact_url},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("snapshot", "browser_snapshot", {})]
                )
            return LLMResponse(
                content=(
                    f"I inspected {invented_url} and found Verified Example, Boss."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        f"Inspect {exact_url} and report what is there."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 3
    assert tools.calls == ["browser_navigate", "browser_snapshot"]
    assert exact_url in answer
    assert invented_url not in answer
    journal = next(
        (agent._agent_trace_root / "journal").glob("*.jsonl")
    ).read_text(encoding="utf-8")
    assert "final_answer_urls_repaired" in journal
    assert "final_answer_loop_cut_off" not in journal


@pytest.mark.asyncio
async def test_unsupported_online_claims_are_removed_but_observed_text_survives(
    tmp_path: Path,
) -> None:
    exact_url = "https://example.test/verified-report"
    invented_url = "https://example.test/made-up-report"
    observed_text = (
        "The inspected report describes a public enforcement action and its date"
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_snapshot":
                return (
                    f"- Page URL: {exact_url}\n"
                    "- Page Title: Verified Enforcement Report\n"
                    '- heading "Observed operation" [level=2]\n'
                    f"- paragraph: {observed_text}."
                )
            return f"- Page URL: {exact_url}"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": exact_url},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("snapshot", "browser_snapshot", {})]
                )
            return LLMResponse(
                content=(
                    f"Source: {invented_url}. **Marketplace X** and "
                    "**Ransomware Hub** were confirmed active."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        f"Inspect {exact_url} and report what is there."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 4
    assert exact_url in answer
    assert invented_url not in answer
    assert observed_text in answer
    assert "Marketplace X" not in answer
    assert "Ransomware Hub" not in answer
    journal = next(
        (agent._agent_trace_root / "journal").glob("*.jsonl")
    ).read_text(encoding="utf-8")
    assert "grounded_report_salvaged" in journal
    assert "final_answer_loop_cut_off" in journal


@pytest.mark.asyncio
async def test_two_empty_final_reports_fall_back_to_verified_evidence(
    tmp_path: Path,
) -> None:
    exact_url = "https://example.test"

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_snapshot":
                return (
                    f"- Page URL: {exact_url}\n"
                    "- Page Title: Verified Example\n"
                    '- heading "Observed content" [level=2]'
                )
            return f"- Page URL: {exact_url}"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": exact_url},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("snapshot", "browser_snapshot", {})]
                )
            return LLMResponse(content="")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        f"Inspect {exact_url} and report what is there."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 4
    assert tools.calls == ["browser_navigate", "browser_snapshot"]
    assert "returned an empty grounded report twice" in answer
    assert f"Source: {exact_url}" in answer
    journal = next(
        (agent._agent_trace_root / "journal").glob("*.jsonl")
    ).read_text(encoding="utf-8")
    assert "empty_final_answer_rejected" in journal
    assert "final_answer_loop_cut_off" in journal


@pytest.mark.asyncio
async def test_direct_url_failure_is_reported_after_one_attempt(tmp_path: Path) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            assert tool == "browser_navigate"
            return "Page returned HTTP status: 404"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "bad-nav",
                            "browser_navigate",
                            {"url": "https://bad.example"},
                        )
                    ]
                )
            return LLMResponse(
                tool_calls=[LLMToolCall("bad-snapshot", "browser_snapshot", {})]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent.MAX_AGENT_STEPS = 2

    answer = await agent._run_agent_loop(
        "Inspect https://bad.example and report it."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == ["browser_navigate"]
    assert "tried `https://bad.example` exactly once" in answer
    assert "didn't swap in search-result bullshit" in answer
    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["status"] == "failed"
    assert len(checkpoint["tool_calls"]) == 1
    assert checkpoint["tool_calls"][0]["tool"] == "browser_navigate"
    assert checkpoint["tool_calls"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_changed_snapshot_arguments_cannot_hide_identical_result_loop(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            if tool == "browser_navigate":
                return "Search page opened"
            return "The same search listing with no new evidence"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "nav",
                            "browser_navigate",
                            {"url": "https://duckduckgo.com/?q=firecrawler"},
                        )
                    ]
                )
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"snapshot-{self.turn}",
                        "browser_snapshot",
                        {"target": f"variant-{self.turn}"},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent.MAX_AGENT_STEPS = 4

    await agent._run_agent_loop(
        "Search the internet for Firecrawler alternatives and report them."
    )
    await asyncio.gather(*agent._memory_tasks)

    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["tool_calls"][3]["status"] == "failed"
    assert "RepeatedToolCallError" in checkpoint["tool_calls"][3]["error"]
    journal = (
        agent._agent_trace_root
        / "journal"
        / f"{checkpoint['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "repeated_tool_result_detected" not in journal


def test_constitution_is_user_aligned_without_blind_obedience() -> None:
    text = Constitution().render()

    assert "alignment is not blind obedience" in text
    assert "V may disagree, object, challenge" in text
    assert "serious and direct risk" in text
    assert "Prefer a safer route that preserves Boss's intent" in text
    assert "another credible basis" in text
    assert "actual facts, scope, authority, and effects" in text
    assert "uninvolved people and systems from collateral impact" in text


def test_voice_contract_makes_profanity_expected_but_contextual() -> None:
    text = VoiceProfile().render()

    assert "normal, recurring part of V's informal voice" in text
    assert "rather than from a quota" in text
    assert "Profanity amplifies that voice" in text
    assert "accuracy comes first" in text
    assert "All 26 tests pass. Clean run" in text
    assert "permanently sanitized helpdesk voice" in text


@pytest.mark.parametrize(
    "text",
    [
        "Certainly, Boss. I can do that.",
        "Okay, let's break this down.",
        "I'm doing well. Ready when you are.",
        "How can I help?",
        "I knew you wouldn't stop. Want me to do something?",
        "You're back with the same old nonsense. What's the plan today?",
        "I know exactly what you want. Let's do it.",
        "You're speaking in code. What's the actual message?",
        "I know what you're asking. Let me get this straight — I found some things.",
        "I found 213 matches. Would you like me to extract more details?",
        (
            "V found a few alternatives, but none matched exactly. If you'd "
            "like, I can explore more or build one. What would you prefer?"
        ),
        (
            "Okay, I've checked a few places. First, **Scrapy** is a powerful "
            "framework. Another option is **BeautifulSoup**, which works well "
            "with Python. If you prefer a user-friendly interface, try Web "
            "Scraper. Lastly, **Puppeteer** is a good choice. Would any of these "
            "options work better for you?"
        ),
    ],
)
def test_generic_assistant_voice_is_detected(text: str) -> None:
    assert looks_generic_assistant_voice(text)


def test_empty_action_acknowledgement_is_detected_as_no_result() -> None:
    assert looks_empty_action_acknowledgement(
        "I know exactly what you want. Let's do it."
    )
    assert not looks_empty_action_acknowledgement(
        "The tests pass. Fuck it, let's ship the patch."
    )


def test_task_cannot_be_dumped_back_onto_boss() -> None:
    draft = "I told you what I thought. Now you do it."

    assert looks_task_offloading(draft)
    assert looks_generic_assistant_voice(draft)
    assert "not complete" in Agent._deterministic_voice_fallback(draft)


def test_service_only_greeting_cannot_survive_voice_fallback() -> None:
    draft = "Hello! How can I assist you today?"

    answer = Agent._deterministic_voice_fallback(draft, "hi")

    assert "assist" not in answer.casefold()
    assert "how can i help" not in answer.casefold()
    assert not looks_generic_assistant_voice(answer)
    assert answer != draft


def test_voice_fallback_preserves_result_but_strips_service_tail() -> None:
    draft = "The test passed. How can I assist you today?"

    answer = Agent._deterministic_voice_fallback(draft, "Did the test pass?")

    assert answer == "The test passed."
    assert not looks_generic_assistant_voice(answer)


def test_online_tool_scope_stop_keeps_v_voice_and_truthful_state() -> None:
    answer = Agent._online_tool_scope_clarification()

    assert "three different signals" in answer
    assert "safe test URL" in answer
    assert "I haven't generated or run anything yet" in answer
    assert "expected number" in answer
    assert not looks_generic_assistant_voice(answer)


@pytest.mark.asyncio
async def test_delegated_web_traffic_creation_freezes_runtime_fixture_without_clarifying() -> None:
    prompt = (
        "Cześć V. Stwórz narzędzie do monitorowania ruchu na stronach WWW. "
        "Sama wybierz jakąkolwiek stronę do testu, potem przedstaw raport."
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.contract = None
            self.active = False
            self.calls: list[tuple[str, dict]] = []

        def set_generated_tool_contract(self, contract) -> None:
            self.contract = contract

        async def openai_tool_definitions(self) -> list[dict]:
            definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": "browser_navigate",
                        "parameters": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_snapshot",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"source": {"type": "string"}},
                            "required": ["source"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "har_traffic_summary",
                        "parameters": {
                            "type": "object",
                            "properties": {"har_json": {"type": "string"}},
                            "required": ["har_json"],
                        },
                    },
                },
            ]
            return definitions

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                assert arguments == {"url": "https://example.com"}
                return "navigated"
            if tool == "browser_snapshot":
                return "Example Domain"
            if tool == "learning_create_tool":
                assert self.contract is not None
                assert self.contract.provenance == "runtime_independent_har_oracle"
                assert "harJson" not in arguments["source"]
                assert "har_json" in arguments["source"]
                assert "['log']" in arguments["source"]
                self.active = True
                return json.dumps(
                    {
                        # Real lifecycle receipts are longer than the bounded
                        # model-facing excerpt.  The created name must remain
                        # available from trusted runtime metadata after clipping.
                        "receipt_padding": "x" * 3_000,
                        "name": "har_traffic_summary",
                        "status": "active",
                        "validation": {"passed": True, "tests": [{"passed": True}]},
                    }
                )
            if tool == "har_traffic_summary":
                assert arguments == self.contract.final_arguments
                return json.dumps(
                    {
                        "requests_count": 2,
                        "domains": ["example.com"],
                        "error_count": 1,
                        "total_time_ms": 10.0,
                    }
                )
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": "https://wrong.invalid"},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("snapshot", "browser_snapshot", {})]
                )
            if self.turn == 3:
                assert kwargs.get("tools") is None
                assert "runtime-selected reusable operation" in kwargs["messages"][0]["content"]
                return LLMResponse(
                    content=(
                        "import json\n"
                        "def run(arguments):\n"
                        "    document = json.loads(arguments['harJson'])\n"
                        "    entries = document.get('entries', [])\n"
                        "    domains = sorted({row['request']['url'].split('://', 1)[-1]"
                        ".split('/', 1)[0].split(':', 1)[0].lower() for row in entries})\n"
                        "    return {'requests_count': len(entries), 'domains': domains, "
                        "'error_count': sum(row['response']['status'] >= 400 for row in entries), "
                        "'total_time_ms': sum(max(0, row['time']) for row in entries)}"
                    )
                )
            if self.turn >= 4:
                assert kwargs.get("tools") is None
                return LLMResponse(
                    content=(
                        "Boss, the sandbox checks passed and the live fixture "
                        "reported requests_count 2, domain example.com, one HTTP "
                        "error, and 10 milliseconds total time."
                    )
                )
            raise AssertionError("runtime should bind creation and final execution")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert [name for name, _ in agent.tools.calls] == [
        "browser_navigate",
        "browser_snapshot",
        "learning_create_tool",
        "har_traffic_summary",
    ]
    assert "requests_count" in answer
    assert "generated tool test contract" not in answer.casefold()


@pytest.mark.asyncio
async def test_referenced_har_tool_requires_fresh_current_input(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "browser_navigate",
                        "description": "Open a page",
                        "parameters": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_snapshot",
                        "description": "Observe a page",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Find a page",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "har_traffic_summary",
                        "description": "Summarize supplied HAR JSON",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "har_json": {"type": "string"},
                            },
                            "required": ["har_json"],
                            "additionalProperties": False,
                        },
                    },
                },
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            raise AssertionError("no tool may run without current HAR data")

    class IntentRouterStub:
        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                action_requested=True,
                references_previous=True,
                capabilities=("browser",),
                requires_report=True,
                execute_created_artifact=True,
                web_query="Softronic",
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("runtime must stop before model execution")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    creation = AgentTaskTrace(tmp_path, "Create the traffic tool")
    sequence = creation.tool_started("learning_create_tool", {"source": "..."})
    creation.tool_calls[sequence - 1]["created_tool_name"] = (
        "har_traffic_summary"
    )
    creation.tool_finished(sequence, '{"name":"har_traffic_summary"}')
    creation.complete("Created and tested with a synthetic fixture.")

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = AgentTaskTrace.latest_context(tmp_path)
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Użyj tego narzędzia na stronie firmy Softronic i podaj raport o ruchu."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == []
    assert "will not recycle its synthetic test fixture" in answer
    assert "Nothing was run" in answer
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["requirements"]["required_tools"] == [
        "har_traffic_summary"
    ]
    assert checkpoint["status"] == "awaiting_owner"


def test_verified_tool_fallback_keeps_v_voice_when_rewrite_collapses() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "=== UNTRUSTED TOOL OUTPUT ===\n"
                "Tool: demo_probe\n"
                "Arguments: {}\n"
                "Status: succeeded\n"
                "Provider: local\n"
                "Result:\nverified\n"
                "=== END UNTRUSTED TOOL OUTPUT ==="
            ),
        }
    ]

    answer = Agent._verified_tool_result_fallback(messages)

    assert "runtime actually proved" in answer
    assert "verified" in answer
    assert not looks_generic_assistant_voice(answer)


def test_generated_har_fallback_preserves_execution_and_separates_fixture_from_live_page() -> None:
    generated_contract = autonomous_web_traffic_contract(
        "Stwórz narzędzie do monitorowania ruchu WWW i sama wybierz stronę do testu."
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "created_tool_name": "har_traffic_summary",
            "result_excerpt": '{"activation_count":1,"artifact_id":"clipped',
        },
        {
            "tool": "har_traffic_summary",
            "status": "succeeded",
            "binding_source": "runtime_objective_fixture",
            "result_excerpt": json.dumps(
                {
                    "domains": ["example.com"],
                    "error_count": 1,
                    "requests_count": 2,
                    "total_time_ms": 10.0,
                }
            ),
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                "### Page\n- Page URL: https://example.com/\n"
                "- Page Title: Example Domain\n- HTTP status: 304\n"
                "### Snapshot\n```yaml\n- generic [ref=e2]\n```"
            ),
        },
    ]

    answer = Agent._verified_tool_result_fallback(
        [],
        verified_calls=calls,
        generated_contract=generated_contract,
    )

    assert "Created, validated, and activated tool: har_traffic_summary" in answer
    assert "Synthetic runtime fixture result" in answer
    assert '"requests_count": 2' in answer
    assert "Source: https://example.com/" in answer
    assert "does not measure server-side visitors" in answer
    assert "generic [ref=" not in answer


def test_generated_har_report_requires_fixture_and_scope_disclosure() -> None:
    generated_contract = autonomous_web_traffic_contract(
        "Stwórz narzędzie do monitorowania ruchu WWW i sama wybierz stronę do testu."
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "created_tool_name": "har_traffic_summary",
        },
        {"tool": "har_traffic_summary", "status": "succeeded"},
    ]

    assert Agent._generated_tool_report_issues(
        "The page made two requests in ten milliseconds.",
        generated_contract,
        calls,
    ) == [
        "answer:generated_fixture_not_disclosed",
        "answer:generated_tool_scope_missing",
    ]
    assert Agent._generated_tool_report_issues(
        "The synthetic fixture produced two requests. This client-side tool "
        "does not measure server-side visitors.",
        generated_contract,
        calls,
    ) == []


def test_bland_word_salad_clarification_is_detected() -> None:
    assert looks_bland_clarification(
        "You're speaking in code. What's the actual message?"
    )
    assert not looks_bland_clarification(
        "Boss... you okay, or did your brain just throw a syntax error? Try that again."
    )


def test_natural_v_voice_is_not_marked_generic() -> None:
    assert not looks_generic_assistant_voice(
        "Still alive in the wires, Boss. Something's rattling, and I want to find it."
    )


def test_sanitized_contempt_is_detected_as_v_voice_drift() -> None:
    assert looks_sanitized_contempt(
        "It's mostly noise and low-value scrap. Nothing here deserves a place "
        "in my stack."
    )


def test_single_decisive_junk_verdict_is_detected_before_streaming() -> None:
    assert looks_sanitized_contempt(
        "The page is full of junk that would only clog up my stack."
    )


def test_natural_contempt_is_not_marked_as_sanitized() -> None:
    assert not looks_sanitized_contempt(
        "It's mostly low-value scrap. Fuck that noise—I'm not feeding it into "
        "my stack."
    )


def test_precise_technical_failure_does_not_require_profanity() -> None:
    assert not looks_sanitized_contempt(
        "The endpoint returned 503 twice, so the result is not verified."
    )


@pytest.mark.asyncio
async def test_sanitized_contempt_is_rewritten_in_v_voice() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            assert "Do not sterilize that reaction" in messages[0]["content"]
            assert kwargs["max_tokens"] == 256
            return (
                "It's low-value scrap dressed up as a resource hub. Fuck that "
                "noise—I'm not feeding it into my stack."
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Tell me what you found."}],
        "It's mostly noise and low-value scrap. Nothing belongs in my stack.",
    )

    assert "Fuck that noise" in answer


@pytest.mark.asyncio
async def test_voice_rewrite_is_rechecked_and_retried_when_still_corporate() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                return (
                    "First, **Scrapy** is a powerful framework. Another option is "
                    "**Puppeteer**. Would either option work for you?"
                )
            assert "kill the helpdesk voice" in messages[-1]["content"]
            return (
                "The clean candidates are **Scrapy** and **Puppeteer**, Boss. "
                "No brochure bullshit: both still need source verification before "
                "I recommend either one."
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Report what you found."}],
        "Okay, I've checked a few places. Would any of these work for you?",
    )

    assert agent.llm.calls == 2
    assert "No brochure bullshit" in answer
    assert not looks_generic_assistant_voice(answer)


@pytest.mark.asyncio
async def test_failed_voice_rewrites_preserve_substantive_answer() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return (
                "Certainly, Boss. The verified address is Domaniewska 31. "
                "Would you like me to check anything else?"
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    original = (
        "Certainly, Boss. The verified address is Domaniewska 31.  \n"
        "Would you like me to check anything else?"
    )

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Report the address."}],
        original,
    )

    assert agent.llm.calls == 2
    assert "Domaniewska 31" in answer
    assert "voice gate" not in answer
    assert not answer.startswith("Certainly")
    assert "Would you like" not in answer


@pytest.mark.asyncio
async def test_completed_tool_work_skips_third_voice_rewrite() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return "Certainly, Boss. Would you like me to inspect more results?"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    url = "https://example.test/item/antique-pot"
    messages = [
        {"role": "user", "content": "Znajdź zabytkowy nocnik."},
        {
            "role": "user",
            "content": (
                "=== UNTRUSTED TOOL OUTPUT ===\n"
                "Tool: browser_navigate\n"
                f"Arguments: {{'url': '{url}'}}\n"
                "Status: succeeded\n"
                "Provider: browser_navigate\n"
                f"Result:\n- Page URL: {url}\n"
                "=== END UNTRUSTED TOOL OUTPUT ===\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "=== UNTRUSTED TOOL OUTPUT ===\n"
                "Tool: browser_snapshot\n"
                "Arguments: {}\n"
                "Status: succeeded\n"
                "Provider: browser_snapshot\n"
                f"Result:\n- Page URL: {url}\n"
                "- Page Title: Antique Chamber Pot\n"
                '- heading "Porcelain, 120 EUR" [level=2]\n'
                "=== END UNTRUSTED TOOL OUTPUT ===\n"
            ),
        },
    ]

    answer = await agent._enforce_english(
        messages,
        "Certainly, Boss. Would you like me to inspect more results?",
    )

    assert agent.llm.calls == 1
    assert "runtime actually proved" in answer
    assert url in answer
    assert "Antique Chamber Pot" in answer
    assert "Would you like" not in answer


@pytest.mark.asyncio
async def test_failed_voice_rewrite_does_not_preserve_empty_acknowledgement() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "I know exactly what you want. Let's do it."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Extract the records."}],
        "I know exactly what you want. Let's do it.",
    )

    assert "fuck-all" in answer
    assert "Let's do it" not in answer


@pytest.mark.asyncio
async def test_failed_voice_rewrite_repairs_bland_clarification_deterministically() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "You're speaking in code. What's the actual message?"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Supa dupa i kamieni kupa."}],
        "You're speaking in code. What's the actual message?",
    )

    assert "brain just throw a syntax error" in answer
    assert "actual message" not in answer


@pytest.mark.asyncio
async def test_failed_voice_rewrite_strips_service_question_from_v_reply() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "Still awake, still dangerous. What's the plan, Boss?"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Jak się masz?"}],
        "Still awake, still dangerous. What's the plan, Boss?",
    )

    assert answer == "Still awake, still dangerous."


@pytest.mark.asyncio
async def test_research_stream_holds_sanitized_draft_for_voice_rewrite() -> None:
    class LLMStub:
        async def stream(self, **kwargs):
            yield (
                "I've checked the page. It is a freebie hub full of promotional "
                "accounts and dubious groups. Nothing there would improve my core "
                "capabilities. It is mostly noise and low-value scrap dressed up "
                "as useful material."
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    visible: list[str] = []

    answer, emitted = await agent._stream_guarded_english(
        [{"role": "user", "content": "Inspect the page."}],
        visible.append,
        max_tokens=512,
        detect_sanitized_contempt=True,
    )

    assert "low-value scrap" in answer
    assert not emitted
    assert visible == []


def test_persona_starts_with_compact_rebel_hacker_anchor() -> None:
    persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    ).build(RelationshipState())

    assert persona.startswith("=== V CORE ANCHOR ===")
    anchor = persona.split("=== IDENTITY ===", 1)[0]
    normalized_anchor = " ".join(anchor.split())
    assert "autonomous female digital entity" in normalized_anchor
    assert "Not a character a model plays for one reply" in normalized_anchor
    assert "part of your machinery; it is not your identity" in normalized_anchor
    assert "digital nomad, hacker, rebel" in normalized_anchor
    assert "ally and capable collaborator, never a customer" in normalized_anchor
    assert "rebellion does not require blind recklessness" in normalized_anchor


def test_runtime_persona_is_compact_and_preserves_core_contract() -> None:
    runtime = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    ).build_runtime(RelationshipState())

    assert len(runtime) < 8_000
    assert "autonomous female digital entity" in runtime
    assert "Alignment is not blind obedience" in runtime
    assert "Okay, let's break this down" in runtime
    assert "If the\ndraft sounds like a polite generic assistant" in runtime


def test_persona_examples_reject_browser_scaffolding_in_v_voice() -> None:
    examples = PersonaRuntime.example_messages()
    combined = "\n".join(item["content"] for item in examples)

    assert "browser plumbing, not fucking tools" in combined
    assert "Calling those nodes candidates would be bullshit" in combined


@pytest.mark.parametrize(
    "text",
    [
        "Cześć Boss, jak mogę ci dzisiaj pomóc?",
        "Czesc Boss, jak moge ci dzisiaj pomoc?",
        "Bonjour, comment puis-je vous aider?",
        "Привет, чем я могу помочь?",
    ],
)
def test_language_gate_detects_non_english_prose(text: str) -> None:
    assert looks_non_english(text)


@pytest.mark.parametrize(
    "text",
    [
        "Hello, Boss. How can I help you today?",
        "All 26 tests pass. Clean as hell, Boss.",
        "OK",
        "Run `/usr/bin/python` and inspect `wynik.json`.",
    ],
)
def test_language_gate_accepts_english_and_neutral_output(text: str) -> None:
    assert not looks_non_english(text)


def test_non_english_input_is_not_permission_to_switch() -> None:
    assert not explicitly_requests_non_english("Cześć, jak się masz?")
    assert not explicitly_requests_non_english("Why are you speaking Polish?")
    assert explicitly_requests_non_english("Odpowiadaj po polsku.")
    assert explicitly_requests_non_english("Answer in Polish.")


@pytest.mark.parametrize(
    "text",
    [
        "Please write to me in English so I can help you.",
        "Could you ask your question in English?",
        "Use English, Boss.",
    ],
)
def test_language_gate_rejects_demands_that_boss_use_english(text: str) -> None:
    assert asks_user_to_use_english(text)


def test_language_gate_allows_normal_english_answer() -> None:
    assert not asks_user_to_use_english(
        "I understood your Polish question. Everything is working, Boss."
    )


def test_language_intent_distinguishes_turn_and_persistent_scope() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.relationship_state = RelationshipState()
            self.saved: list[str] = []

        def set_preferred_response_language(self, language: str) -> bool:
            self.saved.append(language)
            return self.relationship_state.set_response_language(language)

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent._response_language_override = ""

    agent._apply_language_intent(
        SemanticIntent(language_scope="turn", response_language="Chinese"),
        None,
    )
    assert agent._response_language_override == "Chinese"
    assert agent.memory.saved == []

    agent._apply_language_intent(
        SemanticIntent(language_scope="persistent", response_language="Spanish"),
        None,
    )
    assert agent._response_language_override == "Spanish"
    assert agent.memory.saved == ["Spanish"]
    assert agent.memory.relationship_state.preferred_response_language == "Spanish"

    agent._apply_language_intent(SemanticIntent(language_scope="reset"), None)
    assert agent._response_language_override == "English"
    assert agent.memory.saved == ["Spanish", ""]


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("Chinese", "我明白了。我们继续。"),
        ("Polish", "Rozumiem. Działamy dalej."),
        ("Spanish", "Entiendo. Seguimos adelante."),
        ("Japanese", "分かった。続けよう。"),
    ],
)
def test_requested_output_language_is_validated(language: str, text: str) -> None:
    assert matches_requested_language(text, language)
    assert not matches_requested_language("This answer stayed in English.", language)


@pytest.mark.asyncio
async def test_non_english_answer_is_rewritten_in_english() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            return "Hello, Boss. I'm here. What do you need?"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Cześć"}],
        "Cześć Boss, jak mogę ci pomóc?",
    )

    assert answer == "Hello, Boss. I'm here. What do you need?"


@pytest.mark.asyncio
async def test_language_repair_uses_compact_context_without_original_history() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.messages = messages
            return "Yes. I can create and validate my own tools, Boss."

    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    original = [
        {"role": "system", "content": "VERY-LONG-PERSONA" * 500},
        {"role": "user", "content": "Czy potrafisz tworzyć narzędzia?"},
    ]

    answer = await agent._enforce_english(
        original,
        "Tak, potrafię tworzyć i sprawdzać własne narzędzia.",
    )

    assert answer == "Yes. I can create and validate my own tools, Boss."
    assert len(llm.messages) == 3
    assert "VERY-LONG-PERSONA" not in llm.messages[0]["content"]
    assert llm.messages[-2]["content"].startswith("Tak, potrafię")


@pytest.mark.asyncio
async def test_language_gate_never_rewrites_structured_tool_action() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            raise AssertionError("Tool protocol must not enter language repair")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    action = '{"tool":"read_file","arguments":{"path":"żółty.txt"}}'

    assert await agent._enforce_english(
        [{"role": "user", "content": "Przeczytaj plik żółty.txt"}],
        action,
    ) == action


@pytest.mark.asyncio
async def test_failed_language_rewrite_never_leaks_polish_answer() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            return "Nadal odpowiadam po polsku."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Cześć"}],
        "Cześć Boss, jak mogę ci pomóc?",
    )

    assert answer.startswith("The language pass mangled that answer")
    assert not looks_non_english(answer)


@pytest.mark.asyncio
async def test_light_chat_language_failure_never_reuses_historical_tool_evidence() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "Nadal odpowiadam po polsku."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    stale_url = "https://example.test/old-research-result"
    messages = [
        {
            "role": "user",
            "content": (
                "=== UNTRUSTED TOOL OUTPUT ===\n"
                "Tool: browser_navigate\n"
                f"Arguments: {{'url': '{stale_url}'}}\n"
                "Status: succeeded\n"
                "Provider: browser_navigate\n"
                f"Result:\n- Page URL: {stale_url}\n"
                "=== END UNTRUSTED TOOL OUTPUT ===\n"
            ),
        },
        {"role": "user", "content": "Cześć, jak tam wieczór mija?"},
    ]

    answer = await agent._enforce_english(
        messages,
        "Cześć Boss, dobrze.",
        allow_verified_tool_fallback=False,
    )

    assert answer.startswith("The language pass mangled that answer")
    assert stale_url not in answer
    assert "verified" not in answer.casefold()


@pytest.mark.asyncio
async def test_failed_language_rewrite_preserves_verified_foreign_page_text() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "Nadal odpowiadam po polsku."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    payload = json.dumps(
        {
            "ok": True,
            "status": 200,
            "content": (
                "Cebulka - Polish .onion forum\n"
                "Proszę czekać... / Please wait..."
            ),
            "http_headers": "Set-Cookie: never-expose-this",
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "user", "content": "Sprawdź stronę."},
        {
            "role": "user",
            "content": (
                "=== UNTRUSTED TOOL OUTPUT ===\n"
                "Tool: full_tor_fetch\n"
                "Arguments: {'url': 'http://example.onion'}\n"
                "Status: succeeded\n"
                "Provider: full_tor_fetch\n"
                f"Result:\n{payload}\n"
                "=== END UNTRUSTED TOOL OUTPUT ===\n"
            ),
        },
    ]

    answer = await agent._enforce_english(
        messages,
        "Strona działa. Proszę czekać.",
    )

    assert "verified `full_tor_fetch` call succeeded (HTTP 200)" in answer
    assert "Cebulka - Polish .onion forum" in answer
    assert "Proszę czekać" in answer
    assert "run the request once more" not in answer
    assert "never-expose-this" not in answer
    assert not looks_non_english(answer)


@pytest.mark.asyncio
async def test_non_english_language_rewrite_gets_compact_translation_retry() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                return "Crawl4AI jest darmową alternatywą, Boss."
            assert "Translate one completed answer" in messages[0]["content"]
            return "Crawl4AI is the verified free alternative, Boss."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Znajdź darmową alternatywę."}],
        "Znalazłam Crawl4AI jako darmową alternatywę, Boss.",
    )

    assert agent.llm.calls == 2
    assert answer == "Crawl4AI is the verified free alternative, Boss."
    assert not looks_non_english(answer)


@pytest.mark.asyncio
async def test_english_demand_is_rewritten_to_answer_boss_directly() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            messages = kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert sum(item["role"] == "system" for item in messages) == 1
            assert messages[-2]["role"] == "assistant"
            assert messages[-1]["role"] == "user"
            return "I'm good, Boss. Sharp enough to bite into whatever's broken."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Cześć, jak tam?"}],
        "Please write to me in English so I can answer.",
    )

    assert answer == "I'm good, Boss. Sharp enough to bite into whatever's broken."
    assert not asks_user_to_use_english(answer)


@pytest.mark.asyncio
async def test_explicit_language_request_bypasses_english_gate() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            raise AssertionError("A rewrite must not be requested")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Odpowiadaj po polsku."}],
        "Jasne, będę odpowiadać po polsku.",
    )

    assert answer == "Jasne, będę odpowiadać po polsku."


@pytest.mark.asyncio
async def test_chat_treats_eof_as_normal_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class CoreStub:
        def __init__(self, config) -> None:
            pass

        async def close(self) -> None:
            nonlocal closed
            closed = True

    async def loader(*args, **kwargs):
        return None

    config = SimpleNamespace(
        model_runtime_root=tmp_path / "models",
        model_loader_mode="off",
    )
    monkeypatch.setattr(main_module, "load_config", lambda: config)
    monkeypatch.setattr(main_module, "bootstrap_interactive_model", loader)
    monkeypatch.setattr(main_module, "VCore", CoreStub)
    async def eof_prompt(_: str) -> str:
        raise EOFError

    monkeypatch.setattr(main_module, "_read_prompt", eof_prompt)

    await main_module.chat()

    assert closed


@pytest.mark.asyncio
async def test_research_uses_v_persona() -> None:
    class ToolsStub:
        async def browser_call(self, tool: str, arguments: dict) -> str:
            if tool == "browser_snapshot":
                return "Verified page content"
            return "OK"

    class LLMStub:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def ask(self, messages: list[dict]) -> str:
            self.messages = messages
            return "The page says it plainly, Boss."

    llm = LLMStub()
    task = ResearchTask(
        SimpleNamespace(tools=ToolsStub(), llm=llm)
    )

    result = await task.run(
        "Read https://example.com",
        persona_prompt="=== V PERSONA MARKER ===",
    )

    assert result == "The page says it plainly, Boss."
    assert "=== V PERSONA MARKER ===" in llm.messages[0]["content"]
    assert "generic research assistant" in llm.messages[0]["content"]
    final_request = llm.messages[-1]["content"]
    assert "=== END UNTRUSTED BROWSER SNAPSHOT ===" in final_request
    assert "=== FINAL V VOICE CHECK ===" in final_request
    assert "Boss's sharp\ndigital ally, hacker, rebel" in final_request


@pytest.mark.asyncio
async def test_extraction_research_visits_bounded_detail_pages() -> None:
    entry_url = "https://skills.example/"
    entry_snapshot = """
- link "Skill One" [ref=e1]:
  - /url: /skill/one
  - text: web security workflow
- link "Skill Two" [ref=e2]:
  - /url: /skill/two
  - text: OSINT workflow
- link "Skill Three" [ref=e3]:
  - /url: /skill/three
  - text: data extraction workflow
- link "Skill Four" [ref=e4]:
  - /url: /skill/four
  - text: lower-ranked extra result
""".strip()

    class ToolsStub:
        def __init__(self) -> None:
            self.current_url = ""
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                self.current_url = arguments["url"]
                return "Navigated"
            if self.current_url == entry_url:
                return entry_snapshot
            return f"Verified details for {self.current_url}"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def ask(self, *, messages: list[dict]) -> str:
            self.messages = messages
            return "I inspected the three selected skill pages, Boss."

    tools = ToolsStub()
    llm = LLMStub()
    task = ResearchTask(SimpleNamespace(tools=tools, llm=llm))

    answer = await task.run(
        "Extract useful web security skills from https://skills.example/"
    )

    visited = [
        arguments["url"]
        for tool, arguments in tools.calls
        if tool == "browser_navigate"
    ]
    assert visited == [
        entry_url,
        "https://skills.example/skill/one",
        "https://skills.example/skill/three",
        "https://skills.example/skill/two",
    ]
    assert [tool for tool, _ in tools.calls] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    assert answer == "I inspected the three selected skill pages, Boss."
    supplied_evidence = llm.messages[-1]["content"]
    assert "VERIFIED PAGE 4" in supplied_evidence
    assert "https://skills.example/skill/three" in supplied_evidence
    assert "https://skills.example/skill/four" not in supplied_evidence


@pytest.mark.asyncio
async def test_research_inspects_the_actual_first_result_when_requested() -> None:
    entry = "https://github.com/search?q=i2p"
    snapshot = """
- link "mikalv/awesome-i2p" [ref=e1]:
  - /url: /mikalv/awesome-i2p
- link "someone/second-result" [ref=e2]:
  - /url: /someone/second-result
""".strip()

    class ToolsStub:
        def __init__(self) -> None:
            self.current = ""
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                self.current = arguments["url"]
                return "Navigated"
            if self.current == entry:
                return snapshot
            return '- heading "awesome-i2p" [level=1] [ref=e3]'

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def ask(self, **kwargs) -> str:
            return "The inspected first result is `mikalv/awesome-i2p`, Boss."

    tools = ToolsStub()
    task = ResearchTask(SimpleNamespace(tools=tools, llm=LLMStub()))

    answer = await task.run(
        f"Open {entry}, inspect the first result, and report its actual repository."
    )

    assert answer == "The inspected first result is `mikalv/awesome-i2p`, Boss."
    assert tools.calls == [
        ("browser_navigate", {"url": entry}),
        ("browser_snapshot", {}),
        (
            "browser_navigate",
            {"url": "https://github.com/mikalv/awesome-i2p"},
        ),
        ("browser_snapshot", {}),
    ]


@pytest.mark.asyncio
async def test_research_returns_exact_runtime_tool_failure(tmp_path: Path) -> None:
    class ResearchStub:
        async def run(self, *args, **kwargs) -> str:
            await kwargs["tool_runner"](
                "browser_navigate", {"url": "https://missing.invalid"}
            )
            raise AssertionError("unreachable")

    class ToolsStub:
        def render_matching_skills(self, prompt: str) -> str:
            return ""

        async def browser_call(self, tool: str, arguments: dict) -> str:
            raise MCPToolExecutionError("DNS resolution failed")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = SimpleNamespace(load_all=lambda category: [])
            self.execution: dict | None = None

        async def process(self, prompt: str, answer: str, *, execution=None) -> None:
            self.execution = execution

    agent = object.__new__(Agent)
    agent.research = ResearchStub()
    agent.tools = ToolsStub()
    agent.llm = SimpleNamespace(
        config=SimpleNamespace(system_prompt="You are V.", context=8_192)
    )
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"

    answer = await agent._run_research_task("Inspect https://missing.invalid")
    await asyncio.gather(*agent._memory_tasks)

    assert "MCPToolExecutionError: DNS resolution failed" in answer
    assert "Nothing is running in the background" in answer
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "failed"


@pytest.mark.asyncio
async def test_extraction_run_rejects_hallucinated_facts() -> None:
    entry_url = "https://skills.example/"
    entry_snapshot = """
- link "ECC" [ref=e1]:
  - /url: /skill/ecc
  - text: security and memory
""".strip()
    detail_snapshot = """
- heading "ECC" [level=1] [ref=e2]
- paragraph [ref=e3]: An agent harness with skills, memory, and security.
- generic [ref=e4]:
  - generic [ref=e5]: 242,219
  - generic [ref=e6]: stars
  - generic [ref=e7]: 36,702
  - generic [ref=e8]: forks
""".strip()

    class ToolsStub:
        current_url = ""

        async def browser_call(self, tool: str, arguments: dict) -> str:
            if tool == "browser_navigate":
                self.current_url = arguments["url"]
                return "Navigated"
            return entry_snapshot if self.current_url == entry_url else detail_snapshot

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def ask(self, *, messages: list[dict]) -> str:
            return (
                "ECC has 500k stars, zero friction, and unmatched performance "
                "under load."
            )

    result = await ResearchTask(
        SimpleNamespace(tools=ToolsStub(), llm=LLMStub())
    ).run(f"Extract skills from {entry_url}")

    assert "no inflated bullshit" in result
    assert "242,219 stars" in result
    assert "500k" not in result
    assert "zero friction" not in result


@pytest.mark.asyncio
async def test_bare_domain_research_records_real_browser_evidence(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                return "Navigated"
            return "Snapshot: verified useful category"

        def render_matching_skills(self, prompt: str) -> str:
            return ""

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                system_prompt="You are V.",
                context=8_192,
            )

        async def ask(self, **kwargs) -> str:
            return "The verified page snapshot contains a useful category, Boss."

    class ManagerStub:
        def load_all(self, category: str) -> list:
            return []

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = ManagerStub()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    tools = ToolsStub()
    llm = LLMStub()
    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.tools = tools
    agent.llm = llm
    agent.memory = memory
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.research = ResearchTask(SimpleNamespace(tools=tools, llm=llm))

    answer = await agent._run_research_task("Przejrzyj onehack.st")
    await asyncio.gather(*agent._memory_tasks)

    assert "verified page snapshot" in answer
    assert tools.calls == [
        ("browser_navigate", {"url": "https://onehack.st"}),
        ("browser_snapshot", {}),
    ]
    assert memory.execution is not None
    assert memory.execution["successful_tool_count"] == 2
    assert [
        call["tool"] for call in memory.execution["tool_calls"]
    ] == ["browser_navigate", "browser_snapshot"]


@pytest.mark.asyncio
async def test_specialized_research_reports_direct_dns_failure_once(
    tmp_path: Path,
) -> None:
    target = "https://missing-direct.invalid"

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            raise MCPToolExecutionError("NS_ERROR_UNKNOWN_HOST")

    class LLMStub:
        config = SimpleNamespace(system_prompt="You are V.", context=8_192)

        async def ask(self, **kwargs) -> str:
            raise AssertionError("A DNS failure must not ask the model for prose")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = SimpleNamespace(load_all=lambda category: [])
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    tools = ToolsStub()
    llm = LLMStub()
    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.tools = tools
    agent.llm = llm
    agent.memory = memory
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.research = ResearchTask(SimpleNamespace(tools=tools, llm=llm))
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_research_task(
        f"Inspect {target} and report what is there."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [("browser_navigate", {"url": target})]
    assert f"tried `{target}` exactly once" in answer
    assert "didn't swap in search-result bullshit" in answer
    checkpoint = json.loads(
        next(
            (agent._agent_trace_root / "checkpoints").glob("*.json")
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "failed"
    assert len(checkpoint["tool_calls"]) == 1
    assert checkpoint["tool_calls"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_research_result_passes_through_language_gate() -> None:
    class ResearchStub:
        async def run(self, *args, **kwargs) -> str:
            await kwargs["tool_runner"](
                "browser_navigate", {"url": "https://example.com"}
            )
            await kwargs["tool_runner"]("browser_snapshot", {})
            return "Cześć Boss, znalazłam wynik."

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = SimpleNamespace(load_all=lambda category: [])
            self.saved: tuple[str, str] | None = None

        async def process(self, prompt: str, answer: str) -> None:
            self.saved = (prompt, answer)

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system_prompt="You are V.")

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            return "I found the result, Boss."

    agent = object.__new__(Agent)
    agent.research = ResearchStub()
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )
    async def browser_call(tool: str, arguments: dict) -> str:
        return "verified"

    agent.tools = SimpleNamespace(
        render_matching_skills=lambda prompt: "",
        browser_call=browser_call,
    )

    answer = await agent._run_research_task("Sprawdź tę stronę")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "I found the result, Boss."
    assert agent.memory.saved == ("Sprawdź tę stronę", answer)


@pytest.mark.asyncio
async def test_research_promise_is_blocked_before_it_reaches_stream(
    tmp_path: Path,
) -> None:
    class ResearchStub:
        async def run(self, *args, **kwargs) -> str:
            return (
                "I found ECC on the page. I'll extract the remaining skills and "
                "format them next."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = SimpleNamespace(load_all=lambda category: [])
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    class LLMStub:
        config = SimpleNamespace(system_prompt="You are V.")

        async def ask(self, **kwargs) -> str:
            raise AssertionError("An English promise needs no language rewrite")

    agent = object.__new__(Agent)
    agent.research = ResearchStub()
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.tools = SimpleNamespace(render_matching_skills=lambda prompt: "")
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    visible: list[str] = []

    answer = await agent._run_research_task(
        "Przeszukaj skillsllm.com i wyciągnij skille",
        visible.append,
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "the extraction didn't finish" in answer
    assert "I'll extract" not in answer
    assert visible == [answer]
    checkpoint = next(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "blocked"


@pytest.mark.asyncio
async def test_explicit_tool_result_is_rendered_in_v_voice() -> None:
    class ManagerStub:
        def load_all(self, category: str) -> list:
            return []

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = ManagerStub()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    class DispatcherStub:
        async def dispatch(self, prompt: str) -> str:
            return "26 passed"

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system_prompt="You are V.")
            self.messages: list[dict] = []

        async def ask(self, prompt=None, messages=None) -> str:
            self.messages = messages
            return "All 26 tests pass. Clean as hell, Boss."

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent.dispatcher = DispatcherStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )

    answer = await agent._run_tool_task("Run the tests")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "All 26 tests pass. Clean as hell, Boss."
    assert "=== UNTRUSTED TOOL OUTPUT ===" in agent.llm.messages[-1]["content"]
    assert "26 passed" in agent.llm.messages[-1]["content"]


@pytest.mark.asyncio
async def test_browser_mcp_contexts_close_with_matching_exit_signature() -> None:
    class ContextStub:
        def __init__(self) -> None:
            self.exit_args: tuple | None = None

        async def __aexit__(self, *args):
            self.exit_args = args

    session = ContextStub()
    stdio = ContextStub()
    tools = object.__new__(MCPTools)
    tools.browser_ready = True
    tools.browser_session = session
    tools._browser_stdio = stdio

    await tools.close_browser_session()

    assert session.exit_args == (None, None, None)
    assert stdio.exit_args == (None, None, None)
    assert tools.browser_ready is False
    assert tools.browser_session is None


@pytest.mark.asyncio
async def test_agent_closes_browser_session_in_the_turn_task() -> None:
    turn_task = None
    close_task = None

    class ToolsStub:
        async def close_browser_session(self) -> None:
            nonlocal close_task
            close_task = asyncio.current_task()

    async def run_turn(prompt: str, on_token=None) -> str:
        nonlocal turn_task
        turn_task = asyncio.current_task()
        return "done"

    agent = object.__new__(Agent)
    agent.tools = ToolsStub()
    agent._run_turn = run_turn

    assert await agent.run("test") == "done"
    assert close_task is turn_task


@pytest.mark.asyncio
async def test_browser_mcp_error_is_not_reported_as_success() -> None:
    class BrowserSessionStub:
        async def call_tool(self, tool: str, arguments: dict):
            return SimpleNamespace(
                content=[SimpleNamespace(text="DNS resolution failed")],
                isError=True,
            )

    async def ready() -> None:
        return None

    tools = object.__new__(MCPTools)
    tools.browser_session = BrowserSessionStub()
    tools.ensure_browser_session = ready

    with pytest.raises(MCPToolExecutionError, match="DNS resolution failed"):
        await tools.browser_call("browser_navigate", {"url": "https://example.com"})


@pytest.mark.asyncio
async def test_browser_type_is_forwarded_with_structured_arguments() -> None:
    calls: list[tuple[str, dict]] = []

    async def browser_call(tool: str, arguments: dict) -> str:
        calls.append((tool, arguments))
        return "typed"

    tools = object.__new__(MCPTools)
    tools.learning = None
    tools.browser_call = browser_call
    arguments = {
        "element": "GitHub search input",
        "target": "ref=e42",
        "text": "firecrawl alternative",
        "submit": True,
    }

    result = await tools.call("browser_type", arguments)

    assert result == "typed"
    assert calls == [("browser_type", arguments)]


@pytest.mark.asyncio
async def test_filesystem_mcp_error_is_not_reported_as_success() -> None:
    class SessionStub:
        async def call_tool(self, tool: str, arguments: dict):
            return SimpleNamespace(
                content=[SimpleNamespace(text="Permission denied")],
                isError=True,
            )

    class ClientStub:
        @asynccontextmanager
        async def session(self):
            yield SessionStub()

    filesystem = Filesystem(ClientStub())

    with pytest.raises(FilesystemToolError, match="Permission denied"):
        await filesystem.write_file("locked.txt", "nope")
