import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.generated_tool_contract import (
    GeneratedToolContract,
    GeneratedToolContractError,
    GeneratedToolTest,
    autonomous_web_traffic_contract,
    has_natural_fixture_candidate,
    parse_generated_tool_contract,
)
from v_core.learning.runtime import LearningRuntime
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.llm.llm import LLMResponse
from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.sandbox import BubblewrapBackend


@pytest.mark.parametrize(
    "prompt,quotes",
    [
        (
            "Dla 2 ma dać 4, dla 7 ma dać 14. Potem użyj dla n = 9.",
            ("Dla 2 ma dać 4", "dla 7 ma dać 14", "n = 9"),
        ),
        (
            "For 2 return 4; for 7 return 14. Finally run it with n = 9.",
            ("For 2 return 4", "for 7 return 14", "n = 9"),
        ),
        (
            "Test: input 2 daje output 4, input 7 daje output 14; run n = 9.",
            ("input 2 daje output 4", "input 7 daje output 14", "n = 9"),
        ),
    ],
)
def test_contract_parser_is_language_neutral_and_owner_grounded(prompt, quotes):
    response = json.dumps(
        {
            "status": "ready",
            "tests": [
                {"arguments": {"n": 2}, "expected": {"result": 4}, "evidence_quote": quotes[0]},
                {"arguments": {"n": 7}, "expected": {"result": 14}, "evidence_quote": quotes[1]},
            ],
            "final_arguments": {"n": 9},
            "final_evidence_quote": quotes[2],
            "reason": "",
        }
    )

    contract = parse_generated_tool_contract(response, prompt)

    assert [case.arguments for case in contract.tests] == [{"n": 2}, {"n": 7}]
    assert [case.expected for case in contract.tests] == [{"result": 4}, {"result": 14}]
    assert contract.final_arguments == {"n": 9}
    assert has_natural_fixture_candidate(prompt)


def test_contract_parser_rejects_model_invented_value():
    prompt = "Dla 2 ma dać 4. Potem użyj dla n = 9."
    response = json.dumps(
        {
            "status": "ready",
            "tests": [
                {"arguments": {"n": 2}, "expected": {"result": 6}, "evidence_quote": "Dla 2 ma dać 4"}
            ],
            "final_arguments": {"n": 9},
            "final_evidence_quote": "n = 9",
            "reason": "",
        }
    )

    with pytest.raises(GeneratedToolContractError, match="absent from its evidence quote"):
        parse_generated_tool_contract(response, prompt)


def test_contract_parser_rejects_conflicting_oracles():
    prompt = "Dla 2 ma dać 4, ale dla 2 ma dać 5. Potem n = 9."
    response = json.dumps(
        {
            "status": "ready",
            "tests": [
                {"arguments": {"n": 2}, "expected": {"result": 4}, "evidence_quote": "Dla 2 ma dać 4"},
                {"arguments": {"n": 2}, "expected": {"result": 5}, "evidence_quote": "dla 2 ma dać 5"},
            ],
            "final_arguments": {"n": 9},
            "final_evidence_quote": "n = 9",
            "reason": "",
        }
    )

    with pytest.raises(GeneratedToolContractError, match="conflicting"):
        parse_generated_tool_contract(response, prompt)


def test_single_input_alias_is_canonicalized_without_guessing_between_fields():
    prompt = "Dla liczby 2 ma dać 4. Potem użyj dla n = 9."
    response = json.dumps(
        {
            "status": "ready",
            "tests": [
                {"arguments": {"liczba": 2}, "expected": {"wynik": 4}, "evidence_quote": "liczby 2 ma dać 4"}
            ],
            "final_arguments": {"n": 9},
            "final_evidence_quote": "n = 9",
            "reason": "",
        }
    )

    contract = parse_generated_tool_contract(response, prompt)

    assert contract.final_arguments == {"liczba": 9}


@pytest.mark.parametrize(
    "prompt",
    [
        (
            "V, stwórz narzędzie do monitorowania ruchu na stronach WWW. "
            "Sama wybierz dowolną stronę do testu i przedstaw raport."
        ),
        (
            "Create a tool to monitor network traffic on a website. Pick any "
            "safe site yourself, test it, and report the result."
        ),
    ],
)
def test_delegated_web_traffic_tool_gets_runtime_owned_har_oracle(prompt):
    contract = autonomous_web_traffic_contract(prompt)

    assert contract is not None
    assert contract.name_hint == "har_traffic_summary"
    assert contract.provenance == "runtime_independent_har_oracle"
    assert len(contract.tests) == 3
    assert set(contract.tests[0].arguments) == {"har_json"}
    assert contract.tests[0].expected == {
        "requests_count": 3,
        "domains": ["cdn.example.com", "example.com"],
        "error_count": 1,
        "total_time_ms": 20.0,
    }
    assert "server-side visitor analytics" in contract.specification
    assert "example.com" in contract.final_evidence_quote
    assert GeneratedToolContract.from_dict(contract.to_dict()) == contract


def test_web_traffic_default_requires_explicit_delegation_of_test_target():
    assert autonomous_web_traffic_contract(
        "Stwórz narzędzie do monitorowania ruchu na stronach WWW."
    ) is None


def test_delegated_web_traffic_accepts_spoken_polish_past_tense():
    contract = autonomous_web_traffic_contract(
        "Cześć V. Stwórz narzędzie do obserwacji i monitorowania ruchu na "
        "stronach WWW. Chciałbym, żebyś sama podjęła decyzję, na jakiej "
        "stronie je przetestować. Potem przedstaw mi raport."
    )

    assert contract is not None
    assert contract.archetype == "client_har_summary"
    assert contract.name_hint == "har_traffic_summary"


def test_semantic_web_traffic_delegation_does_not_depend_on_prompt_language():
    contract = autonomous_web_traffic_contract(
        "任意の安全なサイトを自分で選んでテストしてください。",
        semantic_archetype="client_web_traffic_monitor",
        delegates_test_target=True,
    )

    assert contract is not None
    assert contract.archetype == "client_har_summary"


def test_semantic_web_traffic_requires_grounded_target_delegation():
    assert autonomous_web_traffic_contract(
        "任意の安全なサイトを自分で選んでテストしてください。",
        semantic_archetype="client_web_traffic_monitor",
        delegates_test_target=False,
    ) is None


@pytest.mark.asyncio
async def test_source_builder_runs_every_frozen_semantic_test(tmp_path: Path):
    envelope = AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))
    learning = LearningRuntime(
        tmp_path / "learning",
        AuthorizationGuard(tmp_path, envelope),
        BubblewrapBackend(),
    )
    contract = GeneratedToolContract(
        tests=(
            GeneratedToolTest({"n": 2}, {"result": 4}, "2 gives 4"),
            GeneratedToolTest({"n": 7}, {"result": 14}, "7 gives 14"),
        ),
        final_arguments={"n": 9},
        final_evidence_quote="n = 9",
    )

    created = await learning.create_tool_from_source(
        "def run(arguments):\n    return {'result': arguments['n'] * 2}",
        objective="Create a tool named frozen_double.",
        generated_contract=contract,
    )

    assert created.status.value == "active"
    assert created.validation["test_counts"] == {"passed": 2, "failed": 0, "not_run": 0}
    assert json.loads(json.dumps(await learning.execute_tool("frozen_double", {"n": 9}))) == {"result": 18}


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_first", [False, True])
async def test_agent_freezes_natural_tests_before_source_and_binds_final_input(empty_first):
    prompt = (
        "Stwórz narzędzie o nazwie podwoj_liczbe. Przetestuj je: "
        "dla 2 ma dać 4, a dla 7 ma dać 14. Potem użyj go dla n = 9 i pokaż wynik."
    )

    class Tools:
        def __init__(self):
            self.active = False
            self.contract = None
            self.calls = []

        def set_generated_tool_contract(self, contract):
            self.contract = contract

        async def openai_tool_definitions(self):
            if not self.active:
                return [{
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
                }]
            return [{
                "type": "function",
                "function": {
                    "name": "podwoj_liczbe",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                        "additionalProperties": False,
                    },
                },
            }]

        async def call(self, tool, arguments):
            self.calls.append((tool, arguments))
            if tool == "learning_create_tool":
                assert self.contract is not None
                assert [case.expected for case in self.contract.tests] == [
                    {"result": 4}, {"result": 14}
                ]
                self.active = True
                return json.dumps({
                    "name": "podwoj_liczbe",
                    "status": "active",
                    "validation": {
                        "passed": True,
                        "tests": [{"passed": True}, {"passed": True}],
                    },
                })
            if tool == "podwoj_liczbe":
                return json.dumps({"result": arguments["n"] * 2})
            raise AssertionError(tool)

    class Model:
        config = SimpleNamespace(context=12_000)
        source_turns = 0

        async def ask(self, **kwargs):
            assert kwargs.get("response_format")
            return json.dumps({
                "status": "ready",
                "tests": [
                    {"arguments": {"n": 2}, "expected": {"result": 4}, "evidence_quote": "dla 2 ma dać 4"},
                    {"arguments": {"n": 7}, "expected": {"result": 14}, "evidence_quote": "dla 7 ma dać 14"},
                ],
                "final_arguments": {"n": 9},
                "final_evidence_quote": "n = 9",
                "reason": "",
            })

        async def respond(self, **kwargs):
            self.source_turns += 1
            if empty_first and self.source_turns == 1:
                return LLMResponse(content="", finish_reason="length")
            assert kwargs["tools"] is None
            assert "arguments['n']" in kwargs["messages"][0]["content"]
            return LLMResponse(
                content="def run(arguments):\n    return {'result': arguments['n'] * 2}"
            )

    class Memory:
        session = Session()

        async def process(self, *args, **kwargs):
            return None

    tools = Tools()
    agent = object.__new__(Agent)
    agent.tools = tools
    agent.llm = Model()
    agent.memory = Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls[0][0] == "learning_create_tool"
    assert tools.calls[1] == ("podwoj_liczbe", {"n": 9})
    assert '"result": 18' in answer
    assert agent.llm.source_turns == (2 if empty_first else 1)
