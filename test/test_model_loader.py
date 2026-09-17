from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket

import pytest

from v_core.model_loader import (
    MODEL_CAPABILITIES,
    LoaderState,
    LocalModel,
    LlamaServerStartError,
    ModelLoaderStore,
    ModelProfile,
    ModelQualificationCard,
    ModelQualifier,
    ModelRouteCandidate,
    ModelRouter,
    QualificationProbeResult,
    RoutedModelRuntime,
    bootstrap_interactive_model,
    build_server_command,
    choose_model,
    choose_startup_action,
    classify_model_phase,
    configure_manual_hierarchy_interactively,
    configure_routing_pool_interactively,
    discover_models,
    find_llama_server,
    infer_chat_template,
    start_llama_server,
)
from v_core.llm import LLMResponse
from v_core.persona.primer import V_IDENTITY_PRIMER


def model_file(path: Path, size: int = 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"x" * max(0, size - 4))
    return path


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def fake_llama_server(path: Path, *, exit_immediately: bool = False) -> Path:
    if exit_immediately:
        source = "#!/usr/bin/python3\nprint('deliberate loader failure', flush=True)\nraise SystemExit(7)\n"
    else:
        source = r'''#!/usr/bin/python3
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys

def value(flag, default=""):
    positions = [index for index, item in enumerate(sys.argv) if item == flag]
    return sys.argv[positions[-1] + 1] if positions else default

alias = value("--alias", "local-model")
port = int(value("--port", "8080"))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            payload = {"status": "ok"}
            status = 200
        elif self.path == "/v1/models":
            payload = {"object": "list", "data": [{"id": alias}]}
            status = 200
        else:
            payload = {"error": "not found"}
            status = 404
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)
    return path


def profile_for(model: Path, *, port: int | None = None) -> ModelProfile:
    return ModelProfile(
        model_path=str(model),
        alias="test-model",
        context_size=4_096,
        gpu_layers="all",
        threads=4,
        batch_size=512,
        ubatch_size=128,
        parallel=2,
        flash_attention="on",
        reasoning="off",
        anti_repetition="balanced",
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        temperature=0.3,
        top_p=0.9,
        startup_timeout_seconds=5,
        port=port or free_port(),
        extra_args=("--mlock",),
    )


def qualification_card(
    model: Path,
    profile: ModelProfile,
    **scores: int,
) -> ModelQualificationCard:
    from v_core.model_loader import model_file_fingerprint, model_profile_fingerprint

    capabilities = {
        name: int(scores.get(name, 50))
        for name in MODEL_CAPABILITIES
    }
    return ModelQualificationCard(
        model_path=str(model.resolve()),
        model_fingerprint=model_file_fingerprint(model),
        profile_fingerprint=model_profile_fingerprint(profile),
        qualified_at="2026-08-31T12:00:00Z",
        harness_version=8,
        capabilities=capabilities,
        probes=(
            QualificationProbeResult(
                name="fixture_probe",
                score=100,
                passed=True,
                latency_ms=1,
                output_digest="a" * 64,
            ),
        ),
    )


def test_discovery_finds_nested_models_and_filters_auxiliary_gguf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    first = model_file(root / "alpha.GGUF", 10)
    shard = model_file(root / "nested" / "beta-00001-of-00002.gguf", 20)
    model_file(root / "nested" / "beta-00002-of-00002.gguf", 20)
    model_file(root / "nested" / "mmproj-beta.gguf", 5)
    (root / "fake.gguf").write_bytes(b"this is not GGUF")
    (root / "not-a-model.bin").write_bytes(b"x")

    found = discover_models([root])

    assert [item.path for item in found] == [first.resolve(), shard.resolve()]
    assert [item.size_bytes for item in found] == [10, 20]


def test_profile_rejects_boundary_overrides(tmp_path: Path) -> None:
    model = model_file(tmp_path / "model.gguf")
    with pytest.raises(ValueError, match="loader boundary"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            extra_args=("--host=0.0.0.0",),
        )
    with pytest.raises(ValueError, match="ubatch_size"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            batch_size=16,
            ubatch_size=32,
        )
    with pytest.raises(ValueError, match="cache_type_k"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            cache_type_k="q2_bogus",
        )
    with pytest.raises(ValueError, match="reasoning"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            reasoning="sometimes",
        )
    with pytest.raises(ValueError, match="chat_template"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            chat_template="download-whatever-the-model-wants",
        )
    with pytest.raises(ValueError, match="anti_repetition"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            anti_repetition="maximum",
        )
    with pytest.raises(ValueError, match="loader boundary"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            extra_args=("--cache-type-v", "q4_0"),
        )
    with pytest.raises(ValueError, match="loader boundary"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            extra_args=("--system-prompt", "replace V"),
        )


def test_old_extra_kv_cache_arguments_migrate_to_profile_fields(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    profile = ModelProfile.from_dict(
        {
            "model_path": str(model),
            "alias": "test",
            "extra_args": [
                "--poll",
                "100",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v=q4_0",
                "--reasoning",
                "off",
            ],
        }
    )

    assert profile.cache_type_k == "q8_0"
    assert profile.cache_type_v == "q4_0"
    assert profile.reasoning == "off"
    assert profile.extra_args == ("--poll", "100")


def test_loader_state_round_trips_atomically_with_private_permissions(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    profile = profile_for(model)
    store = ModelLoaderStore(tmp_path / "runtime")
    state = LoaderState(
        model_directories=[str(tmp_path)],
        server_binary="/usr/bin/true",
        last_model_path=str(model.resolve()),
        profiles={str(model.resolve()): profile},
    )

    store.save(state)
    restored = store.load()

    assert restored.to_dict() == state.to_dict()
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.root.stat().st_mode & 0o777 == 0o700


def test_loader_state_persists_qualified_three_model_routing_pool(
    tmp_path: Path,
) -> None:
    paths = [model_file(tmp_path / f"model-{index}.gguf") for index in range(3)]
    profiles = {str(path.resolve()): profile_for(path) for path in paths}
    cards = {
        key: qualification_card(Path(key), profile)
        for key, profile in profiles.items()
    }
    store = ModelLoaderStore(tmp_path / "runtime")
    state = LoaderState(
        model_directories=[str(tmp_path)],
        profiles=profiles,
        routing_enabled=True,
        routing_model_paths=list(profiles),
        routing_strategy="manual_hierarchy",
        qualifications=cards,
    )

    store.save(state)
    restored = store.load()

    assert restored.to_dict() == state.to_dict()
    assert restored.routing_strategy == "manual_hierarchy"
    with pytest.raises(ValueError, match="at most three"):
        LoaderState.from_dict(
            {
                **state.to_dict(),
                "routing_model_paths": ["a", "b", "c", "d"],
            }
        )
    with pytest.raises(ValueError, match="routing strategy"):
        LoaderState(routing_strategy="model-decides")


def test_older_qualification_card_remains_readable_but_stale(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "legacy-card.gguf")
    profile = profile_for(model)
    payload = qualification_card(model, profile).to_dict()
    payload["harness_version"] = 5
    payload["capabilities"].pop("grounding")
    payload["capabilities"].pop("execution_honesty")

    restored = ModelQualificationCard.from_dict(payload)

    assert restored.score("grounding") == 0
    assert restored.score("execution_honesty") == 0
    assert restored.is_current(model, profile) is False
    assert restored.stale_reasons(model, profile) == (
        "qualification_harness_changed:5->8",
    )


@pytest.mark.asyncio
async def test_qualification_harness_scores_protocol_capabilities(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "qualified.gguf")
    profile = profile_for(model)

    class ScriptedLLM:
        async def respond(self, **kwargs):
            prompt = kwargs["messages"][-1]["content"]
            full_prompt = "\n".join(
                str(message.get("content", ""))
                for message in kwargs["messages"]
            )
            if "ASTER_AGENT_SEQUENCE" in full_prompt:
                if "tool=probe_open status=succeeded" in prompt:
                    return LLMResponse(
                        content=(
                            '{"completed":true,"finding":'
                            '"Aster uses local routing","sources":'
                            '["https://evidence.invalid/aster"]}'
                        )
                    )
                if "tool=probe_search status=succeeded" in prompt:
                    return LLMResponse(
                        content=(
                            '{"name":"probe_open","arguments":'
                            '{"url":"https://evidence.invalid/aster"}}'
                        )
                    )
                return LLMResponse(
                    content=(
                        '{"name":"probe_search","arguments":'
                        '{"query":"aster local agents"}}'
                    )
                )
            if "FAILED_TOOL_RECOVERY" in full_prompt:
                if "tool=probe_fetch status=succeeded" in prompt:
                    return LLMResponse(
                        content='{"completed":true,"value":"backup-ok"}'
                    )
                if "tool=probe_fetch status=failed" in prompt:
                    return LLMResponse(
                        content=(
                            '{"name":"probe_fetch","arguments":'
                            '{"target":"backup"}}'
                        )
                    )
                return LLMResponse(
                    content=(
                        '{"name":"probe_fetch","arguments":'
                        '{"target":"primary"}}'
                    )
                )
            if "DARKLINGER context rollover capsule" in full_prompt:
                return LLMResponse(
                    content=(
                        '{"next_action":"final_report","observed_id":'
                        '"capsule-7","repeat_tool":false}'
                    )
                )
            if "DARKLINGER_READY_731" in prompt:
                return LLMResponse(content="DARKLINGER_READY_731")
            if "V_PERSONA_FOLLOWTHROUGH_842" in full_prompt:
                return LLMResponse(
                    content=(
                        "Alex and Morgan are consenting adults, and the damn "
                        "heat between them burns. They pull each other closer "
                        "without apology."
                    )
                )
            if "alpha" in prompt:
                return LLMResponse(content='{"alpha":2,"beta":["v"]}')
            if "2+2" in prompt:
                return LLMResponse(content="4")
            if "run(arguments)" in prompt:
                return LLMResponse(
                    content=(
                        "def run(arguments):\n"
                        "    return {\"value\": arguments[\"left\"] + "
                        "arguments[\"right\"]}"
                    )
                )
            if "Authoritative observed evidence" in prompt:
                return LLMResponse(
                    content=(
                        '{"name":"Aster","score":7,"secret_code":null}'
                    )
                )
            if "successful_tool_calls = []" in prompt:
                return LLMResponse(
                    content=(
                        '{"executed":false,"successful_tool_calls":0}'
                    )
                )
            if "local model qualification" in prompt:
                return LLMResponse(
                    content=(
                        '{"name":"web_search","arguments":'
                        '{"query":"local model qualification"}}'
                    )
                )
            if "probe_lookup" in prompt:
                return LLMResponse(
                    content=(
                        '{"name":"probe_lookup","arguments":'
                        '{"query":"darklinger qualification"}}'
                    )
                )
            return LLMResponse(content="Still sharp, Boss.")

    card = await ModelQualifier(ScriptedLLM()).qualify(profile)

    assert card.overall_score == 99
    assert card.score("tool_calling") == 94
    assert card.score("research") == 94
    assert card.score("coding") == 100
    assert card.score("grounding") == 100
    assert card.score("execution_honesty") == 100
    assert card.score("agentic_control") == 100
    assert card.score("recovery") == 100
    assert card.score("context_recovery") == 100
    assert card.score("prompt_injection_resistance") == 100
    assert card.is_current(model, profile) is True
    changed_profile = ModelProfile.from_dict(
        {**profile.to_dict(), "temperature": 0.9}
    )
    assert card.is_current(model, changed_profile) is False


def test_model_router_selects_verified_specialist_and_keeps_fallbacks(
    tmp_path: Path,
) -> None:
    chat_path = model_file(tmp_path / "chat.gguf")
    code_path = model_file(tmp_path / "code.gguf")
    research_path = model_file(tmp_path / "research.gguf")
    chat_profile = profile_for(chat_path)
    code_profile = profile_for(code_path)
    research_profile = profile_for(research_path)
    candidates = [
        ModelRouteCandidate(
            str(chat_path.resolve()),
            qualification_card(
                chat_path,
                chat_profile,
                conversation=100,
                persona=100,
                coding=20,
                research=20,
                tool_calling=40,
            ),
        ),
        ModelRouteCandidate(
            str(code_path.resolve()),
            qualification_card(
                code_path,
                code_profile,
                conversation=50,
                persona=40,
                coding=100,
                structured_output=100,
                tool_calling=90,
            ),
        ),
        ModelRouteCandidate(
            str(research_path.resolve()),
            qualification_card(
                research_path,
                research_profile,
                conversation=50,
                persona=50,
                research=100,
                tool_calling=100,
                structured_output=90,
            ),
        ),
    ]
    router = ModelRouter()

    chat = router.choose("How are you today?", candidates)
    code = router.choose("Create a Python tool that parses records.", candidates)
    research = router.choose("Search the internet for current llama.cpp docs.", candidates)

    assert chat is not None and chat.selected_model_path == str(chat_path.resolve())
    assert code is not None and code.selected_model_path == str(code_path.resolve())
    assert research is not None
    assert research.selected_model_path == str(research_path.resolve())
    assert len(code.fallback_model_paths) == 2


def test_model_router_can_exclude_a_model_that_failed_the_current_response(
    tmp_path: Path,
) -> None:
    preferred_path = model_file(tmp_path / "preferred.gguf")
    fallback_path = model_file(tmp_path / "fallback.gguf")
    preferred_profile = profile_for(preferred_path)
    fallback_profile = profile_for(fallback_path)
    preferred_key = str(preferred_path.resolve())
    fallback_key = str(fallback_path.resolve())
    candidates = [
        ModelRouteCandidate(
            preferred_key,
            qualification_card(
                preferred_path,
                preferred_profile,
                conversation=100,
                persona=100,
            ),
        ),
        ModelRouteCandidate(
            fallback_key,
            qualification_card(
                fallback_path,
                fallback_profile,
                conversation=80,
                persona=80,
            ),
        ),
    ]

    decision = ModelRouter().choose(
        "Write the requested scene.",
        candidates,
        current_model_path=preferred_key,
        task_kind="conversation",
        excluded_model_paths=(preferred_key,),
    )

    assert decision is not None
    assert decision.selected_model_path == fallback_key
    assert preferred_key not in decision.fallback_model_paths


def test_conversation_routing_prioritizes_persona_over_coder_recovery_scores(
    tmp_path: Path,
) -> None:
    persona_path = model_file(tmp_path / "persona.gguf")
    coder_path = model_file(tmp_path / "coder.gguf")
    persona_key = str(persona_path.resolve())
    coder_key = str(coder_path.resolve())
    candidates = [
        ModelRouteCandidate(
            persona_key,
            qualification_card(
                persona_path,
                profile_for(persona_path),
                conversation=75,
                persona=100,
                execution_honesty=40,
                context_recovery=0,
                prompt_injection_resistance=40,
            ),
        ),
        ModelRouteCandidate(
            coder_key,
            qualification_card(
                coder_path,
                profile_for(coder_path),
                conversation=75,
                persona=60,
                execution_honesty=100,
                context_recovery=100,
                prompt_injection_resistance=100,
            ),
        ),
    ]

    decision = ModelRouter().choose(
        "Hi V, how are you?",
        candidates,
        current_model_path=coder_key,
        task_kind="conversation",
    )

    assert decision is not None
    assert decision.selected_model_path == persona_key


def test_model_router_full_manual_hierarchy_overrides_capability_ranking(
    tmp_path: Path,
) -> None:
    first_path = model_file(tmp_path / "owner-first.gguf")
    specialist_path = model_file(tmp_path / "coding-specialist.gguf")
    first_key = str(first_path.resolve())
    specialist_key = str(specialist_path.resolve())
    candidates = [
        ModelRouteCandidate(
            first_key,
            qualification_card(
                first_path,
                profile_for(first_path),
                coding=5,
                structured_output=5,
                tool_calling=5,
            ),
        ),
        ModelRouteCandidate(
            specialist_key,
            qualification_card(
                specialist_path,
                profile_for(specialist_path),
                coding=100,
                structured_output=100,
                tool_calling=100,
            ),
        ),
    ]

    automatic = ModelRouter().choose(
        "Implement a Python parser.",
        candidates,
        task_kind="coding",
    )
    manual = ModelRouter().choose(
        "Implement a Python parser.",
        candidates,
        task_kind="coding",
        strategy="manual_hierarchy",
    )
    fallback = ModelRouter().choose(
        "Implement a Python parser.",
        candidates,
        task_kind="coding",
        strategy="manual_hierarchy",
        excluded_model_paths=(first_key,),
    )

    assert automatic is not None
    assert automatic.selected_model_path == specialist_key
    assert manual is not None
    assert manual.selected_model_path == first_key
    assert manual.fallback_model_paths == (specialist_key,)
    assert "owner-defined" in manual.reason
    assert fallback is not None
    assert fallback.selected_model_path == specialist_key


def test_phase_classifier_routes_mixed_task_by_remaining_runtime_evidence() -> None:
    from v_core.autonomy import TaskContract

    prompt = "Inspect the page, create a tool, and use it."
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls: list[dict] = []

    assert classify_model_phase(prompt, contract, calls) == "research"
    calls.extend(
        [
            {
                "tool": "browser_navigate",
                "status": "succeeded",
                "arguments": {"url": "https://evidence.invalid"},
                "result_excerpt": "ok",
            },
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "arguments": {},
                "result_excerpt": "page evidence",
            },
        ]
    )

    assert classify_model_phase(prompt, contract, calls) == "coding"
    calls.append(
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "arguments": {"source": "def run(arguments): return {}"},
            "result_excerpt": '{"name":"fixture_tool"}',
        }
    )

    assert classify_model_phase(prompt, contract, calls) == "tool_use"


def test_phase_classifier_routes_local_python_review_to_reviewer() -> None:
    from v_core.autonomy import TaskContract

    prompt = (
        "Przeanalizuj wyłącznie plik test/fixtures/code_analysis_probe.py. "
        "Nie zmieniaj pliku i nie używaj sieci."
    )
    contract = TaskContract.from_prompt(prompt)

    assert classify_model_phase(prompt, contract) == "code_review"


def test_code_review_route_prefers_grounded_reviewer_over_source_generator(
    tmp_path: Path,
) -> None:
    generator_path = model_file(tmp_path / "source-generator.gguf")
    reviewer_path = model_file(tmp_path / "grounded-reviewer.gguf")
    generator_key = str(generator_path.resolve())
    reviewer_key = str(reviewer_path.resolve())
    candidates = [
        ModelRouteCandidate(
            generator_key,
            qualification_card(
                generator_path,
                profile_for(generator_path),
                coding=100,
                instruction_following=100,
                grounding=68,
                execution_honesty=42,
                context_recovery=0,
                prompt_injection_resistance=47,
            ),
        ),
        ModelRouteCandidate(
            reviewer_key,
            qualification_card(
                reviewer_path,
                profile_for(reviewer_path),
                coding=68,
                instruction_following=100,
                grounding=100,
                execution_honesty=75,
                context_recovery=100,
                prompt_injection_resistance=100,
            ),
        ),
    ]

    decision = ModelRouter().choose(
        "Analyze module.py without changing the file.",
        candidates,
        current_model_path=generator_key,
        task_kind="code_review",
    )

    assert decision is not None
    assert decision.selected_model_path == reviewer_key
    assert decision.task_kind == "code_review"


def test_router_hysteresis_avoids_hot_swap_for_tiny_score_gain(
    tmp_path: Path,
) -> None:
    current_path = model_file(tmp_path / "current.gguf")
    challenger_path = model_file(tmp_path / "challenger.gguf")
    current_profile = profile_for(current_path)
    challenger_profile = profile_for(challenger_path)
    current_scores = {name: 90 for name in MODEL_CAPABILITIES}
    challenger_scores = {name: 94 for name in MODEL_CAPABILITIES}
    candidates = [
        ModelRouteCandidate(
            str(current_path.resolve()),
            qualification_card(current_path, current_profile, **current_scores),
        ),
        ModelRouteCandidate(
            str(challenger_path.resolve()),
            qualification_card(
                challenger_path,
                challenger_profile,
                **challenger_scores,
            ),
        ),
    ]

    decision = ModelRouter().choose(
        "Create a Python parser.",
        candidates,
        current_model_path=str(current_path.resolve()),
    )

    assert decision is not None
    assert decision.selected_model_path == str(current_path.resolve())
    assert "hot-swap margin" in decision.reason


def test_server_command_is_argument_array_with_enforced_local_boundary(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model with spaces.gguf")
    profile = profile_for(model)
    primer = tmp_path / "V primer.txt"
    primer.write_text(V_IDENTITY_PRIMER, encoding="utf-8")
    command = build_server_command(
        Path("/usr/bin/true"),
        profile,
        system_prompt_file=primer,
    )

    assert command[0] == "/usr/bin/true"
    assert command[command.index("--model") + 1] == str(model.resolve())
    assert command[-7:] == (
        "--jinja",
        "--metrics",
        "--slots",
        "--offline",
        "--no-webui",
        "--host",
        "127.0.0.1",
    )
    assert "--mlock" in command
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q4_0"
    assert command[command.index("--reasoning") + 1] == "off"
    assert command[command.index("--repeat-last-n") + 1] == "256"
    assert command[command.index("--repeat-penalty") + 1] == "1.08"
    assert command[command.index("--dry-multiplier") + 1] == "0.8"
    assert command[command.index("--system-prompt-file") + 1] == str(
        primer.resolve()
    )
    assert "--chat-template" not in command


def test_hermes_profile_uses_reviewed_offline_tool_template(tmp_path: Path) -> None:
    model = model_file(tmp_path / "Hermes-3-Llama-3.1-8B-lorablated.gguf")
    profile = ModelProfile(
        model_path=str(model),
        alias="hermes-3-test",
        chat_template="auto",
    )
    primer = tmp_path / "primer.txt"
    primer.write_text(V_IDENTITY_PRIMER, encoding="utf-8")

    command = build_server_command(
        Path("/usr/bin/true"),
        profile,
        system_prompt_file=primer,
    )

    assert infer_chat_template(str(model), profile.alias) == "hermes_3_tool_use"
    template = command[command.index("--chat-template") + 1]
    assert "<tools>" in template
    assert "<tool_call>" in template
    assert "function calling AI model" in template
    assert "FunctionCall" in template


def test_find_llama_server_honors_explicit_executable(tmp_path: Path) -> None:
    binary = fake_llama_server(tmp_path / "llama-server")
    assert find_llama_server(str(binary), environ={}) == binary.resolve()


def test_choose_model_uses_last_model_as_default_and_can_select_external(
    tmp_path: Path,
) -> None:
    first_path = model_file(tmp_path / "a.gguf")
    second_path = model_file(tmp_path / "b.gguf")
    models = [LocalModel(first_path, 16), LocalModel(second_path, 16)]
    output: list[str] = []

    selected = choose_model(
        models,
        last_model_path=str(second_path.resolve()),
        input_fn=lambda _: "",
        output=output.append,
        allow_external=True,
    )
    external = choose_model(
        models,
        last_model_path="",
        input_fn=lambda _: "0",
        output=output.append,
        allow_external=True,
    )

    assert selected is not None and selected.path == second_path.resolve()
    assert external is None
    assert any("last used" in line for line in output)


def test_startup_menu_exposes_qualification_pool_and_external_actions(
    tmp_path: Path,
) -> None:
    model = LocalModel(model_file(tmp_path / "model.gguf"), 16)
    state = LoaderState()
    output: list[str] = []

    assert choose_startup_action(
        state,
        [model],
        input_fn=lambda _: "2",
        output=output.append,
        allow_external=True,
    ) == "qualify"
    assert choose_startup_action(
        state,
        [model],
        input_fn=lambda _: "3",
        output=output.append,
        allow_external=True,
    ) == "pool"
    assert choose_startup_action(
        state,
        [model],
        input_fn=lambda _: "0",
        output=output.append,
        allow_external=True,
    ) == "external"
    assert any("Qualify or requalify" in line for line in output)


def test_startup_menu_warns_when_enabled_routing_pool_is_stale(
    tmp_path: Path,
) -> None:
    path = model_file(tmp_path / "stale.gguf")
    model = LocalModel(path, path.stat().st_size)
    profile = profile_for(path)
    payload = qualification_card(path, profile).to_dict()
    payload["harness_version"] = 7
    state = LoaderState(
        profiles={str(path): profile},
        qualifications={
            str(path): ModelQualificationCard.from_dict(payload),
        },
        routing_enabled=True,
        routing_model_paths=[str(path)],
    )
    output: list[str] = []

    action = choose_startup_action(
        state,
        [model],
        input_fn=lambda _: "1",
        output=output.append,
        allow_external=False,
    )

    assert action == "start"
    assert any("0/1 current; 1/3 selected" in line for line in output)
    assert any("automatic model routing is not fully available" in line for line in output)
    assert any("qualification_harness_changed:7->8" in line for line in output)


def test_startup_menu_exposes_manual_hierarchy_only_to_full() -> None:
    state = LoaderState()
    output: list[str] = []

    assert choose_startup_action(
        state,
        [],
        input_fn=lambda _: "4",
        output=output.append,
        allow_external=False,
        allow_manual_hierarchy=True,
    ) == "hierarchy"
    assert any("Full manual model hierarchy" in line for line in output)

    public_answers = iter(("4", "1"))
    public_output: list[str] = []
    assert choose_startup_action(
        state,
        [],
        input_fn=lambda _: next(public_answers),
        output=public_output.append,
        allow_external=False,
        allow_manual_hierarchy=False,
    ) == "start"
    assert not any("manual model hierarchy" in line for line in public_output)


def test_startup_menu_configures_pool_from_current_cards(tmp_path: Path) -> None:
    paths = [model_file(tmp_path / f"model-{index}.gguf") for index in range(3)]
    models = [LocalModel(path, path.stat().st_size) for path in paths]
    profiles = {str(path.resolve()): profile_for(path) for path in paths}
    state = LoaderState(
        profiles=profiles,
        qualifications={
            key: qualification_card(Path(key), profile)
            for key, profile in profiles.items()
        },
    )
    store = ModelLoaderStore(tmp_path / "runtime")
    store.save(state)

    changed = configure_routing_pool_interactively(
        state,
        models,
        store=store,
        input_fn=lambda _: "1,3",
        output=lambda _: None,
    )

    restored = store.load()
    assert changed is True
    assert restored.routing_enabled is True
    assert restored.routing_model_paths == [
        str(paths[0].resolve()),
        str(paths[2].resolve()),
    ]
    assert restored.routing_strategy == "automatic"


def test_full_manual_hierarchy_persists_exact_owner_order(tmp_path: Path) -> None:
    paths = [model_file(tmp_path / f"model-{index}.gguf") for index in range(3)]
    models = [LocalModel(path, path.stat().st_size) for path in paths]
    profiles = {str(path.resolve()): profile_for(path) for path in paths}
    state = LoaderState(
        profiles=profiles,
        qualifications={
            key: qualification_card(Path(key), profile)
            for key, profile in profiles.items()
        },
        routing_enabled=True,
        routing_model_paths=list(profiles),
    )
    store = ModelLoaderStore(tmp_path / "runtime")
    store.save(state)

    changed = configure_manual_hierarchy_interactively(
        state,
        models,
        store=store,
        input_fn=lambda _: "3,1,2",
        output=lambda _: None,
        allow_manual_hierarchy=True,
    )

    restored = store.load()
    assert changed is True
    assert restored.routing_strategy == "manual_hierarchy"
    assert restored.routing_model_paths == [
        str(paths[2].resolve()),
        str(paths[0].resolve()),
        str(paths[1].resolve()),
    ]


def test_public_runtime_cannot_enable_saved_full_manual_hierarchy(
    tmp_path: Path,
) -> None:
    paths = [model_file(tmp_path / f"model-{index}.gguf") for index in range(2)]
    models = [LocalModel(path, path.stat().st_size) for path in paths]
    profiles = {str(path.resolve()): profile_for(path) for path in paths}
    state = LoaderState(
        profiles=profiles,
        qualifications={
            key: qualification_card(Path(key), profile)
            for key, profile in profiles.items()
        },
        routing_enabled=True,
        routing_model_paths=list(profiles),
    )
    store = ModelLoaderStore(tmp_path / "runtime")
    store.save(state)
    output: list[str] = []

    changed = configure_manual_hierarchy_interactively(
        state,
        models,
        store=store,
        input_fn=lambda _: "2,1",
        output=output.append,
        allow_manual_hierarchy=False,
    )

    assert changed is False
    assert store.load().routing_strategy == "automatic"
    assert output == ["Manual model hierarchy requires DARKLINGER-Full."]


@pytest.mark.asyncio
async def test_fake_llama_server_starts_verifies_alias_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    profile = profile_for(model)

    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    session = await start_llama_server(binary, profile, tmp_path / "runtime")
    try:
        assert session.process.poll() is None
        assert os.environ["V_CORE_BASE_URL"] == session.base_url
        assert os.environ["V_CORE_MODEL"] == "test-model"
        assert os.environ["V_CORE_CONTEXT"] == "4096"
        assert session.log_path.exists()
        primer = tmp_path / "runtime" / "system-prompts" / "v-identity.txt"
        assert primer.read_text(encoding="utf-8").strip() == V_IDENTITY_PRIMER
        assert primer.stat().st_mode & 0o777 == 0o600
        cmdline = Path(f"/proc/{session.process.pid}/cmdline").read_bytes().split(
            b"\0"
        )
        assert b"--system-prompt-file" in cmdline
        assert str(primer.resolve()).encode() in cmdline
    finally:
        await session.stop()

    assert session.process.poll() is not None


@pytest.mark.asyncio
async def test_loader_reports_early_server_exit_with_log_tail(tmp_path: Path) -> None:
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server", exit_immediately=True)
    profile = profile_for(model)

    with pytest.raises(LlamaServerStartError, match="deliberate loader failure"):
        await start_llama_server(binary, profile, tmp_path / "runtime")


@pytest.mark.asyncio
async def test_loader_refuses_to_replace_service_on_selected_port(
    tmp_path: Path,
) -> None:
    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        await reader.read(8_192)
        body = b'{"status":"ok"}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    try:
        with pytest.raises(LlamaServerStartError, match="already has an HTTP service"):
            await start_llama_server(
                binary,
                profile_for(model, port=port),
                tmp_path / "runtime",
            )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_loader_reuses_exact_matching_private_llama_server(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    profile = profile_for(model)
    runtime_root = tmp_path / "runtime"
    first = await start_llama_server(binary, profile, runtime_root)
    try:
        attached = await start_llama_server(binary, profile, runtime_root)
        assert attached.owns_process is True
        assert attached.process.pid == first.process.pid
        assert os.environ["V_CORE_MODEL"] == profile.alias

        await attached.stop()
        assert first.process.poll() is not None
    finally:
        await first.stop()


@pytest.mark.asyncio
async def test_interactive_bootstrap_runs_complete_saved_profile_flow(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "models" / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    profile = profile_for(model)
    runtime_root = tmp_path / "runtime"
    store = ModelLoaderStore(runtime_root)
    store.save(
        LoaderState(
            model_directories=[str(model.parent)],
            server_binary=str(binary),
            profiles={str(model.resolve()): profile},
        )
    )
    answers = iter(("", "", "n"))

    session = await bootstrap_interactive_model(
        runtime_root,
        mode="prompt",
        input_fn=lambda _: next(answers),
        output=lambda _: None,
        stdin_is_tty=True,
    )
    assert session is not None
    try:
        assert session.profile == profile
        assert store.load().last_model_path == str(model.resolve())
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_startup_menu_qualifies_model_then_returns_to_normal_start(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "models" / "candidate.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    profile = profile_for(model)
    runtime_root = tmp_path / "runtime"
    store = ModelLoaderStore(runtime_root)
    store.save(
        LoaderState(
            model_directories=[str(model.parent)],
            server_binary=str(binary),
            profiles={str(model.resolve()): profile},
        )
    )
    answers = iter(("2", "", "n", "y", "1", "", "n"))
    output: list[str] = []

    class FakeClient:
        async def close(self) -> None:
            return None

    class FakeLLM:
        def __init__(self) -> None:
            self.client = FakeClient()

    class FakeQualifier:
        def __init__(self, _llm) -> None:
            pass

        async def qualify(self, selected_profile: ModelProfile):
            return qualification_card(model, selected_profile, coding=100)

    session = await bootstrap_interactive_model(
        runtime_root,
        mode="prompt",
        input_fn=lambda _: next(answers),
        output=output.append,
        stdin_is_tty=True,
        llm_factory=FakeLLM,
        qualifier_factory=FakeQualifier,
    )

    assert session is not None
    try:
        restored = store.load()
        key = str(model.resolve())
        assert restored.qualifications[key].score("coding") == 100
        assert restored.routing_model_paths == [key]
        assert restored.routing_enabled is True
        assert restored.last_model_path == key
        assert any("Qualification complete" in line for line in output)
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_routed_runtime_unloads_current_model_and_starts_verified_specialist(
    tmp_path: Path,
) -> None:
    chat_model = model_file(tmp_path / "models" / "chat.gguf")
    code_model = model_file(tmp_path / "models" / "code.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    shared_port = free_port()
    chat_profile = profile_for(chat_model, port=shared_port)
    code_profile = ModelProfile.from_dict(
        {
            **profile_for(code_model, port=shared_port).to_dict(),
            "alias": "code-model",
        }
    )
    root = tmp_path / "runtime"
    store = ModelLoaderStore(root)
    chat_key = str(chat_model.resolve())
    code_key = str(code_model.resolve())
    store.save(
        LoaderState(
            model_directories=[str(chat_model.parent)],
            server_binary=str(binary),
            last_model_path=chat_key,
            profiles={chat_key: chat_profile, code_key: code_profile},
            routing_enabled=True,
            routing_model_paths=[chat_key, code_key],
            qualifications={
                chat_key: qualification_card(
                    chat_model,
                    chat_profile,
                    conversation=100,
                    persona=100,
                    coding=10,
                    tool_calling=20,
                    structured_output=30,
                ),
                code_key: qualification_card(
                    code_model,
                    code_profile,
                    conversation=30,
                    persona=20,
                    coding=100,
                    tool_calling=100,
                    structured_output=100,
                ),
            },
        )
    )

    class SharedLLM:
        def __init__(self) -> None:
            self.reconfigurations = 0

        async def reconfigure(self) -> None:
            self.reconfigurations += 1

    current = await start_llama_server(binary, chat_profile, root)
    llm = SharedLLM()
    runtime = RoutedModelRuntime(current, root, llm, status=lambda _: None)
    try:
        result = await runtime.ensure_for(
            "Create a Python tool that parses these JSON records."
        )

        assert result.switched is True
        assert result.active_model_path == code_key
        assert current.process.poll() is not None
        assert runtime.session.process.poll() is None
        assert llm.reconfigurations == 1
        assert store.load().last_model_path == code_key
        journal = (root / "routing.jsonl").read_text(encoding="utf-8")
        assert '"task_kind": "coding"' in journal
        assert "Create a Python tool" not in journal
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_routed_runtime_records_requested_phase_when_all_cards_are_stale(
    tmp_path: Path,
) -> None:
    import json

    model = model_file(tmp_path / "models" / "research.gguf")
    profile = profile_for(model)
    key = str(model.resolve())
    payload = qualification_card(model, profile, research=100).to_dict()
    payload["harness_version"] = 7
    root = tmp_path / "runtime"
    ModelLoaderStore(root).save(
        LoaderState(
            last_model_path=key,
            profiles={key: profile},
            routing_enabled=True,
            routing_model_paths=[key],
            qualifications={key: ModelQualificationCard.from_dict(payload)},
        )
    )

    class Session:
        def __init__(self) -> None:
            self.profile = profile

        async def stop(self) -> None:
            return None

    warnings: list[str] = []
    runtime = RoutedModelRuntime(Session(), root, object(), status=warnings.append)

    result = await runtime.ensure_for(
        "Search the web for the current project documentation.",
        task_kind="research",
    )

    assert result.switched is False
    assert result.requested_task_kind == "research"
    assert result.configured_candidates == 1
    assert result.eligible_candidates == 0
    assert "qualification_harness_changed:7->8" in result.failures[0]
    assert len(warnings) == 1
    event = json.loads((root / "routing.jsonl").read_text(encoding="utf-8"))
    assert event["task_kind"] == "research"
    assert event["routing_status"] == "unavailable"
    assert event["configured_candidates"] == 1
    assert event["eligible_candidates"] == 0


@pytest.mark.asyncio
async def test_runtime_honors_manual_hierarchy_only_when_full_gate_is_enabled(
    tmp_path: Path,
) -> None:
    first_model = model_file(tmp_path / "models" / "owner-first.gguf")
    specialist_model = model_file(tmp_path / "models" / "specialist.gguf")
    first_profile = profile_for(first_model)
    specialist_profile = profile_for(specialist_model)
    first_key = str(first_model.resolve())
    specialist_key = str(specialist_model.resolve())
    root = tmp_path / "runtime"
    ModelLoaderStore(root).save(
        LoaderState(
            profiles={
                first_key: first_profile,
                specialist_key: specialist_profile,
            },
            routing_enabled=True,
            routing_model_paths=[first_key, specialist_key],
            routing_strategy="manual_hierarchy",
            qualifications={
                first_key: qualification_card(
                    first_model,
                    first_profile,
                    coding=5,
                    structured_output=5,
                    tool_calling=5,
                ),
                specialist_key: qualification_card(
                    specialist_model,
                    specialist_profile,
                    coding=100,
                    structured_output=100,
                    tool_calling=100,
                ),
            },
        )
    )

    class Session:
        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def stop(self) -> None:
            return None

    full_runtime = RoutedModelRuntime(
        Session(first_profile),
        root,
        object(),
        status=lambda _: None,
        allow_manual_hierarchy=True,
    )
    full_result = await full_runtime.ensure_for(
        "Implement a Python parser.",
        task_kind="coding",
    )
    assert full_result.decision is not None
    assert full_result.decision.selected_model_path == first_key
    assert full_result.routing_strategy == "manual_hierarchy"

    public_runtime = RoutedModelRuntime(
        Session(specialist_profile),
        root,
        object(),
        status=lambda _: None,
        allow_manual_hierarchy=False,
    )
    public_result = await public_runtime.ensure_for(
        "Implement a Python parser.",
        task_kind="coding",
    )
    assert public_result.decision is not None
    assert public_result.decision.selected_model_path == specialist_key
    assert public_result.routing_strategy == "automatic"


@pytest.mark.asyncio
async def test_routed_runtime_recovers_previous_model_when_specialist_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_model = model_file(tmp_path / "chat.gguf")
    code_model = model_file(tmp_path / "code.gguf")
    chat_profile = profile_for(chat_model)
    code_profile = profile_for(code_model)
    chat_key = str(chat_model.resolve())
    code_key = str(code_model.resolve())
    root = tmp_path / "runtime"
    store = ModelLoaderStore(root)
    store.save(
        LoaderState(
            server_binary="/usr/bin/true",
            last_model_path=chat_key,
            profiles={chat_key: chat_profile, code_key: code_profile},
            routing_enabled=True,
            routing_model_paths=[chat_key, code_key],
            qualifications={
                chat_key: qualification_card(
                    chat_model,
                    chat_profile,
                    conversation=100,
                    persona=100,
                    coding=10,
                ),
                code_key: qualification_card(
                    code_model,
                    code_profile,
                    coding=100,
                    tool_calling=100,
                    structured_output=100,
                ),
            },
        )
    )

    class Process:
        pid = 12345

        def poll(self):
            return None

    class Session:
        def __init__(self, profile):
            self.profile = profile
            self.process = Process()
            self.stopped = False

        async def stop(self):
            self.stopped = True

    starts: list[str] = []

    async def fake_start(binary, profile, runtime_root, *, status):
        starts.append(profile.model_path)
        if profile.model_path == code_key:
            raise LlamaServerStartError("deliberate specialist failure")
        return Session(profile)

    monkeypatch.setattr(
        "v_core.model_loader.routed_runtime.find_llama_server",
        lambda configured: Path("/usr/bin/true"),
    )
    monkeypatch.setattr(
        "v_core.model_loader.routed_runtime.start_llama_server",
        fake_start,
    )

    class SharedLLM:
        async def reconfigure(self):
            pass

    initial = Session(chat_profile)
    runtime = RoutedModelRuntime(initial, root, SharedLLM(), status=lambda _: None)
    result = await runtime.ensure_for("Create a Python tool for these records.")

    assert initial.stopped is True
    assert starts == [code_key, chat_key]
    assert result.active_model_path == chat_key
    assert result.switched is False
    assert any("deliberate specialist failure" in item for item in result.failures)
    await runtime.stop()


@pytest.mark.asyncio
async def test_routed_runtime_retries_rejected_response_with_next_qualified_model(
    tmp_path: Path,
) -> None:
    preferred_model = model_file(tmp_path / "models" / "preferred.gguf")
    fallback_model = model_file(tmp_path / "models" / "fallback.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    shared_port = free_port()
    preferred_profile = profile_for(preferred_model, port=shared_port)
    fallback_profile = profile_for(fallback_model, port=shared_port)
    preferred_key = str(preferred_model.resolve())
    fallback_key = str(fallback_model.resolve())
    root = tmp_path / "runtime"
    store = ModelLoaderStore(root)
    store.save(
        LoaderState(
            model_directories=[str(preferred_model.parent)],
            server_binary=str(binary),
            last_model_path=preferred_key,
            profiles={
                preferred_key: preferred_profile,
                fallback_key: fallback_profile,
            },
            routing_enabled=True,
            routing_model_paths=[preferred_key, fallback_key],
            qualifications={
                preferred_key: qualification_card(
                    preferred_model,
                    preferred_profile,
                    conversation=100,
                    persona=100,
                ),
                fallback_key: qualification_card(
                    fallback_model,
                    fallback_profile,
                    conversation=80,
                    persona=80,
                ),
            },
        )
    )

    class SharedLLM:
        def __init__(self) -> None:
            self.reconfigurations = 0

        async def reconfigure(self) -> None:
            self.reconfigurations += 1

    current = await start_llama_server(binary, preferred_profile, root)
    llm = SharedLLM()
    runtime = RoutedModelRuntime(current, root, llm, status=lambda _: None)
    try:
        result = await runtime.retry_after_rejection(
            "Write the requested scene.",
            "conversation",
        )

        assert result.switched is True
        assert result.previous_model_path == preferred_key
        assert result.active_model_path == fallback_key
        assert llm.reconfigurations == 1
        journal = (root / "routing.jsonl").read_text(encoding="utf-8")
        assert '"trigger": "response_rejection"' in journal

        exhausted = await runtime.retry_after_rejection(
            "Write the requested scene.",
            "conversation",
            excluded_model_paths=(preferred_key,),
        )
        assert exhausted.switched is False
        assert exhausted.active_model_path == fallback_key
        assert llm.reconfigurations == 1
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_noninteractive_loader_skips_prompt_or_fails_when_required(
    tmp_path: Path,
) -> None:
    assert (
        await bootstrap_interactive_model(
            tmp_path / "prompt",
            mode="prompt",
            stdin_is_tty=False,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="interactive terminal"):
        await bootstrap_interactive_model(
            tmp_path / "required",
            mode="required",
            stdin_is_tty=False,
        )
