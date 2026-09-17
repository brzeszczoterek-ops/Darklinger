from __future__ import annotations

from types import SimpleNamespace

import pytest

from v_core.inference import (
    InferenceController,
    PUBLIC_INFERENCE_PROFILES,
)
from v_core.llm import LLM


def test_public_profiles_cover_distinct_task_modes() -> None:
    assert {
        "conversation",
        "creative",
        "coding",
        "analysis",
        "research",
        "tool_use",
        "document",
    }.issubset(PUBLIC_INFERENCE_PROFILES)
    assert (
        PUBLIC_INFERENCE_PROFILES["conversation"].temperature
        > PUBLIC_INFERENCE_PROFILES["coding"].temperature
    )


def test_public_controller_selects_profiles_but_rejects_arbitrary_tuning() -> None:
    controller = InferenceController(edition_name="public")

    assert controller.begin_turn("coding") == "coding"
    assert controller.parameters().temperature == pytest.approx(0.08)
    with pytest.raises(PermissionError):
        controller.configure_full(request={"temperature": 1.2})


def test_full_controller_accepts_turn_and_session_request_tuning() -> None:
    controller = InferenceController(edition_name="full")
    controller.begin_turn("conversation")

    status = controller.configure_full(
        profile="creative",
        request={"temperature": 1.1, "top_k": 96},
        scope="turn",
    )

    assert status["profile"] == "creative"
    assert status["request"]["temperature"] == pytest.approx(1.1)
    assert status["request"]["top_k"] == 96
    controller.begin_turn("coding")
    assert controller.parameters().temperature == pytest.approx(0.08)

    controller.configure_full(
        request={"presence_penalty": 0.4},
        scope="session",
    )
    controller.begin_turn("research")
    assert controller.parameters().presence_penalty == pytest.approx(0.4)


def test_full_controller_stages_server_tuning_once() -> None:
    controller = InferenceController(edition_name="full")

    status = controller.configure_full(
        server={"context_size": 65_536, "reasoning": "on"},
    )

    assert status["server_restart_pending"] is True
    staged = controller.consume_server_tuning()
    assert staged is not None
    assert staged.values() == {"context_size": 65_536, "reasoning": "on"}
    assert controller.consume_server_tuning() is None


def test_invalid_full_request_tuning_is_rejected_without_partial_update() -> None:
    controller = InferenceController(edition_name="full")
    controller.begin_turn("conversation")

    with pytest.raises(ValueError):
        controller.configure_full(request={"temperature": 99})

    assert controller.parameters().temperature == pytest.approx(0.65)


def test_invalid_full_server_tuning_is_rejected_before_restart() -> None:
    controller = InferenceController(edition_name="full")

    with pytest.raises(ValueError):
        controller.configure_full(server={"context_size": 1})
    with pytest.raises(ValueError):
        controller.configure_full(
            server={"batch_size": 128, "ubatch_size": 256}
        )

    assert controller.consume_server_tuning() is None


@pytest.mark.asyncio
async def test_llm_sends_selected_sampling_profile_to_local_provider() -> None:
    class Completions:
        def __init__(self) -> None:
            self.request = {}

        async def create(self, **kwargs):
            self.request = kwargs
            message = SimpleNamespace(content="ok", tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    completions = Completions()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.95)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None
    llm.inference = InferenceController(edition_name="public")
    llm.inference.begin_turn("coding")

    response = await llm.respond(messages=[{"role": "user", "content": "code"}])

    assert response.content == "ok"
    assert completions.request["temperature"] == pytest.approx(0.08)
    assert completions.request["top_p"] == pytest.approx(0.82)
    assert completions.request["extra_body"] == {
        "top_k": 20,
        "min_p": 0.02,
        "typical_p": 1.0,
        "repeat_penalty": 1.08,
    }
