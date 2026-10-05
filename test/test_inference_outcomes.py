from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from v_core.inference import InferenceController, InferenceParameters, PUBLIC_INFERENCE_PROFILES, ServerTuning
from v_core.model_loader.performance import InferenceOutcome, InferenceOutcomeStore, OutcomePolicy, task_context
from v_core.model_loader.outcome_runtime import OutcomeRun, current_run, request_snapshot
from v_core.llm import LLM


H = "a" * 64
J = "b" * 64


def observation(task="task", profile="coding", outcome="success", **kw):
    data = dict(task_id=task, task_kind="coding", context=H, edition="full", model_fingerprint=H,
                server_profile_fingerprint=H, profile=profile, request=asdict(PUBLIC_INFERENCE_PROFILES[profile]),
                request_context=H, outcome=outcome, verified=outcome != "unverified",
                failure_domain="model" if outcome == "failure" else "none", latency_ms=100)
    data.update(kw)
    return InferenceOutcome(**data)


def recommend(store, **kw):
    args = dict(task_kind="coding", context=H, model_fingerprint=H, server_profile_fingerprint=H,
                edition="full", default_profile="coding", default_request=asdict(PUBLIC_INFERENCE_PROFILES["coding"]),
                request_context=H)
    args.update(kw)
    return store.recommend(**args)


def seed(store, profile="analysis", n=5, **kw):
    for i in range(n):
        store.append(observation(f"{profile}-{i}", profile, **kw))


def test_no_evidence_and_one_success_do_not_promote(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    assert recommend(store).profile == "coding"
    seed(store, n=1)
    assert recommend(store).profile == "coding"


def test_restart_promotion_and_durable_rollback(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store)
    first = recommend(store)
    assert first.profile == "analysis"
    assert first.decision_id
    store = InferenceOutcomeStore(tmp_path)
    assert recommend(store).profile == "analysis"
    store.append(observation("bad1", "analysis", "failure"))
    store.append(observation("bad2", "analysis", "failure"))
    rolled = recommend(store)
    assert rolled.rolled_back and rolled.profile == "coding"
    # Quarantine survives later successes and process restart.
    seed(store, n=30)
    assert recommend(InferenceOutcomeStore(tmp_path)).profile == "coding"


def test_feedback_is_separate_deduplicated_and_cannot_promote(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    for i in range(8):
        store.append(observation(str(i), "analysis", "unverified"))
        store.record_feedback(str(i), True)
    assert recommend(store).profile == "coding"
    with pytest.raises(ValueError, match="already recorded"):
        store.record_feedback("0", True)
    assert len(store.load()) == 16


def test_negative_feedback_rolls_back_without_changing_runtime_verdict(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store)
    assert recommend(store).profile == "analysis"
    store.record_feedback("analysis-0", False)
    result = recommend(store)
    assert result.profile == "coding" and result.rolled_back
    assert store.load()[0].outcome == "success"


def test_fast_wrong_never_beats_slower_correct(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store, "coding", n=10, latency_ms=5000)
    seed(store, "analysis", n=8, latency_ms=1)
    for i in range(2):
        store.append(observation(f"bad-{i}", "analysis", "failure", latency_ms=1))
    assert recommend(store).profile == "coding"


def test_equal_quality_efficiency_requires_margin_and_cooldown(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store, "coding", latency_ms=100)
    assert recommend(store).profile == "coding"
    seed(store, "analysis", latency_ms=90)
    assert recommend(store).profile == "coding"
    seed(store, "research", latency_ms=10)
    assert recommend(store).profile == "research"
    assert recommend(store).profile == "research"


@pytest.mark.parametrize("field", ["model_fingerprint", "server_profile_fingerprint", "context", "request_context", "edition"])
def test_versions_contexts_and_editions_never_mix(tmp_path, field):
    store = InferenceOutcomeStore(tmp_path)
    seed(store)
    assert recommend(store, **{field: "public" if field == "edition" else J}).profile == "coding"


def test_tool_environment_and_cancelled_events_do_not_penalize_model(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    for domain in ("tool", "environment", "cancelled"):
        for i in range(8):
            store.append(observation(f"{domain}{i}", "analysis", "unverified", failure_domain=domain))
    assert recommend(store).profile == "coding"
    assert store.model_adjustment(task_kind="coding", context=H, model_fingerprint=H,
                                  server_profile_fingerprint=H, edition="full") == 0


def test_full_replays_validated_tuning_public_uses_exact_presets(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    request = asdict(PUBLIC_INFERENCE_PROFILES["analysis"])
    request["temperature"] = 0.33
    seed(store, request=request)
    assert recommend(store).request["temperature"] == 0.33
    seed(store, edition="public", request=request)
    assert recommend(store, edition="public").profile == "coding"


@pytest.mark.parametrize("bad", [{"verified": "true"}, {"signal": "self_rating"}, {"request": {"prompt": "secret"}}, {"latency_ms": -1}, {"schema_version": 99}, {"cost": float("nan"), "currency": "USD"}])
def test_invalid_evidence_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        observation(**bad)


def test_corrupt_history_fails_closed_and_retains_evidence(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE outcomes SET data='{}' WHERE id=1")
    assert recommend(store).profile == "coding"
    assert store.error
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 5


def test_future_schema_and_corrupt_policy_disable_learning(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute("PRAGMA user_version=999")
    assert InferenceOutcomeStore(tmp_path).error
    other = tmp_path / "policy"
    other.mkdir()
    (other / "outcome-policy.json").write_text('{"min_observations": 0}')
    assert InferenceOutcomeStore(other).error


def test_repeated_task_does_not_inflate_support(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    for _ in range(20):
        store.append(observation(profile="analysis"))
    assert len(store.load()) == 1
    assert recommend(store).profile == "coding"


def test_ambiguous_feedback_is_not_assigned_to_last_model(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    store.append(observation())
    store.append(observation(model_fingerprint=J))
    with pytest.raises(ValueError, match="one attributable"):
        store.record_feedback("task", False)


@pytest.mark.parametrize("values", [{"temperature": "0.3"}, {"top_k": 2.5}, {"seed": True}, {"top_p": float("nan")}])
def test_sampling_types_are_strict(values):
    with pytest.raises(ValueError):
        InferenceParameters(**values)


@pytest.mark.parametrize("values", [{"extra_args": ["--host=0.0.0.0"]}, {"extra_args": "--verbose"}, {"threads": 2.5}, {"gpu_layers": "-1"}])
def test_invalid_server_tuning_is_atomic(values):
    controller = InferenceController(edition_name="full")
    before = controller.status()
    with pytest.raises(ValueError):
        controller.configure_full(profile="creative", request={"temperature": 0.7}, server=values)
    assert controller.status() == before


def test_server_support_matches_model_profile_cache_types():
    assert ServerTuning(cache_type_k="q5_1").cache_type_k == "q5_1"


class Completions:
    def __init__(self):
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]), finish_reason="stop")],
                               usage=SimpleNamespace(total_tokens=12))


def llm_for(tmp_path, edition="full", qualified=True):
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.95)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm._native_tools_supported = None
    llm.inference = InferenceController(edition_name=edition)
    llm.inference.identity_provider = lambda: (H, H, qualified)
    llm.inference.outcome_store = InferenceOutcomeStore(tmp_path)
    llm.inference.begin_turn("coding")
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize("edition", ["full", "public"])
async def test_real_request_boundary_learns_and_reuses_parameters(tmp_path, edition):
    llm = llm_for(tmp_path, edition=edition)
    controller = llm.inference
    prompt = "write code"
    # Collect measured requests through the real LLM boundary. Verification is
    # supplied by the deterministic runtime contract, not provider prose.
    for i in range(5):
        run = OutcomeRun(controller, prompt)
        controller.select_profile("analysis")
        await llm.respond(messages=[{"role": "user", "content": prompt}])
        run.verdict = ("success", True, "none")
        run.finish()
    assert len(controller.outcome_store.load()) == 5
    controller.begin_turn("coding")
    run = OutcomeRun(controller, prompt)
    await llm.respond(messages=[{"role": "user", "content": prompt}])
    run.finish()
    assert llm.client.chat.completions.requests[-1]["temperature"] == PUBLIC_INFERENCE_PROFILES["analysis"].temperature
    assert controller.last_recommendation.profile == "analysis"
    records = controller.outcome_store.load()
    assert records[-1].total_tokens == 12 and records[-1].latency_ms >= 0
    assert all(o.cost is None for o in records)
    assert "write code" not in controller.outcome_store.path.read_bytes().decode(errors="ignore")


@pytest.mark.asyncio
async def test_unqualified_model_cannot_learn_or_adapt(tmp_path):
    llm = llm_for(tmp_path, qualified=False)
    for i in range(6):
        run = OutcomeRun(llm.inference, "code")
        await llm.respond(messages=[{"role": "user", "content": "code"}])
        run.verdict = ("success", True, "none")
        run.finish()
    assert all(not o.verified for o in llm.inference.outcome_store.load())
    assert llm.inference.last_recommendation is None


@pytest.mark.asyncio
async def test_background_and_mixed_requests_do_not_get_task_credit(tmp_path):
    llm = llm_for(tmp_path)
    run = OutcomeRun(llm.inference, "code")
    async def background():
        assert current_run() is None
        await llm.respond(messages=[{"role": "user", "content": "reflection"}])
    await asyncio.create_task(background())
    assert not run.samples
    await llm.respond(messages=[{"role": "user", "content": "code"}])
    await llm.respond(messages=[{"role": "user", "content": "code"}], temperature=0.4)
    run.verdict = ("success", True, "none")
    run.finish()
    assert len(llm.inference.outcome_store.load()) == 2
    assert all(not o.verified for o in llm.inference.outcome_store.load())


@pytest.mark.asyncio
async def test_completion_prose_does_not_verify_success(tmp_path):
    llm = llm_for(tmp_path)
    run = OutcomeRun(llm.inference, "hello")
    trace = SimpleNamespace(objective="hello", status="completed", tool_calls=[])
    run.trace_finished(trace)
    assert run.verdict == ("unverified", False, "none")
    run.finish()


def test_feedback_on_tool_failure_does_not_blame_model(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store, "analysis", outcome="unverified", failure_domain="tool")
    for i in range(5):
        store.record_feedback(f"analysis-{i}", False)
    assert store.model_adjustment(task_kind="coding", context=H, model_fingerprint=H,
        server_profile_fingerprint=H, edition="full") == 0


@pytest.mark.asyncio
async def test_agent_run_records_contract_proof_and_handles_feedback_without_generation(tmp_path):
    from v_core.agent import Agent
    from v_core.autonomy.agent_trace import AgentTaskTrace
    from v_core.autonomy.task_contract import TaskContract
    llm = llm_for(tmp_path / "outcomes")
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = SimpleNamespace()
    async def turn(prompt, on_token=None):
        trace = AgentTaskTrace(tmp_path / "traces", prompt)
        trace.set_requirements(TaskContract(required_tools=("probe",)).to_dict())
        trace.tool_calls = [{"tool": "probe", "status": "succeeded"}]
        response = await llm.respond(messages=[{"role": "user", "content": prompt}])
        agent._finish_agent_trace(trace, response.content)
        return response.content
    agent._run_turn = turn
    assert await agent.run("code") == "ok"
    outcomes = llm.inference.outcome_store.load()
    assert len(outcomes) == 1 and outcomes[0].outcome == "success"
    assert "saved" in await agent.run("/feedback good")
    assert len(llm.client.chat.completions.requests) == 1
    assert len(llm.inference.outcome_store.load()) == 2


@pytest.mark.asyncio
async def test_streaming_records_elapsed_time_and_actual_preset(tmp_path):
    llm = llm_for(tmp_path)
    class Stream:
        closed = False
        def __aiter__(self):
            return self.chunks()
        async def chunks(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])
        async def close(self):
            self.closed = True
    stream = Stream()
    async def create(**kwargs):
        return stream
    llm.client.chat.completions.create = create
    run = OutcomeRun(llm.inference, "code")
    result = "".join([part async for part in llm.stream(messages=[{"role": "user", "content": "code"}])])
    run.finish()
    assert result == "ok" and stream.closed
    item = llm.inference.outcome_store.load()[0]
    assert item.request == asdict(PUBLIC_INFERENCE_PROFILES["coding"])
    assert item.total_tokens is None and item.latency_ms >= 0


@pytest.mark.asyncio
async def test_provider_error_is_measured_as_environment_failure(tmp_path):
    llm = llm_for(tmp_path)
    async def create(**kwargs):
        raise ConnectionError("offline")
    llm.client.chat.completions.create = create
    run = OutcomeRun(llm.inference, "code")
    with pytest.raises(ConnectionError):
        await llm.respond(messages=[{"role": "user", "content": "code"}])
    run.finish()
    item = llm.inference.outcome_store.load()[0]
    assert not item.verified and item.failure_domain == "environment"


@pytest.mark.asyncio
async def test_router_uses_outcomes_but_stale_card_cannot_be_resurrected(tmp_path, monkeypatch):
    from v_core.model_loader import (ModelProfile, ModelLoaderStore, LoaderState, ModelQualificationCard,
        ModelRouteCandidate, ModelRouter, MODEL_CAPABILITIES, QualificationProbeResult, RoutedModelRuntime,
        model_file_fingerprint, model_profile_fingerprint)
    import v_core.model_loader.routed_runtime as module
    profiles, cards = {}, {}
    for name in ("a", "b"):
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(b"GGUF" + name.encode() * 16)
        profile = ModelProfile(model_path=str(path), alias=name)
        key = str(path)
        profiles[key] = profile
        cards[key] = ModelQualificationCard(model_path=key, model_fingerprint=model_file_fingerprint(path),
            profile_fingerprint=model_profile_fingerprint(profile), qualified_at="2026-09-17T00:00:00Z",
            harness_version=9, capabilities={k: 90 for k in MODEL_CAPABILITIES},
            probes=(QualificationProbeResult(name="fixture", score=90, passed=True, latency_ms=1, output_digest=H),))
    a, b = profiles
    root = tmp_path / "runtime"
    state = LoaderState(profiles=profiles, qualifications=cards, routing_enabled=True,
                        routing_model_paths=[a, b], last_model_path=a)
    ModelLoaderStore(root).save(state)
    llm = llm_for(tmp_path / "outcomes")
    async def reconfigure():
        pass
    llm.reconfigure = reconfigure
    class Session:
        def __init__(self, profile):
            self.profile = profile
            self.stopped = False
        async def stop(self):
            self.stopped = True
    started = []
    async def start(binary, profile, runtime_root, status):
        started.append(profile.model_path)
        return Session(profile)
    monkeypatch.setattr(module, "start_llama_server", start)
    monkeypatch.setattr(module, "find_llama_server", lambda _: Path("/fake/server"))
    runtime = RoutedModelRuntime(Session(profiles[a]), root, llm, status=lambda _: None)
    prompt = "write code"
    for i in range(2):
        llm.inference.outcome_store.append(observation(str(i), outcome="failure", context=task_context(prompt),
            model_fingerprint=cards[a].model_fingerprint, server_profile_fingerprint=cards[a].profile_fingerprint))
    result = await runtime.ensure_for(prompt, task_kind="coding")
    assert result.switched and result.active_model_path == b and started == [b]
    assert dict(result.decision.outcome_adjustments)[a] == -10
    # Current model bytes change: history cannot unlock its now-stale card.
    Path(b).write_bytes(b"GGUFchanged")
    result = await runtime.ensure_for(prompt, task_kind="coding")
    assert result.active_model_path == a
    assert result.eligible_candidates == 1


@pytest.mark.asyncio
async def test_invalid_merged_server_profile_never_stops_active_process(tmp_path):
    from v_core.model_loader import ModelProfile, ModelLoaderStore, LoaderState, RoutedModelRuntime
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF1234")
    profile = ModelProfile(model_path=str(model), alias="test", batch_size=1024, ubatch_size=512)
    class Session:
        stopped = False
        async def stop(self):
            self.stopped = True
    session = Session()
    session.profile = profile
    llm = llm_for(tmp_path / "outcomes")
    root = tmp_path / "runtime"
    ModelLoaderStore(root).save(LoaderState(profiles={str(model): profile}))
    runtime = RoutedModelRuntime(session, root, llm, status=lambda _: None)
    llm.inference.configure_full(server={"batch_size": 128})
    result = await runtime.ensure_for("write code")
    assert result.failures and "before restart" in result.failures[0]
    assert not session.stopped


def test_quarantined_default_is_not_silently_retried_without_safe_alternative(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store, "coding", n=2, outcome="failure")
    result = recommend(store)
    assert not result.allowed
    assert "no safe fallback" in result.reason


def test_corrupted_scope_index_cannot_mix_model_versions(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store)
    with sqlite3.connect(store.path) as db:
        raw = json.loads(db.execute("SELECT data FROM outcomes WHERE id=1").fetchone()[0])
        raw["model_fingerprint"] = J
        db.execute("UPDATE outcomes SET data=? WHERE id=1", (json.dumps(raw),))
    assert recommend(store).profile == "coding"
    assert store.error


@pytest.mark.asyncio
async def test_corrupt_model_identity_disables_learning_without_breaking_response(tmp_path):
    from v_core.model_loader.storage import ModelLoaderStorageError
    llm = llm_for(tmp_path)
    def broken_identity():
        raise ModelLoaderStorageError("corrupt loader")
    llm.inference.identity_provider = broken_identity
    run = OutcomeRun(llm.inference, "code")
    assert (await llm.respond(messages=[{"role": "user", "content": "code"}])).content == "ok"
    run.finish()
    assert llm.inference.outcome_store.load() == []


def test_first_promotion_reason_and_saved_audit_explain_actual_selection(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store)
    result = recommend(store)
    assert result.profile == "analysis" and "proven configuration" in result.reason
    audit = InferenceOutcomeStore(tmp_path).inspect()["last_decision"]
    assert audit["profile"] == "analysis"
    assert audit["request"] == asdict(PUBLIC_INFERENCE_PROFILES["analysis"])
    assert audit["model_fingerprint"] == H


@pytest.mark.asyncio
async def test_blocked_tool_trace_preserves_domain_for_later_feedback(tmp_path):
    llm = llm_for(tmp_path)
    run = OutcomeRun(llm.inference, "code")
    run.trace = SimpleNamespace(status="blocked", tool_calls=[{"status": "failed"}], objective="code")
    await llm.respond(messages=[{"role": "user", "content": "code"}])
    run.finish()
    item = llm.inference.outcome_store.load()[0]
    assert item.failure_domain == "tool" and not item.verified
    assert llm.inference.outcome_store.record_feedback(run.task_id, False).failure_domain == "tool"


@pytest.mark.asyncio
async def test_cancelled_turn_does_not_become_provider_or_model_failure(tmp_path):
    llm = llm_for(tmp_path)
    run = OutcomeRun(llm.inference, "code")
    async def create(**kwargs):
        raise asyncio.CancelledError()
    llm.client.chat.completions.create = create
    with pytest.raises(asyncio.CancelledError):
        await llm.respond(messages=[{"role": "user", "content": "code"}])
    run.finish(domain="cancelled")
    item = llm.inference.outcome_store.load()[0]
    assert item.failure_domain == "cancelled" and not item.verified


@pytest.mark.asyncio
async def test_real_intent_classifier_is_auxiliary_not_a_second_task_configuration(tmp_path):
    from v_core.autonomy.intent import MultilingualIntentRouter
    llm = llm_for(tmp_path)
    run = OutcomeRun(llm.inference, "code")
    # The actual classifier can retry invalid classifier JSON. Those calls must
    # neither receive task credit nor prevent attribution of the task response.
    await MultilingualIntentRouter(llm).classify("write code")
    assert not run.samples
    assert current_run() is run
    await llm.respond(messages=[{"role": "user", "content": "code"}])
    run.verdict = ("success", True, "none")
    run.finish()
    assert len(llm.inference.outcome_store.load()) == 1
    assert llm.inference.outcome_store.load()[0].verified


def test_bad_configuration_does_not_demote_healthy_model_configuration(tmp_path):
    store = InferenceOutcomeStore(tmp_path)
    seed(store, "coding", n=5)
    seed(store, "analysis", n=5)
    store.append(observation("bad-analysis-1", "analysis", "failure"))
    store.append(observation("bad-analysis-2", "analysis", "failure"))

    assert store.model_adjustment(
        task_kind="coding",
        context=H,
        model_fingerprint=H,
        server_profile_fingerprint=H,
        edition="full",
    ) >= 0



def test_full_balanced_uses_model_fallback_but_public_and_named_presets_do_not():
    full = InferenceController(edition_name="full")
    full.bind_model_defaults(temperature=0.44, top_p=0.77)
    assert full.parameters().temperature == 0.44
    assert full.parameters().top_p == 0.77
    full.select_profile("coding")
    assert full.parameters() == PUBLIC_INFERENCE_PROFILES["coding"]

    public = InferenceController(edition_name="public")
    public.bind_model_defaults(temperature=0.44, top_p=0.77)
    assert public.parameters() == PUBLIC_INFERENCE_PROFILES["balanced"]



@pytest.mark.asyncio
@pytest.mark.parametrize("error_text,domain", [
    ("failed to parse tool call arguments as JSON", "model"),
    ("tools unsupported by template", "environment"),
])
async def test_native_tool_fallback_attributes_only_generated_errors(tmp_path, error_text, domain):
    import httpx
    from openai import BadRequestError
    llm = llm_for(tmp_path)
    completion = llm.client.chat.completions
    success = completion.create
    count = 0
    async def create(**request):
        nonlocal count
        count += 1
        if count == 1:
            raise BadRequestError(error_text, response=httpx.Response(400,
                request=httpx.Request("POST", "http://localhost/v1/chat/completions")), body={})
        return await success(**request)
    completion.create = create
    run = OutcomeRun(llm.inference, "code")
    try:
        response = await llm.respond(messages=[{"role": "user", "content": "code"}],
            tools=[{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}])
        assert response.content == "ok" and count == 2
        assert run.samples[0][0].failure_domain == domain
        assert run.samples[0][0].verified == (domain == "model")
    finally:
        run.finish()
    records = llm.inference.outcome_store.load()
    assert all(record.outcome != "success" for record in records)
    assert any(record.failure_domain == domain for record in records)


@pytest.mark.asyncio
async def test_unmeasurable_tool_failure_cannot_relabel_previous_request(tmp_path):
    import httpx
    from openai import BadRequestError
    llm = llm_for(tmp_path)
    run = OutcomeRun(llm.inference, "code")
    try:
        await llm.respond(messages=[{"role": "user", "content": "code"}])
        llm.inference.identity_provider = lambda: None
        success = llm.client.chat.completions.create
        count = 0
        async def create(**request):
            nonlocal count
            count += 1
            if count == 1:
                raise BadRequestError("failed to parse tool call arguments as JSON",
                    response=httpx.Response(400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")), body={})
            return await success(**request)
        llm.client.chat.completions.create = create
        await llm.respond(messages=[{"role": "user", "content": "code"}],
            tools=[{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}])
        assert len(run.samples) == 1
        assert run.samples[0][0].failure_domain == "none"
        assert not run.samples[0][0].verified
    finally:
        run.finish()
    assert all(item.outcome == "unverified" for item in llm.inference.outcome_store.load())
