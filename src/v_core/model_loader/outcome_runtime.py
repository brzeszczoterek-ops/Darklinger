"""Task-scoped measurement and conservative attribution at runtime boundaries."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import asdict, replace
from time import perf_counter
from uuid import uuid4

from .performance import InferenceOutcome, fingerprint, task_context

_auxiliary = ContextVar("inference_auxiliary", default=False)


@contextmanager
def auxiliary_inference():
    """Exclude explicitly identified classifier calls from task attribution."""
    token = _auxiliary.set(True)
    try:
        yield
    finally:
        _auxiliary.reset(token)


_active: ContextVar[OutcomeRun | None] = ContextVar("inference_outcome_run", default=None)


def current_run():
    run = _active.get()
    return run if not _auxiliary.get() and run is not None and run.owner is asyncio.current_task() and not run.closed else None


def request_snapshot(request):
    from ..inference import InferenceParameters
    values = asdict(InferenceParameters())
    for key in values:
        if key in request:
            values[key] = request[key]
        elif key in request.get("extra_body", {}):
            values[key] = request["extra_body"][key]
    context = fingerprint({"version": 1, "max_tokens": request.get("max_tokens"),
        "tools": request.get("tools", []), "tool_choice": request.get("tool_choice"),
        "response_format": request.get("response_format"), "stream": bool(request.get("stream")),
        "input_bucket": min(20, sum(len(str(m.get("content", ""))) for m in request.get("messages", [])).bit_length() // 3)})
    return values, context


class OutcomeRun:
    def __init__(self, controller, prompt: str):
        self.controller = controller
        self.store = controller.outcome_store
        self.task_id = uuid4().hex
        self.context = task_context(prompt)
        self.owner = asyncio.current_task()
        self.closed = False
        self.samples = []
        self.attributable = True
        self.trace = None
        self.verdict = ("unverified", False, "none")
        self.token = _active.set(self)
        controller.last_task_id = self.task_id

    def snapshot(self, request):
        identity = self.controller.outcome_identity()
        if identity is None:
            self.attributable = False
            return None
        model, server, qualified = identity
        values, context = request_snapshot(request)
        profile = self.controller.active_profile
        applied = self.controller.last_recommendation
        if applied and applied.request == values:
            profile = applied.profile
        try:
            return InferenceOutcome(task_id=self.task_id, task_kind=self.controller.task_kind,
            context=self.context, edition=self.controller.edition_name,
            model_fingerprint=model, server_profile_fingerprint=server,
            profile=profile, request=values, request_context=context), qualified
        except (ValueError, TypeError):
            self.attributable = False
            return None

    def measure(self, snapshot, started, response=None, *, domain="none"):
        if snapshot is None:
            return
        item, qualified = snapshot
        usage = getattr(response, "usage", None)
        total = getattr(usage, "total_tokens", None)
        if type(total) is not int or total < 0:
            total = None
        item = replace(item, latency_ms=max(0, round((perf_counter() - started) * 1000)),
                       total_tokens=total, failure_domain=domain)
        self.samples.append((item, qualified))

    def reject_last(self, *, proven_provider_output: bool = False):
        if self.samples:
            item, qualified = self.samples[-1]
            if item.failure_domain == "none" or (proven_provider_output and item.failure_domain == "environment"):
                self.samples[-1] = (replace(item, outcome="failure", verified=True, failure_domain="model"), qualified)

    def trace_finished(self, trace):
        self.trace = trace
        from ..autonomy.task_contract import TaskContract
        if any(call.get("status") == "failed" for call in trace.tool_calls):
            self.verdict = ("unverified", False, "tool")
            return
        # Completion itself and model prose are NOT verifiers. Only a nonempty
        # deterministic contract with runtime tool evidence can verify execution.
        contract = (TaskContract.from_dict(trace.requirements) if getattr(trace, "requirements", None)
                    else TaskContract.from_prompt(trace.objective))
        empty_missing = contract.unmet([])
        if trace.status == "completed" and empty_missing and not contract.unmet(trace.tool_calls):
            self.verdict = ("success", True, "none")

    def finish(self, *, domain="none"):
        if self.trace is not None:
            self.trace_finished(self.trace)
        self.closed = True
        _active.reset(self.token)
        grouped = {}
        for item, qualified in self.samples:
            grouped.setdefault((item.scope, item.config_fingerprint), []).append((item, qualified))
        for samples in grouped.values():
            base = samples[-1][0]
            verdict = self.verdict if len(grouped) == 1 and self.attributable else ("unverified", False, "none")
            sample_domain = domain if domain != "none" else next((o.failure_domain for o, _ in samples if o.failure_domain in {"environment", "tool", "cancelled"}), "none")
            if sample_domain != "none":
                verdict = ("unverified", False, sample_domain)
            elif any(o.outcome == "failure" for o, _ in samples):
                verdict = ("failure", True, "model")
            # We retain measurement without qualification, but cannot promote it.
            if not all(qualified for _, qualified in samples):
                verdict = ("unverified", False, verdict[2] if not verdict[1] else "none")
            totals = [o.total_tokens for o, _ in samples]
            self.store.append(replace(base, outcome=verdict[0], verified=verdict[1], failure_domain=verdict[2],
                latency_ms=sum(o.latency_ms or 0 for o, _ in samples),
                total_tokens=sum(totals) if all(t is not None for t in totals) else None))
