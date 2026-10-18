from __future__ import annotations

import asyncio
import ast
import copy
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal, TYPE_CHECKING
from urllib.parse import quote_plus, unquote, urldefrag, urljoin, urlparse, urlsplit, urlunparse, urlunsplit
from uuid import uuid4

from openai import APIStatusError

from ..autonomy import (
    AgentTaskTrace,
    ContextWindowManager,
    MultilingualIntentRouter,
    SemanticIntent,
    TaskContract,
)
from ..config import Config
from ..content_policy import render_content_policy
from ..creation_budget import (
    CREATION_ATTEMPT_TOOLS,
    CREATION_TOOLS,
    SOURCE_DRAFT_TOOL,
    TOTAL_ATTEMPT_LIMIT,
    TOTAL_SECONDS,
    evaluate as evaluate_creation_budget,
)
from ..generated_tool_contract import (
    GeneratedToolContract,
    GeneratedToolContractError,
    autonomous_web_traffic_contract,
    extract_generated_tool_contract,
    has_natural_fixture_candidate,
    source_generation_objective,
)
from ..response_preview import response_preview
from ..autonomy.local_read_scope import literal_paths, resolve_read_scope
from ..autonomy.search_reference import resolve_search_reference
from ..autonomy.research_facts import (
    capture_source,
    evidence_for_model,
    publisher,
    fact_missing,
    render_release_facts,
    catalog_target,
)
from ..tool_catalog_review import review_catalog
from ..tool_self_test import test_local_tools
from ..tools.image_metadata import literal_image_paths
from ..tools.media_input import image_paths as photo_paths, IMAGE_SUFFIXES
from ..perception.evidence import perception_inputs, next_media_request, records as media_records
from ..perception.reports import preserve_literal_report
from ..autonomy.photo_context import (
    owner_photo_context,
    photo_location_request,
    location_query,
    relevant_image_evidence,
)
from ..llm import LLM
from ..llm.llm import IncompleteGenerationError
from ..learning.source_builder import (
    MissingToolInputError,
    SourceBlueprintError,
    build_source_blueprint,
    repair_generated_source_argument_alias,
    repair_generated_source_json_wrapper,
)
from ..mcp_tools import MCPTools
from ..model_loader.router import classify_model_phase
from ..tool_dispatcher import ToolDispatcher
from ..memory.memory_engine import MemoryEngine
from ..memory.manager import clip_text
from ..memory.memoir import recalled_memoirs, record_memoir_execution
from ..capability_dispatcher import CapabilityDispatcher
from ..capabilities.research import ResearchTask
from ..capabilities.web_target import extract_web_target, extract_web_targets
from ..execution_claims import (
    claim_has_runtime_capability,
    tool_supports_claim,
    unsupported_execution_claims,
)
from ..persona.kernel import IdentityKernel
from ..persona.voice import (
    VoiceProfile,
    is_identity_inquiry,
    is_origin_inquiry,
    render_identity_response,
    render_origin_response,
    is_direct_greeting,
    render_greeting_response,
)
from ..persona.runtime import PersonaRuntime
from ..persona.context import PersonaContext
from ..persona.grounding import claims_physical_experience, claims_unobserved_runtime_status
from ..persona.language import (
    is_english,
    is_polish,
    contains_polish,
    detect_language_name,
    is_target_language,
    normalize_language_target,
    prompt_requests_language,
)


class TasksMixin:
    """Mixin providing tasks capabilities for Agent."""

    async def _run_tool_task(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:

        tool_result = await self.dispatcher.dispatch(
            prompt
        )

        if tool_result is None:
            return await self._run_agent_loop(
                prompt,
                on_token,
            )

        answer = str(
            tool_result
        )

        answer = await self._render_tool_result(
            prompt,
            answer,
            on_token,
        )

        await self._remember_task(
            prompt,
            answer,
        )

        return answer

    async def _run_research_task(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:

        trace = self._start_agent_trace(prompt)
        contract = TaskContract.from_prompt(prompt)
        if trace is not None:
            trace.set_requirements(contract.to_dict())
        research_calls: list[dict[str, Any]] = []

        async def run_browser_tool(tool: str, arguments: dict) -> str:
            sequence: int | None = None
            if trace is not None:
                sequence = trace.tool_started(tool, arguments)
                print(f"[Task {trace.task_id}] tool {sequence} started: {tool}")
            error: str | None = None
            try:
                result = await self.tools.browser_call(tool, arguments)
            except BaseException as exception:
                error = f"{type(exception).__name__}: {exception}"
                if trace is not None and sequence is not None:
                    trace.tool_finished(sequence, error, error=error)
                    print(f"[Task {trace.task_id}] tool {sequence} failed: {tool}")
                raise
            if trace is not None and sequence is not None:
                trace.tool_finished(sequence, result)
                print(f"[Task {trace.task_id}] tool {sequence} completed: {tool}")
            research_calls.append(
                {
                    "tool": tool,
                    "arguments": arguments,
                    "status": "succeeded",
                    "result_excerpt": str(result)[:2_000],
                }
            )
            return result

        candidate_passed_stream_guards = False

        async def answerer(messages: list[dict[str, str]]) -> str:
            nonlocal candidate_passed_stream_guards
            # Research reports remain private until the full candidate passes
            # completion, evidence, language, and voice checks. Tool progress is
            # still visible, but a trailing "I'll extract it next" must never be
            # streamed and then recorded as completed work.
            answer, candidate_passed_stream_guards = (
                await self._stream_guarded_english(
                    messages,
                    lambda _chunk: None,
                    max_tokens=2048,
                    detect_sanitized_contempt=True,
                )
            )
            return answer

        try:
            answer = await self.research.run(
                prompt,
                persona_prompt=self._build_system_prompt(
                    prompt,
                    agent_mode=False,
                ),
                persona_examples=self.persona.example_messages(),
                answerer=answerer if on_token is not None else None,
                tool_runner=run_browser_tool,
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            if trace is not None:
                trace.stop("interaction interrupted")
                print(f"[Task {trace.task_id}] stopped")
            raise
        except BaseException as error:
            direct_target = extract_web_target(prompt)
            if direct_target and not contract.requires_web_discovery:
                answer = self._direct_navigation_failure_answer(
                    direct_target,
                    f"{type(error).__name__}: {error}",
                )
                if trace is not None:
                    trace.record_event(
                        "direct_navigation_failure_reported",
                        {
                            "url": direct_target,
                            "tool_sequence": (
                                trace.tool_calls[-1].get("sequence")
                                if trace.tool_calls
                                else None
                            ),
                            "retried": False,
                        },
                    )
                    trace.fail("direct navigation failed after one attempt")
                    self._last_execution_context = AgentTaskTrace.latest_context(
                        trace.root
                    )
                    print(f"[Task {trace.task_id}] failed")
                    evidence = trace.evidence()
                else:
                    evidence = None
                await self._remember_task(prompt, answer, execution=evidence)
                if on_token is not None:
                    on_token(answer)
                return answer
            if trace is not None:
                trace.fail(f"{type(error).__name__}: {error}")
                self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
                print(f"[Task {trace.task_id}] failed")
            answer = (
                "The browser task crashed. The runtime returned "
                f"`{type(error).__name__}: {error}`. Nothing is running in "
                "the background, and I'm not inventing a prettier result."
            )
            evidence = trace.evidence() if trace is not None else None
            await self._remember_task(prompt, answer, execution=evidence)
            if on_token is not None:
                on_token(answer)
            return answer

        if not candidate_passed_stream_guards:
            answer = await self._enforce_english(
                [{"role": "user", "content": prompt}],
                answer,
            )

        missing_evidence = [
            *contract.unmet(research_calls),
            *contract.answer_issues(answer, research_calls),
        ]
        if missing_evidence:
            answer = self._incomplete_task_answer(missing_evidence, [])
            evidence = self._block_agent_trace(
                trace,
                "required research evidence was not produced: "
                + ", ".join(missing_evidence),
            )
            await self._remember_task(
                prompt,
                answer,
                execution=evidence,
            )
            if on_token is not None:
                on_token(answer)
            return answer

        if self._claims_unverified_work(answer):
            answer = (
                "I gathered real browser evidence, but the extraction didn't "
                "finish. Nothing is still running in the background, and I'm "
                "not dressing a promise up as completed work."
            )
            evidence = self._block_agent_trace(
                trace,
                "research candidate promised unfinished follow-up work",
            )
            await self._remember_task(
                prompt,
                answer,
                execution=evidence,
            )
            if on_token is not None:
                on_token(answer)
            return answer

        unsupported = unsupported_execution_claims(
            answer,
            self._successful_trace_tools(trace),
        )
        if unsupported:
            answer = self._unverified_execution_answer(unsupported)
            evidence = self._block_agent_trace(
                trace,
                "research candidate claimed execution without matching tool evidence: "
                + ", ".join(unsupported),
            )
            await self._remember_task(
                prompt,
                answer,
                execution=evidence,
            )
            if on_token is not None:
                on_token(answer)
            return answer

        evidence = self._finish_agent_trace(trace, answer)

        await self._remember_task(
            prompt,
            answer,
            execution=evidence,
        )

        if on_token is not None:
            on_token(answer)

        return answer

    def _photo_routing_prompt(self, prompt: str) -> str:
        session = getattr(getattr(self, "memory", None), "session", None)
        events = getattr(session, "events", [])
        start = max(0, int(getattr(session, "_current_session_start", 0)))
        owners = [{"role": "user", "content": e.data.get("task", "")}
                  for e in events[start:] if e.event_type == "task"]
        return owner_photo_context(prompt, owners)

