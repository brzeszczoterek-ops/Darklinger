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

from .chat import ChatMixin
from .tasks import TasksMixin
from .runner import RunnerMixin
from .reports import ReportsMixin
from .web_discovery import WebDiscoveryMixin
from .prompts import PromptsMixin
from .tools_handler import ToolsHandlerMixin
from .memory import MemoryMixin


class _ContinueAgentBatch(Exception):
    """Internal control flow for an owner-authorized continuous task."""


class Agent(
    ChatMixin,
    TasksMixin,
    RunnerMixin,
    ReportsMixin,
    WebDiscoveryMixin,
    PromptsMixin,
    ToolsHandlerMixin,
    MemoryMixin,
):

    MAX_AGENT_STEPS = 32
    MAX_STAGNANT_SUCCESSFUL_CALLS = 6
    READ_ONLY_TOOL_NAMES = frozenset(
        {
            "read_file",
            "file_read",
            "image_metadata",
            "image_analyze",
            "document_read", "audio_transcribe", "audio_analyze", "perception_status",
            "list_directory",
            "directory_tree",
            "get_file_info",
            "search_files",
            "memory_recall",
            "learning_list_artifacts",
            "learning_recall_failures",
            "runtime_review_task",
        }
    )
    READ_ONLY_WEB_TOOL_NAMES = frozenset(
        {
            "web_read",
            "web_search",
            "browser_navigate",
            "browser_find",
            "browser_snapshot",
            "browser_vision",
            "full_tor_search",
            "full_tor_fetch",
            "full_tor_inventory",
            "full_tor_browser_inventory",
            "full_tor_browser_close",
        }
    )
    NETWORK_TOOL_NAMES = frozenset(
        {
            "browser_vision",
            "web_read",
            "web_search",
            "browser_navigate",
            "browser_snapshot",
            "browser_find",
            "browser_click",
            "browser_press_key",
            "browser_type",
            "full_tor_search",
            "full_tor_fetch",
            "full_tor_inventory",
            "full_tor_browser_inventory",
            "full_tor_browser_close",
        }
    )

    def __init__(
        self,
        config: Config,
        memory: MemoryEngine,
        llm: LLM | None = None,
        phase_router: Callable[[str, str], Awaitable[Any]] | None = None,
        response_fallback_router: (
            Callable[[str, str], Awaitable[Any]] | None
        ) = None,
    ):
        self.config = config

        self.llm = llm or LLM()
        set_edition = getattr(self.llm, "set_edition", None)
        if callable(set_edition):
            set_edition(
                getattr(getattr(config, "edition", None), "name", "public")
            )
        from .inference import InferenceController
        from .model_loader.performance import InferenceOutcomeStore
        controller = getattr(getattr(self, "llm", None), "inference", None)
        if isinstance(controller, InferenceController):
            controller.outcome_store = InferenceOutcomeStore(config.model_runtime_root / "outcomes")
        self.phase_router = phase_router
        self.response_fallback_router = response_fallback_router

        self.intent_router = MultilingualIntentRouter(self.llm)

        self.context_window = ContextWindowManager()

        self.tools = MCPTools(config)
        bind_inference = getattr(
            self.tools.edition_extension,
            "bind_inference_controller",
            None,
        )
        inference = getattr(self.llm, "inference", None)
        if callable(bind_inference) and inference is not None:
            bind_inference(inference)

        self.dispatcher = ToolDispatcher(
            llm=self.llm,
            tools=self.tools,
        )

        self.memory = memory

        self.capabilities = CapabilityDispatcher()

        self.research = ResearchTask(
            self.dispatcher,
        )

        self.persona = PersonaRuntime(
            identity=IdentityKernel(),
            voice=VoiceProfile(),
        )

        self._memory_tasks: set[asyncio.Task[Any]] = set()
        self._agent_trace_root = config.autonomy_root / "interactive"
        recovered = AgentTaskTrace.recover_stale_running(self._agent_trace_root)
        for task_id in recovered:
            print(f"[Task {task_id}] recovered as interrupted")
        self._last_execution_context = AgentTaskTrace.latest_context(
            self._agent_trace_root
        )

    async def run(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:

        from .model_loader.outcome_runtime import OutcomeRun
        from .inference import InferenceController
        controller = getattr(getattr(self, "llm", None), "inference", None)
        store = controller.outcome_store if isinstance(controller, InferenceController) else None
        if prompt.strip().startswith("/feedback") or prompt.strip() == "/inference-memory":
            if store is None:
                return "Inference outcome memory is unavailable."
            if prompt.strip() == "/inference-memory":
                return json.dumps(controller.status(), ensure_ascii=False, default=str)
            parts = prompt.strip().split()
            if len(parts) not in {2, 3} or parts[1] not in {"good", "bad"}:
                return "Usage: /feedback good|bad [task_id]"
            try:
                feedback = store.record_feedback(parts[2] if len(parts) == 3 else controller.last_task_id, parts[1] == "good")
                return f"Explicit feedback saved for task {feedback.task_id}: {feedback.outcome}."
            except ValueError as error:
                return str(error)
        if isinstance(controller, InferenceController):
            from .model_loader.router import classify_model_phase
            controller.begin_turn(classify_model_phase(prompt))
        run = OutcomeRun(controller, prompt) if store is not None else None
        domain = "none"
        try:
            return await self._run_turn(prompt, on_token)
        except asyncio.CancelledError:
            domain = "cancelled"
            raise
        except Exception:
            domain = "environment"
            raise
        finally:
            if run is not None:
                run.finish(domain=domain)
            # MCP stdio contexts are task-affine.  The browser session is
            # opened while executing this turn, so it must be closed by this
            # same task rather than deferred to application shutdown.
            close_browser = getattr(self.tools, "close_browser_session", None)
            if callable(close_browser):
                await close_browser()

    async def _run_turn(
        self,
        prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:

        prompt = prompt.strip()
        supervisor = getattr(getattr(self, "tools", None), "supervisor", None)
        if supervisor is not None and prompt == "/jobs":
            return json.dumps(supervisor.snapshot(), ensure_ascii=False)
        if supervisor is not None and prompt.startswith("/job "):
            job_id = prompt.split(maxsplit=1)[1].strip()
            job = supervisor.get(job_id)
            return json.dumps({"job": supervisor.status(job_id),
                               "result": str(getattr(job.outcome, "result", ""))}, ensure_ascii=False)
        if supervisor is not None and prompt.startswith("/stop-job "):
            return json.dumps(supervisor.cancel(prompt.split(maxsplit=1)[1].strip()), ensure_ascii=False)

        if not prompt:
            return ""

        # A turn-level language request must never leak into the next turn.
        self._response_language_override = ""
        self._memory_recall_requested = False
        self._memory_recall_query = ""

        # Visible user work always has priority over best-effort background
        # reflection from the previous turn.
        await self.cancel_background_memory()

        begin_interaction = getattr(self.tools, "begin_interaction", None)
        if callable(begin_interaction):
            begin_interaction(uuid4().hex, prompt)

        owner_control = self._owner_control_command(prompt)
        if owner_control is not None:
            controlled = await self._handle_owner_control(
                owner_control,
                on_token,
            )
            if controlled is not None:
                return controlled

        if self._photo_routing_prompt(prompt) != prompt or TaskContract.from_prompt(prompt).required_media_paths or (photo_paths(prompt) and photo_location_request(prompt)):
            return await self._run_agent_loop(prompt, on_token)

        # A short answer to V's immediately preceding question belongs to the
        # visible conversation.  Do not send an isolated "yes"/"of course"
        # through the expensive semantic task classifier without the question
        # it answers.  This rule is structural rather than language-specific:
        # the current turn must be short, non-executable, and follow an actual
        # question in the durable dialogue ledger.
        short_dialogue_question = self._short_dialogue_question(prompt)
        if short_dialogue_question:
            if (
                self._claims_active_chat_work(short_dialogue_question)
                or unsupported_execution_claims(short_dialogue_question, ())
            ):
                trace = self._start_agent_trace(prompt)
                answer = (
                    "No. There are no results to show, Boss. The previous reply "
                    "claimed work without matching tool evidence. That was a "
                    "hallucinated status, not executed work."
                )
                if trace is not None:
                    trace.record_event(
                        "conversation_false_offer_rejected",
                        {"reason": "previous_reply_had_no_tool_evidence"},
                    )
                execution = self._finish_agent_trace(trace, answer)
                await self._remember_task(prompt, answer, execution=execution)
                if on_token is not None:
                    on_token(answer)
                return answer
            return await self._run_light_chat(
                prompt,
                on_token,
                trace=self._start_agent_trace(prompt),
            )

        capability = self.capabilities.dispatch(
            prompt
        )

        #
        # Explicit tool path
        #

        if capability == CapabilityDispatcher.TOOLS:

            return await self._run_tool_task(
                prompt,
                on_token,
            )

        #
        # Research path
        #

        if capability == CapabilityDispatcher.RESEARCH:

            return await self._run_research_task(
                prompt,
                on_token,
            )

        if TaskContract.asks_about_creation_capability(prompt):
            return await self._run_light_chat(
                prompt,
                on_token,
                trace=self._start_agent_trace(prompt),
            )

        if self._is_light_conversation(prompt):
            return await self._run_light_chat(
                prompt,
                on_token,
                trace=self._start_agent_trace(prompt),
            )

        #
        # Agent / chat path
        #

        return await self._run_agent_loop(
            prompt,
            on_token,
        )

    async def close(self) -> None:
        await self.cancel_background_memory()
        close_supervisor = getattr(self.tools, "close_supervisor", None)
        if callable(close_supervisor):
            await close_supervisor()
        try:
            await self.tools.close_browser_session()
        finally:
            extension = getattr(self.tools, "edition_extension", None)
            close = getattr(extension, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def cancel_background_memory(self) -> None:
        tasks = list(getattr(self, "_memory_tasks", set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)



__all__ = ['Agent']
