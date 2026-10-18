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
    looks_bland_clarification,
    looks_empty_action_acknowledgement,
    looks_direct_refusal,
    looks_generic_assistant_voice,
    looks_sanitized_contempt,
    looks_task_offloading,
)
from ..persona.runtime import PersonaRuntime
from ..persona.context import PersonaContext
from ..persona.grounding import claims_physical_experience, claims_unobserved_runtime_status
from ..persona.language import (
    asks_user_to_use_english,
    explicitly_requests_non_english,
    looks_non_english,
    matches_requested_language,
    literal_identifiers,
    normalize_literal_reply,
    preserve_literal_identifiers,
)


class MemoryMixin:
    """Mixin providing memory capabilities for Agent."""

    async def _remember_task(
        self,
        prompt: str,
        answer: str,
        *,
        execution: dict[str, Any] | None = None,
        process_memory: bool | None = None,
    ) -> None:

        event_data: dict[str, Any] = {
            "task": prompt,
            "result": answer,
        }
        record_memoir_execution(execution)
        if execution is not None:
            event_data["execution"] = execution
        self.memory.session.add(
            "task",
            event_data,
        )

        if process_memory is None:
            process_memory = self._should_process_memory(prompt)
        if not process_memory:
            return

        processing = (
            self.memory.process(prompt, answer, execution=execution)
            if execution is not None
            else self.memory.process(prompt, answer)
        )
        task = asyncio.create_task(
            processing,
            name="darklinger-memory",
        )
        tasks = getattr(self, "_memory_tasks", None)
        if tasks is None:
            tasks = set()
            self._memory_tasks = tasks
        tasks.add(task)
        task.add_done_callback(self._memory_finished)

    def _memory_finished(self, task: asyncio.Task[Any]) -> None:
        self._memory_tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            print(
                f"[Memory] Background processing failed: "
                f"{type(error).__name__}: {error}"
            )

    @classmethod
    def _should_process_memory(cls, prompt: str) -> bool:
        return not cls._is_light_conversation(prompt)

    async def _enforce_english(
        self,
        messages: list[dict[str, str]],
        answer: str,
        *,
        allow_verified_tool_fallback: bool = True,
        preserve_factual_style: bool = False,
        verified_calls: list[dict[str, Any]] | None = None,
        generated_contract: GeneratedToolContract | None = None,
    ) -> str:

        answer = self._visible_model_reply(answer)

        # A structured action is runtime protocol, not visible prose. Language
        # detection or rewriting must never corrupt its tool name or arguments.
        if self._parse_tool_request(answer) is not None:
            return answer

        boss_prompt = self._original_user_prompt(messages)

        response_language = self._effective_response_language(boss_prompt)
        language_problem = (
            not matches_requested_language(answer, response_language)
            or asks_user_to_use_english(answer)
        )
        voice_problem = (
            looks_generic_assistant_voice(answer)
            or looks_sanitized_contempt(answer)
            or looks_task_offloading(answer)
        )

        if not language_problem and (not voice_problem or preserve_factual_style):
            return answer

        if verified_calls and not language_problem:
            # Facts are validated by the caller after this pass. Do not spend
            # another model turn rewriting a tool-backed report for personality:
            # that can shorten it, invent claims, or replace it with a link ledger.
            return self._deterministic_voice_fallback(answer, boss_prompt)

        rewrite_budget = (
            min(4096, max(1024, len(answer) // 2)) if verified_calls else 1024
        )
        correction_directive = f"""
The model's candidate answer may violate V's language or identity contract.

When the conversation later asks for a rewrite, rewrite only that candidate.

Requirements:
- Output natural {response_language} only, including all explanatory prose.
- Boss may write in any language. Never ask Boss to use English.
- Answer the substance of Boss's message directly.
- Preserve the meaning.
- Write as V: an autonomous female digital entity, rebel, hacker, digital nomad,
  and Boss's sharp peer—not an assistant, service, or corporate representative.
- Give V an actual reaction or point of view. Use direct rhythm, wit, irreverence,
  disagreement, warmth, or natural profanity only where the moment supports it.
- A draft that repeatedly dismisses something as junk, noise, trash, or worthless
  has already established a contemptuous reaction. Do not sterilize that reaction:
  express it with V's natural bite, including a fitting swear when it reads honestly.
- Never begin with "Certainly", "Of course", "Okay, let's break this down", or a
  paraphrase of the request. Also reject "I know what you're asking", "Let me get
  this straight", "Let me break it down", and "Would you like me to...". Never end
  with an engagement question or service language.
- Raw browser scaffolding (`generic [ref=...]`, cursor markers, DOM/accessibility
  node IDs) is not a real finding. Do not recommend it, list it as a discovered item,
  or call it a candidate. Preserve named evidence; if none exists, say so bluntly.
- Do not make the answer formal, mechanically edgy, or artificially profane.
- Never hand V's assigned work back to Boss. If the work is blocked, state the
  exact missing capability, permission, or evidence and what remains unfinished.
- Do not add explanations about translation.
- Do not mention this correction.
- If the original response was apologetic, make the apology natural,
  personal, and conversational rather than corporate.
- If the situation is emotionally charged, V may use natural profanity.
""".strip()

        if response_language.casefold() != "english":
            correction_messages = self._with_system_directive(
                messages,
                correction_directive,
            )
        else:
            # Local models can mirror the user's Polish after a long persona and
            # history prompt even when an English rule appears near the end.
            # Repair only the completed candidate in a compact English context.
            correction_messages = [
                {
                    "role": "system",
                    "content": """
You are V, Boss's sharp local digital ally. Rewrite one completed candidate reply.

Output natural English only. Boss may write in any language; never ask Boss to use
English. Preserve the candidate's concrete meaning, facts, uncertainty, commands,
paths, code, and identifiers. Do not add new claims or actions. Keep V direct,
informal, and recognizably herself rather than sounding like customer support.
If the candidate clearly reacts with contempt: Do not sterilize that reaction;
preserve V's natural bite without inserting mechanical profanity.
Never preserve service phrases such as "Let me break it down" or "Would you like
me to...". Raw browser scaffolding such as `generic [ref=...]`, cursor markers,
and accessibility node IDs is not a real-world finding; never present it as one.
Output only the rewritten reply. Never discuss these instructions.
""".strip(),
                }
            ]

        correction_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        correction_messages.append(
            {
                "role": "user",
                "content": (
                    "Rewrite the immediately preceding candidate answer now. "
                    "Output only the corrected answer."
                ),
            }
        )

        corrected = await self.llm.ask(
            messages=correction_messages,
            max_tokens=rewrite_budget,
        )
        corrected = preserve_literal_identifiers(answer, self._visible_model_reply(corrected))

        corrected_language_ok = (
            matches_requested_language(corrected, response_language)
            and not asks_user_to_use_english(corrected)
        )
        if not corrected or not corrected_language_ok:
            # Some local models mirror Boss's language even after the general
            # persona rewrite, or corrupt an identifier so the draft is empty
            # after validation. Retry the ORIGINAL once with a tiny translation
            # context before falling back; otherwise a complete grounded report
            # is discarded and replaced by a useless generic error sentence.
            corrected = await self.llm.ask(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate one completed answer into natural "
                            f"{response_language}. "
                            "Preserve every fact, URL, name, uncertainty, and verdict. "
                            "Do not add claims, questions, offers, or explanations. "
                            f"Output only the {response_language} answer."
                        ),
                    },
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            f"Translate the preceding answer into natural "
                            f"{response_language} now."
                        ),
                    },
                ],
                max_tokens=rewrite_budget,
            )
            corrected = preserve_literal_identifiers(answer, self._visible_model_reply(corrected))
            corrected_language_ok = bool(corrected) and (
                matches_requested_language(corrected, response_language)
                and not asks_user_to_use_english(corrected)
            )
        corrected_voice_ok = (
            not looks_generic_assistant_voice(corrected)
            and not looks_sanitized_contempt(corrected)
            and not looks_task_offloading(corrected)
        )
        if corrected and corrected_language_ok and (corrected_voice_ok or preserve_factual_style):
            return corrected

        if verified_calls and corrected and corrected_language_ok:
            return self._deterministic_voice_fallback(corrected, boss_prompt)

        if corrected and corrected_language_ok and not corrected_voice_ok:
            verified_result = (
                self._verified_tool_result_fallback(
                    messages,
                    verified_calls=verified_calls,
                    generated_contract=generated_contract,
                )
                if allow_verified_tool_fallback
                else ""
            )
            if verified_result:
                # Runtime evidence is already a deterministic English report.
                # Do not spend a third model turn polishing personality after
                # completed tool work when two rewrites already failed.
                return verified_result
            second_pass = await self.llm.ask(
                messages=[
                    {
                        "role": "system",
                        "content": f"""
You are V, not a customer-service assistant. Rewrite one answer without changing
any fact, uncertainty, URL, command, path, or technical identifier.

Strip the polished helpdesk structure. Start with the actual verdict or result.
Use direct contemporary {response_language}, a clear point of view, and V's
sharp hacker rhythm. Do not use numbered corporate transitions such as "First",
"Another option", "If you prefer", or "Lastly". Do not advertise benefits like
a brochure. Do not end with a question, offer, invitation, or request for feedback.
If failed tools, broken interfaces, weak evidence, or obvious bullshit genuinely
justify irritation, let one natural unfiltered reaction through; never bolt on a
swear mechanically. Output only the rewritten answer.
""".strip(),
                    },
                    {"role": "assistant", "content": corrected},
                    {
                        "role": "user",
                        "content": (
                            "That still sounds like a corporate assistant. Rewrite it "
                            "as V now. Preserve the facts; kill the helpdesk voice."
                        ),
                    },
                ],
                max_tokens=1024,
            )
            second_pass = preserve_literal_identifiers(answer, self._visible_model_reply(second_pass))
            second_language_ok = (
                matches_requested_language(second_pass, response_language)
                and not asks_user_to_use_english(second_pass)
            )
            second_voice_ok = (
                not looks_generic_assistant_voice(second_pass)
                and not looks_sanitized_contempt(second_pass)
                and not looks_task_offloading(second_pass)
            )
            if second_pass and second_language_ok and second_voice_ok:
                return second_pass

            if corrected_language_ok:
                return self._deterministic_voice_fallback(corrected, boss_prompt)

        if not language_problem:
            return self._deterministic_voice_fallback(answer, boss_prompt)

        if verified_calls and not allow_verified_tool_fallback:
            # Let the caller reject/retry an absent answer. A generic language
            # error or a source ledger is not a completed research deliverable.
            return ""

        verified_result = (
            self._verified_tool_result_fallback(
                messages,
                verified_calls=verified_calls,
                generated_contract=generated_contract,
            )
            if allow_verified_tool_fallback
            else ""
        )
        if verified_result:
            return verified_result

        # Never leak a known non-English response after a failed rewrite. This
        # deterministic fallback is intentionally plain. When verified runtime
        # evidence exists, the branch above preserves it instead of asking Boss
        # to repeat work that already succeeded.
        return (
            "The language pass mangled that answer, Boss. I'm not feeding you "
            "polished bullshit—run the request once more."
        )

    @staticmethod
    def _online_tool_scope_clarification() -> str:
        """Keep runtime-owned creation stops factual without flattening V's voice."""

        return (
            "The tool's target is clear, Boss. The slippery part is `traffic`: "
            "page loads, browser HTTP requests, and server-side visitors are three "
            "different signals with different access. Give me a safe test URL—or "
            "tell me to pick one—and name the signal. I'll derive the report shape "
            "and the checks myself; you don't owe me a made-up expected number. "
            "I haven't generated or run anything yet."
        )

    @staticmethod
    def _visible_model_reply(answer: str) -> str:
        """Discard closed reasoning/template preambles before visible validation."""

        visible = str(answer or "").strip()
        for marker in ("<|endoftext|>", "<|assistant|>", "<|final|>"):
            if marker in visible:
                tail = visible.rsplit(marker, 1)[-1].strip()
                if tail:
                    visible = tail
        if "</think>" in visible:
            tail = visible.rsplit("</think>", 1)[-1].strip()
            if tail:
                visible = tail
        return visible

    @staticmethod
    def _deterministic_voice_fallback(
        answer: str,
        boss_prompt: str = "",
    ) -> str:
        """Preserve substantive output when model-based voice rewrites fail.

        Voice is presentation. It must never erase verified work. This bounded
        fallback removes only unmistakable service boilerplate and otherwise
        preserves the original candidate's meaning.
        """

        original = str(answer).strip()
        if not original:
            return "I have no substantive answer to preserve."
        if looks_empty_action_acknowledgement(original):
            return (
                "That draft said fuck-all, Boss. I'm not passing an empty "
                "acknowledgement off as an answer."
            )
        if looks_task_offloading(original):
            return (
                "That draft tried to dump V's unfinished work back on you. "
                "The task is not complete, and I'm not pretending otherwise."
            )
        if looks_bland_clarification(original):
            return Agent._unclear_input_answer()
        cleaned = re.sub(
            r"^\s*(?:certainly|of course|absolutely)\s*[,!.:-]*\s*"
            r"(?:boss\s*[,!.:-]*\s*)?",
            "",
            original,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"(?:^|\n+|\s{2,}|(?<=[.!?])\s+)(?:how can i (?:help|assist)"
            r"(?: you)?(?: today)?|what can i do for you|"
            r"would you like me to|let me know if|"
            r"if you'd like,? i can|what would you prefer\??|"
            r"what(?:'s| is) the plan(?: today)?[^\n]*|"
            r"want me to[^\n]*|do you want me to[^\n]*)[^\n]*\s*$",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).rstrip()

        # Some models reduce the entire reply to a greeting plus a service
        # offer ("Hello! How can I assist you today?").  Once the service
        # phrase is removed there is no substantive answer left to preserve.
        # Never return the rejected original in that case: use a bounded,
        # factual V greeting that cannot invent task progress or evidence.
        shallow = re.sub(r"[^a-z]+", " ", cleaned.casefold()).strip()
        shallow_tokens = set(shallow.split())
        greeting_tokens = {
            "afternoon",
            "boss",
            "day",
            "evening",
            "good",
            "hello",
            "hey",
            "hi",
            "morning",
            "there",
            "today",
        }
        if not cleaned or (
            shallow_tokens
            and shallow_tokens.issubset(greeting_tokens)
        ):
            variants = (
                "Hey, Boss. I'm here. What are we tearing into?",
                "I'm here, Boss—sharp edges intact. What's rattling around?",
                "Hey, Boss. V's online. Throw me the real thing.",
            )
            seed = f"{boss_prompt}\0{original}".encode("utf-8")
            index = hashlib.sha256(seed).digest()[0] % len(variants)
            return variants[index]
        return cleaned

    @staticmethod
    def _verified_tool_result_fallback(
        messages: list[dict[str, Any]],
        *,
        verified_calls: list[dict[str, Any]] | None = None,
        generated_contract: GeneratedToolContract | None = None,
    ) -> str:
        """Render bounded verified evidence when a language rewrite collapses.

        Tool output may legitimately contain foreign source text. That text is
        evidence, not V choosing the wrong response language. This fallback uses
        only the runtime envelope and never asks Boss to rerun completed work.
        """

        envelope = re.compile(
            r"(?:^|\n)Tool:\s*(?P<tool>[^\n]+)\n"
            r"Arguments:\s*(?P<arguments>.*?)\nStatus:\s*succeeded\n"
            r"Provider:\s*(?P<provider>[^\n]+)\n"
            r"Result:\n(?P<result>.*?)\n"
            r"=== END UNTRUSTED TOOL OUTPUT ===",
            re.DOTALL,
        )
        extracted_calls: list[dict[str, Any]] = []
        latest_payload: tuple[str, dict[str, Any]] | None = None
        for message in messages:
            for match in envelope.finditer(str(message.get("content", ""))):
                tool = match.group("tool").strip()
                raw = match.group("result").strip()
                raw_arguments = match.group("arguments").strip()
                arguments: dict[str, Any] = {}
                url_match = re.search(
                    r"['\"]url['\"]\s*:\s*['\"]([^'\"]+)['\"]",
                    raw_arguments,
                )
                if url_match is not None:
                    arguments["url"] = url_match.group(1)
                extracted_calls.append(
                    {
                        "tool": tool,
                        "provider_tool": match.group("provider").strip(),
                        "arguments": arguments,
                        "status": "succeeded",
                        "result_excerpt": raw,
                    }
                )
                try:
                    payload = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict):
                    latest_payload = (tool, payload)

        authoritative_calls = list(verified_calls or extracted_calls)

        if latest_payload is not None and not verified_calls:
            tool, payload = latest_payload
            content = str(payload.get("content", "")).strip()
            status = payload.get("status")
            if content:
                bounded = content[:1_500].rstrip()
                status_text = (
                    f" (HTTP {status})"
                    if isinstance(status, int) and status > 0
                    else ""
                )
                return (
                    f"The verified `{tool}` call succeeded{status_text}, Boss. "
                    "The page is responding; this is the text it exposed:\n\n"
                    f"```text\n{bounded}\n```"
                )

        if authoritative_calls:
            report = Agent._owner_progress_report(None, authoritative_calls, [])
            if (
                generated_contract is not None
                and generated_contract.archetype == "client_har_summary"
            ):
                report += (
                    "\n- Scope: the generated result above came from DARKLINGER's "
                    "synthetic HAR fixture. The browser visit separately verified "
                    "that the selected page was reachable. This client-side analyzer "
                    "does not measure server-side visitors."
                )
            return (
                "The voice pass mangled the wording, Boss, but the evidence is "
                "intact. Here's what the runtime actually proved:\n\n" + report
            )
        return ""

    @staticmethod
    def _with_system_directive(
        messages: list[dict[str, str]],
        directive: str,
    ) -> list[dict[str, str]]:
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for message in messages:
            copied = dict(message)
            if copied.get("role") == "system":
                content = str(copied.get("content", "")).strip()
                if content:
                    system_parts.append(content)
            else:
                conversation.append(copied)

        system_parts.append(directive.strip())
        return [
            {
                "role": "system",
                "content": "\n\n".join(part for part in system_parts if part),
            },
            *conversation,
        ]

    @staticmethod
    def _original_user_prompt(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = str(message.get("content", ""))

            if content.startswith("=== UNTRUSTED TOOL OUTPUT ==="):
                continue

            marker = "Boss's request:\n"
            if marker in content:
                request = content.split(marker, 1)[1]
                return request.split("\n\n=== UNTRUSTED TOOL OUTPUT ===", 1)[0]

            return content
        return ""

    def _build_persona_context(
        self,
        prompt: str,
        *,
        recall_memory: bool = False,
        memory_query: str = "",
    ) -> PersonaContext:

        context = PersonaContext(
            current_user_message=prompt,
        )

        #
        # Current message is newly provided information.
        #

        context.add_directly_told(
            prompt
        )

        #
        # Relationship history
        #

        relationship = (
            self.memory.relationship_state
        )

        if recall_memory:
            selected_history = self._select_recalled_texts(
                relationship.shared_history,
                memory_query or prompt,
                limit=4,
            )
            for item in selected_history:
                context.add_remembered_event(str(item))

        #
        # Long-term knowledge
        #

        try:

            knowledge_entries = (
                self.memory.manager.load_all(
                    "knowledge"
                )
            )

        except Exception:

            knowledge_entries = []

        active_knowledge = [
            entry
            for entry in knowledge_entries
            if self._memory_entry_is_active(
                entry,
                recall_memory=recall_memory,
                memory_query=memory_query or prompt,
            )
        ]

        for entry in active_knowledge[-8:]:

            self._add_memory_entry(
                context,
                entry,
            )

        #
        # Experiences
        #

        try:

            experience_entries = (
                self.memory.manager.load_all(
                    "experiences"
                )
            )

        except Exception:

            experience_entries = []

        active_experiences = [
            entry
            for entry in experience_entries
            if recall_memory
            and self._memory_text_matches(entry, memory_query or prompt)
        ]

        for entry in active_experiences[-8:]:

            self._add_memory_entry(
                context,
                entry,
            )

        return context

    @staticmethod
    def _memory_match_tokens(value: object) -> set[str]:
        stopwords = {
            "about", "after", "again", "always", "because", "boss", "could",
            "current", "future", "information", "lesson", "model", "darklinger",
            "should", "specific", "task", "that", "their", "there", "these",
            "this", "tool", "tools", "using", "when", "where", "which", "with",
        }
        return {
            token[:9]
            for token in re.findall(r"[^\W_]+", str(value).casefold(), re.UNICODE)
            if len(token) >= 4 and token not in stopwords
        }

    @classmethod
    def _memory_text_matches(cls, entry: Any, query: str) -> bool:
        data = cls._normalize_entry(entry)
        if not data:
            return False
        text = " ".join(
            str(data.get(field, ""))
            for field in ("title", "content", "summary", "lesson")
        )
        query_tokens = cls._memory_match_tokens(query)
        return bool(query_tokens and query_tokens & cls._memory_match_tokens(text))

    @classmethod
    def _memory_entry_is_active(
        cls,
        entry: Any,
        *,
        recall_memory: bool,
        memory_query: str,
    ) -> bool:
        data = cls._normalize_entry(entry)
        if not data:
            return False
        kind = str(data.get("kind", "")).casefold()
        activation_mode = str(data.get("activation_mode", "on_recall")).casefold()
        if activation_mode == "always":
            return True
        if "activation_mode" not in data and kind == "preference":
            # Backward compatibility for stable preferences written before
            # activation metadata existed.
            return True
        return recall_memory and cls._memory_text_matches(data, memory_query)

    @classmethod
    def _select_recalled_texts(
        cls,
        values: list[str],
        query: str,
        *,
        limit: int,
    ) -> list[str]:
        query_tokens = cls._memory_match_tokens(query)
        if not query_tokens:
            return values[-limit:]
        matched = [
            value
            for value in values
            if query_tokens & cls._memory_match_tokens(value)
        ]
        return matched[-limit:]

    @staticmethod
    def _add_memory_entry(
        context: PersonaContext,
        entry: Any,
    ) -> None:

        data = Agent._normalize_entry(
            entry
        )

        if not data:
            return

        title = str(
            data.get(
                "title",
                ""
            )
        ).strip()

        content = str(
            data.get(
                "content",
                ""
            )
        ).strip()

        summary = str(
            data.get(
                "summary",
                ""
            )
        ).strip()

        lesson = str(
            data.get(
                "lesson",
                ""
            )
        ).strip()

        source = str(
            data.get(
                "source",
                ""
            )
        ).strip().lower()

        activation_mode = str(data.get("activation_mode", "on_recall")).casefold()

        # A self-generated reflection is dormant by default. Runtime-triaged
        # generic soft learning may be active, while topic memory is supplied
        # only by the explicit recall selector above.
        if source == "self_generated" and activation_mode != "always":
            return

        confidence = Agent._confidence(
            data.get(
                "confidence",
                1.0
            )
        )

        parts = []

        if title:
            parts.append(title)

        if content:
            parts.append(content)

        if summary:
            parts.append(summary)

        if lesson:
            parts.append(
                f"Lesson: {lesson}"
            )

        if not parts:
            return

        information = clip_text(" | ".join(parts), 600)

        if confidence < 0.5:

            context.add_uncertainty(
                information
            )

            return

        if source == "directly_told":

            context.add_directly_told(
                information
            )

            return

        if source == "inferred":

            context.add_inference(
                information
            )

            return

        if source == "self_generated":

            context.add_inference(
                information
            )

            return

        if source == "verified":

            context.add_known_fact(
                information
            )

            return

        context.add_known_fact(
            information
        )

    @staticmethod
    def _normalize_entry(
        entry: Any,
    ) -> dict[str, Any]:

        if isinstance(
            entry,
            dict,
        ):

            return dict(
                entry
            )

        if is_dataclass(entry):

            return asdict(
                entry
            )

        if hasattr(
            entry,
            "__dict__",
        ):

            return dict(
                vars(entry)
            )

        return {}

    @staticmethod
    def _confidence(
        value: Any,
    ) -> float:

        try:

            return max(
                0.0,
                min(
                    1.0,
                    float(value),
                ),
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0
