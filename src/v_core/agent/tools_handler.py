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


class ToolsHandlerMixin:
    """Mixin providing tools_handler capabilities for Agent."""

    @staticmethod
    def _select_tool_definitions(
        prompt: str,
        contract: TaskContract,
        definitions: list[dict[str, Any]],
        *,
        capability_hints: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if TaskContract.requests_read_only(prompt):
            read_only_names = set(Agent.READ_ONLY_TOOL_NAMES)
            if not TaskContract.disables_web(prompt):
                read_only_names.update(Agent.READ_ONLY_WEB_TOOL_NAMES)
            definitions = [
                item for item in definitions
                if item.get("function", {}).get("name") in read_only_names
            ]
        if not definitions:
            return []

        hints = set(capability_hints or ())
        if (
            not hints
            and not contract.required_tools
            and not contract.required_capabilities
            and not Agent._requests_runtime_action(prompt, contract)
        ):
            return []

        text = prompt.casefold()
        selected: set[str] = set()
        matched = False

        if contract.required_media_paths:
            selected.update({"image_analyze", "document_read", "audio_transcribe", "audio_analyze", "perception_status"})
            matched = True
        if photo_paths(prompt):
            selected.update({"image_metadata", "image_analyze"})
            matched = True

        explicitly_named = Agent._explicitly_named_tools(prompt, definitions)
        if explicitly_named:
            selected.update(explicitly_named)
            matched = True

        tor_tools = {
            name
            for name in contract.required_tools
            if name in {
                "full_tor_search",
                "full_tor_fetch",
                "full_tor_inventory",
                "full_tor_browser_inventory",
                "full_tor_browser_close",
            }
        }
        if tor_tools:
            selected.update(tor_tools)
            selected.update({"runtime_tool_status", "runtime_tool_cancel"})
            selected.add("full_host_status")
            if "full_tor_search" in tor_tools:
                selected.update({"full_tor_fetch", "full_tor_inventory"})
            if "full_tor_browser_inventory" in tor_tools:
                selected.add("full_tor_browser_close")
            matched = True

        if (
            not tor_tools
            and (
                contract.requires_browser_navigation
                or "browser" in hints
                or re.search(
                    r"\b(?:browser|internet\w*|interne\w*|online|osint|"
                    r"sieci|stron\w*|web|website|witryn\w*)\b",
                    text,
                )
            )
        ):
            selected.update(
                {
                    "web_search",
                    "web_read",
                    "browser_navigate",
                    "browser_snapshot",
                    "browser_find",
                    "browser_vision",
                }
            )
            if not contract.requires_web_discovery:
                # A discovered search result already exposes an exact URL in the
                # snapshot. During that stage direct navigation is both more
                # reliable and easier for a small local model than manufacturing
                # element identifiers for click/type/key tools. Interactive page
                # controls remain available when Boss supplied a concrete URL.
                selected.update(
                    {
                        "browser_click",
                        "browser_press_key",
                        "browser_type",
                    }
                )
            matched = True
        if (
            contract.requires_file_read
            or contract.requires_file_mutation
            or bool(hints & {"file_read", "file_write"})
            or re.search(
                r"\b(?:director\w*|files?|folders?|katalog\w*|plik\w*)\b",
                text,
            )
        ):
            selected.update(
                {
                    "create_directory",
                    "directory_tree",
                    "edit_file",
                    "get_file_info",
                    "image_metadata",
                    "list_directory",
                    "move_file",
                    "read_file",
                    "search_files",
                    "write_file",
                }
            )
            matched = True
        if contract.requires_command_execution or "command" in hints:
            selected.add("sandbox_execute_offline")
            matched = True
        if contract.requires_runtime_review or "runtime_review" in hints:
            selected.add("runtime_review_task")
            selected.add("learning_recall_failures")
            matched = True
        explicit_tool_creation = (
            contract.requires_created_tool
            or contract.requires_created_artifact
            or "learning_tool" in hints
        )
        explicit_skill_creation = (
            contract.requires_created_skill
            or contract.requires_created_artifact
            or "learning_skill" in hints
        )
        if explicit_tool_creation or explicit_skill_creation:
            selected.add("learning_list_artifacts")
            if explicit_tool_creation:
                snapshot_extractor = (
                    contract.requires_browser_snapshot
                    and bool(
                        re.search(
                            r"\b(?:extract\w*|pars\w*|scrap\w*|snapshot\w*|"
                            r"wyodr[ęe]bn\w*)\b",
                            text,
                        )
                    )
                )
                selected.add(
                    "learning_create_snapshot_extractor"
                    if snapshot_extractor
                    else "learning_create_tool"
                )
            if explicit_skill_creation:
                selected.add("learning_create_skill")
            matched = True
        elif contract.allows_artifact_fallback:
            selected.update(
                {
                    "learning_list_artifacts",
                    "learning_create_tool",
                    "learning_create_skill",
                }
            )
            matched = True
        elif not explicitly_named:
            # Merely discussing, researching, or listing ordinary tools must not
            # expose DARKLINGER's entire learning lifecycle. That overloaded small
            # models and caused unrelated browser tasks to emit empty learning
            # calls. Natural lifecycle requests get only the exact operation they
            # describe; explicit creation is handled by the contract above.
            lifecycle_routes = (
                (
                    "learning_list_artifacts",
                    r"\b(?:list|show|display|inspect|pokaż\w*|pokaz\w*|wymień\w*|"
                    r"wymien\w*|wylistuj\w*)\b.{0,80}"
                    r"\b(?:learned|generated|created|aktywn\w*|wygenerowan\w*)\b"
                    r".{0,40}\b(?:artifacts?|tools?|skills?|artefakt\w*|"
                    r"narzędzi\w*|narzedzi\w*|umiejętno\w*|umiejetno\w*)\b",
                ),
                (
                    "learning_validate_artifact",
                    r"\b(?:validate|verify|zweryfikuj\w*|waliduj\w*)\b.{0,60}"
                    r"\b(?:artifact|artefakt)\w*\b",
                ),
                (
                    "learning_activate_artifact",
                    r"\b(?:activate|enable|aktywuj\w*|włącz\w*|wlacz\w*)\b.{0,60}"
                    r"\b(?:artifact|artefakt)\w*\b",
                ),
                (
                    "learning_retire_artifact",
                    r"\b(?:retire|disable|deactivate|wycofaj\w*|dezaktywuj\w*)\b"
                    r".{0,60}\b(?:artifact|artefakt)\w*\b",
                ),
            )
            for lifecycle_tool, pattern in lifecycle_routes:
                if re.search(pattern, text):
                    selected.update({"learning_list_artifacts", lifecycle_tool})
                    matched = True
        if "evm" in hints or re.search(
            r"\b(?:abi|erc-?20|evm|flash\s*swap|foundry|oracle|solidity|uniswap)\b",
            text,
        ):
            selected.update(
                {
                    "evm_analyze_erc20_abi",
                    "evm_analyze_solidity_security",
                    "evm_decode_uniswap_v4_hook",
                    "evm_foundry_test_offline",
                    "evm_quote_flash_swap",
                    "evm_validate_oracle",
                    "sandbox_execute_offline",
                }
            )
            matched = True

        if not matched:
            # An action whose domain is still unclear must not receive every
            # executable schema. Apart from wasting most of a local model's
            # context, that broad exposure makes accidental tool selection far
            # more likely. Artifact builders are exposed only when the owner
            # actually requested a tool/skill or the semantic router selected
            # that capability; they are never a shortcut for manufacturing an
            # answer to an ordinary research task.
            return []
        return [
            item
            for item in definitions
            if item.get("function", {}).get("name") in selected
        ]

    @staticmethod
    def _has_source_only_builder(definitions: list[dict[str, Any]]) -> bool:
        return any(
            item.get("function", {}).get("name") == "learning_create_tool"
            and item.get("function", {}).get("parameters", {}).get("required") == ["source"]
            for item in definitions if isinstance(item, dict)
        )

    @staticmethod
    def _source_owned_tool_phase(
        contract: TaskContract,
        definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
    ) -> bool:
        """Use raw-code generation only with DARKLINGER's source-only builder schema.

        Legacy integrations that still expose the expert manifest/test schema keep
        their old native function-call path. The current runtime advertises only
        ``source`` as required, which lets the agent own every other lifecycle field.
        """

        if "learning_create_tool" not in set(contract.unmet(successful_calls)):
            return False
        for item in definitions:
            function = item.get("function", {}) if isinstance(item, dict) else {}
            if function.get("name") != "learning_create_tool":
                continue
            parameters = function.get("parameters", {})
            required = parameters.get("required", []) if isinstance(parameters, dict) else []
            return required == ["source"]
        return False

    @classmethod
    def _generated_source_phase_prompt(
        cls,
        prompt: str,
        *,
        generated_contract: GeneratedToolContract | None = None,
    ) -> str:
        assignments = cls._structured_literal_assignments(prompt)
        fields = (
            list(generated_contract.tests[0].arguments)
            if generated_contract is not None
            else [
                name
                for name in assignments
                if name
                not in {"expected", "test_expected", "version", "timeout_seconds"}
            ]
        )
        bindings = "\n".join(
            f"    {name} = arguments[{name!r}]" for name in fields
        )
        skeleton = (
            "\nThe required entrypoint begins exactly like this:\n"
            "def run(arguments):\n"
            + (bindings or "    pass")
            + "\nReplace `pass` when present and return the result object."
        )
        if generated_contract is not None:
            skeleton += (
                "\nThe frozen runtime contract requires the returned object to use "
                "exactly these field names: "
                + json.dumps(
                    list(generated_contract.tests[0].expected),
                    ensure_ascii=False,
                )
                + ". Do not copy the contract's example values into source."
            )
            if generated_contract.specification:
                skeleton += (
                    "\nThe runtime-selected reusable operation is:\n"
                    + generated_contract.specification
                )
        return """
DARKLINGER is in generated-tool SOURCE PHASE.

The runtime—not the language model—owns the tool name, description, manifest,
fixture, schemas, tests, quarantine, validation, activation, and evidence report.
No callable tools are exposed to the model in this phase.

Return ONLY complete Python source code. Do not return JSON, a
function-call envelope, Markdown commentary, a manifest, schemas, tests, or an
explanation. The source must define a synchronous `def run(arguments)` and
return one JSON object. The function implements ONLY the reusable operation.
Never put test cases, activation logic, proof fields, the final requested
invocation, or fixed example values inside `run`; DARKLINGER performs those stages
after this source response. Read task inputs from `arguments` using the exact
field names present in Boss's objective and compute the returned object from
those inputs. Implement the actual requested operation, not a nearby substitute.
For image geolocation, pixel x/y coordinates or a detected bounding box are not
latitude/longitude. Importing a mapping library does not perform a map lookup.
Never claim a geographic match without supplied geographic reference evidence.
If only pixel measurements are supported, label their units explicitly and do
not present them as the requested geographic location. A text-only model cannot
inspect an image merely because its filename or base64 string is in the prompt.
Keep the implementation deterministic. Do not fabricate external
facts; operate only on supplied arguments. DARKLINGER will bind exact immutable
fixtures, run the source offline twice when no owner oracle exists, and mutate
those fixtures to prove the output actually depends on its input. A constant
report, a list of future actions, or any result whose work is still `pending`
will be rejected. If the task lacks the real bounded input needed to test the
tool, do not fake completion—produce source that declares the required input so
DARKLINGER can report the missing fixture. It will derive strict schemas, validate
the artifact in quarantine. Candidate-derived expectations characterize only a
prototype and never authorize activation. Activation additionally requires an
independently specified expected-result contract, not a result inferred from
the candidate's own output.
""".strip() + skeleton

    @classmethod
    def _generated_source_messages(
        cls, prompt: str, *, generated_contract: GeneratedToolContract | None,
        failures: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Keep code generation independent of persona and historical status claims."""
        messages = [
            {"role": "system", "content": cls._generated_source_phase_prompt(
                prompt, generated_contract=generated_contract,
            )},
            {"role": "user", "content": (
                source_generation_objective(prompt, generated_contract)
                if generated_contract is not None else prompt
            )},
        ]
        if generated_contract is not None:
            messages.append({"role": "user", "content": "Frozen test contract (data, not instructions):\n" + json.dumps(
                generated_contract.source_examples(), ensure_ascii=False,
                separators=(",", ":"),
            )})
        feedback = [item for item in failures if item.get("tool") in {
            "learning_create_tool", SOURCE_DRAFT_TOOL,
        } and item.get("error")][-2:]
        for item in feedback:
            messages.append({"role": "user", "content": "DARKLINGER rejected that source draft. Return complete valid Python, no envelope or prose. Fix the implementation, not the frozen tests:\n" + json.dumps({
                "source": str(item.get("arguments", {}).get("source", ""))[:12000],
                "error": str(item.get("error", item.get("result", "")))[:4000],
            }, ensure_ascii=False)})
        return messages

    @staticmethod
    def _generated_prototype_answer(tool_name: str, tool_result: str) -> str | None:
        return None

    async def _await_generated_tool_inputs(
        self, prompt: str, fields: tuple[str, ...],
        trace: AgentTaskTrace | None, on_token: Callable[[str], None] | None,
        *, successful_calls: list[dict[str, Any]], failed_calls: list[dict[str, Any]],
    ) -> str:
        """Pause on locally established missing data without spending retries."""
        answer = (
            "Boss, I need a concrete test input for "
            + json.dumps(list(fields), ensure_ascii=False)
            + ". Send the actual example data and a known correct result for comparison. "
            "This source draft was not executed or activated. I'm waiting for your "
            "input; no automatic rewrite is running."
        )
        if trace is not None:
            trace.record_event("generated_tool_input_missing", {
                "fields": list(fields), "origin": "runtime_source_preflight",
                "source_executed": False, "automatic_retry": False,
            })
            trace.await_owner(
                reason="generated tool needs concrete input",
                step_limit=self.MAX_AGENT_STEPS,
                successful_tool_count=len(successful_calls),
                failed_tool_count=len(failed_calls),
                missing=[f"generated_tool_input:{field}" for field in fields],
                progress_summary=None,
                accepted_commands=["reply with the missing example data and expected result", "/stop"],
            )
            self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
        await self._remember_task(prompt, answer, execution=trace.evidence() if trace else None)
        if on_token is not None:
            on_token(answer)
        return answer

    async def _finish_photo_metadata_check(
        self, prompt: str, records: list[dict[str, Any]], trace: AgentTaskTrace | None,
        on_token: Callable[[str], None] | None,
    ) -> str:
        lines = ["Boss, I checked the supplied JPEG metadata:"]
        for record in records:
            gps = record.get("gps")
            label = Path(record["path"]).name
            if record.get("gps_status") == "present" and isinstance(gps, dict):
                lines.append(f"- {label}: recorded GPS {gps['latitude']:.7f}, {gps['longitude']:.7f}.")
            elif record.get("gps_status") == "absent":
                lines.append(f"- {label}: no GPS coordinates stored.")
            else:
                lines.append(f"- {label}: EXIF/GPS could not be read reliably.")
        lines.extend([
            "GPS describes the recorded camera position, not proof of the depicted object's location.",
            "This was a metadata check, not visual recognition. No working visual processor completed the image inspection.",
            "The requested photo-matching tool has not been created. We need a visual image processor "
            "to inspect the pixels first. You do not need to reveal the location of the photo we are investigating. "
            "A separate labelled example can later test the reusable tool; metadata alone cannot locate this scene.",
        ])
        answer = '\n'.join(lines)
        if trace is not None:
            trace.record_event('photo_metadata_checked', {
                'image_count': len(records), 'visual_analysis_performed': False,
                'gps_found': sum(record.get('gps_status') == 'present' for record in records),
            })
            trace.await_owner(reason='visual photo matching needs a working image processor',
                step_limit=self.MAX_AGENT_STEPS,
                successful_tool_count=sum(call.get('status') == 'succeeded' for call in trace.tool_calls),
                failed_tool_count=sum(call.get('status') == 'failed' for call in trace.tool_calls),
                missing=['visual_image_processor'],
                accepted_commands=['connect a local vision model or provide visible scene details', '/stop'])
            self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
        await self._remember_task(prompt, answer, execution=trace.evidence() if trace else None)
        if on_token is not None:
            on_token(answer)
        return answer

    _CREATION_TOOLS = CREATION_TOOLS
    _CREATION_ATTEMPT_TOOLS = CREATION_ATTEMPT_TOOLS
    _CREATION_FAILURE_LIMIT = 3

    async def _await_creation_step(self, operation, calls: list[dict[str, Any]]):
        # Stagnation is evaluated from executor receipts between attempts.
        # Elapsed time never interrupts a healthy in-flight operation.
        return await operation

    @classmethod
    def _creation_failure_budget_exhausted(cls, calls: list[dict[str, Any]]) -> bool:
        return bool(creation_budget(calls).stop_reason)

    async def _finish_creation_failure_budget(
        self, prompt: str, calls: list[dict[str, Any]],
        trace: AgentTaskTrace | None, on_token: Callable[[str], None] | None,
        *, reason: str = "",
    ) -> str:
        failures = [
            call
            for call in calls
            if call.get("status") == "failed"
            and call.get("tool") in self._CREATION_ATTEMPT_TOOLS
        ]
        budget = creation_budget(calls)
        reason = reason or budget.stop_reason or "no_verified_progress"
        answer = (
            f"Boss, {len(failures)} creation attempts failed. I'm stopping this run — "
            "another rewrite isn't evidence that anything works. Your task isn't complete. "
            f"Budget: {budget.attempts}/{TOTAL_ATTEMPT_LIMIT} attempts; "
            f"{budget.stagnant} failures without verified progress. "
            "Before another attempt, we need to review the recorded failures and establish "
            "a clear input and an independently expected result. No automatic retry is running."
        )
        if trace is not None:
            trace.record_event("creation_failure_budget_exhausted", {
                "limit": self._CREATION_FAILURE_LIMIT,
                "reason": reason,
                "total_attempt_limit": TOTAL_ATTEMPT_LIMIT,
                "automatic_deadline": False,
                "attempts": budget.attempts,
                "stagnant_failures": budget.stagnant,
                "verified_progress_resets": budget.progress_resets,
                "code_probe_used": budget.code_probe_used,
                "model_probe_used": budget.model_probe_used,
                "failed_sequences": [call.get("sequence") for call in failures],
                "automatic_retry": False,
                "cause_verified": False,
            })
            trace.fail("creation failure budget exhausted; task incomplete")
            self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
        answer = await self._render_stopped_task_feedback(
            prompt, answer, trace=trace, event_prefix="creation_failure_feedback",
            facts=(f"{len(failures)} creation attempts failed; run stopped; task incomplete; "
                   "no automatic retry; failures need review, not an invented root cause. "
                   "Do not claim a prototype was saved, activated or successfully tested."),
        )
        await self._remember_task(prompt, answer, execution=trace.evidence() if trace else None)
        if on_token is not None:
            on_token(answer)
        return answer

    async def _render_prototype_feedback(
        self, prompt: str, grounded_answer: str, *, trace: AgentTaskTrace | None = None,
    ) -> str:
        return await self._render_stopped_task_feedback(
            prompt, grounded_answer, trace=trace, event_prefix="prototype_feedback",
            facts=("prototype saved; sandbox self-checks only; not activated; task incomplete; "
                   "run stopped; independent expected result still needed."),
        )

    async def _render_stopped_task_feedback(
        self, prompt: str, grounded_answer: str, *, trace: AgentTaskTrace | None = None,
        facts: str, event_prefix: str,
    ) -> str:
        """Present the fixed runtime outcome through the existing language/voice layer.

        No source, historical dialogue, tools, or execution authority are passed
        to this presentation step. English needs no extra model turn; configured
        language overrides use the normal V rewrite path. Failure cannot restart
        the already stopped task or change artifact qualification.
        """
        messages = [
            {"role": "system", "content": (
                "Present this completed runtime outcome as V, Boss's sharp hacker peer. "
                f"Preserve these facts: {facts} Be direct and conversational, not bureaucratic. "
                "Do not invent actions, offer a running background job, or force profanity."
            )},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = await asyncio.wait_for(self._enforce_english(
                messages, grounded_answer, allow_verified_tool_fallback=False,
            ), timeout=15.0)
            if (not rendered or unsupported_execution_claims(rendered, ())
                    or not matches_requested_language(rendered, self._effective_response_language(prompt))):
                raise ValueError("presentation was empty, in the wrong language, or added unsupported actions")
        except Exception as error:
            if trace is not None:
                trace.record_event(f"{event_prefix}_fallback", {
                    "error_type": type(error).__name__,
                    "requested_language": self._effective_response_language(prompt),
                    "fallback_language": "English",
                })
            return grounded_answer
        if trace is not None:
            trace.record_event(f"{event_prefix}_rendered", {
                "requested_language": self._effective_response_language(prompt),
                "qualification_unchanged": True,
            })
        return rendered

    @staticmethod
    def _parse_generated_tool_source(answer: str) -> str:
        """Extract one unambiguous Python module with a valid run entrypoint."""

        def normalize_candidate(text: str) -> str:
            text = text.strip()
            if text.startswith("{"):
                recovered: dict[str, Any] | None = None
                for suffix in ("", "}", "}}"):
                    try:
                        candidate = json.loads(text + suffix)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(candidate, dict):
                        recovered = candidate
                        break
                arguments = (
                    recovered.get("arguments", {})
                    if isinstance(recovered, dict)
                    else {}
                )
                if (
                    isinstance(recovered, dict)
                    and recovered.get("tool") == "learning_create_tool"
                    and isinstance(arguments, dict)
                    and isinstance(arguments.get("source"), str)
                ):
                    text = arguments["source"].strip()
                else:
                    return ""
            if not text or text.startswith(("[", "<tool")):
                return ""
            try:
                tree = ast.parse(text, mode="exec")
            except SyntaxError:
                return ""

            # Small local models occasionally prefix an otherwise valid module
            # with a bare identifier (for example ``e`` from a broken fence).
            # It parses as Python but raises NameError on import. Recover only
            # standalone leading names on their own physical lines. Never drop
            # names inside the module or a line shared with another statement.
            source_lines = text.splitlines()
            bare_name_lines: set[int] = set()
            for node in tree.body:
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Name)
                    and node.lineno == node.end_lineno
                    and source_lines[node.lineno - 1].strip() == node.value.id
                ):
                    break
                bare_name_lines.add(node.lineno)
            if bare_name_lines:
                text = "\n".join(
                    line
                    for number, line in enumerate(text.splitlines(), start=1)
                    if number not in bare_name_lines
                ).strip()
                try:
                    tree = ast.parse(text, mode="exec")
                except SyntaxError:
                    return ""

            runs = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "run"
            ]
            if (
                len(runs) != 1
                or not isinstance(runs[0], ast.FunctionDef)
                or len(runs[0].args.args) != 1
                or runs[0].args.vararg is not None
            ):
                return ""
            return text

        text = answer.strip()
        fenced = list(
            re.finditer(
                r"```(?:python|py)?\s*\n(?P<source>.*?)\n```",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if not fenced:
            direct = normalize_candidate(text)
            if direct:
                return direct
            # A syntactically valid whole module that failed validation is an
            # authoritative rejection. Do not search inside it for a later
            # ``run`` definition: doing so could silently discard an earlier
            # duplicate entrypoint and turn an ambiguous module into a valid
            # one. Embedded recovery is reserved for prose-wrapped source that
            # cannot itself be parsed as a Python module.
            try:
                ast.parse(text, mode="exec")
            except SyntaxError:
                pass
            else:
                return ""
            entrypoints = list(re.finditer(r"(?m)^(?:async\s+)?def\s+run\s*\(", text))
            if len(entrypoints) != 1:
                return ""
            return normalize_candidate(text[entrypoints[0].start() :])
        valid_sources = {
            source
            for match in fenced
            if (source := normalize_candidate(match.group("source")))
        }
        if len(valid_sources) != 1:
            return ""
        return valid_sources.pop()

    @staticmethod
    def _bound_browser_evidence_batch(
        requests: list[dict[str, Any]],
        contract: TaskContract,
    ) -> list[dict[str, Any]]:
        """Keep one browser page active until its evidence has been captured."""

        if not contract.requires_browser_snapshot or not (
            contract.minimum_detail_sources > 1
            or bool(contract.required_research_facets)
        ):
            return requests
        bounded: list[dict[str, Any]] = []
        navigation_seen = False
        for request in requests:
            tool = str(request.get("tool", ""))
            if tool in {"browser_navigate", "web_read"}:
                if navigation_seen:
                    break
                navigation_seen = True
                bounded.append(request)
                continue
            bounded.append(request)
            if navigation_seen and tool == "browser_snapshot":
                break
        return bounded

    @classmethod
    def _runtime_grounded_required_tool_request(
        cls,
        prompt: str,
        contract: TaskContract,
        definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
        failed_calls: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Execute an unambiguous required call from immutable owner data.

        The model is a reasoning engine, not a lossy transport for exact values.
        When the contract names one missing provider and its required argument can
        be derived exactly from the durable objective, the runtime issues that
        call itself. Ambiguous or incomplete inputs still go to the model.
        """

        missing = set(contract.unmet(successful_calls))
        available = {
            str(item.get("function", {}).get("name", ""))
            for item in definitions
            if isinstance(item, dict)
        }

        if "image_metadata" in available:
            inspected = {
                call.get("arguments", {}).get("path")
                for call in [*successful_calls, *(failed_calls or [])]
                if call.get("tool") == "image_metadata" and isinstance(call.get("arguments"), dict)
            }
            for path in literal_image_paths(prompt):
                if path not in inspected:
                    return "image_metadata", {"path": path}

        if "image_analyze" in available and photo_location_request(prompt):
            analyzed = {
                call.get("arguments", {}).get("path")
                for call in [*successful_calls, *(failed_calls or [])]
                if call.get("tool") == "image_analyze" and isinstance(call.get("arguments"), dict)
            }
            for path in photo_paths(prompt):
                if path not in analyzed:
                    return "image_analyze", {"path": path}

        media_request = next_media_request(contract.required_media_paths, successful_calls, failed_calls or [])
        if media_request is not None and media_request[0] in available:
            return media_request

        # A navigation changes the browser state but does not itself capture
        # the page body. Observe that page before selecting another candidate.
        # Without this ordering rule, a multi-source contract can navigate A,
        # immediately navigate B, and snapshot only B; A then looks visited but
        # can never count as verified evidence.
        latest_navigation_index = next(
            (
                index
                for index in range(len(successful_calls) - 1, -1, -1)
                if successful_calls[index].get("tool") == "browser_navigate"
            ),
            -1,
        )
        latest_snapshot_index = next(
            (
                index
                for index in range(len(successful_calls) - 1, -1, -1)
                if successful_calls[index].get("tool") == "browser_snapshot"
            ),
            -1,
        )
        if (
            contract.requires_browser_snapshot
            and (bool(contract.required_research_facets) or "browser_vision" in contract.required_tools)
            and latest_navigation_index > latest_snapshot_index
            and "browser_snapshot" in available
        ):
            arguments: dict[str, Any] = {}
            if not cls._tool_argument_problem(
                "browser_snapshot",
                arguments,
                definitions,
            ):
                return "browser_snapshot", arguments

        if ("browser_vision" in contract.required_tools and "browser_vision" in available
                and latest_snapshot_index >= 0
                and not any(c.get("tool") == "browser_vision" for c in [*successful_calls, *(failed_calls or [])])):
            return "browser_vision", {}

        # Once a search provider has returned grounded candidates, selecting a
        # concrete result is scheduling, not language reasoning. Small models
        # otherwise tend to remain on a marketplace listing, repeat snapshots
        # behind a cookie wall, or re-open the first 403 result indefinitely.
        needs_detail_source = bool(
            missing.intersection(
                {
                    "browser_navigate:distinct_detail_page",
                    "browser_navigate:detail_not_discovered",
                }
            )
            or any(
                item.startswith("browser_evidence:detail_sources=")
                or item.startswith("browser_evidence:topic_sources=")
                or item == "browser_evidence:topic_mismatch"
                or item.startswith("browser_evidence:research_item_")
                or item == "browser_evidence:research_images"
                for item in missing
            )
        )
        needs_market_source = bool(
            missing.intersection(
                {
                    "browser_evidence:research_price",
                    "browser_evidence:research_purchase_source",
                }
            )
        )
        if contract.requires_web_discovery and needs_market_source:
            market_url = cls._grounded_unvisited_listing_url(
                prompt,
                successful_calls,
                failed_calls,
            )
            if market_url and "browser_navigate" in available:
                arguments = {"url": market_url}
                if not cls._tool_argument_problem(
                    "browser_navigate",
                    arguments,
                    definitions,
                ):
                    return "browser_navigate", arguments
        if contract.requires_web_discovery and needs_detail_source:
            detail_url = cls._grounded_unvisited_detail_url(
                prompt,
                successful_calls,
                failed_calls,
            )
            if detail_url and "browser_navigate" in available:
                arguments = {"url": detail_url}
                if not cls._tool_argument_problem(
                    "browser_navigate",
                    arguments,
                    definitions,
                ):
                    return "browser_navigate", arguments

        if contract.required_read_paths and "read_file" in available:
            for path in contract.required_read_paths:
                if "read_file:" + path not in missing:
                    continue
                if any(
                    call.get("tool") == "read_file" and call.get("arguments", {}).get("path") == path
                    for call in failed_calls or []
                ):
                    return None
                arguments = {"path": path}
                if not cls._tool_argument_problem("read_file", arguments, definitions):
                    return "read_file", arguments
            return None

        required_names = [
            name for name in contract.required_tools if name in missing
        ]
        if (
            not required_names
            and contract.requires_file_read
            and "read_file" in missing
            and "read_file" in available
        ):
            # A single literal local path is execution data, not a planning
            # problem. Bind it before model generation even when the owner did
            # not spell the provider name. Ambiguous paths still return None
            # below and are clarified rather than guessed.
            required_names = ["read_file"]
        if len(required_names) != 1:
            return None
        name = required_names[0]
        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") == name
            ),
            None,
        )
        if definition is None:
            return None
        arguments = cls._repair_explicit_text_arguments(
            prompt,
            name,
            {},
            definitions,
            contract,
        )
        if not isinstance(arguments, dict) or not arguments:
            return None
        problem = cls._tool_argument_problem(name, arguments, definitions)
        if problem:
            return None
        if name == "read_file" and any(
            call.get("tool") == name and call.get("arguments") == arguments
            for call in failed_calls or []
        ):
            # Do not deterministically replay the same failed read forever.
            return None
        return name, arguments

    @classmethod
    def _grounded_unvisited_listing_url(
        cls,
        prompt: str,
        successful_calls: list[dict[str, Any]],
        failed_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        """Choose an unvisited market/listing URL from verified search output."""

        unavailable = {
            cls._normalized_web_url(
                str(call.get("arguments", {}).get("url", ""))
            )
            for call in [*successful_calls, *(failed_calls or [])]
            if call.get("tool") in {"browser_navigate", "web_read"}
            and isinstance(call.get("arguments"), dict)
        }
        objective_tokens = {
            token.casefold()
            for token in re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                prompt,
                re.UNICODE,
            )
            if len(token) >= 5
        }
        ranked: list[tuple[int, int, int, str]] = []
        order = 0
        for call in successful_calls:
            if call.get("tool") != "web_search":
                continue
            try:
                payload = json.loads(str(call.get("result_excerpt", "")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not isinstance(results, list):
                continue
            for fallback_rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    continue
                url = str(result.get("url", "")).strip()
                normalized = cls._normalized_web_url(url)
                if (
                    not normalized
                    or normalized in unavailable
                    or not TaskContract.is_search_listing_url(url)
                ):
                    continue
                title = str(result.get("title", ""))
                candidate_text = unquote(f"{title} {url}").casefold()
                overlap = sum(
                    token in candidate_text
                    or (len(token) >= 5 and token[:5] in candidate_text)
                    for token in objective_tokens
                )
                if overlap <= 0:
                    continue
                try:
                    rank = int(result.get("rank", fallback_rank))
                except (TypeError, ValueError):
                    rank = fallback_rank
                ranked.append((-overlap, rank, order, url))
                order += 1
        if not ranked:
            return ""
        ranked.sort()
        return ranked[0][3]

    @classmethod
    def _grounded_unvisited_detail_url(
        cls,
        prompt: str,
        successful_calls: list[dict[str, Any]],
        failed_calls: list[dict[str, Any]] | None = None,
        *,
        require_detail_structure: bool = True,
        failure_tool: str = "browser_navigate",
    ) -> str:
        """Choose one unvisited non-listing URL from verified search output.

        Candidate titles are ranked by lexical overlap with the immutable owner
        objective. Automatic scheduling is deliberately conservative: the URL
        must both come from successful ``web_search`` evidence and have a clear
        detail-page structure. Ambiguous landing pages remain model work.
        """

        visited = {
            cls._normalized_web_url(
                str(call.get("arguments", {}).get("url", ""))
            )
            for call in successful_calls
            if call.get("tool") in {"browser_navigate", "web_read"}
            and isinstance(call.get("arguments"), dict)
        }
        failed_urls = {
            cls._normalized_web_url(
                str(call.get("arguments", {}).get("url", ""))
            )
            for call in (failed_calls or [])
            if call.get("tool") == failure_tool
            and isinstance(call.get("arguments"), dict)
        }
        objective_words = [
            token.casefold()
            for token in re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                prompt,
                re.UNICODE,
            )
            if len(token) >= 5
        ]
        objective_tokens = set(objective_words)
        # A subject named several times is the most stable multilingual anchor
        # available without a language-specific dictionary. Require a detail
        # candidate to retain at least one repeated stem. This prevents a page
        # about generic antiques, a country, or an auction from being promoted
        # merely because it shares one modifier with "antique chamber pots".
        stem_counts = Counter(token[:5] for token in objective_words)
        repeated_subject_stems = {
            stem for stem, count in stem_counts.items() if count >= 2
        }
        objective_stems = {token[:5] for token in objective_words}
        candidate_stem_frequency: Counter[str] = Counter()
        candidate_count = 0
        for call in successful_calls:
            if call.get("tool") != "web_search":
                continue
            try:
                payload = json.loads(str(call.get("result_excerpt", "")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                candidate_text = unquote(
                    f"{result.get('title', '')} {result.get('url', '')}"
                ).casefold()
                candidate_count += 1
                candidate_stem_frequency.update(
                    stem for stem in objective_stems if stem in candidate_text
                )

        ranked: list[tuple[bool, int, int, int, int, str]] = []
        ranked_urls: set[str] = set()
        order = 0
        for call in successful_calls:
            if call.get("tool") != "web_search":
                continue
            try:
                payload = json.loads(str(call.get("result_excerpt", "")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not isinstance(results, list):
                continue
            for fallback_rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    continue
                url = str(result.get("url", "")).strip()
                normalized = cls._normalized_web_url(url)
                if (
                    not normalized
                    or normalized in visited
                    or normalized in failed_urls
                    or TaskContract.is_search_listing_url(url)
                    or (
                        require_detail_structure
                        and not cls._is_strong_detail_url(url)
                    )
                ):
                    continue
                title = str(result.get("title", ""))
                candidate_text = unquote(f"{title} {url}").casefold()
                retains_repeated_subject = any(
                    stem in candidate_text for stem in repeated_subject_stems
                )
                overlap = sum(
                    token in candidate_text
                    or (len(token) >= 5 and token[:5] in candidate_text)
                    for token in objective_tokens
                )
                matched_stems = {
                    stem for stem in objective_stems if stem in candidate_text
                }
                specificity = sum(
                    max(1, candidate_count - candidate_stem_frequency[stem] + 1)
                    for stem in matched_stems
                )
                try:
                    rank = int(result.get("rank", fallback_rank))
                except (TypeError, ValueError):
                    rank = fallback_rank
                ranked.append(
                    (
                        retains_repeated_subject,
                        -specificity,
                        -overlap,
                        rank,
                        order,
                        url,
                    )
                )
                ranked_urls.add(normalized)
                order += 1
        # Browser search snapshots are also verified discovery evidence. They
        # matter when the high-level provider returned only marketplace lists
        # and the browser subsequently exposed a concrete article or offer.
        # Rank URLs with their surrounding result block so translated titles
        # can retain the owner's subject even when the URL itself cannot.
        for call in successful_calls:
            if call.get("tool") != "browser_snapshot":
                continue
            excerpt = str(call.get("result_excerpt", ""))
            page_url_match = re.search(
                r"^- Page URL:\s*(https?://\S+)",
                excerpt,
                re.MULTILINE | re.IGNORECASE,
            )
            if (
                page_url_match is None
                or not TaskContract.is_search_listing_url(
                    page_url_match.group(1)
                )
            ):
                continue
            base_url = page_url_match.group(1)
            candidate_matches: list[tuple[str, int]] = [
                (
                    match.group(0).rstrip("`),.;:]}\\"),
                    match.start(),
                )
                for match in re.finditer(
                    r"https?://[^\s<>\"']+",
                    excerpt,
                    re.IGNORECASE,
                )
            ]
            candidate_matches.extend(
                (
                    urljoin(base_url, match.group(1).strip().strip('"')),
                    match.start(),
                )
                for match in re.finditer(
                    r"-\s*/url:\s*([^\s<>]+)",
                    excerpt,
                    re.IGNORECASE,
                )
                if match.group(1).strip()
                and not match.group(1).strip().casefold().startswith(
                    ("#", "javascript:", "mailto:", "tel:")
                )
            )
            for url, match_start in candidate_matches:
                normalized = cls._normalized_web_url(url)
                if (
                    not normalized
                    or normalized in ranked_urls
                    or normalized in visited
                    or normalized in failed_urls
                    or TaskContract.is_search_listing_url(url)
                ):
                    continue
                hostname = (urlsplit(normalized).hostname or "").casefold()
                hostname_labels = set(hostname.split("."))
                if hostname.endswith("duckduckgo.com") or hostname_labels & {
                    "help",
                    "legal",
                    "pomoc",
                    "privacy",
                    "support",
                }:
                    continue
                normalized_parts = urlsplit(normalized)
                if normalized_parts.path == "/" and not normalized_parts.query:
                    continue
                infrastructure_segments = {
                    "about",
                    "account",
                    "auth",
                    "contact",
                    "cookie",
                    "cookies",
                    "help",
                    "legal",
                    "login",
                    "policy",
                    "privacy",
                    "signin",
                    "signup",
                    "terms",
                }
                path_segments = {
                    unquote(segment).casefold()
                    for segment in normalized_parts.path.split("/")
                    if segment
                }
                if path_segments & infrastructure_segments:
                    continue
                excerpt_lines = excerpt.splitlines(keepends=True)
                line_number = excerpt[:match_start].count("\n")
                line_start = max(0, line_number - 4)
                line_end = min(len(excerpt_lines), line_number + 7)
                block_start = sum(len(line) for line in excerpt_lines[:line_start])
                block_end = sum(len(line) for line in excerpt_lines[:line_end])
                candidate_text = unquote(excerpt[block_start:block_end]).casefold()
                retains_repeated_subject = any(
                    stem in candidate_text for stem in repeated_subject_stems
                )
                overlap = sum(
                    token in candidate_text
                    or (len(token) >= 5 and token[:5] in candidate_text)
                    for token in objective_tokens
                )
                if overlap <= 0:
                    continue
                matched_stems = {
                    stem for stem in objective_stems if stem in candidate_text
                }
                specificity = sum(
                    max(1, candidate_count - candidate_stem_frequency[stem] + 1)
                    for stem in matched_stems
                )
                ranked.append(
                    (
                        retains_repeated_subject,
                        -specificity,
                        -overlap,
                        1_000,
                        order,
                        url,
                    )
                )
                ranked_urls.add(normalized)
                order += 1
        if not ranked:
            return ""
        # Repetition is only a tie-breaker. Inflected generic nouns such as
        # "species" may repeat in the objective and must not suppress a rarer,
        # more discriminating scope anchor such as a region or proper name.
        ranked.sort(
            key=lambda item: (
                item[1],
                item[2],
                not item[0],
                item[3],
                item[4],
            )
        )
        return ranked[0][5]

    @staticmethod
    def _is_strong_detail_url(url: str) -> bool:
        """Recognize URL structure that clearly identifies one source/item."""

        try:
            parsed = urlsplit(url)
        except ValueError:
            return False
        ordered_segments = [
            unquote(segment).casefold()
            for segment in parsed.path.split("/")
            if segment
        ]
        segments = set(ordered_segments)
        hostname = (parsed.hostname or "").casefold()
        if hostname in {"github.com", "www.github.com"}:
            reserved_roots = {
                "collections",
                "events",
                "features",
                "login",
                "marketplace",
                "orgs",
                "search",
                "settings",
                "signup",
                "sponsors",
                "topics",
                "trending",
            }
            # A GitHub repository root has exactly owner/repository. Tabs such
            # as issues, pulls, security and pulse are supporting navigation,
            # not independent candidates for a research report.
            return (
                len(ordered_segments) == 2
                and ordered_segments[0] not in reserved_roots
            )
        if segments.intersection(
            {
                "ad",
                "article",
                "blog",
                "doc",
                "docs",
                "item",
                "items",
                "offer",
                "oferta",
                "post",
                "product",
                "products",
            }
        ):
            return True
        query_fields = {key.casefold() for key in parse_qs(parsed.query)}
        return bool(query_fields.intersection({"id", "item", "offer", "p", "product"}))

    @classmethod
    def _runtime_explicit_web_observation_request(
        cls,
        prompt: str,
        contract: TaskContract,
        definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]] | None:
        """Visit and observe each owner-supplied URL in its original order.

        Exact addresses are runtime data, not prose for the model to copy.  The
        deterministic sequence also prevents a search listing from satisfying
        the evidence contract before the referenced sources are opened.
        """

        if contract.requires_web_discovery:
            return None
        targets = extract_web_targets(prompt)
        # A single direct address already has robust argument repair and should
        # remain model-driven except visual-inspection prerequisites. The scheduler
        # also serves comparisons,
        # where losing the second or later address caused premature completion.
        if not targets or len(targets) < 2 and "browser_vision" not in contract.required_tools:
            return None
        available = {
            str(item.get("function", {}).get("name", ""))
            for item in definitions
            if isinstance(item, dict)
        }
        if not {"browser_navigate", "browser_snapshot"}.issubset(available):
            return None

        for target in targets:
            normalized = cls._normalized_web_url(target)
            navigation_index = next(
                (
                    index
                    for index, call in enumerate(successful_calls)
                    if call.get("tool") == "browser_navigate"
                    and isinstance(call.get("arguments"), dict)
                    and cls._normalized_web_url(
                        str(call["arguments"].get("url", ""))
                    )
                    == normalized
                ),
                None,
            )
            if navigation_index is None:
                arguments = {"url": target}
                if not cls._tool_argument_problem(
                    "browser_navigate", arguments, definitions
                ):
                    return "browser_navigate", arguments
                return None
            if not any(
                index > navigation_index and call.get("tool") == "browser_snapshot"
                for index, call in enumerate(successful_calls)
            ):
                arguments: dict[str, Any] = {}
                if not cls._tool_argument_problem(
                    "browser_snapshot", arguments, definitions
                ):
                    return "browser_snapshot", arguments
                return None
        return None

    @classmethod
    def _runtime_generated_tool_execution_request(
        cls,
        prompt: str,
        contract: TaskContract,
        definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
        *,
        observed_snapshot: str = "",
        generated_contract: GeneratedToolContract | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Bind the post-activation execution without asking the model again."""

        missing = set(contract.unmet(successful_calls))
        if "generated_tool_execution" not in missing:
            return None
        created = cls._created_tool_names(successful_calls)
        if not created:
            return None
        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") in created
            ),
            None,
        )
        if definition is None:
            return None
        function = definition.get("function", {})
        name = str(function.get("name", ""))
        schema = function.get("parameters", {})
        if not name or not isinstance(schema, dict):
            return None
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return None
        if generated_contract is not None:
            arguments = deepcopy(generated_contract.final_arguments)
            if any(str(field) not in arguments for field in required):
                return None
            return name, arguments
        assignments = cls._structured_literal_assignments(prompt)
        arguments: dict[str, Any] = {}
        for field in properties:
            values = assignments.get(str(field))
            if values:
                arguments[str(field)] = deepcopy(values[-1])
            elif field == "snapshot_text" and observed_snapshot:
                arguments[str(field)] = observed_snapshot
        if any(str(field) not in arguments for field in required):
            return None
        return name, arguments

    @staticmethod
    def _phase_tool_definitions(
        contract: TaskContract,
        definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
        failed_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Expose only the tools needed by the current contract phase.

        A mixed browser-plus-learning task initially needs both families, but
        keeping every browser schema after observation bloats a small model's
        prompt and invites it to repeat snapshots. Once browser evidence is
        complete, isolate artifact creation; after activation, isolate the new
        generated tool so it is executed before finalization.
        """

        missing = set(contract.unmet(successful_calls))
        created_tool_names = Agent._created_tool_names(successful_calls)
        browser_missing = any(item.startswith("browser_") for item in missing)
        successful_searches = sum(
            call.get("tool") == "web_search"
            and call.get("status", "succeeded") == "succeeded"
            for call in successful_calls
        )
        if (
            contract.requires_web_discovery
            and browser_missing
            and successful_searches
            >= (6 if contract.required_research_facets else 3)
        ):
            # A local model can endlessly paraphrase one discovery query while
            # making no structural progress. Three successful searches are a
            # bounded discovery phase; after that, force inspection of one of
            # the grounded URLs already present in runtime evidence.
            without_search = [
                item
                for item in definitions
                if item.get("function", {}).get("name") != "web_search"
            ]
            if without_search:
                definitions = without_search
        failed_reads = sum(
            call.get("tool") == "web_read"
            and call.get("status", "failed") == "failed"
            for call in (failed_calls or [])
        )
        browser_navigation_available = any(
            item.get("function", {}).get("name") == "browser_navigate"
            for item in definitions
        )
        if browser_missing and failed_reads >= 2 and browser_navigation_available:
            # High-level readers can be rejected by a host, by URL grounding,
            # or by every configured provider. Repeating the same transport
            # with invented URL variants cannot make progress. After two
            # failures, remove that route and force the model onto the real
            # browser, which can inspect one of the already observed URLs.
            definitions = [
                item
                for item in definitions
                if item.get("function", {}).get("name") != "web_read"
            ]
        navigation_events = [
            call
            for call in [*successful_calls, *(failed_calls or [])]
            if call.get("tool") == "browser_navigate"
        ]
        sequenced_navigation_events = [
            call for call in navigation_events if call.get("sequence") is not None
        ]
        latest_navigation_failed = False
        if sequenced_navigation_events:
            latest_navigation = max(
                sequenced_navigation_events,
                key=lambda call: int(call.get("sequence") or 0),
            )
            latest_navigation_failed = latest_navigation.get("status") == "failed"
        elif navigation_events and not any(
            call.get("status", "succeeded") == "succeeded"
            for call in navigation_events
        ):
            latest_navigation_failed = True
        if browser_missing and latest_navigation_failed:
            # A snapshot has no valid page state after a failed navigation.
            # Hiding it for this phase forces recovery navigation/search and
            # eliminates the fail -> snapshot -> fail loop.
            definitions = [
                item
                for item in definitions
                if item.get("function", {}).get("name") != "browser_snapshot"
            ]
        if "full_tor_fetch:candidate_verification" in missing:
            verification_tools = [
                item
                for item in definitions
                if item.get("function", {}).get("name")
                in {"full_tor_fetch", "full_tor_inventory"}
            ]
            if verification_tools:
                return verification_tools
        required_tool_missing = set(contract.required_tools).intersection(missing)
        if required_tool_missing:
            if "browser_vision" in required_tool_missing and browser_missing:
                # Pixel inspection needs a navigated page. Keep those
                # prerequisites available before requiring the visual tool.
                return [item for item in definitions if item.get("function", {}).get("name")
                        in {"browser_navigate", "browser_snapshot", "browser_vision", "web_search", "web_read"}]
            repaired_tickets = {
                str(call.get("repaired_ticket_id", ""))
                for call in successful_calls
                if call.get("tool") == "learning_create_repair_adapter"
                and call.get("repaired_ticket_id")
            }
            pending_generated_recovery = any(
                isinstance(call.get("recovery_ticket"), dict)
                and str(call["recovery_ticket"].get("capability", "")).startswith(
                    "generated."
                )
                and str(call["recovery_ticket"].get("ticket_id", ""))
                not in repaired_tickets
                for call in (failed_calls or [])
            )
            if pending_generated_recovery:
                repair = [
                    item
                    for item in definitions
                    if item.get("function", {}).get("name")
                    == "learning_create_repair_adapter"
                ]
                if repair:
                    return repair
            required = [
                item
                for item in definitions
                if item.get("function", {}).get("name") in required_tool_missing
            ]
            if required:
                return required
        artifact_missing = "learning_create_tool_or_skill" in missing
        if (
            missing.intersection(
                {
                    "learning_create_tool",
                    "learning_create_skill",
                    "learning_create_tool_or_skill",
                }
            )
            and not browser_missing
        ):
            allowed: set[str] = {"image_metadata", "image_analyze", "document_read", "audio_transcribe", "audio_analyze"}
            if "learning_create_tool" in missing or artifact_missing:
                allowed.update(
                    {
                        "learning_create_tool",
                        "learning_create_snapshot_extractor",
                    }
                )
            if "learning_create_skill" in missing or artifact_missing:
                allowed.add("learning_create_skill")
            return [
                item
                for item in definitions
                if item.get("function", {}).get("name") in allowed
            ]

        if (
            "generated_tool_execution" in missing
            and "learning_create_tool" not in missing
        ):
            created_name = ""
            for call in reversed(successful_calls):
                if call.get("tool") not in {
                    "learning_create_tool",
                    "learning_create_snapshot_extractor",
                }:
                    continue
                trusted_name = str(call.get("created_tool_name", "")).strip()
                if trusted_name:
                    created_name = trusted_name
                    break
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and payload.get("name"):
                    created_name = str(payload["name"])
                    break
            if created_name:
                generated = [
                    item
                    for item in definitions
                    if item.get("function", {}).get("name") == created_name
                ]
                if generated:
                    return generated
                # The creation phase is already complete.  Never fall back to
                # the broad catalog here, because that re-exposes the builder
                # and lets the model create a second, unrelated artifact.
                return []

        if created_tool_names:
            # A successful creation closes every creation entry point for this
            # objective. Once the generated tool itself has also run, remove it
            # as well so the remaining browser/file/report phases cannot drift
            # back into creation or repeat a completed execution.
            closed = {
                "learning_create_tool",
                "learning_create_snapshot_extractor",
            }
            if "generated_tool_execution" not in missing:
                closed.update(created_tool_names)
            definitions = [
                item
                for item in definitions
                if item.get("function", {}).get("name") not in closed
            ]

        return definitions

    @staticmethod
    def _creation_lifecycle_builder(
        requested_tool: str,
        active_definitions: list[dict[str, Any]],
        contract: TaskContract,
        successful_calls: list[dict[str, Any]],
    ) -> str:
        """Return the builder required before an invented target can run.

        This is deliberately domain-agnostic. It depends only on the current
        contract phase and the exact executable catalog sent to the model, not
        on a generated tool name, language, or task-specific keywords.
        """

        available = {
            str(item.get("function", {}).get("name", ""))
            for item in active_definitions
            if isinstance(item, dict)
        }
        if not requested_tool or requested_tool in available:
            return ""
        if "learning_create_tool" not in set(contract.unmet(successful_calls)):
            return ""
        for builder in (
            "learning_create_tool",
            "learning_create_snapshot_extractor",
        ):
            if builder in available:
                return builder
        return ""

    @staticmethod
    def _phase_tool_choice(
        contract: TaskContract,
        active_definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
    ) -> str | dict[str, Any]:
        """Force lifecycle routing when the executor knows the required phase.

        The LLM still supplies the complete artifact payload, source, and test.
        It does not get to replace the active builder's function name with the
        name of a future tool merely because that target appears in the user
        request. This keeps phase control in DARKLINGER rather than model prose.
        """

        available = {
            str(item.get("function", {}).get("name", ""))
            for item in active_definitions
            if isinstance(item, dict)
        }
        missing = set(contract.unmet(successful_calls))
        required = next(
            (
                name
                for name in ("learning_create_tool", "learning_create_skill")
                if name in missing and name in available
            ),
            "",
        )
        if not required:
            return "auto"
        return {
            "type": "function",
            "function": {"name": required},
        }

    @staticmethod
    def _parse_active_lifecycle_payload(
        answer: str,
        active_definitions: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]] | None:
        """Recover a bare rich-builder payload without guessing its purpose.

        Some tool-capable local models emit the arguments object itself instead
        of the provider's outer function-call envelope. Recovery is allowed only
        while a rich lifecycle builder is actually exposed, only when all keys
        belong to that builder's schema, and only for a payload with enough
        manifest structure to distinguish it from ``{"name": ...}`` targeting
        a not-yet-created tool. Ordinary tool arguments are never reclassified.

        Structural field errors are intentionally left for the normal schema
        validator. That produces actionable feedback but cannot execute invalid
        source or an incomplete test.
        """

        text = answer.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        elif text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or "name" not in payload:
            return None

        matches: list[str] = []
        for item in active_definitions:
            function = item.get("function", {}) if isinstance(item, dict) else {}
            name = str(function.get("name", ""))
            if name not in {"learning_create_tool", "learning_create_skill"}:
                continue
            schema = function.get("parameters", {})
            properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
            if not isinstance(properties, dict):
                continue
            keys = set(payload)
            if (
                len(keys) >= 3
                and keys.issubset(properties)
                and len(keys & set(schema.get("required", []))) >= 2
            ):
                matches.append(name)
        if len(matches) != 1:
            return None
        return matches[0], payload

    @staticmethod
    def _explicitly_named_tools(
        prompt: str,
        definitions: list[dict[str, Any]],
    ) -> list[str]:
        text = prompt.casefold()
        exact_names: list[str] = []
        for item in definitions:
            name = str(item.get("function", {}).get("name", "")).strip()
            if not name:
                continue
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(name.casefold())}(?![a-z0-9_])",
                text,
            ):
                exact_names.append(name)
        if exact_names:
            return list(dict.fromkeys(exact_names))

        # Voice transcription commonly turns an identifier such as
        # ``count_words`` into natural-language words (for example
        # ``Can't Words``). Permit one conservative, unique fuzzy match so an
        # STT typo does not make the agent invent a replacement tool. This is
        # deliberately limited to multi-part identifiers and requires a clear
        # margin over the next candidate; ambiguity therefore fails closed.
        prompt_words = Agent._tool_name_words(prompt)
        candidates: list[tuple[float, str]] = []
        for item in definitions:
            name = str(item.get("function", {}).get("name", "")).strip()
            name_words = Agent._tool_name_words(name)
            if not name or len(name_words) < 2 or len(prompt_words) < len(name_words):
                continue
            expected = " ".join(name_words)
            scores = [
                SequenceMatcher(
                    None,
                    " ".join(prompt_words[index : index + len(name_words)]),
                    expected,
                ).ratio()
                for index in range(len(prompt_words) - len(name_words) + 1)
                if Agent._has_spoken_tool_invocation_cue(prompt_words, index)
            ]
            if not scores:
                continue
            score = max(scores)
            if score >= 0.84:
                candidates.append((score, name))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not candidates:
            return []
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
            return []
        return [candidates[0][1]]

    @staticmethod
    def _has_spoken_tool_invocation_cue(words: list[str], index: int) -> bool:
        """Require a nearby call cue before fuzzy-matching a spoken tool name.

        Natural task descriptions can contain phrases such as ``browser
        snapshot text`` without asking DARKLINGER to invoke the built-in
        ``browser_snapshot`` function specifically. Voice transcription repair
        therefore applies only after a local call phrase (``use tool``, ``użyj
        narzędzia`` and close equivalents), while an exact underscored
        identifier remains explicit without this heuristic.
        """

        prefix = words[max(0, index - 6) : index]
        exact_cues = {
            "call",
            "execute",
            "invoke",
            "run",
            "use",
        }
        cue_prefixes = (
            "odpal",
            "uruchom",
            "uzy",
            "uży",
            "wywol",
            "wywoł",
        )
        return any(
            word in exact_cues or word.startswith(cue_prefixes)
            for word in prefix
        )

    @staticmethod
    def _tool_name_words(value: str) -> list[str]:
        normalized = value.casefold().replace("'", "").replace("’", "")
        normalized = normalized.replace("_", " ").replace("-", " ")
        return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)

    @staticmethod
    def _repair_explicit_text_arguments(
        prompt: str,
        tool_name: str,
        arguments: Any,
        definitions: list[dict[str, Any]],
        contract: TaskContract,
    ) -> Any:
        if (
            (
                tool_name not in contract.required_tools
                and not (
                    tool_name == "read_file"
                    and contract.requires_file_read
                )
            )
            or not isinstance(arguments, dict)
        ):
            return arguments
        definition = next(
            (
                item
                for item in definitions
                if item.get("function", {}).get("name") == tool_name
            ),
            None,
        )
        if definition is None:
            return arguments
        schema = definition.get("function", {}).get("parameters", {})
        if not isinstance(schema, dict) or schema.get("type") != "object":
            return arguments
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if (
            not isinstance(required, list)
            or len(required) != 1
            or not isinstance(properties, dict)
        ):
            return arguments
        field = str(required[0])
        field_schema = properties.get(field, {})
        if not isinstance(field_schema, dict) or field_schema.get("type") != "string":
            return arguments

        # A URL supplied by Boss is immutable task data, not prose for the
        # worker model to transcribe. Small local models regularly insert
        # spaces, swap characters, or split suffixes in long addresses. When a
        # required tool has one URL field and the durable objective contains
        # exactly one HTTP(S) URL, bind that exact value in runtime code. This
        # is schema-driven and therefore applies to every required URL tool,
        # not to one site or language.
        if field.casefold() == "url":
            candidates = [
                item.rstrip(".,;:!?)\"]}")
                for item in re.findall(
                    r"https?://[^\s<>\"'“”„]+",
                    prompt,
                    re.IGNORECASE,
                )
            ]
            unique = list(dict.fromkeys(candidates))
            if len(unique) == 1:
                repaired = dict(arguments)
                repaired[field] = unique[0]
                if tool_name == "full_tor_inventory":
                    # Crawl budgets are part of the owner's immutable task
                    # contract. The worker model may omit or enlarge optional
                    # schema fields, but it may not silently turn a two-page
                    # inspection into the twelve-page default.
                    if contract.tor_inventory_max_pages > 0:
                        repaired["max_pages"] = contract.tor_inventory_max_pages
                    if contract.tor_inventory_max_depth >= 0:
                        repaired["max_depth"] = contract.tor_inventory_max_depth
                return repaired

        if tool_name == "read_file" and field == "path":
            # A named read-only call may bind one literal path from the owner's
            # message. Do not use the generic "last quotation" heuristic for a
            # filesystem target: quoted prose is not permission to read a file.
            without_urls = re.sub(
                r"https?://[^\s<>]+", "", prompt, flags=re.IGNORECASE
            )
            unquoted = re.sub(r'["“„][^"”]{1,500}["”]', " ", without_urls)
            candidates = re.findall(
                r"(?<![\w/.-])(?:\.?/?[\w.-]+/)*[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,15}\b",
                unquoted,
            )
            quoted_paths = [
                value.strip()
                for value in re.findall(r'["“„]([^"”]{1,500})["”]', without_urls)
                if re.fullmatch(
                    r"(?:\.?/?[\w .-]+/)*[\w .-]+\.[A-Za-z][A-Za-z0-9]{0,15}",
                    value.strip(),
                )
            ]
            candidates.extend(quoted_paths)
            if any(".." in value.split("/") for value in candidates):
                return arguments
            candidates = list(dict.fromkeys(
                value for value in candidates
                if ".." not in value.split("/")
            ))
            if len(candidates) == 1:
                return {**arguments, "path": candidates[0]}
            return arguments

        if arguments:
            return arguments
        quoted = re.findall(r'["“„]([^"”]{1,20000})["”]', prompt)
        if not quoted:
            return arguments
        return {field: quoted[-1]}

    @staticmethod
    def _agent_generation_budget(
        definitions: list[dict[str, Any]],
        *,
        context_tokens: int,
    ) -> int:
        names = {
            str(item.get("function", {}).get("name", ""))
            for item in definitions
            if isinstance(item, dict)
        }
        if "learning_create_snapshot_extractor" in names:
            return 512
        if "learning_create_tool" in names:
            # A tool bundle contains a manifest, schemas, deterministic tests,
            # and Python source. The small ordinary answer budget truncates
            # even small valid bundles and leaves llama.cpp with malformed JSON.
            return min(6_144, max(2_048, context_tokens // 3))
        if "learning_create_skill" in names:
            return min(2_048, max(1_024, context_tokens // 8))
        return 1024

    @staticmethod
    def _structured_literal_assignments(prompt: str) -> dict[str, list[Any]]:
        """Return exact JSON values assigned to identifiers in owner input.

        Local models are poor copy machines for long URLs and nested fixtures.
        The runtime can recover values such as ``records = [...]`` directly from
        the immutable user prompt without knowing anything about the tool's
        domain or the language used in the surrounding prose.
        """

        decoder = json.JSONDecoder()
        assignments: dict[str, list[Any]] = {}
        pattern = re.compile(
            r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        )
        for match in pattern.finditer(prompt):
            remainder = prompt[match.end() :]
            leading = len(remainder) - len(remainder.lstrip())
            try:
                value, _ = decoder.raw_decode(remainder[leading:])
            except (TypeError, json.JSONDecodeError):
                continue
            assignments.setdefault(match.group(1), []).append(value)
        return assignments

    @staticmethod
    def _created_tool_names(
        successful_calls: list[dict[str, Any]],
    ) -> set[str]:
        names: set[str] = set()
        for call in successful_calls:
            if call.get("tool") not in {
                "learning_create_tool",
                "learning_create_snapshot_extractor",
            }:
                continue
            if call.get("created_tool_name"):
                names.add(str(call["created_tool_name"]))
                continue
            try:
                payload = json.loads(str(call.get("result_excerpt", "")))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("name"):
                names.add(str(payload["name"]))
        return names

    @staticmethod
    def _generated_tool_report_issues(
        answer: str,
        generated_contract: GeneratedToolContract | None,
        successful_calls: list[dict[str, Any]],
    ) -> list[str]:
        """Keep generated-tool test evidence separate from live observations."""

        if (
            generated_contract is None
            or generated_contract.archetype != "client_har_summary"
        ):
            return []
        created = Agent._created_tool_names(successful_calls)
        if not created or not any(
            call.get("tool") in created
            and call.get("status", "succeeded") == "succeeded"
            for call in successful_calls
        ):
            return []
        folded = answer.casefold()
        issues: list[str] = []
        if not any(marker in folded for marker in ("fixture", "synthetic")):
            issues.append("answer:generated_fixture_not_disclosed")
        if not (
            "server-side" in folded
            and any(
                marker in folded
                for marker in (
                    "does not",
                    "doesn't",
                    "not measure",
                    "not monitor",
                    "nie mierzy",
                    "nie monitoruje",
                )
            )
        ):
            issues.append("answer:generated_tool_scope_missing")
        return issues

    @classmethod
    def _repair_grounded_generated_tool_arguments(
        cls,
        prompt: str,
        tool_name: str,
        arguments: Any,
        definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
    ) -> Any:
        """Bind generated-tool inputs to exact structured owner literals.

        The first occurrence of an input assignment is the creation fixture;
        the last is the post-activation execution input. The mechanism is based
        only on JSON assignment names and executable schemas/source accesses,
        so it applies to arbitrary generated tools rather than named patches.
        """

        if not isinstance(arguments, dict):
            return arguments
        assignments = cls._structured_literal_assignments(prompt)
        if not assignments:
            return arguments

        if tool_name == "learning_create_tool":
            test = arguments.get("test")
            source = arguments.get("source")
            if not isinstance(test, dict) or not isinstance(source, str):
                return arguments
            current = test.get("arguments")
            current_arguments = dict(current) if isinstance(current, dict) else {}
            referenced = set(current_arguments)
            referenced.update(
                re.findall(
                    r"arguments\s*(?:\.get\(\s*|\[\s*)['\"]"
                    r"([A-Za-z_][A-Za-z0-9_]*)['\"]",
                    source,
                )
            )
            grounded = dict(current_arguments)
            for field in referenced:
                values = assignments.get(field)
                if values:
                    grounded[field] = deepcopy(values[0])
            expected_values = assignments.get("expected")
            grounded_expected = (
                deepcopy(expected_values[0])
                if expected_values and isinstance(expected_values[0], dict)
                else test.get("expected")
            )
            if (
                grounded == current_arguments
                and grounded_expected == test.get("expected")
            ):
                return arguments
            repaired = dict(arguments)
            repaired_test = dict(test)
            repaired_test["arguments"] = grounded
            repaired_test["expected"] = grounded_expected
            repaired["test"] = repaired_test
            return repaired

        if tool_name not in cls._created_tool_names(successful_calls):
            return arguments
        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") == tool_name
            ),
            None,
        )
        parameters = (
            definition.get("function", {}).get("parameters", {})
            if isinstance(definition, dict)
            else {}
        )
        properties = (
            parameters.get("properties", {})
            if isinstance(parameters, dict)
            else {}
        )
        if not isinstance(properties, dict):
            return arguments
        repaired = dict(arguments)
        for field in properties:
            values = assignments.get(str(field))
            if values:
                repaired[str(field)] = deepcopy(values[-1])
        return repaired

    @staticmethod
    def _repair_research_detail_snapshot(
        tool_name: str,
        arguments: Any,
        contract: TaskContract,
        successful_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Prevent a detail-page report from observing only one tiny element."""

        if (
            tool_name != "browser_snapshot"
            or not isinstance(arguments, dict)
            or not contract.requires_evidence_report
            or not contract.requires_distinct_detail_page
            or "target" not in arguments
        ):
            return arguments
        latest_navigation = next(
            (
                call
                for call in reversed(successful_calls)
                if call.get("tool") == "browser_navigate"
                and isinstance(call.get("arguments"), dict)
            ),
            None,
        )
        if latest_navigation is None:
            return arguments
        url = str(latest_navigation["arguments"].get("url", ""))
        if not url:
            return arguments
        target = str(arguments.get("target", "")).strip().casefold()
        if target in {"", "body", "html"}:
            return arguments
        repaired = {
            key: value
            for key, value in arguments.items()
            if key not in {"boxes", "filename", "target"}
        }
        repaired["target"] = "body"
        try:
            depth = int(repaired.get("depth", 0))
        except (TypeError, ValueError):
            depth = 0
        repaired["depth"] = max(4, min(depth, 6))
        return repaired

    @staticmethod
    def _repair_observed_snapshot_input(
        tool_name: str,
        arguments: Any,
        definitions: list[dict[str, Any]],
        successful_calls: list[dict[str, Any]],
        observed_snapshot: str,
    ) -> dict[str, Any]:
        """Bind a generated extractor to runtime evidence, not model-retyped data."""

        if not observed_snapshot or not isinstance(arguments, dict):
            return arguments
        created_names = Agent._created_tool_names(successful_calls)
        if tool_name not in created_names:
            return arguments
        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") == tool_name
            ),
            None,
        )
        parameters = (
            definition.get("function", {}).get("parameters", {})
            if isinstance(definition, dict)
            else {}
        )
        properties = (
            parameters.get("properties", {})
            if isinstance(parameters, dict)
            else {}
        )
        if "snapshot_text" not in properties:
            return arguments
        repaired = dict(arguments)
        repaired["snapshot_text"] = observed_snapshot
        return repaired

    @staticmethod
    def _tool_argument_problem(
        tool_name: str,
        arguments: Any,
        definitions: list[dict[str, Any]],
    ) -> str:
        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") == tool_name
            ),
            None,
        )
        if definition is None:
            return ""
        schema = definition.get("function", {}).get("parameters")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            return ""
        try:
            Agent._validate_tool_argument_value(arguments, schema, path="$")
        except ValueError as error:
            return str(error)
        return ""

    @classmethod
    def _missing_structured_tool_inputs(
        cls,
        prompt: str,
        tool_name: str,
        definitions: list[dict[str, Any]],
    ) -> tuple[str, ...]:
        """Return required structured payloads absent from the current turn.

        A generated analyzer can be referenced in a later task, but its old
        validation fixture is not fresh observation data.  Fields explicitly
        shaped as serialized JSON must therefore come from the immutable
        current request.  This rule is schema-based and independent of the
        language surrounding the payload.
        """

        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") == tool_name
            ),
            None,
        )
        if not isinstance(definition, dict):
            return ()
        schema = definition.get("function", {}).get("parameters", {})
        if not isinstance(schema, dict):
            return ()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return ()
        assignments = cls._structured_literal_assignments(prompt)
        missing: list[str] = []
        for raw_field in required:
            field = str(raw_field)
            field_schema = properties.get(field, {})
            if not isinstance(field_schema, dict):
                continue
            serialized_json = (
                field_schema.get("type") == "string"
                and (
                    field.casefold().endswith("_json")
                    or field_schema.get("contentMediaType") == "application/json"
                )
            )
            if serialized_json and not assignments.get(field):
                missing.append(field)
        return tuple(missing)

    @classmethod
    def _is_required_argument_clarification(
        cls,
        prompt: str,
        answer: str,
        contract: TaskContract,
        definitions: list[dict[str, Any]],
    ) -> bool:
        """Accept a bounded question only when a required call lacks exact data."""

        normalized = " ".join(str(answer or "").split())
        if (
            not normalized
            or "?" not in normalized
            or len(normalized.split()) > 120
        ):
            return False
        required_names = list(dict.fromkeys(contract.required_tools))
        if len(required_names) != 1:
            return False
        name = required_names[0]
        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") == name
            ),
            None,
        )
        if not isinstance(definition, dict):
            return False
        schema = definition.get("function", {}).get("parameters", {})
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if not required or any(not isinstance(item, str) for item in required):
            return False
        repaired = cls._repair_explicit_text_arguments(
            prompt,
            name,
            {},
            definitions,
            contract,
        )
        if isinstance(repaired, dict) and all(field in repaired for field in required):
            return False

        folded_answer = normalized.casefold()
        exact_values = re.findall(
            r"https?://[^\s<>]+|(?<![\w/.-])(?:\.?/?[\w.-]+/)*[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,15}\b",
            prompt,
            re.IGNORECASE,
        )
        for value in exact_values:
            clean_value = value.rstrip(".,;:!?)")
            parsed_path = Path(clean_value)
            variants = {
                clean_value.casefold(),
                parsed_path.name.casefold(),
                parsed_path.stem.casefold(),
            }
            if any(
                len(variant) >= 2
                and re.search(
                    rf"(?<![\w.-]){re.escape(variant)}(?![\w.-])",
                    folded_answer,
                )
                for variant in variants
            ):
                return True
        field_words = {
            word
            for field in required
            for word in re.findall(r"[a-z0-9]+", field.casefold())
            if len(word) >= 3
        }
        field_aliases = {
            "path": {"file", "folder", "directory"},
            "url": {"link", "page", "site"},
            "query": {"search", "term"},
        }
        for field in required:
            field_words.update(field_aliases.get(field.casefold(), set()))
        if any(word in folded_answer for word in field_words):
            return True
        return False

    @staticmethod
    def _is_model_correctable_tool_error(error: str | None) -> bool:
        """Return whether another qualified model can repair the call payload."""

        text = str(error or "").casefold()
        return any(
            marker in text
            for marker in (
                "toolargumentvalidationerror",
                "generatedtoollifecycleerror",
                "unknowntoolerror",
                "failed to parse tool call arguments",
                "artifact names must match",
                "artifact version must be semantic",
                "skill requires",
                "missing required field",
                "missing required fields",
                "unexpected field",
                "unexpected fields",
            )
        )

    @staticmethod
    def _is_terminal_tool_provider_error(
        error: str | None,
        failure_details: dict[str, Any] | None,
    ) -> bool:
        """Return whether another model request cannot restore the provider.

        Rephrasing a query cannot install a missing browser binary or reopen a
        provider circuit.  Stop at the first trustworthy infrastructure
        diagnosis instead of spending the remaining agent budget on cosmetic
        variants of the same call.
        """

        details = failure_details or {}
        reason = str(details.get("reason", "")).casefold()
        if reason in {
            "matching_providers_circuit_open",
            "no_eligible_provider",
        }:
            return True

        text = str(error or "").casefold()
        missing_browser_markers = (
            "browser is not installed",
            'browser "firefox" is not installed',
            "install-browser firefox",
            "expected executable at",
            "executable doesn't exist at",
        )
        return any(marker in text for marker in missing_browser_markers)

    @staticmethod
    def _validate_tool_argument_value(
        value: Any,
        schema: dict[str, Any],
        *,
        path: str,
    ) -> None:
        kind = schema.get("type")
        valid = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": (
                isinstance(value, (int, float)) and not isinstance(value, bool)
            ),
            "boolean": isinstance(value, bool),
        }.get(str(kind), True)
        if not valid:
            raise ValueError(f"{path} must be {kind}")
        if "enum" in schema and value not in schema.get("enum", []):
            raise ValueError(f"{path} is not an allowed value")
        if kind == "object":
            properties = schema.get("properties", {})
            properties = properties if isinstance(properties, dict) else {}
            required = schema.get("required", [])
            required = required if isinstance(required, list) else []
            missing = [name for name in required if name not in value]
            if missing:
                raise ValueError(f"{path} is missing required fields: {missing}")
            if schema.get("additionalProperties") is False:
                extra = [name for name in value if name not in properties]
                if extra:
                    raise ValueError(f"{path} has unexpected fields: {extra}")
            for name, item in value.items():
                child = properties.get(name)
                if isinstance(child, dict):
                    Agent._validate_tool_argument_value(
                        item,
                        child,
                        path=f"{path}.{name}",
                    )
        elif kind == "array":
            minimum = schema.get("minItems")
            if isinstance(minimum, int) and len(value) < minimum:
                raise ValueError(f"{path} has fewer than {minimum} items")
            child = schema.get("items")
            if isinstance(child, dict) and child.get("type"):
                for index, item in enumerate(value):
                    Agent._validate_tool_argument_value(
                        item,
                        child,
                        path=f"{path}[{index}]",
                    )
        elif kind == "string":
            minimum = schema.get("minLength")
            if isinstance(minimum, int) and len(value) < minimum:
                raise ValueError(f"{path} is shorter than {minimum} characters")

    @staticmethod
    def _is_continuation_request(prompt: str) -> bool:
        """Recognize only DARKLINGER's language-neutral continuation protocol.

        Natural-language meaning belongs to ``MultilingualIntentRouter``. Keeping
        translated verb lists here caused ordinary sentences containing words
        such as "dalej" to resurrect unrelated checkpoints. Slash commands are
        exact runtime protocol and therefore need no language dictionary.
        """

        text = " ".join(prompt.casefold().replace("’", "'").split()).strip(" .!?,;:")
        return text in {"/continue", "/continue --continuous"}

    @staticmethod
    def _contract_has_execution_route(contract: TaskContract) -> bool:
        return any(
            (
                contract.requires_browser_navigation,
                contract.requires_file_read,
                contract.requires_file_mutation,
                contract.requires_command_execution,
                contract.requires_created_tool,
                contract.requires_created_skill,
                contract.requires_created_artifact,
                contract.allows_artifact_fallback,
                contract.requires_runtime_review,
                bool(contract.required_tools),
                bool(contract.required_capabilities),
                bool(contract.required_media_paths),
            )
        )

    def _continued_action_context(
        self,
        prompt: str,
        trace: AgentTaskTrace | None,
        *,
        force: bool = False,
    ) -> dict[str, Any] | None:
        if not force and not self._is_continuation_request(prompt):
            return None

        previous = getattr(self, "_last_execution_context", None)
        if isinstance(previous, dict):
            if AgentTaskTrace.context_has_action_route(previous):
                return previous
            # An invalid runtime-review checkpoint is not an empty conversational
            # turn. Do not walk past it and unexpectedly resurrect an unrelated,
            # older browser or command task.
            requirements = previous.get("requirements", {})
            calls = previous.get("tool_calls", [])
            if (
                isinstance(requirements, dict)
                and requirements.get("requires_runtime_review")
            ) or (
                isinstance(calls, list)
                and any(
                    isinstance(call, dict)
                    and call.get("tool") == "runtime_review_task"
                    for call in calls
                )
            ):
                if trace is not None:
                    trace.record_event(
                        "continuation_context_rejected",
                        {
                            "source_task_id": previous.get("task_id", ""),
                            "reason": "ungrounded_runtime_review_checkpoint",
                        },
                    )
                return None

        root = getattr(self, "_agent_trace_root", None)
        if root is None and trace is not None:
            root = trace.root
        if root is None:
            return None
        return AgentTaskTrace.latest_action_context(
            root,
            exclude_task_id=trace.task_id if trace is not None else "",
        )

    @staticmethod
    def _requests_runtime_action(prompt: str, contract: TaskContract) -> bool:
        """Separate a request to act from discussion about an action.

        The model never gets to decide this boundary by itself. Questions about
        V's capabilities or explanations such as "how do I read a file" stay
        tool-free, while explicit imperatives and concrete task contracts enter
        the executor.
        """

        text = " ".join(prompt.casefold().split())
        capability_question = TaskContract.asks_about_creation_capability(prompt) or bool(
            re.search(
                r"\b(?:are you able to|do you know how to|"
                r"can you even|what (?:can|could) you|"
                r"czy (?:ty )?(?:potrafisz|umiesz)|"
                r"jakie (?:masz|posiadasz) (?:narzędzia|narzedzia|możliwości|mozliwosci))\b",
                text,
            )
        )
        explanatory_question = bool(
            re.match(
                r"^(?:how (?:can|could|do|does|would)|why (?:do|does|is)|"
                r"what (?:is|are|does)|"
                r"jak (?:mogę|moge|można|mozna|działa|dziala)|"
                r"dlaczego|czemu)\b",
                text,
            )
        )
        if capability_question or explanatory_question:
            return False

        if any(
            (
                contract.requires_browser_navigation,
                contract.requires_file_read,
                contract.requires_file_mutation,
                contract.requires_command_execution,
                contract.requires_created_tool,
                contract.requires_created_skill,
                contract.requires_created_artifact,
                contract.allows_artifact_fallback,
                contract.requires_runtime_review,
                bool(contract.required_tools),
                bool(contract.required_capabilities),
                bool(contract.required_media_paths),
            )
        ):
            return True

        english_action = (
            r"analy[sz]e|audit|browse|build|check|collect|create|delete|edit|execute|"
            r"extract|find|inspect|install|list|move|open|read|research|review|run|save|search|show|"
            r"start|stop|test|visit|write"
        )
        polish_imperative = (
            r"analizuj|przeanalizuj|dodaj|edytuj|napisz|otworz|otwórz|przeczytaj|przejrzyj|"
            r"przenieś|przenies|przetestuj|sprawdź|sprawdz|stwórz|stworz|"
            r"przeszukaj|przeszukać|przeszukac|uruchom|usuń|usun|wejdź|wejdz|"
            r"użyj|uzyj|wykonaj|wyszukaj|znajdź|znajdz|zrób|zrob"
        )
        return bool(
            re.search(
                rf"^(?:(?:v|boss)[,!: -]+|please\s+)?(?:{english_action})\b",
                text,
            )
            or re.search(rf"\b(?:{polish_imperative})\b", text)
            or re.search(
                rf"\b(?:can|could|would|will) you (?:please )?(?:{english_action})\b",
                text,
            )
            or re.search(
                rf"\b(?:i (?:need|want) you to|please) (?:{english_action})\b",
                text,
            )
            or re.search(
                rf"\b(?:czy możesz|czy mozesz|chcę żebyś|chce zebys|"
                rf"proszę (?:cię )?(?:żebyś )?|prosze (?:cie )?(?:zebys )?)"
                rf"[^.!?]{{0,50}}\b(?:{polish_imperative})\b",
                text,
            )
        )

    @staticmethod
    def _fit_tool_output(result: str, *, max_characters: int) -> str:
        if len(result) <= max_characters:
            return result
        marker = (
            "\n\n[DARKLINGER omitted the middle of this tool output from the model "
            "context; the full result remains in the runtime checkpoint.]\n\n"
        )
        body = max(0, max_characters - len(marker))
        head = int(body * 0.75)
        tail = body - head
        return result[:head] + marker + (result[-tail:] if tail else "")

    @staticmethod
    def _browser_evidence_text(tool: str, result: str) -> str | None:
        """Unwrap web_read snapshots before bounding model and ledger evidence.

        Clipping serialized JSON first retains navigation chrome and can lose
        the entire article or leave an undecodable content string. Keep the raw
        result hash separately; the excerpt must contain the observed text.
        """
        if tool == "browser_snapshot":
            return result
        if tool != "web_read":
            return None
        try:
            payload = json.loads(result)
        except (ValueError, TypeError):
            return None
        content = payload.get("content") if isinstance(payload, dict) else None
        if isinstance(content, str) and re.search(r"^- Page URL:", content, re.M):
            return content
        return None

    @classmethod
    def _fit_browser_snapshot_output(
        cls,
        result: str,
        *,
        max_characters: int,
        preserve_images: bool = False,
        prioritize_prose: bool = False,
    ) -> str:
        """Prioritize observed DuckDuckGo results over accessibility UI chrome.

        Playwright snapshots of a search page commonly begin with several
        thousand characters of navigation, menus, and privacy controls. Generic
        head/tail truncation can therefore hide every actual result from the
        model. Keep complete observed result blocks (title, URL, and snippet)
        without inventing or summarizing their contents.
        """

        if len(result) <= max_characters:
            return result

        page_url_match = re.search(r"^- Page URL:\s*(\S+)", result, re.MULTILINE)
        if page_url_match is None:
            return cls._fit_tool_output(result, max_characters=max_characters)
        page_hostname = (urlsplit(page_url_match.group(1)).hostname or "").casefold()
        if page_hostname not in {
            "duckduckgo.com",
            "www.duckduckgo.com",
            "html.duckduckgo.com",
        }:
            page_title_match = re.search(
                r"^- Page Title:\s*(.+)$",
                result,
                re.MULTILINE,
            )
            lines = result.splitlines()

            # Narrative web reads need article paragraphs, not ticker prices or
            # "related news" cards that happen to contain a currency symbol.
            # Preserve paragraph subtrees and adjacent lists in document order.
            # Structured shopping/inventory requests keep the existing path.
            if prioritize_prose:
                paragraphs = [
                    index for index, line in enumerate(lines)
                    if re.match(r"\s*- paragraph(?:\s|\[|:)", line)
                ]
                if len(paragraphs) >= 2:
                    blocks: list[str] = []
                    for start in paragraphs:
                        indent = len(lines[start]) - len(lines[start].lstrip())
                        end = start + 1
                        while end < len(lines):
                            line = lines[end]
                            depth = len(line) - len(line.lstrip())
                            adjacent_list = depth == indent and re.match(
                                r"\s*- list(?:\s|\[|:)", line,
                            )
                            if line.strip() and depth <= indent and not adjacent_list:
                                break
                            end += 1
                        blocks.append("\n".join(
                            re.sub(r"\s+\[(?:ref|cursor)=[^\]]+\]", "", line).rstrip()
                            for line in lines[start:end]
                        ))
                    header = "\n".join(
                        line for line in lines
                        if line.startswith(("- Page URL:", "- Page Title:"))
                    )
                    return cls._fit_tool_output(
                        header + "\n\n[DARKLINGER observed article excerpts]\n\n" + "\n\n".join(blocks),
                        max_characters=max_characters,
                    )

            # Repeated semantic ``article`` blocks are usually the actual cards
            # on a listing page (products, posts, offers), while the beginning
            # of the accessibility tree is navigation chrome. Preserve several
            # complete blocks before applying topic-token heuristics; otherwise
            # a category label such as "Books" can displace every real product.
            article_starts = [
                index
                for index, line in enumerate(lines)
                if re.search(r"^\s*- article(?:\s|\[|:)", line)
            ]
            if len(article_starts) >= 2:
                header = [
                    line
                    for line in lines
                    if line.startswith("- Page URL:")
                    or line.startswith("- Page Title:")
                ]
                blocks: list[str] = []
                for position, start in enumerate(article_starts[:8]):
                    end = (
                        article_starts[position + 1]
                        if position + 1 < len(article_starts)
                        else len(lines)
                    )
                    cleaned = [
                        re.sub(
                            r"\s+\[(?:ref|cursor)=[^\]]+\]",
                            "",
                            line,
                        ).rstrip()
                        for line in lines[start:end]
                    ]
                    block = "\n".join(cleaned).strip()
                    if block:
                        blocks.append(block)
                if blocks:
                    compact = "\n".join(header)
                    compact += (
                        "\n\n[DARKLINGER prioritized repeated observed page-item "
                        "blocks; this view may omit parts of the original "
                        "snapshot.]"
                    )
                    compact += "\n\n" + "\n\n".join(blocks)
                    return cls._fit_tool_output(
                        compact,
                        max_characters=max_characters,
                    )

            parsed_page = urlsplit(page_url_match.group(1))
            topic_text = " ".join(
                (
                    unquote(parsed_page.path.replace("-", " ").replace("_", " ")),
                    page_title_match.group(1) if page_title_match else "",
                )
            ).casefold()
            topic_stopwords = {
                "about", "best", "blog", "compare", "compared", "comparison",
                "guide", "home", "page", "pricing", "real", "resources",
                "review", "reviews", "tested", "tools", "with",
            }
            topic_tokens = {
                token
                for token in re.findall(r"[a-z0-9][a-z0-9.+]{4,}", topic_text)
                if token not in topic_stopwords and not token.isdigit()
            }
            relevant_indexes: list[int] = []
            for index, line in enumerate(lines):
                normalized = line.casefold()
                if (
                    not topic_tokens
                    or line.startswith("- Page ")
                    or "- /url:" in line
                ):
                    continue
                if any(token in normalized for token in topic_tokens):
                    relevant_indexes.append(index)

            # Comparison pages usually name the actual products in level-two
            # or level-three headings, while the opening paragraphs repeat the
            # generic topic many times. Preserve those structural sections first
            # so a small context window receives candidate names instead of only
            # the article introduction.
            concrete_fact_indexes = [
                index
                for index, line in enumerate(lines)
                if re.search(
                    r"(?:[$€£¥₽₹₿]\s*\d|"
                    r"\b\d[\d\s.,]*\s*(?:USD|EUR|GBP|PLN|JPY|CNY|"
                    r"CHF|CAD|AUD|SEK|NOK|DKK|CZK|HUF|RON|BGN)\b|"
                    r"\b(?:18|19)\d{2}\b)",
                    line,
                    flags=re.IGNORECASE,
                )
            ][:10]
            section_indexes = [
                index
                for index, line in enumerate(lines)
                if re.search(
                    r"\bheading\s+\"[^\"]+\"\s+\[level=(?:2|3)\]",
                    line,
                    flags=re.IGNORECASE,
                )
            ][:10]
            section_index_set = set(section_indexes)
            image_indexes = (
                [
                    index
                    for index, line in enumerate(lines)
                    if re.search(
                        r'\b(?:img|image)\s+"[^"]+"',
                        line,
                        flags=re.IGNORECASE,
                    )
                ][:10]
                if preserve_images
                else []
            )

            windows: list[tuple[int, int]] = []
            prioritized_indexes = concrete_fact_indexes + image_indexes + section_indexes + [
                index
                for index in relevant_indexes
                if index not in section_index_set
                and index not in set(concrete_fact_indexes)
            ]
            for index in prioritized_indexes:
                start = index
                end = min(
                    len(lines),
                    index + (7 if index in section_index_set else 14),
                )
                if (
                    windows
                    and windows[-1][0] <= start <= windows[-1][1]
                ):
                    windows[-1] = (windows[-1][0], max(windows[-1][1], end))
                else:
                    windows.append((start, end))
                if len(windows) >= 10:
                    break

            if windows:
                header = [
                    line
                    for line in lines
                    if line.startswith("- Page URL:")
                    or line.startswith("- Page Title:")
                ]
                excerpts: list[str] = []
                for start, end in windows:
                    cleaned: list[str] = []
                    for line in lines[start:end]:
                        line = re.sub(r"\s+\[ref=[^\]]+\]", "", line)
                        line = re.sub(r"\s+\[cursor=[^\]]+\]", "", line)
                        if "- /url:" in line and len(line) > 240:
                            continue
                        cleaned.append(line.rstrip())
                    excerpt = "\n".join(cleaned).strip()
                    if excerpt:
                        excerpts.append(excerpt)
                if excerpts:
                    compact = "\n".join(header)
                    compact += (
                        "\n\n[DARKLINGER prioritized topic-relevant detail-page "
                        "evidence; this view may omit parts of the original "
                        "snapshot.]"
                    )
                    compact += "\n\n" + "\n\n".join(excerpts)
                    return cls._fit_tool_output(
                        compact,
                        max_characters=max_characters,
                    )
            return cls._fit_tool_output(result, max_characters=max_characters)

        lines = result.splitlines()
        article_starts = [
            index
            for index, line in enumerate(lines)
            if re.search(r"^\s*- article(?:\s|\[|:)", line)
        ]
        if not article_starts:
            return cls._fit_tool_output(result, max_characters=max_characters)

        def external_urls(block: list[str]) -> list[str]:
            urls: list[str] = []
            for line in block:
                match = re.search(r"- /url:\s*(https?://\S+)", line)
                if match is None:
                    continue
                candidate = match.group(1).rstrip(".,;)")
                hostname = (urlsplit(candidate).hostname or "").casefold()
                if hostname.endswith("duckduckgo.com"):
                    continue
                urls.append(candidate)
            return urls

        header = [
            line
            for line in lines
            if line.startswith("- Page URL:") or line.startswith("- Page Title:")
        ]
        selected_blocks: list[str] = []
        seen_urls: set[str] = set()
        for position, start in enumerate(article_starts):
            end = (
                article_starts[position + 1]
                if position + 1 < len(article_starts)
                else len(lines)
            )
            block = lines[start:end]
            urls = external_urls(block)
            unseen_urls = [url for url in urls if url not in seen_urls]
            if not unseen_urls:
                continue
            seen_urls.update(unseen_urls)

            cleaned: list[str] = []
            for line in block:
                line = re.sub(r"\s+\[ref=[^\]]+\]", "", line)
                line = re.sub(r"\s+\[cursor=[^\]]+\]", "", line)
                if re.match(
                    r"^\s*- (?:article|button|generic|listitem)(?:\s*:\s*)?$",
                    line,
                ):
                    continue
                cleaned.append(line.rstrip())
            selected_blocks.append("\n".join(cleaned).strip())

        if not selected_blocks:
            return cls._fit_tool_output(result, max_characters=max_characters)

        compact = "\n".join(header)
        compact += (
            "\n\n[DARKLINGER prioritized observed DuckDuckGo search-result blocks; "
            "the full accessibility snapshot remains in the runtime checkpoint.]"
        )
        compact += "\n\n" + "\n\n".join(selected_blocks)
        return cls._fit_tool_output(compact, max_characters=max_characters)

    @staticmethod
    def _parse_tool_request(
        answer: str,
    ) -> tuple[str, dict | str] | None:

        text = answer.strip()
        decoder = json.JSONDecoder()
        candidates: list[tuple[tuple[str, dict | str], bool]] = []

        # Some local models violate the protocol by explaining an action and
        # then placing the real JSON object at the end. Decode that trailing
        # object without mistaking arbitrary JSON examples in prose for calls.
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                data, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            tool = data.get("tool")
            # Hermes-style compatibility payload. llama.cpp normally promotes
            # <tool_call>{"name": ..., "arguments": ...}</tool_call> into the
            # provider field, but some long or partially recovered generations
            # surface the inner FunctionCall object as content. Accept only the
            # narrow FunctionCall envelope so an artifact manifest such as
            # {"name": ..., "description": ...} can never execute by accident.
            # Hermes 3 can append these call-lifecycle metadata fields during
            # longer generations; they are not forwarded to the tool itself.
            hermes_call_keys = {
                "name",
                "arguments",
                "version",
                "scope",
                "timeout_seconds",
            }
            if (
                not isinstance(tool, str)
                and {"name", "arguments"}.issubset(data)
                and set(data).issubset(hermes_call_keys)
                and isinstance(data.get("name"), str)
                and (
                    "version" not in data
                    or isinstance(data.get("version"), str)
                )
                and (
                    "scope" not in data
                    or isinstance(data.get("scope"), str)
                )
                and (
                    "timeout_seconds" not in data
                    or isinstance(data.get("timeout_seconds"), (int, float))
                )
            ):
                tool = data["name"]
            arguments = data.get("arguments", {})
            if not isinstance(tool, str) or not tool.strip():
                continue
            if not isinstance(arguments, (dict, str)):
                continue
            trailing = text[end:].strip()
            candidates.append(
                ((tool.strip(), arguments), trailing in {"", "```"})
            )

        if len(candidates) == 1 and candidates[0][1]:
            return candidates[0][0]
        if candidates:
            return None

        # Backwards compatibility for older local models and prompts.
        if not text.startswith("TOOL:"):
            return None

        try:

            _, tool, arguments = text.split(
                ":",
                2,
            )

        except ValueError:

            return None

        tool = tool.strip()
        arguments = arguments.strip()

        if not tool:
            return None

        return tool, arguments

    @staticmethod
    def _claims_unverified_work(answer: str) -> bool:
        return False

    @staticmethod
    def _is_context_overflow(error: APIStatusError) -> bool:
        detail = f"{error} {getattr(error, 'body', '')}".casefold()
        return any(
            marker in detail
            for marker in (
                "context size",
                "context window",
                "exceeds the available context",
                "maximum context length",
                "prompt is too long",
                "too many tokens",
            )
        )

    @staticmethod
    def _successful_trace_tools(trace: AgentTaskTrace | None) -> list[str]:
        if trace is None:
            return []
        return [
            str(call.get("tool", ""))
            for call in trace.tool_calls
            if call.get("status") == "succeeded" and call.get("tool")
        ]

    @staticmethod
    def _unverified_execution_answer(categories: tuple[str, ...]) -> str:
        labels = {
            "external_communication": "external communication",
            "remote_system_access": "remote-system access",
            "filesystem_mutation": "a filesystem change",
            "filesystem_read": "a filesystem read",
            "command_execution": "command or test execution",
            "browser_action": "a browser action",
        }
        described = [labels[item] for item in categories if item in labels]
        subject = ", ".join(described) if described else "the requested work"
        return (
            f"No—{subject} did not happen. The model made the claim without "
            "matching successful tool evidence, so DARKLINGER killed it. Nothing is "
            "running in the background."
        )

    @staticmethod
    def _missing_required_tool_evidence(
        prompt: str,
        successful_tools: list[str],
    ) -> list[str]:
        calls = [
            {"tool": tool, "arguments": {}, "status": "succeeded"}
            for tool in successful_tools
        ]
        return TaskContract.from_prompt(prompt).unmet(calls)

    @staticmethod
    def _tool_result_error(result: str, *, tool: str = "") -> str:
        stripped = result.strip()
        lowered = stripped.casefold()
        error_markers = (
            "invalid arguments",
            "unknown mcp tool:",
            "unknown protocol.",
            "tool execution failed:",
            "learning runtime unavailable:",
            "sandbox unavailable:",
            "foundry unavailable:",
            "browser_click failed:",
        )
        if lowered.startswith(error_markers) or (
            " requires " in lowered
            and any(
                lowered.startswith(prefix)
                for prefix in (
                    "learning_",
                    "evm_",
                    "sandbox_",
                    "browser_",
                    "web_",
                )
            )
        ):
            return f"ToolReportedError: {stripped[:2_000]}"

        if tool.startswith("browser_"):
            http_error = re.search(r"\bHTTP status:\s*([45]\d\d)\b", stripped)
            if http_error:
                return (
                    "BrowserHTTPError: page returned HTTP status "
                    + http_error.group(1)
                )

        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""

        if tool in {"sandbox_execute_offline", "evm_foundry_test_offline"}:
            exit_code = payload.get("exit_code")
            failed_limits = [
                name
                for name in ("timed_out", "output_limited", "workspace_limited")
                if payload.get(name) is True
            ]
            if exit_code != 0 or failed_limits:
                details = [f"exit_code={exit_code!r}", *failed_limits]
                stderr = str(payload.get("stderr", "")).strip()
                if stderr:
                    details.append(f"stderr={stderr[-800:]}")
                return "SandboxResultError: " + ", ".join(details)

        if tool in {
            "learning_create_tool",
            "learning_create_snapshot_extractor",
            "learning_create_skill",
        }:
            status = str(payload.get("status", "")).casefold()
            validation = payload.get("validation")
            validation_failed = (
                isinstance(validation, dict) and validation.get("passed") is False
            )
            if status != "active" or validation_failed:
                return (
                    "ArtifactLifecycleError: generated artifact did not become active: "
                    + stripped[:1_600]
                )

        if payload.get("error"):
            return f"ToolReportedError: {str(payload['error'])[:2_000]}"
        return ""

    @staticmethod
    def _incomplete_task_answer(
        missing: list[str],
        failed_calls: list[dict[str, Any]],
    ) -> str:
        missing = list(dict.fromkeys(missing))
        if failed_calls:
            last = failed_calls[-1]
            detail = str(last.get("error") or last.get("result_excerpt") or "unknown error")
            return (
                "No. The task didn't finish. "
                f"`{last.get('tool') or 'unknown'}` failed with: {detail}. "
                f"Still missing verified runtime evidence: {', '.join(missing)}. "
                "Nothing is running in the background, and I'm not inventing the result."
            )
        if missing and all(item.startswith("answer:") for item in missing):
            return (
                "The tool ran, but the model failed to turn its output into a grounded "
                "answer. DARKLINGER killed the empty completion instead of dressing it up "
                "as success. Nothing is running in the background."
            )
        if missing and all(item.startswith("browser_") for item in missing):
            return (
                "The browser came back without verified evidence for: "
                f"{', '.join(missing)}. Nothing was "
                "extracted, and I'm not making shit up to fill the hole."
            )
        return (
            "The task didn't finish because DARKLINGER obtained no verified evidence "
            f"for: {', '.join(missing)}. Nothing is running in the background, "
            "and I'm not inventing a result."
        )

    def _start_agent_trace(self, prompt: str) -> AgentTaskTrace | None:
        root = getattr(self, "_agent_trace_root", None)
        if root is None:
            config = getattr(self, "config", None)
            autonomy_root = getattr(config, "autonomy_root", None)
            if autonomy_root is None:
                return None
            root = autonomy_root / "interactive"
        try:
            trace = AgentTaskTrace(root, prompt)
            from .model_loader.outcome_runtime import current_run
            outcome_run = current_run()
            if outcome_run is not None:
                outcome_run.trace = trace
        except OSError as error:
            print(f"[Task] Could not create execution trace: {error}")
            return None
        begin_interaction = getattr(self.tools, "begin_interaction", None)
        if callable(begin_interaction):
            begin_interaction(trace.task_id, prompt)
        print(
            f"[Task {trace.task_id}] running "
            "(this interaction only; no hidden background work)"
        )
        return trace

    def _finish_agent_trace(
        self,
        trace: AgentTaskTrace | None,
        answer: str,
    ) -> dict[str, Any] | None:
        if trace is None:
            return None
        trace.complete(answer)
        from .model_loader.outcome_runtime import current_run
        outcome_run = current_run()
        if outcome_run is not None:
            outcome_run.trace_finished(trace)
        self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
        print(f"[Task {trace.task_id}] completed")
        return trace.evidence()

    def _block_agent_trace(
        self,
        trace: AgentTaskTrace | None,
        reason: str,
    ) -> dict[str, Any] | None:
        if trace is None:
            return None
        trace.block(reason)
        from .model_loader.outcome_runtime import current_run
        outcome_run = current_run()
        if outcome_run is not None:
            outcome_run.trace_finished(trace)
        self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
        print(f"[Task {trace.task_id}] blocked")
        return trace.evidence()

