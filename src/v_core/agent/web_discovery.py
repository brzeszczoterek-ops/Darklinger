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


class WebDiscoveryMixin:
    """Mixin providing web_discovery capabilities for Agent."""

    @staticmethod
    def _discovery_search_query(prompt: str) -> str:
        """Derive a bounded, language-agnostic query from the primary clause."""

        normalized = " ".join(prompt.split()).strip()
        if not normalized:
            return "web research"
        # Conditional fallback instructions normally follow the primary request
        # as another sentence. A short salutation may precede it. Search engines
        # need the first substantive clause, not either piece of conversational
        # scaffolding. The preamble check uses punctuation, position and token
        # shape rather than requiring the user's language to be known.
        clauses = [
            clause.strip()
            for clause in re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
            if clause.strip()
        ]
        primary = clauses[0]
        if len(clauses) > 1:
            primary_words = re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                primary,
                re.UNICODE,
            )
            # A generic conversational lead-in may not address V by name at
            # all (for example, "I have an important task for you"). When it
            # contains no concrete product/protocol anchor, prefer the first
            # following clause that does. This stays language-agnostic and
            # avoids teaching the parser a list of translated lead-in phrases.
            anchored_following = next(
                (
                    clause
                    for clause in clauses[1:]
                    if Agent._discovery_anchor_index(
                        re.findall(
                            r"[^\W_]+(?:[-'][^\W_]+)*",
                            clause,
                            re.UNICODE,
                        )
                    )
                    is not None
                ),
                None,
            )
            if (
                Agent._discovery_anchor_index(primary_words) is None
                and anchored_following is not None
            ):
                primary = anchored_following
            else:
                # Voice input often contains more than one conversational
                # sentence before the actual request (for example a greeting
                # followed by "how is your evening?"). Skipping at most one
                # sentence turned that second pleasantry into the web query.
                # Advance through every structurally short preamble until the
                # first information-dense clause, without relying on a list of
                # greeting words or on the language being Polish or English.
                primary_index = 0
                while (
                    primary_index + 1 < len(clauses)
                    and Agent._looks_like_discovery_preamble(
                        clauses[primary_index],
                        clauses[primary_index + 1],
                    )
                ):
                    primary_index += 1
                primary = clauses[primary_index]
        # Voice requests often name the discovery surface in one sentence and
        # put the actual subject in the next one: "find tools on GitHub. Tools
        # for darknet OSINT...".  Do not let the platform anchor amputate that
        # adjacent scope.  The continuation is joined only when it contains a
        # second concrete identifier, so a following instruction such as "do
        # not install anything" cannot become search material.
        joined_platform_scope = False
        try:
            selected_index = clauses.index(primary)
        except ValueError:
            selected_index = -1
        if (
            selected_index >= 0
            and selected_index + 1 < len(clauses)
            and re.search(
                r"\b(?:github|gitlab|bitbucket)\w*\b",
                primary,
                re.IGNORECASE,
            )
        ):
            continuation = clauses[selected_index + 1]
            continuation_words = re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                continuation,
                re.UNICODE,
            )
            if Agent._discovery_anchor_index(continuation_words) is not None:
                primary = f"{primary} {continuation}"
                joined_platform_scope = True
        primary_words = re.findall(
            r"[^\W_]+(?:[-'][^\W_]+)*",
            primary,
            re.UNICODE,
        )
        if len(primary_words) > 18 and not joined_platform_scope:
            anchor = Agent._discovery_anchor_index(primary_words)
            if anchor is not None:
                first_mentions: dict[str, int] = {}
                for index, word in enumerate(primary_words):
                    if index > 0 and Agent._discovery_anchor_index(["", word]) is not None:
                        first_mentions.setdefault(word.casefold(), index)
                anchors = list(first_mentions.values())
                primary = " ".join(primary_words[max(0, anchor - 3) : anchors[-1] + 1])
            else:
                # With no product/protocol name, the request subject normally
                # sits at the end of the primary clause while greetings,
                # imperatives and politeness sit at the beginning. Keeping the
                # bounded tail is language-independent and prevents those lead-
                # ins from becoming the search topic.
                # Keep enough left context to retain the noun governed by a
                # later pronoun ("this forum ... take it down"). Fourteen
                # tokens dropped that noun and left only the ambiguous verb.
                primary = " ".join(primary_words[-18:])
        primary = re.sub(
            r"^(?:(?:v|boss|darklinger)\s*[,!:—-]\s*)+",
            "",
            primary,
            flags=re.IGNORECASE,
        ).strip(" .!?;:")
        if not primary:
            primary = normalized
        if len(primary) > 220:
            primary = primary[:220].rsplit(" ", 1)[0]
        return primary

    @staticmethod
    def _query_remainder_overlap(query: str, prompt: str) -> int:
        """Score how strongly a query is grounded beyond a greeting clause.

        This is deliberately lexical and language-neutral. It does not decide
        what words mean; it only detects that a short first sentence such as a
        greeting has no material overlap with the actual request that follows.
        """

        clauses = [
            clause.strip()
            for clause in re.split(r"(?<=[.!?])\s+|[\r\n]+", prompt)
            if clause.strip()
        ]
        if len(clauses) < 2:
            return 0
        remainder_tokens = {
            token.casefold()
            for token in re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                " ".join(clauses[1:]),
                re.UNICODE,
            )
            if len(token) >= 4
        }
        query_tokens = {
            token.casefold()
            for token in re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                query,
                re.UNICODE,
            )
            if len(token) >= 4
        }
        return sum(
            any(
                query_token == prompt_token
                or (
                    len(query_token) >= 5
                    and len(prompt_token) >= 5
                    and query_token[:5] == prompt_token[:5]
                )
                for prompt_token in remainder_tokens
            )
            for query_token in query_tokens
        )

    @staticmethod
    def _query_prompt_overlap(query: str, prompt: str) -> int:
        """Count query terms grounded anywhere in the immutable request."""

        prompt_tokens = {
            token.casefold()
            for token in re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                prompt,
                re.UNICODE,
            )
            if len(token) >= 4
        }
        query_tokens = {
            token.casefold()
            for token in re.findall(
                r"[^\W_]+(?:[-'][^\W_]+)*",
                query,
                re.UNICODE,
            )
            if len(token) >= 4
        }
        return sum(
            any(
                query_token == prompt_token
                or (
                    len(query_token) >= 5
                    and len(prompt_token) >= 5
                    and query_token[:5] == prompt_token[:5]
                )
                for prompt_token in prompt_tokens
            )
            for query_token in query_tokens
        )

    @staticmethod
    def _topic_relevance_missing(
        query: str,
        successful_calls: list[dict[str, Any]],
        *,
        minimum_sources: int = 1,
        strict_each_source: bool = False,
    ) -> list[str]:
        """Reject a structurally valid detail page about the wrong subject.

        A literal query-to-page word match is only one signal. An alternative,
        translated result or proper name may share no words with the request.
        In that case the runtime also accepts a detail URL that was actually
        observed in the preceding result set for this execution. This keeps the
        check language-neutral without allowing an unrelated, invented URL.
        """

        query_tokens = sorted(
            {
                token.casefold()
                for token in re.findall(
                    r"[^\W_][\w-]*",
                    query,
                    re.UNICODE,
                )
                if len(token) >= 3
            },
            key=lambda token: (-len(token), token),
        )[:16]
        if not query_tokens:
            return []

        source_evidence: dict[str, list[str]] = {}
        current_page = ""
        for call in successful_calls:
            tool = str(call.get("tool", ""))
            arguments = call.get("arguments", {})
            url = (
                str(arguments.get("url", ""))
                if isinstance(arguments, dict)
                else ""
            )
            if tool == "browser_navigate":
                current_page = Agent._normalized_web_url(url)
                if current_page and not TaskContract.is_search_listing_url(url):
                    source_evidence.setdefault(current_page, [])
                else:
                    current_page = ""
                continue
            if tool == "browser_snapshot" and current_page:
                source_evidence.setdefault(current_page, []).append(
                    str(call.get("result_excerpt", ""))
                )
                continue
            if tool == "web_read" and url and not TaskContract.is_search_listing_url(url):
                normalized = Agent._normalized_web_url(url)
                if normalized:
                    source_evidence.setdefault(normalized, []).append(
                        str(call.get("result_excerpt", ""))
                    )
        if not source_evidence:
            return []

        discovered_text: dict[str, str] = {}
        for call in successful_calls:
            if call.get("tool") != "web_search":
                continue
            try:
                payload = json.loads(str(call.get("result_excerpt", "")))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not isinstance(results, list):
                continue
            for item in results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", ""))
                normalized = Agent._normalized_web_url(url)
                if not normalized:
                    continue
                discovered_text[normalized] = " ".join(
                    str(item.get(field, ""))
                    for field in ("title", "snippet", "description")
                ).casefold()

        required_overlap = (
            2 if strict_each_source and len(query_tokens) > 1 else 1
        )
        relevant_sources: set[str] = set()
        named_subjects = Agent._discovery_named_subjects(query)
        comparison_subjects = (
            named_subjects if len(named_subjects) > 1 and minimum_sources > 1 else set()
        )
        covered_subjects: set[str] = set()
        saw_detail_evidence = False
        for normalized, excerpts in source_evidence.items():
            observed = "\n".join(excerpts).casefold()
            if not observed:
                continue
            saw_detail_evidence = True
            observed_overlap = sum(
                token in observed or (len(token) >= 5 and token[:5] in observed)
                for token in query_tokens
            )
            discovery = discovered_text.get(normalized, "")
            discovery_overlap = sum(
                token in discovery
                or (len(token) >= 5 and token[:5] in discovery)
                for token in query_tokens
            )
            source_words = set(re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", observed, re.UNICODE))
            source_subjects = comparison_subjects & source_words
            if max(observed_overlap, discovery_overlap) >= required_overlap or source_subjects:
                relevant_sources.add(normalized)
                covered_subjects.update(source_subjects)
                continue
            # For a single-source task, a URL copied exactly from the current
            # search result may legitimately lead to a translated source with
            # no lexical overlap. Exhaustive/multi-source contracts are stricter:
            # every counted corroborating source must retain the subject.
            if minimum_sources <= 1 and normalized in discovered_text:
                relevant_sources.add(normalized)
                continue
            if (
                minimum_sources <= 1
                and "verified" in observed
                and ("result" in observed or "evidence" in observed)
            ):
                relevant_sources.add(normalized)
        if not saw_detail_evidence:
            return []
        required_sources = max(1, minimum_sources)
        if len(relevant_sources) >= required_sources:
            if comparison_subjects - covered_subjects:
                return ["browser_evidence:topic_subjects=" + "|".join(sorted(comparison_subjects - covered_subjects))]
            return []
        if required_sources == 1:
            return ["browser_evidence:topic_mismatch"]
        return [
            "browser_evidence:topic_sources="
            f"{len(relevant_sources)}/{required_sources}"
        ]

    @staticmethod
    def _public_fact_search_query(prompt: str) -> str:
        """Derive a concise entity/location query without trusting the router.

        Capitalized names and places are useful cross-language anchors. If the
        request has none, retain the normal bounded discovery query.
        """

        words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", prompt, re.UNICODE)
        anchors = [
            word
            for index, word in enumerate(words)
            if index > 0 and len(word) >= 3 and word[0].isupper()
        ]
        query = " ".join(dict.fromkeys(anchors)).strip()
        return query or Agent._discovery_search_query(prompt)

    @staticmethod
    def _public_fact_recovery_request(
        prompt: str,
        contract: TaskContract,
        successful_calls: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        *,
        preferred_query: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        """Build a new grounded search when the model stops before public facts.

        Completion fields belong to DARKLINGER's runtime contract, not to the local
        model's judgment. If the model tries to finish without a requested
        address, contact, opening time, or location count, the runtime schedules
        a different focused DuckDuckGo query instead of accepting the empty
        answer. Query variants are bounded and never repeat within one task.
        """

        available = {
            str(item.get("function", {}).get("name", ""))
            for item in tool_definitions
            if isinstance(item, dict)
            and isinstance(item.get("function"), dict)
        }
        if "web_search" not in available:
            return None

        missing = {
            item.removeprefix("public_fact:")
            for item in contract.unmet(successful_calls)
            if item.startswith("public_fact:")
        }
        if not missing:
            return None

        previous_queries = {
            " ".join(str(call.get("arguments", {}).get("query", "")).split())
            .strip()
            .casefold()
            for call in successful_calls
            if call.get("tool") == "web_search"
            and isinstance(call.get("arguments"), dict)
        }
        previous_queries.discard("")

        base = " ".join(preferred_query.split()).strip()
        if not base:
            base = Agent._discovery_search_query(prompt)
        if not base:
            return None

        field_terms: list[str] = []
        for field, terms in (
            ("count", ("locations",)),
            ("address", ("address",)),
            ("contact", ("phone", "contact")),
            ("opening_hours", ("opening", "hours")),
        ):
            if field in missing:
                field_terms.extend(terms)
        suffix = " ".join(dict.fromkeys(field_terms))

        # Proper-name pairs survive across languages and give search engines a
        # useful exact anchor without maintaining a dictionary of user phrases.
        entity_matches = re.findall(
            r"\b[A-ZÀ-ÖØ-ÞĄĆĘŁŃÓŚŹŻ][^\W_]{2,}"
            r"(?:[ -]+[A-ZÀ-ÖØ-ÞĄĆĘŁŃÓŚŹŻ][^\W_]{2,}){1,3}\b",
            prompt,
            re.UNICODE,
        )
        entity = (
            contract.required_public_subject
            or (entity_matches[-1].strip() if entity_matches else "")
        )
        anchored = base
        if entity and entity.casefold() in base.casefold():
            anchored = re.sub(
                re.escape(entity),
                f'"{entity}"',
                base,
                count=1,
                flags=re.IGNORECASE,
            )

        candidates = [
            f"{base} {suffix}",
            f"{anchored} {suffix} official",
            f"{anchored} {suffix} business directory",
            f"{anchored} {suffix} map listing",
        ]
        for candidate in candidates:
            normalized = " ".join(candidate.split()).strip()
            if normalized and normalized.casefold() not in previous_queries:
                return "web_search", {"query": normalized, "max_results": 10}
        return None

    @staticmethod
    def _discovery_named_subjects(query: str) -> set[str]:
        words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", query, re.UNICODE)
        subjects: set[str] = set()
        # Reuse the same identifier heuristic as clause selection; include a
        # leading project name by prepending an empty scaffold word.
        for word in words:
            if word.casefold() in {"znajdź", "wynajdź", "wyszukaj", "sprawdź", "podaj", "porównaj", "find", "search", "inspect", "compare", "research", "tell"}:
                continue
            if Agent._discovery_anchor_index(["", word]) is not None:
                subjects.add(word.casefold())
        return subjects

    @staticmethod
    def _query_preserves_named_subjects(candidate: str, focused: str) -> bool:
        subjects = Agent._discovery_named_subjects(focused)
        candidate_words = {
            word.casefold()
            for word in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", candidate, re.UNICODE)
        }
        return bool(subjects) and subjects <= candidate_words

    @staticmethod
    def _discovery_anchor_index(words: list[str]) -> int | None:
        """Locate a grounded product/protocol-like token without knowing language."""

        for index, word in enumerate(words):
            folded = word.casefold()
            if folded in {"darklinger", "boss", "please", "ciebie", "tobie", "your", "url", "urls"} or index == 0:
                continue
            has_digit = any(character.isdigit() for character in word)
            has_letter = any(character.isalpha() for character in word)
            # Mixed identifiers such as Q8 or Llama3 are useful anchors. A
            # standalone year/number is chronology, not a product identity.
            if has_digit and has_letter:
                return index
            # Roman numerals commonly describe wars, centuries and editions.
            # Treating ``II`` as camel-case once reduced an entire antiques
            # request to the meaningless query "were made before II".
            roman_numeral = re.fullmatch(r"[IVXLCDM]+", word) is not None
            if (
                any(character.isupper() for character in word[1:])
                and not roman_numeral
            ):
                return index
            if len(word) >= 4 and word[0].isupper():
                return index
        return None

    @staticmethod
    def _looks_like_discovery_preamble(
        clause: str,
        following_clause: str,
    ) -> bool:
        """Recognize a short persona-directed preamble before a real request."""

        words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", clause, re.UNICODE)
        if not words or len(words) > 14:
            return False
        if re.search(r"https?://|www\.|[\"“”'«»]", clause, re.IGNORECASE):
            return False

        # A product name, protocol, version, filename-like token, or other
        # concrete anchor means this is already useful search material.
        if Agent._discovery_anchor_index(words) is not None:
            return False
        for index, word in enumerate(words):
            if any(character.isdigit() for character in word):
                return False
            if any(character.isupper() for character in word[1:]):
                return False
            if (
                index > 0
                and len(word) >= 8
                and word[0].isupper()
                and word.casefold() not in {"darklinger"}
            ):
                return False

        addressed = re.match(
            r"^\s*(?:(?:[^\s,!.?:—-]+)\s+){0,2}(?:v|boss|darklinger)\s*[,!.?:—-]",
            clause,
            re.IGNORECASE,
        )
        following_words = re.findall(
            r"[^\W_]+(?:[-'][^\W_]+)*",
            following_clause,
            re.UNICODE,
        )
        # A short greeting may omit V's name entirely. If the next sentence is
        # at least twice as information-dense, the short first clause is
        # conversational scaffolding, not a search topic. This structural rule
        # avoids maintaining translated greeting dictionaries.
        much_richer_following = (
            len(words) <= 8
            and len(following_words) >= 6
            and len(following_words) >= len(words) * 2
        )
        return (addressed is not None and len(following_words) >= 2) or much_richer_following

    @staticmethod
    def _repair_web_discovery_navigation(
        prompt: str,
        tool_name: str,
        arguments: Any,
        contract: TaskContract,
        successful_calls: list[dict[str, Any]],
        failed_calls: list[dict[str, Any]],
        preferred_query: str = "",
        preferred_target: str = "",
    ) -> Any:
        """Force discovery through verified search results before candidate URLs."""

        if (tool_name == 'web_read' and isinstance(arguments, dict)
            and contract.requires_current_release
            and catalog_target(contract.required_research_subjects, str(arguments.get('url','')), prompt)):
            # The immutable publisher catalogue/explicit owner designation is
            # the address discovery provenance. It supplies no observed facts.
            return arguments

        if tool_name == "web_search" and isinstance(arguments, dict):
            arguments = dict(arguments)
            if isinstance(arguments.get("query"), str):
                # Decode prose before subject-overlap checks. Literal escapes
                # otherwise make a valid Polish refinement look unrelated.
                arguments["query"] = MCPTools._normalized_search_text(
                    arguments["query"], owner_text=prompt
                )
            if "exhaustive_coverage" in contract.required_research_facets:
                try:
                    requested_limit = int(arguments.get("max_results", 0))
                except (TypeError, ValueError):
                    requested_limit = 0
                if requested_limit < 8:
                    arguments["max_results"] = 10
            focused_query = MCPTools._normalized_search_text(preferred_query, owner_text=prompt)
            if (
                contract.requires_web_discovery
                and focused_query
                and not contract.required_public_fields
            ):
                requested_query = " ".join(
                    str(arguments.get("query", "")).split()
                ).strip()
                if photo_paths(prompt) and photo_location_request(prompt):
                    area = location_query(prompt).removesuffix(' street photographs landmarks')
                    area_words = re.findall(r'[^\W_]+', area.casefold())
                    if area_words and not all(word in requested_query.casefold() for word in area_words):
                        return {**arguments, "query": focused_query}
                    if area_words:
                        return arguments

                focused_words = re.findall(
                    r"[^\W_]+(?:[-'][^\W_]+)*",
                    focused_query,
                    re.UNICODE,
                )
                anchor_index = Agent._discovery_anchor_index(focused_words)
                anchor = (
                    focused_words[anchor_index].casefold()
                    if anchor_index is not None
                    else ""
                )
                # Named subjects may be refined, but the concrete name must
                # survive. Generic multilingual work has no stable lexical
                # anchor, so keep the runtime-owned query verbatim.
                grounded_refinement = bool(
                    anchor and anchor in requested_query.casefold()
                )
                named_subjects = Agent._discovery_named_subjects(focused_query)
                named_comparison_refinement = False
                if len(named_subjects) > 1:
                    requested_words = {
                        word.casefold()
                        for word in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", requested_query, re.UNICODE)
                    }
                    # A comparison can gather evidence separately for each
                    # named item. Forcing every subquery back to the whole
                    # request turns useful per-item searches into duplicates.
                    grounded_refinement = bool(named_subjects & requested_words)
                    named_comparison_refinement = grounded_refinement
                owner_grounded_refinement = (
                    # Three independently grounded terms are enough for a
                    # useful follow-up query. Requiring four rejected the
                    # concise refinement "darknet OSINT GitHub repositories"
                    # even though every material term came from Boss's request,
                    # and replaced it with a long voice transcript that search
                    # engines interpreted as generic Git/GitHub help.
                    Agent._query_prompt_overlap(requested_query, prompt) >= 3
                    and Agent._query_prompt_overlap(
                        requested_query,
                        focused_query,
                    )
                    >= 2
                )
                # Once a result set exists, terms repeated across its titles
                # are empirical subject anchors. A refinement may add scope,
                # but dropping one of those anchors changes the task (for
                # example, "wild cats" silently becoming domestic cat breeds).
                focused_stems = {
                    token.casefold()[:5]
                    for token in focused_words
                    if len(token) >= 5
                }
                result_stem_frequency: Counter[str] = Counter()
                for call in successful_calls:
                    if call.get("tool") != "web_search":
                        continue
                    try:
                        payload = json.loads(
                            str(call.get("result_excerpt", ""))
                        )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    results = (
                        payload.get("results", [])
                        if isinstance(payload, dict)
                        else []
                    )
                    if not isinstance(results, list):
                        continue
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        observed_result = unquote(
                            f"{result.get('title', '')} {result.get('url', '')}"
                        ).casefold()
                        result_stem_frequency.update(
                            stem
                            for stem in focused_stems
                            if stem in observed_result
                        )
                grounded_anchor_stems = {
                    stem
                    for stem, frequency in result_stem_frequency.items()
                    if frequency >= 2
                }
                requested_folded = unquote(requested_query).casefold()
                dropped_grounded_anchor = any(
                    stem not in requested_folded
                    for stem in grounded_anchor_stems
                )
                if (
                    requested_query != focused_query
                    and dropped_grounded_anchor
                    and not named_comparison_refinement
                ):
                    repaired = dict(arguments)
                    repaired["query"] = focused_query
                    return repaired
                if requested_query != focused_query and not grounded_refinement:
                    if not owner_grounded_refinement:
                        repaired = dict(arguments)
                        repaired["query"] = focused_query
                        return repaired
            return arguments

        if tool_name not in {"browser_navigate", "web_read"} or not isinstance(
            arguments,
            dict,
        ):
            return arguments

        failed_navigation = any(
            call.get("tool") in {"browser_navigate", "web_read"}
            and call.get("status") == "failed"
            for call in failed_calls
        )
        raw_url = str(arguments.get("url", "")).strip()
        try:
            parsed_url = urlsplit(raw_url)
            hostname = (parsed_url.hostname or "").casefold()
            query = parsed_url.query.casefold()
            requested_search_query = str(
                parse_qs(parsed_url.query).get("q", [""])[0]
            ).strip()
        except ValueError:
            hostname = ""
            query = ""
            requested_search_query = ""
        google_host = hostname in {
            "google.com",
            "www.google.com",
        }
        must_use_search = contract.requires_web_discovery or failed_navigation
        focused_query = " ".join(preferred_query.split()).strip()
        if len(focused_query) > 220:
            focused_query = focused_query[:220].rsplit(" ", 1)[0]
        if google_host and must_use_search:
            repaired = dict(arguments)
            repaired["url"] = (
                "https://duckduckgo.com/?q="
                + quote_plus(
                    focused_query
                    or requested_search_query
                    or Agent._discovery_search_query(prompt)
                )
            )
            return repaired

        known_search = (
            hostname in {"duckduckgo.com", "html.duckduckgo.com"}
            and "q=" in query
        )

        direct_target = extract_web_target(preferred_target)
        if (
            not must_use_search
            and direct_target
            and not any(
                call.get("tool") == "browser_navigate"
                for call in [*successful_calls, *failed_calls]
            )
        ):
            repaired = dict(arguments)
            repaired["url"] = direct_target
            return repaired

        if any(
            call.get("tool") in {"browser_navigate", "web_search"}
            and call.get("status") == "succeeded"
            for call in successful_calls
        ):
            if contract.requires_web_discovery:
                observed = Agent._observed_detail_urls(successful_calls)
                exact = Agent._normalized_web_url(raw_url)
                visited_urls = {
                    Agent._normalized_web_url(
                        str(call.get("arguments", {}).get("url", ""))
                    )
                    for call in successful_calls
                    if call.get("tool") in {"browser_navigate", "web_read"}
                    and isinstance(call.get("arguments"), dict)
                }
                failed_urls = {
                    Agent._normalized_web_url(
                        str(call.get("arguments", {}).get("url", ""))
                    )
                    for call in failed_calls
                    if call.get("tool") == tool_name
                    and isinstance(call.get("arguments"), dict)
                }
                if not known_search and not exact:
                    # A fragment, relative path, plain label, or malformed URL
                    # is not an executable navigation target. Do not send it to
                    # Playwright and do not let the model burn retries on nearby
                    # spellings. Continue from a URL already observed in search
                    # evidence, or return to the runtime-owned query.
                    alternative = Agent._grounded_unvisited_detail_url(
                        prompt,
                        successful_calls,
                        failed_calls,
                        require_detail_structure=(tool_name != "web_read"),
                        failure_tool=tool_name,
                    )
                    repaired = dict(arguments)
                    repaired["url"] = (
                        alternative
                        or "https://duckduckgo.com/?q="
                        + quote_plus(
                            focused_query
                            or Agent._discovery_search_query(prompt)
                        )
                    )
                    return repaired
                if exact and exact in failed_urls:
                    alternative = Agent._grounded_unvisited_detail_url(
                        prompt,
                        successful_calls,
                        failed_calls,
                        require_detail_structure=(tool_name != "web_read"),
                        failure_tool=tool_name,
                    )
                    repaired = dict(arguments)
                    repaired["url"] = (
                        alternative
                        or "https://duckduckgo.com/?q="
                        + quote_plus(
                            focused_query
                            or Agent._discovery_search_query(prompt)
                        )
                    )
                    return repaired
                if exact and exact in visited_urls:
                    alternative = Agent._grounded_unvisited_detail_url(
                        prompt,
                        successful_calls,
                        failed_calls,
                        require_detail_structure=(tool_name != "web_read"),
                        failure_tool=tool_name,
                    )
                    repaired = dict(arguments)
                    repaired["url"] = (
                        alternative
                        or "https://duckduckgo.com/?q="
                        + quote_plus(
                            focused_query
                            or Agent._discovery_search_query(prompt)
                        )
                    )
                    return repaired
                if (
                    exact
                    and exact in {
                        Agent._normalized_web_url(candidate)
                        for candidate in observed
                    }
                    and exact not in visited_urls
                    and exact not in failed_urls
                    and tool_name == "browser_navigate"
                    and not Agent._is_strong_detail_url(raw_url)
                ):
                    concrete = Agent._grounded_unvisited_detail_url(
                        prompt,
                        successful_calls,
                        failed_calls,
                        require_detail_structure=True,
                        failure_tool=tool_name,
                    )
                    if concrete and Agent._normalized_web_url(concrete) != exact:
                        repaired = dict(arguments)
                        repaired["url"] = concrete
                        return repaired
                if (
                    exact
                    and exact in {
                        Agent._normalized_web_url(candidate)
                        for candidate in observed
                    }
                    and exact not in visited_urls
                    and exact not in failed_urls
                    and "exhaustive_coverage"
                    in contract.required_research_facets
                ):
                    strongest = Agent._grounded_unvisited_detail_url(
                        prompt,
                        successful_calls,
                        failed_calls,
                        require_detail_structure=(tool_name != "web_read"),
                        failure_tool=tool_name,
                    )
                    if (
                        strongest
                        and Agent._normalized_web_url(strongest) != exact
                    ):
                        repaired = dict(arguments)
                        repaired["url"] = strongest
                        return repaired
                if not known_search and exact and exact not in {
                    Agent._normalized_web_url(candidate) for candidate in observed
                }:
                    closest = Agent._closest_observed_detail_url(raw_url, observed)
                    if (
                        closest
                        and Agent._normalized_web_url(closest) not in failed_urls
                    ):
                        repaired = dict(arguments)
                        repaired["url"] = closest
                        return repaired
                    # A discovery model may invent a plausible-looking domain
                    # after seeing a results page. Never spend tool calls on a
                    # URL that is absent from verified search evidence. Return
                    # to a grounded result (or the query page when none can be
                    # selected). web_read must never receive a search URL.
                    alternative = Agent._grounded_unvisited_detail_url(
                        prompt,
                        successful_calls,
                        failed_calls,
                        require_detail_structure=(tool_name != "web_read"),
                        failure_tool=tool_name,
                    )
                    repaired = dict(arguments)
                    repaired["url"] = (
                        alternative
                        or "https://duckduckgo.com/?q="
                        + quote_plus(
                            focused_query
                            or Agent._discovery_search_query(prompt)
                        )
                    )
                    return repaired
            return arguments
        if not must_use_search:
            return arguments

        if known_search:
            if focused_query and requested_search_query != focused_query:
                repaired = dict(arguments)
                repaired["url"] = (
                    "https://duckduckgo.com/?q=" + quote_plus(focused_query)
                )
                return repaired
            return arguments

        repaired = dict(arguments)
        repaired["url"] = (
            "https://duckduckgo.com/?q="
            + quote_plus(focused_query or Agent._discovery_search_query(prompt))
        )
        return repaired

    @staticmethod
    def _direct_navigation_failure(
        preferred_target: str,
        contract: TaskContract,
        failed_calls: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the first failed attempt at Boss's exact direct URL.

        A failed owner-supplied address is evidence in its own right. Searching
        for similarly spelled pages after that failure changes the task and was
        the source of the Wikipedia/``dwukropek`` loop.
        """

        if contract.requires_web_discovery or not preferred_target:
            return None
        target = Agent._normalized_web_url(preferred_target)
        if not target:
            return None
        return next(
            (
                call
                for call in failed_calls
                if call.get("tool") == "browser_navigate"
                and Agent._normalized_web_url(
                    str(
                        call.get("arguments", {}).get("url", "")
                        if isinstance(call.get("arguments"), dict)
                        else ""
                    )
                )
                == target
            ),
            None,
        )

    @staticmethod
    def _direct_navigation_failure_answer(target: str, error: str) -> str:
        """Render one bounded, truthful report for a failed direct URL."""

        failure_detail = re.sub(
            r"\x1b\[[0-9;]*[A-Za-z]",
            "",
            " ".join(str(error or "navigation failed").split()),
        )[:500]
        return (
            f"That address didn't open, Boss. I tried `{target}` exactly once; "
            f"navigation failed: `{failure_detail}`. No page loaded, so there "
            "was nothing honest to report. I didn't swap in search-result "
            "bullshit or a different site. Nothing is running in the background."
        )

    @staticmethod
    def _normalized_web_url(url: str) -> str:
        try:
            parsed = urlsplit(url.strip())
        except ValueError:
            return ""
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            return ""
        return urlunsplit(
            (
                parsed.scheme.casefold(),
                parsed.netloc.casefold(),
                parsed.path.rstrip("/") or "/",
                parsed.query,
                "",
            )
        )

    @staticmethod
    def _observed_detail_urls(successful_calls: list[dict[str, Any]]) -> list[str]:
        """Extract exact external URLs exposed by successful browser snapshots."""

        found: list[str] = []
        normalized_found: set[str] = set()
        for call in successful_calls:
            if call.get("tool") not in {"browser_snapshot", "web_search"}:
                continue
            text = str(call.get("result_excerpt", ""))
            page_url_match = re.search(
                r"^- Page URL:\s*(https?://\S+)",
                text,
                re.MULTILINE | re.IGNORECASE,
            )
            base_url = page_url_match.group(1) if page_url_match else ""
            candidates = [
                match.group(0).rstrip("`),.;:]}\\")
                for match in re.finditer(
                    r"https?://[^\s<>\"']+",
                    text,
                    re.IGNORECASE,
                )
            ]
            if base_url:
                candidates.extend(
                    urljoin(base_url, match.group(1).strip().strip('"'))
                    for match in re.finditer(
                        r"-\s*/url:\s*([^\s<>]+)",
                        text,
                        re.IGNORECASE,
                    )
                    if match.group(1).strip()
                    and not match.group(1).strip().casefold().startswith(
                        ("#", "javascript:", "mailto:", "tel:")
                    )
                )
            for candidate in candidates:
                normalized = Agent._normalized_web_url(candidate)
                if not normalized:
                    continue
                hostname = (urlsplit(normalized).hostname or "").casefold()
                if hostname.endswith("duckduckgo.com"):
                    continue
                if normalized not in normalized_found:
                    found.append(candidate)
                    normalized_found.add(normalized)
        return found

    @staticmethod
    def _closest_observed_detail_url(
        requested_url: str,
        observed_urls: list[str],
    ) -> str:
        """Repair a model-copied URL only from a close same-host observed URL."""

        requested_fingerprint = Agent._url_copy_fingerprint(requested_url)
        fingerprint_matches = {
            candidate
            for candidate in observed_urls
            if requested_fingerprint
            and Agent._url_copy_fingerprint(candidate) == requested_fingerprint
        }
        if len(fingerprint_matches) == 1:
            return next(iter(fingerprint_matches))

        requested = Agent._normalized_web_url(requested_url)
        if not requested:
            return ""
        requested_parts = urlsplit(requested)
        requested_tokens = set(
            re.findall(
                r"[a-z0-9]{4,}",
                unquote(
                    f"{requested_parts.path}?{requested_parts.query}"
                ).casefold(),
            )
        )
        requested_numeric = {
            token for token in requested_tokens if token.isdigit()
        }
        ranked: list[tuple[float, int, str]] = []
        for candidate in observed_urls:
            normalized = Agent._normalized_web_url(candidate)
            if not normalized:
                continue
            parts = urlsplit(normalized)
            if parts.hostname != requested_parts.hostname:
                continue
            path_similarity = SequenceMatcher(
                None,
                requested_parts.path.casefold(),
                parts.path.casefold(),
            ).ratio()
            candidate_tokens = set(
                re.findall(
                    r"[a-z0-9]{4,}",
                    unquote(f"{parts.path}?{parts.query}").casefold(),
                )
            )
            shared = requested_tokens & candidate_tokens
            token_similarity = (
                len(shared) / max(1, len(requested_tokens | candidate_tokens))
            )
            shared_numeric = bool(requested_numeric & candidate_tokens)
            if path_similarity >= 0.72 or (
                len(shared) >= 3
                and token_similarity >= 0.35
                and (shared_numeric or len(shared) >= 5)
            ):
                ranked.append(
                    (
                        max(path_similarity, token_similarity),
                        len(shared),
                        candidate,
                    )
                )
        if not ranked:
            return ""
        ranked.sort(key=lambda item: (-item[0], -item[1], len(item[2]), item[2]))
        return ranked[0][2]

    @staticmethod
    def _url_copy_fingerprint(url: str) -> str:
        """Collapse separators to repair only copied, already observed URLs."""

        try:
            parsed = urlsplit(str(url).strip())
        except ValueError:
            return ""
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            return ""
        evidence = unquote(
            f"{parsed.netloc}{parsed.path}?{parsed.query}"
        ).casefold()
        return "".join(character for character in evidence if character.isalnum())

    @staticmethod
    def _owner_control_command(prompt: str) -> str | None:
        """Return an explicit control command for an awaiting task."""

        text = " ".join(prompt.casefold().split()).strip(" .!?,;:")
        if text in {
            "/continue --continuous",
            "/continue continuous",
            "/autonomous",
            "/unlimited",
            "continue continuously",
            "continue without stopping",
            "kontynuuj bez limitu",
            "kontynuuj bez zatrzymywania",
            "działaj bez zatrzymywania",
            "dzialaj bez zatrzymywania",
            "leć dalej bez zatrzymywania",
            "lec dalej bez zatrzymywania",
        }:
            return "continuous"
        if text in {"/continue", "continue", "kontynuuj", "dalej"}:
            return "continue"
        if text in {"/stop", "stop", "przerwij", "zakończ", "zakoncz"}:
            return "stop"
        return None

    async def _handle_owner_control(
        self,
        command: str,
        on_token: Callable[[str], None] | None,
    ) -> str | None:
        """Resume or stop the exact checkpoint currently awaiting Boss."""

        previous = getattr(self, "_last_execution_context", None)
        if not isinstance(previous, dict) or previous.get("status") != "awaiting_owner":
            return None
        task_id = str(previous.get("task_id", "")).strip()
        root = getattr(self, "_agent_trace_root", None)
        if not task_id or root is None:
            return None
        trace = AgentTaskTrace.load(root, task_id)
        if trace is None or trace.status != "awaiting_owner":
            return None

        if command == "stop":
            trace.stop_from_owner()
            self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
            answer = (
                f"Stopped task `{trace.task_id}` at the saved checkpoint. "
                "Its verified tool evidence remains on disk. Nothing is running "
                "in the background."
            )
            await self._remember_task(
                trace.objective,
                answer,
                execution=trace.evidence(),
            )
            if on_token is not None:
                on_token(answer)
            print(f"[Task {trace.task_id}] stopped by owner")
            return answer

        if command == "continuous":
            if getattr(getattr(self.config, "edition", None), "name", "") == "full_access":
                return "Full Access requires a user decision for each additional batch. Use /continue for one batch."
            trace.authorize_continuous_from_owner()
            mode = "continuous owner-authorized mode"
        else:
            trace.resume_from_owner()
            mode = "one additional batch"
        self._last_execution_context = AgentTaskTrace.latest_context(trace.root)
        print(f"[Task {trace.task_id}] resumed by owner: {mode}")
        try:
            while True:
                try:
                    return await self._run_agent_steps(
                        trace.objective,
                        on_token,
                        trace,
                    )
                except _ContinueAgentBatch:
                    continue
        except (asyncio.CancelledError, KeyboardInterrupt):
            trace.stop("interaction interrupted")
            print(f"[Task {trace.task_id}] stopped")
            raise
        except BaseException as error:
            trace.fail(f"{type(error).__name__}: {error}")
            print(f"[Task {trace.task_id}] failed")
            raise

