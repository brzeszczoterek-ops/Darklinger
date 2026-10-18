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


class ReportsMixin:
    """Mixin providing reports capabilities for Agent."""

    @staticmethod
    def _owner_progress_report(
        summary: dict[str, list[str]] | None,
        successful_calls: list[dict[str, Any]],
        missing: list[str],
        *,
        objective: str = "",
        failed_calls: list[dict[str, Any]] | None = None,
        stop_reason: str = "",
    ) -> str:
        """Render a reusable task checkpoint without inventing new facts.

        A fallback report is not merely an owner-facing apology.  It is the
        hand-off artifact for the next attempt, so it must distinguish search
        results from opened pages and pages whose content was actually
        captured.  That distinction lets a later run continue from verified
        evidence instead of repeating discovery or treating a URL as proof.
        """

        tool_names = {
            str(call.get("tool", "")).strip()
            for call in successful_calls
            if call.get("tool")
        }

        def clean(value: Any) -> str:
            raw = str(value).strip()
            try:
                structured = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                structured = None
            if isinstance(structured, dict):
                if structured.get("engine") == "duckduckgo" and isinstance(
                    structured.get("results"), list
                ):
                    results = []
                    for item in structured["results"][:8]:
                        if not isinstance(item, dict) or not item.get("url"):
                            continue
                        title = " ".join(str(item.get("title", "")).split()).strip()
                        url = str(item["url"]).strip()
                        results.append(f"{title} — {url}" if title else url)
                    if results:
                        query = " ".join(str(structured.get("query", "")).split())
                        return (f"Search for “{query}” found: " + "; ".join(results))[
                            :600
                        ]
                content = structured.get("content")
                if isinstance(content, str) and content.strip():
                    return clean(content)

            # Immutable trace excerpts are intentionally bounded and can cut a
            # JSON web_read payload in the middle of its content string. Unwrap
            # that leading field before browser-scaffolding filtering so the
            # fallback never prints escaped JSON/YAML garbage to the owner.
            if structured is None and raw.lstrip().startswith("{"):
                truncated_content = re.search(
                    r'"content"\s*:\s*"(.*)', raw, flags=re.DOTALL
                )
                if truncated_content is not None:
                    raw = (
                        truncated_content.group(1)
                        .replace(r"\n", "\n")
                        .replace(r'\"', '"')
                        .replace(r"\/", "/")
                    )

            text = " ".join(raw.split()).strip()
            for name in tool_names:
                prefix = f"{name}:"
                if text.casefold().startswith(prefix.casefold()):
                    text = text[len(prefix):].strip()
                    break

            no_match = re.search(
                r"no\s+matches?\s+found\s+for\s+[\"“']?([^\"”']+)",
                text,
                flags=re.IGNORECASE,
            )
            if no_match:
                query = no_match.group(1).strip().rstrip(". ")
                return (
                    f"No match for \N{LEFT DOUBLE QUOTATION MARK}{query}"
                    f"\N{RIGHT DOUBLE QUOTATION MARK} was found on the inspected page."
                )[:600]

            # Browser adapters return useful page metadata together with large
            # Playwright/accessibility dumps. Owner reports must preserve the
            # verified URL/title without leaking YAML, element refs, or executed
            # JavaScript as if it were a research finding.
            page_url = re.search(
                r"Page URL:\s*\[?(https?://[^\s\])]+)",
                text,
                flags=re.IGNORECASE,
            )
            page_title = re.search(
                r"Page Title:\s*(.+?)(?=\s+-\s+Console:|\s+###\s+Snapshot|"
                r"\s+\[DARKLINGER\s|$)",
                text,
                flags=re.IGNORECASE,
            )
            raw_browser_output = bool(
                page_url
                or re.search(
                    r"(?:###\s+(?:Ran Playwright code|Snapshot|Page)|"
                    r"```ya?ml|\bgeneric\s*\[(?:active\]\s*)?\[?ref=|"
                    r"\[ref=[^\]]+\]|\bcursor=(?:pointer|text)\b)",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            if raw_browser_output:
                if page_url:
                    url = page_url.group(1).rstrip(".,")
                    title = page_title.group(1).strip(" -.") if page_title else ""
                    if title.casefold().startswith("loading http"):
                        title = ""
                    page_summary = (
                        f"Source: {url} — {title}." if title else f"Source: {url}."
                    )
                    headings: list[str] = []
                    for heading in re.findall(
                        r'\bheading\s+"([^"]+)"\s+\[level=(?:2|3)\]',
                        text,
                        flags=re.IGNORECASE,
                    ):
                        compact_heading = " ".join(heading.split()).strip(" -.")
                        if (
                            compact_heading
                            and compact_heading.casefold() != title.casefold()
                            and compact_heading.casefold()
                            not in {item.casefold() for item in headings}
                        ):
                            headings.append(compact_heading)
                    if headings:
                        numbered = [
                            heading
                            for heading in headings
                            if re.match(r"^\d+[.)]\s+", heading)
                        ]
                        if numbered:
                            selected = []
                            for heading in numbered:
                                candidate = re.sub(
                                    r"^\d+[.)]\s+", "", heading
                                ).split(":", 1)[0].strip()
                                if candidate and candidate.casefold() not in {
                                    item.casefold() for item in selected
                                }:
                                    selected.append(candidate)
                        else:
                            selected = headings
                        label = (
                            "Observed candidate labels"
                            if numbered
                            else "Verified sections"
                        )
                        page_summary += f" {label}: " + "; ".join(selected[:10]) + "."
                    return page_summary[:600]
                return ""

            # Failed calls can be present in a model-produced rollover summary
            # even though this function also receives the successful-call ledger.
            # Do not promote adapter diagnostics into verified findings.
            if re.search(
                r"^browser_[a-z0-9_]+:\s*browser_[a-z0-9_]+\s+requires\b|"
                r"^(?:error(?:\s+code)?|tool\s+error|failed)\b",
                text,
                flags=re.IGNORECASE,
            ):
                return ""
            return text[:600]

        raw_findings = (
            summary.get("findings", [])
            if isinstance(summary, dict)
            and isinstance(summary.get("findings", []), list)
            else []
        )
        findings: list[str] = []
        finding_indexes: dict[str, int] = {}

        def finding_key(finding: str) -> str:
            source = re.match(r"(?:Visited\s+|Source:\s+)(https?://\S+)", finding)
            if source:
                return "url:" + source.group(1).rstrip(".,").casefold()
            return "text:" + finding.casefold()

        def add_finding(item: Any) -> None:
            finding = clean(item)
            if not finding:
                return
            key = finding_key(finding)
            existing = finding_indexes.get(key)
            if existing is None:
                finding_indexes[key] = len(findings)
                findings.append(finding)
            elif len(finding) > len(findings[existing]):
                # Context summaries deliberately truncate evidence. When the
                # immutable call ledger still contains a richer description of
                # the same page, keep that one (including verified headings)
                # instead of reducing the owner report to a bare URL.
                findings[existing] = finding

        # A model-written or bounded rollover summary is planning context, not
        # the evidence source of record. When the immutable call ledger exists,
        # derive owner-visible findings from it alone. This also prevents a
        # truncated summary URL from becoming a duplicate of the real URL.
        has_detail_source = False
        if successful_calls:
            for call in successful_calls:
                tool = str(call.get("tool", ""))
                if tool not in {"browser_navigate", "web_read"}:
                    continue
                arguments = call.get("arguments", {})
                url = str(arguments.get("url", "")) if isinstance(arguments, dict) else ""
                parsed = urlsplit(url)
                hostname = (parsed.hostname or "").casefold()
                if url and not (
                    hostname.endswith("duckduckgo.com")
                    or hostname in {
                        "google.com",
                        "www.google.com",
                        "bing.com",
                        "www.bing.com",
                        "search.brave.com",
                        "search.yahoo.com",
                    }
                ):
                    has_detail_source = True
                    break

        if successful_calls:
            for call in successful_calls[-12:]:
                tool = str(call.get("tool", ""))
                if has_detail_source and call.get("tool") == "web_search":
                    continue
                if tool in {
                    "learning_create_tool",
                    "learning_create_snapshot_extractor",
                }:
                    created_name = str(call.get("created_tool_name", "")).strip()
                    if created_name:
                        add_finding(
                            f"Created, validated, and activated tool: {created_name}."
                        )
                    continue
                excerpt = call.get("result_excerpt", "")
                if call.get("binding_source") == "runtime_objective_fixture":
                    fixture_result = clean(excerpt)
                    if fixture_result:
                        add_finding(
                            f"Synthetic runtime fixture result from {tool}: "
                            f"{fixture_result}"
                        )
                    continue
                finding = clean(excerpt)
                if has_detail_source and finding.startswith("Source: http"):
                    match = re.match(r"Source:\s+(https?://\S+)", finding)
                    hostname = (
                        (urlsplit(match.group(1)).hostname or "").casefold()
                        if match
                        else ""
                    )
                    if hostname.endswith("duckduckgo.com") or hostname in {
                        "google.com",
                        "www.google.com",
                        "bing.com",
                        "www.bing.com",
                        "search.brave.com",
                        "search.yahoo.com",
                    }:
                        continue
                add_finding(excerpt)
        else:
            for item in raw_findings:
                add_finding(item)

        # A rollover summary is planning context written before later tool
        # calls. Its open questions become stale as the task progresses. The
        # current runtime contract is the sole authority for unfinished work.
        raw_next_steps: list[Any] = list(missing)
        next_steps: list[str] = []
        for item in raw_next_steps:
            step = clean(item)
            if step:
                step = Agent._owner_missing_descriptions([step])[0]
            if step and step.casefold() not in {
                "continue the original objective using real tools.",
            }:
                next_steps.append(step)
        lines: list[str] = []
        compact_objective = " ".join(str(objective).split()).strip()
        if compact_objective:
            lines.append("Recovery checkpoint:")
            lines.append(f"Objective: {compact_objective[:1_200]}")
            if stop_reason:
                lines.append(f"Stop reason: {' '.join(stop_reason.split())[:800]}")

        lines.append("Verified findings:")
        if findings:
            lines.extend(f"- {item}" for item in findings[-8:])
        else:
            lines.append("- No verified task findings yet.")

        if compact_objective:
            def normalized_url(value: Any) -> str:
                raw = str(value or "").strip().rstrip(".,;:!?")
                try:
                    parsed = urlsplit(raw)
                except ValueError:
                    return ""
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
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

            discovered: dict[str, str] = {}
            opened: dict[str, str] = {}
            inspected: dict[str, str] = {}
            current_opened_url = ""
            for call in successful_calls:
                tool = str(call.get("tool", ""))
                arguments = call.get("arguments", {})
                excerpt = str(call.get("result_excerpt", ""))
                if tool == "web_search":
                    try:
                        payload = json.loads(excerpt)
                    except (TypeError, json.JSONDecodeError):
                        payload = None
                    if isinstance(payload, dict) and isinstance(
                        payload.get("results"), list
                    ):
                        for item in payload["results"]:
                            if not isinstance(item, dict):
                                continue
                            url = normalized_url(item.get("url"))
                            if not url:
                                continue
                            title = " ".join(str(item.get("title", "")).split())
                            discovered.setdefault(url, title)
                if tool in {"browser_navigate", "web_read"} and isinstance(
                    arguments, dict
                ):
                    url = normalized_url(arguments.get("url"))
                    if url:
                        opened.setdefault(url, "")
                        current_opened_url = url
                        if tool == "web_read":
                            inspected.setdefault(url, "")
                if tool == "browser_snapshot":
                    page_match = re.search(
                        r"Page URL:\s*\[?(https?://[^\s\])]+)",
                        excerpt,
                        flags=re.IGNORECASE,
                    )
                    url = normalized_url(
                        page_match.group(1) if page_match else current_opened_url
                    )
                    if url:
                        inspected.setdefault(url, "")

            # Search/listing pages are transport, not subject-matter evidence.
            inspected_detail = {
                url: title
                for url, title in inspected.items()
                if not TaskContract.is_search_listing_url(url)
            }
            opened_only = {
                url: discovered.get(url, title)
                for url, title in opened.items()
                if url not in inspected and not TaskContract.is_search_listing_url(url)
            }
            discovered_only = {
                url: title
                for url, title in discovered.items()
                if url not in opened and url not in inspected
            }

            lines.append("Reusable source ledger:")
            if inspected_detail:
                for url in list(inspected_detail)[:8]:
                    title = discovered.get(url, "")
                    label = f" — {title}" if title else ""
                    lines.append(
                        f"- Inspected (page content captured): {url}{label}"
                    )
            if opened_only:
                for url, title in list(opened_only.items())[:8]:
                    label = f" — {title}" if title else ""
                    lines.append(
                        f"- Opened only (content not captured): {url}{label}"
                    )
            if discovered_only:
                for url, title in list(discovered_only.items())[:8]:
                    label = f" — {title}" if title else ""
                    lines.append(
                        f"- Discovered only (not opened or evaluated): {url}{label}"
                    )
            if not inspected_detail and not opened_only and not discovered_only:
                lines.append("- No reusable online source evidence was captured.")

            limitations: list[str] = []
            requested_hosts: list[tuple[str, str]] = []
            # Product and host names are commonly inflected in natural speech
            # (for example Polish "GitHuba").  Match the stable brand stem,
            # while the actual proof check remains the exact github.com host.
            if re.search(r"\bgithub[\w-]*\b", compact_objective, re.IGNORECASE):
                requested_hosts.append(("GitHub", "github.com"))
            for label, host in requested_hosts:
                matching = [
                    url
                    for url in inspected_detail
                    if (urlsplit(url).hostname or "").casefold().endswith(host)
                ]
                if not matching:
                    limitations.append(
                        f"The objective required {label}, but no {label} page "
                        "content was inspected. Search snippets and third-party "
                        "articles do not satisfy that requirement."
                    )
            if opened_only:
                limitations.append(
                    "Opening a URL proves navigation only; those pages cannot be "
                    "used as factual support until their content is captured."
                )
            if discovered_only:
                limitations.append(
                    "Discovery results are leads, not verified findings or accepted "
                    "candidates."
                )
            if limitations:
                lines.append("Evidence limits:")
                lines.extend(f"- {item}" for item in limitations)

            resume_steps: list[str] = []
            if requested_hosts and any(
                "no GitHub page content" in item for item in limitations
            ):
                resume_steps.append(
                    "Search GitHub directly, open concrete repository pages, and "
                    "capture each repository's README or documentation before "
                    "ranking candidates."
                )
            if opened_only:
                resume_steps.append(
                    "Capture and inspect the pages marked 'Opened only'; do not "
                    "repeat the searches that already produced the leads above."
                )
            if failed_calls:
                last_failure = failed_calls[-1]
                detail = " ".join(
                    str(
                        last_failure.get("error")
                        or last_failure.get("result_excerpt")
                        or "unknown failure"
                    ).split()
                )[:500]
                lines.append("Last execution failure:")
                lines.append(
                    f"- {last_failure.get('tool') or 'unknown tool'}: {detail}"
                )
                resume_steps.append(
                    "Continue after the recorded failure using the preserved ledger; "
                    "do not discard successful earlier calls."
                )
            if resume_steps:
                lines.append("Resume from here:")
                lines.extend(f"- {item}" for item in resume_steps)

        if next_steps:
            lines.append("Still open:")
            lines.extend(f"- {item}" for item in next_steps[:8])
        return "\n".join(lines)

    @staticmethod
    def _repair_unambiguous_observed_urls(
        answer: str,
        successful_calls: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Replace a hallucinated path only when the observed URL is unique.

        Language models occasionally reproduce a real page with a translated,
        truncated, or otherwise invented path.  A single bad path must not
        discard an otherwise grounded report.  Host equality plus exactly one
        observed URL on that host makes the repair deterministic. A Markdown
        link whose literal URL label is observed also identifies its intended
        target; other ambiguous links are left to the grounding validator.
        """

        url_pattern = re.compile(r"https?://[^\s<>\[\]{}()\"'`]+")
        observed_by_host: dict[str, set[str]] = {}
        observed_urls: set[str] = set()
        evidence_tools = {
            "browser_snapshot",
            "web_search",
            "web_read",
            "full_tor_search",
            "full_tor_fetch",
            "full_tor_inventory",
            "full_tor_browser_inventory",
        }
        for call in successful_calls:
            if (
                call.get("status", "succeeded") != "succeeded"
                or call.get("tool") not in evidence_tools
            ):
                continue
            for match in url_pattern.finditer(
                str(call.get("result_excerpt", ""))
            ):
                observed = match.group(0).rstrip(".,;:!?")
                hostname = (urlsplit(observed).hostname or "").casefold()
                if not hostname:
                    continue
                observed_urls.add(observed)
                observed_by_host.setdefault(hostname, set()).add(observed)

        repairs: list[dict[str, str]] = []

        def repair_literal_url_link(match: re.Match[str]) -> str:
            label, target = match.groups()
            if (
                label in observed_urls
                and target not in observed_urls
                and urlsplit(label).hostname == urlsplit(target).hostname
            ):
                repairs.append({"from": target, "to": label})
                return f"[{label}]({label})"
            return match.group(0)

        repaired = re.sub(
            r"\[(https?://[^\s\[\]]+)\]\((https?://[^\s()]+)\)",
            repair_literal_url_link,
            answer,
        )
        replacements: dict[str, str] = {}
        for match in url_pattern.finditer(repaired):
            candidate = match.group(0).rstrip(".,;:!?")
            if candidate in observed_urls:
                continue
            hostname = (urlsplit(candidate).hostname or "").casefold()
            matches = observed_by_host.get(hostname, set())
            if len(matches) == 1:
                replacements[candidate] = next(iter(matches))

        for candidate, observed in replacements.items():
            repaired = repaired.replace(candidate, observed)
            repairs.append({"from": candidate, "to": observed})
        return repaired, repairs

    @staticmethod
    def _owner_missing_descriptions(missing: list[str]) -> list[str]:
        """Translate runtime contract keys into owner-facing work items."""

        labels = {
            "browser_navigate": "open the requested online source",
            "browser_snapshot": "inspect and capture the source's actual content",
            "browser_navigate:distinct_detail_page": (
                "open the relevant result or offer page, not only the listing"
            ),
            "browser_navigate:detail_not_discovered": (
                "open a detail-page URL actually present in the inspected search results"
            ),
            "browser_snapshot:detail_page": (
                "inspect the actual content of that result or offer page"
            ),
            "read_file": "read the requested local file",
            "filesystem_mutation": "complete and verify the requested file change",
            "command_execution": "run and verify the requested command or tests",
            "learning_create_tool": "create, validate, and activate the needed tool",
            "learning_create_skill": "create, validate, and activate the needed skill",
            "learning_create_tool_or_skill": (
                "create and validate either the needed tool or the needed skill"
            ),
            "public_fact:address": (
                "find a source containing the requested street or postal address"
            ),
            "public_fact:contact": (
                "find a source containing the requested phone number or email address"
            ),
            "public_fact:opening_hours": (
                "find a source containing explicit opening days and times"
            ),
            "public_fact:count": (
                "find evidence stating the number of matching locations"
            ),
            "public_fact:subject": (
                "verify that the source describes the exact requested subject"
            ),
            "browser_evidence:research_price": (
                "open a grounded market or offer source containing an explicit price"
            ),
            "browser_evidence:research_purchase_source": (
                "inspect a grounded listing, shop, auction, or concrete offer"
            ),
            "browser_evidence:research_item_list": (
                "capture a source that names multiple concrete requested items"
            ),
            "browser_evidence:research_item_descriptions": (
                "capture item-level descriptions, not only a generic introduction"
            ),
            "browser_evidence:research_images": (
                "capture a source containing visual examples of the requested items"
            ),
            "answer:evidence_observation_missing": (
                "collect concrete evidence needed for the final report"
            ),
            "answer:first_heading_missing": (
                "include the verified first heading in the final report"
            ),
            "answer:evidence_not_reflected": (
                "write the final report using the verified findings"
            ),
            "answer:research_item_list_missing": (
                "present the requested items as a concrete list"
            ),
            "answer:research_item_descriptions_missing": (
                "describe each reported item instead of only naming the source"
            ),
        }
        return [
            (
                "inspect another independent, concrete result page"
                if item.startswith("browser_evidence:detail_sources=")
                else "inspect another independent source that matches the requested subject"
                if item.startswith("browser_evidence:topic_sources=")
                else (
                    "copy every GitHub owner/repository identifier exactly from "
                    "the observed repository URL"
                )
                if item.startswith("answer:ungrounded_repository_identifiers=")
                else labels.get(item, item.replace("_", " "))
            )
            for item in missing
        ]

    @classmethod
    def _owner_verified_final_report(
        cls,
        summary: dict[str, list[str]] | None,
        successful_calls: list[dict[str, Any]],
        contract: TaskContract,
        *,
        objective: str = "",
        stop_reason: str = "",
    ) -> str:
        """Render structural research evidence when prose generation fails.

        The fallback must not collapse a completed multi-item observation into
        two page titles. It reports only labels and image descriptions present
        in browser snapshots and states their evidentiary limit explicitly.
        """

        report = cls._owner_progress_report(
            summary,
            successful_calls,
            [],
            objective=objective,
            stop_reason=stop_reason,
        )

        item_records: list[tuple[str, str]] = []
        image_records: list[tuple[str, str]] = []
        text_records: list[tuple[str, str]] = []
        seen_items: set[str] = set()
        seen_images: set[str] = set()
        seen_text: set[str] = set()
        current_url = ""
        for call in successful_calls:
            tool = str(call.get("tool", ""))
            arguments = call.get("arguments", {})
            if tool in {"browser_navigate", "web_read"} and isinstance(
                arguments,
                dict,
            ):
                candidate_url = str(arguments.get("url", "")).strip()
                if candidate_url and not TaskContract.is_search_listing_url(
                    candidate_url
                ):
                    current_url = candidate_url
            if tool not in {"browser_snapshot", "web_read"}:
                continue
            excerpt = str(call.get("result_excerpt", ""))
            page_match = re.search(
                r"^- Page URL:\s*(https?://\S+)",
                excerpt,
                re.MULTILINE | re.IGNORECASE,
            )
            source_url = page_match.group(1) if page_match else current_url

            for line in excerpt.splitlines():
                text_match = re.search(
                    r"\b(?:paragraph|blockquote|text)\s*(?:\"([^\"]+)\"|:\s*(.+))$",
                    line,
                    re.IGNORECASE,
                )
                if text_match is None:
                    continue
                observed_text = " ".join(
                    next(
                        value
                        for value in text_match.groups()
                        if value is not None
                    ).split()
                ).strip(" .:-")
                key = observed_text.casefold()
                if (
                    len(observed_text) < 30
                    or key in seen_text
                    or observed_text.startswith("[DARKLINGER ")
                ):
                    continue
                seen_text.add(key)
                text_records.append((observed_text[:360], source_url))

            article_blocks = re.split(
                r"(?=^\s*-\s+article\s*:)",
                excerpt,
                flags=re.MULTILINE | re.IGNORECASE,
            )
            structured_blocks = [
                block
                for block in article_blocks
                if re.match(
                    r"^\s*-\s+article\s*:",
                    block,
                    re.IGNORECASE,
                )
            ]
            if len(structured_blocks) >= 2:
                for block in structured_blocks:
                    headings = re.findall(
                        r'\bheading\s+["\']([^"\']{3,160})["\']'
                        r"\s+\[level=[2-6]\]",
                        block,
                        re.IGNORECASE,
                    )
                    if not headings:
                        continue
                    label = " ".join(headings[-1].split()).strip(" .:-")
                    key = label.casefold()
                    if label and key not in seen_items:
                        seen_items.add(key)
                        item_records.append((label, source_url))

            for image_label in re.findall(
                r'\bimg\s+["\']([^"\']{3,220})["\']',
                excerpt,
                re.IGNORECASE,
            ):
                label = " ".join(image_label.split()).strip(" .:-")
                key = label.casefold()
                if label and key not in seen_images:
                    seen_images.add(key)
                    image_records.append((label, source_url))

        lines = [report]
        if text_records:
            lines.append("Observed source excerpts:")
            for observed_text, source_url in text_records[:6]:
                source_note = f" Source: {source_url}." if source_url else ""
                lines.append(f"- \u201c{observed_text}\u201d{source_note}")
        if "item_list" in contract.required_research_facets and item_records:
            lines.append("Observed item records:")
            for label, source_url in item_records[:12]:
                source_note = f" Source: {source_url}." if source_url else ""
                lines.append(
                    f"- {label} — this item label was present in the inspected "
                    f"page structure.{source_note}"
                )
        if "images" in contract.required_research_facets and image_records:
            lines.append("Observed image labels:")
            for label, source_url in image_records[:8]:
                source_note = f" Source: {source_url}." if source_url else ""
                lines.append(f"- {label}.{source_note}")
        if "exhaustive_coverage" in contract.required_research_facets:
            lines.append(
                "Coverage note: these are the records preserved in the bounded "
                "runtime evidence; the fallback does not claim that omitted "
                "records were verified."
            )
        return "\n".join(lines)

    @staticmethod
    def _tool_evidence_identity(call: dict[str, Any]) -> str:
        """Identify repeated tool evidence across continuous step batches."""

        return json.dumps(
            {
                "tool": call.get("tool", ""),
                "arguments": call.get("arguments", {}),
                "status": call.get("status", ""),
                "result_excerpt": call.get("result_excerpt", ""),
                "error": call.get("error", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _tool_request_identity(tool: str, arguments: Any) -> str:
        if tool == "browser_navigate" and isinstance(arguments, dict):
            arguments = dict(arguments)
            raw_url = str(arguments.get("url", "")).strip()
            try:
                parts = urlsplit(raw_url)
            except ValueError:
                parts = None
            if parts is not None and parts.scheme and parts.netloc:
                path = parts.path.rstrip("/")
                arguments["url"] = urlunsplit(
                    (
                        parts.scheme.casefold(),
                        parts.netloc.casefold(),
                        path,
                        parts.query,
                        "",
                    )
                )
        return json.dumps(
            {"tool": tool, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _normalize_runtime_tool_arguments(
        self,
        tool: str,
        arguments: Any,
    ) -> Any:
        """Apply runtime-owned argument normalization before loop identity.

        Equivalent authorized relative/absolute paths share one ledger identity.
        Scope denials preserve their original arguments for the execution boundary
        to report; normalization never substitutes a different filesystem object.
        """

        normalize = getattr(self.tools, "normalize_arguments", None)
        if not callable(normalize):
            return arguments
        try:
            normalized = normalize(tool, arguments)
        except (OSError, TypeError, ValueError):
            return arguments
        return normalized

    @staticmethod
    def _repair_observed_file_path(
        tool: str,
        arguments: Any,
        successful_calls: list[dict[str, Any]],
    ) -> Any:
        """Preserve the requested object; guessed path aliases are not authority.

        Kept as a compatibility hook. A typo must be corrected explicitly after
        inspecting the exact-target failure, never by redirecting to another file.
        """
        return arguments

    @staticmethod
    def _repair_schema_argument_alias(
        tool: str,
        arguments: Any,
        definitions: list[dict[str, Any]],
    ) -> Any:
        """Repair one unambiguous argument-key alias from the active schema."""

        if not isinstance(arguments, dict):
            return arguments
        definition = next(
            (
                item
                for item in definitions
                if isinstance(item, dict)
                and item.get("function", {}).get("name") == tool
            ),
            None,
        )
        if definition is None:
            return arguments
        schema = definition.get("function", {}).get("parameters", {})
        if not isinstance(schema, dict) or schema.get("type") != "object":
            return arguments
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return arguments
        missing = [str(field) for field in required if str(field) not in arguments]
        unexpected = [field for field in arguments if field not in properties]
        if len(missing) != 1 or len(unexpected) != 1:
            return arguments
        target = missing[0]
        source = unexpected[0]
        target_schema = properties.get(target, {})
        if not isinstance(target_schema, dict):
            return arguments
        try:
            Agent._validate_tool_argument_value(
                arguments[source],
                target_schema,
                path=f"$.{target}",
            )
        except ValueError:
            return arguments
        repaired = dict(arguments)
        repaired[target] = repaired.pop(source)
        return repaired

