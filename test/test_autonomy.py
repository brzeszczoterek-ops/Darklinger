from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from v_core.autonomy import (
    AuthorizationEnvelope,
    AgentTaskTrace,
    AutonomousRunner,
    AutonomousTask,
    CheckpointStore,
    ChordDetector,
    ControlChannel,
    ControlSignal,
    ContextWindowManager,
    ExecutionMode,
    GlobalControlChannel,
    RuntimeRegistry,
    StepOutcome,
    StepResult,
    TaskBudget,
    TaskContract,
    MultilingualIntentRouter,
    SemanticIntent,
    TaskJournal,
    TaskStatus,
    parse_chord,
    review_task,
    review_task_checkpoint,
)
from v_core.autonomy.policy import AuthorizationDenied, AuthorizationGuard
from v_core.capabilities.web_target import extract_web_target, extract_web_targets
from v_core.config import load_config


def test_task_contract_detects_polish_local_tool_and_required_use() -> None:
    contract = TaskContract.from_prompt(
        "Stwórz lokalne narzędzie count_words, a następnie użyj go."
    )

    assert contract.requires_created_tool is True
    assert contract.requires_created_tool_execution is True
    assert contract.unmet([]) == [
        "learning_create_tool",
        "generated_tool_execution",
    ]

    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": json.dumps({"name": "count_words"}),
        },
        {
            "tool": "count_words",
            "status": "succeeded",
            "result_excerpt": json.dumps({"word_count": 6}),
        },
    ]
    assert contract.unmet(calls) == []


def test_task_contract_tracks_created_tool_when_lifecycle_receipt_is_clipped() -> None:
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "created_tool_name": "har_traffic_summary",
            "result_excerpt": '{"activation_count":1,"artifact_id":"clipped',
        },
        {
            "tool": "har_traffic_summary",
            "status": "succeeded",
            "result_excerpt": '{"requests_count":2}',
        },
    ]

    assert contract.unmet(calls) == []


def test_created_tool_results_request_requires_real_execution() -> None:
    contract = TaskContract.from_prompt(
        "Stwórz takie narzędzie i pokaż mi rezultaty."
    )

    assert contract.requires_created_tool is True
    assert contract.requires_created_tool_execution is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Powiedz mi, czy potrafiłabyś stworzyć narzędzie do rejestrowania ruchu?",
        "V, czy potrafisz tworzyć własne narzędzia?",
        "Are you able to create a page monitoring tool?",
        "Do you know how to build an agent skill?",
    ],
)
def test_creation_capability_questions_do_not_create_artifacts(prompt: str) -> None:
    contract = TaskContract.from_prompt(prompt)

    assert TaskContract.asks_about_creation_capability(prompt) is True
    assert contract.requires_created_tool is False
    assert contract.requires_created_skill is False
    assert contract.requires_created_tool_execution is False


@pytest.mark.parametrize(
    "prompt",
    [
        "Stwórz narzędzie do rejestrowania ruchu.",
        "Czy możesz stworzyć narzędzie do rejestrowania ruchu?",
        "Could you create a page monitoring tool?",
        "Build an agent skill now.",
    ],
)
def test_explicit_creation_requests_remain_actionable(prompt: str) -> None:
    contract = TaskContract.from_prompt(prompt)

    assert contract.requires_created_tool or contract.requires_created_skill


def test_semantic_created_artifact_results_require_execution_in_any_language() -> None:
    intent = SemanticIntent(
        action_requested=True,
        capabilities=("learning_tool",),
        requires_report=True,
        execute_created_artifact=True,
    )

    contract = intent.to_contract("Create the requested capability.")

    assert contract.requires_created_tool is True
    assert contract.requires_created_tool_execution is True


def test_created_tool_validation_report_names_tool_not_contract() -> None:
    contract = TaskContract(requires_created_tool=True)
    answer = contract.deterministic_answer(
        [
            {
                "tool": "learning_create_tool",
                "status": "succeeded",
                "result_excerpt": json.dumps(
                    {
                        "name": "darknet_observer",
                        "status": "active",
                        "validation": {
                            "passed": True,
                            "tests": [
                                {
                                    "name": "deterministic tool smoke test",
                                    "passed": True,
                                }
                            ],
                        },
                    }
                ),
            }
        ]
    )

    assert answer is not None
    assert "built the generated tool" in answer
    assert "built the contract" not in answer


def test_behaviorally_tested_tool_is_not_reported_as_semantically_proven() -> None:
    contract = TaskContract(requires_created_tool=True)
    answer = contract.deterministic_answer(
        [
            {
                "tool": "learning_create_tool",
                "status": "succeeded",
                "result_excerpt": json.dumps(
                    {
                        "name": "candidate_solver",
                        "status": "active",
                        "successful_runs": 0,
                        "validation": {
                            "passed": True,
                            "validation_strength": (
                                "behavioral_input_sensitivity"
                            ),
                            "tests": [
                                {
                                    "name": (
                                        "runtime-derived input sensitivity probe"
                                    ),
                                    "passed": True,
                                }
                            ],
                        },
                    }
                ),
            }
        ]
    )

    assert answer is not None
    assert "experimental tool `candidate_solver`" in answer
    assert "no independent semantic oracle" in answer
    assert "not yet run on a real task input" in answer
    assert "validated" not in answer


def test_tor_inventory_deterministic_answer_reports_only_observed_facts() -> None:
    payload = {
        "ok": True,
        "seed_url": "http://" + "a" * 56 + ".onion/",
        "coverage": "partial",
        "max_pages": 2,
        "max_depth": 2,
        "pages_observed": 2,
        "pages_succeeded": 2,
        "pages_failed": 0,
        "budget_exhausted": True,
        "timed_out": False,
        "javascript_enabled": False,
        "forms_submitted": False,
        "credentials_used": False,
        "redirects_followed": False,
        "records": [
            {
                "url": "http://" + "a" * 56 + ".onion/",
                "title": "Observed seed",
                "status": 200,
            },
            {
                "url": "http://" + "a" * 56 + ".onion/about",
                "title": "About",
                "status": 200,
            },
        ],
    }
    contract = TaskContract(required_tools=("full_tor_inventory",))

    answer = contract.deterministic_answer(
        [
            {
                "tool": "full_tor_inventory",
                "status": "succeeded",
                "result_excerpt": json.dumps(payload),
            }
        ],
        language="Polish",
    )

    assert answer is not None
    assert "zaobserwowano 2 stron" in answer
    assert "Przerwano zgodnie z ustawionym limitem" in answer
    assert "Observed seed — HTTP 200" in answer
    assert "Nie dowodzi bezpieczeństwa serwisu" in answer
    assert "bezpieczny" not in answer.casefold()


def test_task_contract_accepts_deterministic_snapshot_tool_builder() -> None:
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_snapshot_extractor",
            "status": "succeeded",
            "result_excerpt": json.dumps({"name": "extract_book_cards"}),
        },
        {
            "tool": "extract_book_cards",
            "status": "succeeded",
            "result_excerpt": json.dumps({"records": []}),
        },
    ]

    assert contract.unmet(calls) == []


def test_task_contract_renders_generated_records_without_model_rewrite() -> None:
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
        requires_evidence_report=True,
    )
    records = [
        {
            "title": "A Light in the Attic",
            "price": "£51.77",
            "availability": "In stock",
            "relative_product_url": (
                "catalogue/a-light-in-the-attic_1000/index.html"
            ),
        },
        {
            "title": "Tipping the Velvet",
            "price": "£53.74",
            "availability": "In stock",
            "relative_product_url": "catalogue/tipping-the-velvet_999/index.html",
        },
        {
            "title": "Soumission",
            "price": "£50.10",
            "availability": "In stock",
            "relative_product_url": "catalogue/soumission_998/index.html",
        },
    ]
    calls = [
        {
            "tool": "learning_create_snapshot_extractor",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "name": "extract_book_cards",
                    "status": "active",
                    "validation": {"passed": True},
                }
            ),
        },
        {
            "tool": "extract_book_cards",
            "status": "succeeded",
            "result_excerpt": json.dumps({"records": records}),
        },
    ]

    answer = contract.deterministic_answer(calls)

    assert answer is not None
    assert "validated and activated" in answer
    assert "catalogue/a-light-in-the-attic_1000/index.html" in answer
    assert "attica-1000" not in answer


def test_task_contract_round_trip_preserves_dynamic_required_tools() -> None:
    contract = TaskContract(
        requires_evidence_report=True,
        required_public_subject="Cud Malina",
    ).with_required_tools(["count_words", "count_words"])

    restored = TaskContract.from_dict(contract.to_dict())
    merged = restored.merged(
        TaskContract().with_required_tools(["extract_domains"])
    )

    assert restored.required_tools == ("count_words",)
    assert restored.required_public_subject == "Cud Malina"
    assert merged.required_tools == ("count_words", "extract_domains")
    assert merged.required_public_subject == "Cud Malina"


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    task = AutonomousTask(objective="Build a report", task_id="report-1")
    task.transition(TaskStatus.RUNNING)
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path / "workspace"),
        capabilities={"read_workspace"},
    )
    store = CheckpointStore(tmp_path / "checkpoints")

    store.save(task, envelope)
    loaded = store.load("report-1")

    assert loaded is not None
    restored_task, restored_envelope = loaded
    assert restored_task.objective == "Build a report"
    assert restored_task.status == TaskStatus.RUNNING
    assert restored_envelope.capabilities == {"read_workspace"}


def test_journal_is_append_only_jsonl(tmp_path: Path) -> None:
    journal = TaskJournal(tmp_path / "journal")

    journal.append("task-1", "started", {"value": 1})
    journal.append("task-1", "completed", {"value": 2})

    records = journal.read("task-1")
    assert [record["event"] for record in records] == ["started", "completed"]
    assert records[1]["data"]["value"] == 2
    assert (tmp_path / "journal").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "journal" / "task-1.jsonl").stat().st_mode & 0o777 == 0o600


def test_interactive_agent_trace_records_runtime_evidence(tmp_path: Path) -> None:
    trace = AgentTaskTrace(tmp_path / "interactive", "Inspect a page")
    sequence = trace.tool_started(
        "browser_navigate",
        {"url": "https://example.com"},
    )
    trace.tool_finished(sequence, "Example Domain")
    trace.complete("The page title is Example Domain.")

    checkpoint = (
        tmp_path
        / "interactive"
        / "checkpoints"
        / f"{trace.task_id}.json"
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    records = trace._journal.read(trace.task_id)

    assert payload["status"] == "completed"
    assert payload["tool_calls"][0]["status"] == "succeeded"
    assert payload["tool_calls"][0]["result_sha256"]
    assert [record["event"] for record in records] == [
        "task_started",
        "tool_started",
        "tool_completed",
        "task_completed",
    ]
    assert trace.evidence()["successful_tool_count"] == 1
    assert checkpoint.stat().st_mode & 0o777 == 0o600


def test_interactive_trace_persists_prioritized_evidence_excerpt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interactive"
    trace = AgentTaskTrace(root, "Inspect search results")
    sequence = trace.tool_started("browser_snapshot", {})
    trace.tool_finished(
        sequence,
        "raw navigation chrome " * 500,
        evidence_excerpt=(
            "Observed result: Crawl4AI — https://docs.crawl4ai.com/"
        ),
    )

    restored = AgentTaskTrace.load(root, trace.task_id)

    assert restored is not None
    assert restored.tool_calls[0]["result_excerpt"] == (
        "Observed result: Crawl4AI — https://docs.crawl4ai.com/"
    )
    assert restored.tool_calls[0]["result_sha256"]


def test_interactive_trace_recovers_latest_runtime_context(tmp_path: Path) -> None:
    root = tmp_path / "interactive"
    trace = AgentTaskTrace(root, "Inspect the first result")
    trace.set_requirements({"requires_distinct_detail_page": True})
    sequence = trace.tool_started("browser_navigate", {"url": "https://bad.invalid"})
    trace.tool_finished(
        sequence,
        "DNS resolution failed",
        error="MCPToolExecutionError: DNS resolution failed",
    )
    trace.block("navigation failed")

    context = AgentTaskTrace.latest_context(root)

    assert context is not None
    assert context["objective"] == "Inspect the first result"
    assert context["status"] == "blocked"
    assert context["requirements"]["requires_distinct_detail_page"] is True
    assert context["tool_calls"][0]["error"].endswith("DNS resolution failed")


def test_trace_recovers_latest_created_tool_across_later_turns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interactive"
    creation = AgentTaskTrace(root, "Create a traffic analyzer")
    sequence = creation.tool_started("learning_create_tool", {"source": "..."})
    creation.tool_calls[sequence - 1]["created_tool_name"] = (
        "har_traffic_summary"
    )
    creation.tool_finished(sequence, '{"name":"har_traffic_summary"}')
    creation.complete("Created the tool.")

    later = AgentTaskTrace(root, "Discuss the result")
    later.complete("Done.")

    created = AgentTaskTrace.latest_created_tool(root)

    assert created == {
        "name": "har_traffic_summary",
        "task_id": creation.task_id,
    }


def test_runtime_review_is_grounded_in_checkpoint_events(tmp_path: Path) -> None:
    payload = {
        "task_id": "interactive-review-1",
        "objective": "Find a Firecrawler alternative and report it.",
        "status": "running",
        "finished_at": None,
        "requirements": TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
            requires_distinct_detail_page=True,
            requires_evidence_report=True,
        ).to_dict(),
        "tool_calls": [
            {
                "sequence": 1,
                "tool": "browser_navigate",
                "status": "succeeded",
                "arguments": {"url": "https://duckduckgo.com/?q=firecrawler"},
                "result_sha256": "nav-search",
            },
            {
                "sequence": 2,
                "tool": "browser_snapshot",
                "status": "succeeded",
                "arguments": {},
                "result_sha256": "search-results",
                "result_excerpt": "Scrapy https://scrapy.org",
            },
            {
                "sequence": 3,
                "tool": "browser_navigate",
                "status": "failed",
                "arguments": {"url": "https://bad.example/docs"},
                "error": "BrowserHTTPError: HTTP status 404",
                "result_sha256": "http-404",
            },
            {
                "sequence": 4,
                "tool": "browser_snapshot",
                "status": "failed",
                "arguments": {},
                "error": "BrowserHTTPError: HTTP status 404",
                "result_sha256": "error-page",
            },
            {
                "sequence": 5,
                "tool": "browser_navigate",
                "status": "succeeded",
                "arguments": {"url": "https://scrapy.org"},
                "result_sha256": "nav-detail",
            },
            {
                "sequence": 6,
                "tool": "browser_snapshot",
                "status": "succeeded",
                "arguments": {},
                "result_sha256": "detail-page",
                "result_excerpt": "Scrapy is an open source web crawling framework.",
            },
            {
                "sequence": 7,
                "tool": "browser_find",
                "status": "succeeded",
                "arguments": {"text": "Firecrawler"},
                "result_sha256": "no-match",
            },
            {
                "sequence": 8,
                "tool": "browser_find",
                "status": "succeeded",
                "arguments": {"text": "Firecrawler"},
                "result_sha256": "no-match",
            },
        ],
        "context_rollovers": [
            {
                "sequence": 1,
                "context_size": 12_000,
                "estimated_tokens_after": 12_100,
                "evidence_count": 0,
            }
        ],
    }

    report = review_task_checkpoint(payload)
    findings = {item["code"]: item for item in report["findings"]}

    assert report["metrics"] == {
        "tool_calls": 8,
        "successful_tool_calls": 6,
        "failed_tool_calls": 2,
        "context_rollovers": 1,
        "finding_count": len(findings),
    }
    assert findings["snapshot_after_failed_navigation"]["tool_calls"] == [4]
    assert findings["repeated_identical_result"]["tool_calls"] == [7, 8]
    assert findings["rollover_without_evidence"]["rollovers"] == [1]
    assert findings["rollover_still_over_context"]["rollovers"] == [1]
    assert findings["tooling_after_contract_satisfied"]["tool_calls"] == [7, 8]
    assert "Only describe faults listed" in report["grounding_rule"]


def test_runtime_review_uses_previous_task_and_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interactive"
    prior = AgentTaskTrace(root, "Prior task")
    prior.complete("done")
    current = AgentTaskTrace(root, "Review the prior task")

    report = review_task(root, exclude_task_id=current.task_id)

    assert report["task_id"] == prior.task_id
    with pytest.raises(ValueError, match="invalid DARKLINGER"):
        review_task(root, task_id="../../secrets")


def test_trace_recovery_marks_dead_runtime_as_interrupted(tmp_path: Path) -> None:
    root = tmp_path / "interactive"
    trace = AgentTaskTrace(root, "Interrupted work")
    checkpoint = root / "checkpoints" / f"{trace.task_id}.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["runtime_pid"] = 999_999_999
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    recovered = AgentTaskTrace.recover_stale_running(root)
    restored = AgentTaskTrace.load(root, trace.task_id)

    assert recovered == [trace.task_id]
    assert restored is not None
    assert restored.status == "interrupted"
    assert restored.finished_at is not None
    assert restored._journal.read(trace.task_id)[-1]["event"] == (
        "task_recovered_as_interrupted"
    )
    report = review_task(root, task_id=trace.task_id)
    assert report["findings"][0]["code"] == "interrupted_checkpoint"


def test_task_contract_detects_runtime_self_review() -> None:
    contract = TaskContract.from_prompt(
        "Przeanalizuj swoje logi z ostatniej sesji i pokaż, gdzie były błędy."
    )

    assert contract.requires_runtime_review is True
    assert contract.unmet([]) == ["runtime_review_task"]
    assert contract.unmet(
        [{"tool": "runtime_review_task", "status": "succeeded"}]
    ) == []


def test_task_contract_detects_generic_online_work_without_explicit_url() -> None:
    contract = TaskContract.from_prompt(
        "Znajdź w sieci narzędzia do monitorowania darknetu i podaj wyniki."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_web_discovery is True
    assert contract.requires_distinct_detail_page is True
    assert contract.requires_evidence_report is True
    assert contract.unmet([]) == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate:distinct_detail_page",
    ]


def test_local_code_analysis_explicitly_disables_network() -> None:
    prompt = (
        "Przeanalizuj wyłącznie plik test/fixtures/code_analysis_probe.py. "
        "Nie zmieniaj pliku i nie używaj sieci. Podaj dwa błędy wykonania."
    )

    contract = TaskContract.from_prompt(prompt)

    assert TaskContract.requests_read_only(prompt)
    assert TaskContract.disables_web(prompt)
    assert contract.requires_file_read
    assert not contract.requires_file_mutation
    assert not contract.requires_browser_navigation
    assert not contract.requires_browser_snapshot
    assert not contract.requires_web_discovery


def test_task_contract_routes_polish_tor_research_to_report_artifact() -> None:
    contract = TaskContract.from_prompt(
        "Zgromadź wszystkie wiadomości jakie dasz radę na temat cebulki, "
        "forum na torze. Zapisz je do pliku, bez zgadywania, tylko naprawdę "
        "przejrzyj co tam się dzieje i znajdź wszystkie możliwe informacje."
    )

    assert contract.required_tools == ("full_tor_search",)
    assert contract.required_capabilities == ("network.tor.search",)
    assert contract.requires_evidence_report is True
    assert contract.requires_file_mutation is True
    assert contract.requires_file_read is False


def test_task_contract_routes_english_tor_research_to_report_artifact() -> None:
    contract = TaskContract.from_prompt(
        "Gather all available information about this forum on Tor and save "
        "the verified report to a file."
    )

    assert contract.required_tools == ("full_tor_search",)
    assert contract.requires_evidence_report is True
    assert contract.requires_file_mutation is True
    assert contract.requires_file_read is False


def test_github_source_wins_over_darknet_tool_topic_in_polish_request() -> None:
    prompt = (
        "Z uwagi na to, że brakuje ci narzędzi i umiejętności do "
        "przeszukiwania darknetu, wejdź na GitHuba i poszukaj na GitHubie "
        "zestawów narzędzi do OSINT-u w darknecie. Przeszukaj GitHuba, "
        "a to, co może się przydać, najpierw mi pokaż."
    )

    contract = TaskContract.from_prompt(prompt)

    assert TaskContract.prefers_tor(prompt) is False
    assert contract.requires_browser_navigation is True
    assert contract.requires_web_discovery is True
    assert contract.required_tools == ()
    assert contract.required_capabilities == ()


def test_github_candidate_request_requires_two_observed_repositories() -> None:
    prompt = (
        "Znajdź na GitHubie przydatne narzędzia i umiejętności dotyczące "
        "darknetu, OSINT i cyber security. Przedstaw konkretną listę do "
        "mojego zatwierdzenia, zanim cokolwiek przyswoisz."
    )

    contract = TaskContract.from_prompt(prompt)

    assert contract.required_research_facets == ("github_repositories",)
    assert contract.minimum_detail_sources == 2


def test_explicit_first_github_result_remains_a_single_source_task() -> None:
    contract = TaskContract.from_prompt(
        "Search GitHub for I2P tools, inspect the first result, and report its actual repo."
    )

    assert contract.required_research_facets == ()
    assert contract.minimum_detail_sources == 1


def test_github_candidate_contract_does_not_count_article_as_repository() -> None:
    contract = TaskContract.from_prompt(
        "Find useful GitHub tools for darknet OSINT and present a report for approval."
    )
    repo_one = "https://github.com/thamore/master-osintv4-"
    article = "https://brandefense.io/blog/dark-web-osint-tools"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": "darknet OSINT GitHub repositories"},
            "result_excerpt": f"{repo_one}\n{article}",
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": repo_one},
            "result_excerpt": f"- Page URL: {repo_one}",
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {},
            "result_excerpt": (
                f"- Page URL: {repo_one}\n"
                "- Page Title: thamore/master-osintv4- · GitHub\n"
                "- paragraph: Curated OSINT resources."
            ),
        },
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": article},
            "result_excerpt": f"- Page URL: {article}",
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {},
            "result_excerpt": (
                f"- Page URL: {article}\n"
                "- Page Title: Dark Web OSINT Tools\n"
                "- paragraph: A survey article."
            ),
        },
    ]

    assert "browser_evidence:detail_sources=1/2" in contract.unmet(calls)


def test_github_candidate_answer_uses_only_observed_repository_pages() -> None:
    contract = TaskContract.from_prompt(
        "Find useful GitHub tools for darknet OSINT and present a report for approval."
    )
    repo_one = "https://github.com/thamore/master-osintv4-"
    repo_two = "https://github.com/tekcin/robin_scrapper"
    article = "https://brandefense.io/blog/dark-web-osint-tools"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": "darknet OSINT GitHub repositories"},
            "result_excerpt": f"{repo_one}\n{repo_two}\n{article}",
        },
    ]
    for url, title in (
        (repo_one, "thamore/master-osintv4- · GitHub"),
        (article, "Dark Web OSINT Tools"),
        (repo_two, "tekcin/robin_scrapper · GitHub"),
    ):
        calls.extend(
            [
                {
                    "tool": "browser_navigate",
                    "status": "succeeded",
                    "arguments": {"url": url},
                    "result_excerpt": f"- Page URL: {url}",
                },
                {
                    "tool": "browser_snapshot",
                    "status": "succeeded",
                    "arguments": {},
                    "result_excerpt": (
                        f"- Page URL: {url}\n"
                        f"- Page Title: {title}\n"
                        "- paragraph: Observed source details."
                    ),
                },
            ]
        )

    assert contract.unmet(calls) == []
    answer = contract.deterministic_answer(calls, language="Polish")

    assert answer is not None
    assert "thamore/master-osintv4-" in answer
    assert "tekcin/robin_scrapper" in answer
    assert "brandefense" not in answer.casefold()
    assert "Niczego jeszcze nie instalowałam" in answer
    assert "zatwierdzasz" in answer


def test_github_source_preceding_review_action_wins_in_spoken_polish() -> None:
    prompt = (
        "Ponieważ nie masz narzędzi do pracy w darknecie, chciałbym, żebyś "
        "weszła na GitHuba, przejrzała repozytoria z zestawami narzędzi pod "
        "darknet i do OSINT-u. Najpierw przedstaw mi listę."
    )

    contract = TaskContract.from_prompt(prompt)

    assert TaskContract.prefers_tor(prompt) is False
    assert contract.requires_browser_navigation is True
    assert contract.requires_web_discovery is True
    assert contract.requires_file_read is False
    assert contract.required_tools == ()


def test_github_source_wins_over_darknet_tool_topic_in_english_request() -> None:
    prompt = (
        "Search GitHub for tools for searching the dark web. Show me what "
        "you choose before adopting anything."
    )

    contract = TaskContract.from_prompt(prompt)

    assert TaskContract.prefers_tor(prompt) is False
    assert contract.requires_browser_navigation is True
    assert contract.requires_web_discovery is True
    assert contract.required_tools == ()


def test_direct_darknet_source_wins_over_github_repository_topic() -> None:
    prompt = "Przeszukaj darknet pod kątem repozytoriów GitHub do OSINT-u."

    contract = TaskContract.from_prompt(prompt)

    assert TaskContract.prefers_tor(prompt) is True
    assert contract.required_tools == ("full_tor_search",)
    assert contract.requires_browser_navigation is False


def test_explicit_tor_transport_wins_for_github_destination() -> None:
    prompt = "Używając Tora, przeszukaj GitHuba pod kątem narzędzi OSINT."

    assert TaskContract.prefers_tor(prompt) is True


def test_output_report_file_is_not_misread_as_an_input_file() -> None:
    contract = TaskContract.from_prompt(
        "Inspect the forum, verify the findings, and save them to report.md."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_file_mutation is True
    assert contract.requires_file_read is False


def test_public_business_facts_recover_when_semantic_router_calls_it_chat() -> None:
    prompt = (
        "Cześć mi. Słuchaj, sprawdź mi ile w Warszawie jest cukierni "
        "Cud Malina i sprawdź mi również godziny otwarcia. "
        "I sprawdź mi również gdzie. Podaj mi tylko sprawdzone informacje."
    )
    false_chat_intent = SemanticIntent(
        action_requested=False,
        capabilities=(),
    )

    assert TaskContract.from_prompt(prompt).requires_browser_navigation is False
    assert TaskContract.implies_public_web_lookup(prompt) is True
    assert MultilingualIntentRouter._usable(false_chat_intent, prompt) is False


def test_public_fact_lookup_recovery_does_not_capture_creative_request() -> None:
    prompt = "Napisz opowiadanie o cukierni Cud Malina w Warszawie."

    assert TaskContract.implies_public_web_lookup(prompt) is False


def test_purchase_source_question_does_not_require_postal_address() -> None:
    prompt = (
        "Znajdź w internecie przedwojenne nocniki, opisz ich rodzaje, "
        "gdzie można je nabyć i jakie są ceny."
    )

    contract = TaskContract.from_prompt(prompt)

    assert contract.requires_web_discovery is True
    assert contract.required_public_fields == ()


def test_price_question_does_not_require_a_location_count() -> None:
    prompt = (
        "Znajdź w internecie przedwojenne nocniki, sprawdź ile kosztują "
        "i gdzie można je kupić."
    )

    contract = TaskContract.from_prompt(prompt)

    assert contract.required_public_fields == ()


def test_explicit_shop_address_remains_a_required_public_fact() -> None:
    prompt = (
        "Znajdź w internecie sklepy z antykami, gdzie można nabyć "
        "przedwojenne nocniki, i podaj ich adresy."
    )

    contract = TaskContract.from_prompt(prompt)

    assert contract.required_public_fields == ("address",)


@pytest.mark.parametrize(
    "prompt",
    [
        "Znajdź nazwy i adresy działających marketów w darknecie.",
        "Find the names and addresses of active services on the dark web.",
        "List the website addresses and domains of the online services.",
    ],
)
def test_online_service_addresses_are_not_postal_public_facts(
    prompt: str,
) -> None:
    contract = TaskContract.from_prompt(prompt)

    assert "address" not in contract.required_public_fields


def test_explicit_physical_address_wins_over_online_context() -> None:
    contract = TaskContract.from_prompt(
        "Find the physical street address of the online shop's headquarters."
    )

    assert contract.required_public_fields == ("address",)


def test_public_business_contract_stays_open_on_namesake_product_page() -> None:
    prompt = (
        "Sprawdź ile znajduje się w Warszawie cukierni, które nazywają się "
        "Cud Malina, i znajdź wszystkie adresy i numery telefonów."
    )
    intent = SemanticIntent(
        action_requested=True,
        capabilities=("browser",),
        requires_report=True,
    )
    contract = intent.to_contract(prompt)
    product_page = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": (
                "Cud malina - cena | Cukiernia Sweet Home; "
                "url: https://www.sweet-home.pl/994,Cud-malina.html"
            ),
        },
        {
            "tool": "web_read",
            "status": "succeeded",
            "arguments": {
                "url": "https://www.sweet-home.pl/994,Cud-malina.html"
            },
            "result_excerpt": (
                "Cud malina. Torty kompozycje smaków. 120 zł za kg. "
                "Zadzwoń pod nr 698 314 125."
            ),
        },
    ]

    missing = contract.unmet(product_page)

    assert contract.required_public_fields == ("count", "address", "contact")
    assert "public_fact:address" in missing
    assert "public_fact:contact" not in missing


def test_search_query_metadata_cannot_satisfy_public_address_or_hours() -> None:
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_evidence_report=True,
        required_public_fields=("address", "opening_hours"),
    )
    search_only = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {
                "query": "address and opening hours Cud Malina Warsaw"
            },
            "result_excerpt": json.dumps(
                {
                    "query": "address and opening hours Cud Malina Warsaw",
                    "result_count": 10,
                    "results": [
                        {
                            "rank": 1,
                            "title": "Cud malina cake, price per kg",
                            "url": "https://example.test/product-994",
                        }
                    ],
                }
            ),
        }
    ]

    missing = contract.unmet(search_only)

    assert "public_fact:address" in missing
    assert "public_fact:opening_hours" in missing


def test_public_facts_from_wrong_subject_do_not_complete_contract() -> None:
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_evidence_report=True,
        required_public_fields=("address", "opening_hours"),
        required_public_subject="Cud Malina",
    )
    wrong_business = [
        {
            "tool": "web_read",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/cud-miod"},
            "result_excerpt": (
                "Cud Miód Warszawa. Adres: ul. Senatorska 13/15. "
                "Poniedziałek: 09:00 - 00:00."
            ),
        }
    ]

    missing = contract.unmet(wrong_business)

    assert "public_fact:subject" in missing
    assert "public_fact:address" not in missing
    assert "public_fact:opening_hours" not in missing


def test_public_business_contract_accepts_observed_address_and_contact() -> None:
    prompt = "Znajdź adres i numer telefonu cukierni Cud Malina."
    intent = SemanticIntent(
        action_requested=True,
        capabilities=("browser",),
        requires_report=True,
    )
    contract = intent.to_contract(prompt)
    observed_business_page = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": (
                "Cukiernia Cud Malina; "
                "url: https://example.test/cud-malina-kontakt"
            ),
        },
        {
            "tool": "web_read",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/cud-malina-kontakt"},
            "result_excerpt": (
                "Cukiernia Cud Malina. Adres: ul. Domaniewska 31, "
                "02-672 Warszawa. Telefon: +48 22 123 45 67."
            ),
        },
    ]

    missing = contract.unmet(observed_business_page)

    assert "public_fact:address" not in missing
    assert "public_fact:contact" not in missing


def test_task_contract_skips_discovery_when_owner_supplies_url() -> None:
    contract = TaskContract.from_prompt(
        "Inspect https://example.com/docs and report what is there."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_web_discovery is False


def test_all_owner_supplied_web_targets_are_preserved() -> None:
    prompt = (
        "Compare www.invicti.com/product with "
        "https://www.acunetix.com/product and inspect both."
    )

    assert extract_web_targets(prompt) == (
        "https://www.invicti.com/product",
        "https://www.acunetix.com/product",
    )
    assert extract_web_target(prompt) == "https://www.invicti.com/product"


def test_search_result_alone_does_not_count_as_page_observation() -> None:
    contract = TaskContract.from_prompt(
        "Inspect https://www.invicti.com/product and report what is there."
    )
    search_only = [
        {
            "tool": "web_search",
            "arguments": {"query": "Invicti product"},
            "status": "succeeded",
            "result_excerpt": "https://www.invicti.com/product",
        }
    ]

    assert contract.unmet(search_only) == [
        "browser_navigate",
        "browser_snapshot",
    ]


def test_multiple_owner_urls_require_distinct_observed_pages() -> None:
    contract = TaskContract.from_prompt(
        "Compare https://www.invicti.com/product and "
        "https://www.acunetix.com/product."
    )

    assert contract.requires_distinct_detail_page is True


def test_spoken_url_is_reconstructed_without_search_discovery() -> None:
    prompt = (
        "Otwórz HTTPS, dwukropek, łamane, łamane, this, minus, domain, "
        "minus, definitely, minus, those, minus, not, minus, exist, "
        "kropka, invalid. I powiedz co znalazłaś."
    )

    assert extract_web_target(prompt) == (
        "https://this-domain-definitely-those-not-exist.invalid"
    )
    contract = TaskContract.from_prompt(prompt)
    assert contract.requires_browser_navigation is True
    assert contract.requires_web_discovery is False


def test_semantic_browser_contract_enables_discovery_without_url() -> None:
    intent = SemanticIntent(
        action_requested=True,
        capabilities=("browser",),
        requires_report=True,
    )

    discovery = intent.to_contract("Znajdź alternatywę dla Firecrawlera.")
    direct = intent.to_contract("Inspect https://example.com/docs")

    assert discovery.requires_web_discovery is True
    assert discovery.requires_browser_snapshot is False
    assert discovery.requires_distinct_detail_page is False
    assert direct.requires_web_discovery is False
    assert direct.requires_browser_snapshot is True


def test_semantic_current_rate_lookup_does_not_become_deep_research() -> None:
    prompt = (
        "Cześć V, jak ci mija wieczór? Słuchajcie, możesz dla mnie "
        "sprawdzić, jaki jest aktualny kurs Monero do złotówki?"
    )
    intent = SemanticIntent(
        action_requested=True,
        capabilities=("browser",),
        requires_report=True,
        distinct_detail_page=False,
        web_query="actual Monero to zloty exchange rate",
    )

    contract = intent.to_contract(prompt)

    assert contract.requires_browser_navigation is True
    assert contract.requires_web_discovery is True
    assert contract.requires_browser_snapshot is False
    assert contract.requires_distinct_detail_page is False
    assert contract.minimum_detail_sources == 0


def test_semantic_deep_research_still_requires_detail_observation() -> None:
    intent = SemanticIntent(
        action_requested=True,
        capabilities=("browser",),
        requires_report=True,
        distinct_detail_page=True,
    )

    contract = intent.to_contract("Compare current offers and inspect the details.")

    assert contract.requires_browser_snapshot is True
    assert contract.requires_distinct_detail_page is True


def test_discovery_report_rejects_second_search_as_detail_page() -> None:
    contract = TaskContract.from_prompt(
        "Znajdź w internecie alternatywę dla Firecrawlera i podaj wyniki."
    )
    listing_only = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://duckduckgo.com/?q=firecrawl"},
            "status": "succeeded",
        },
        {"tool": "browser_snapshot", "arguments": {}, "status": "succeeded"},
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://duckduckgo.com/?q=firecrawl+alternative"},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Result: https://docs.crawl4ai.com/ - Crawl4AI docs",
        },
    ]

    assert contract.unmet(listing_only) == [
        "browser_navigate:distinct_detail_page"
    ]


def test_discovery_report_rejects_marketplace_category_as_detail_page() -> None:
    contract = TaskContract.from_prompt(
        "Search online markets for antique chamber pots and report prices."
    )
    category = (
        "https://www.olx.pl/antyki-i-kolekcje/kolekcje/"
        "q-ii-wojna-swiatowa/"
    )
    calls = [
        {
            "tool": "web_search",
            "arguments": {"query": "antique chamber pots"},
            "status": "succeeded",
            "result_excerpt": f"Market category: {category}",
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": category},
            "status": "succeeded",
            "result_excerpt": f"- Page URL: {category}",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Filters, categories, and a cookie banner.",
        },
    ]

    assert contract.unmet(calls) == [
        "browser_navigate:distinct_detail_page"
    ]


def test_product_under_collection_path_is_a_valid_detail_page() -> None:
    contract = TaskContract.from_prompt(
        "Search online for an antique chamber pot and report the result."
    )
    product = (
        "https://example.test/collections/antiques/products/"
        "victorian-chamber-pot"
    )
    calls = [
        {
            "tool": "web_search",
            "arguments": {"query": "antique chamber pot"},
            "status": "succeeded",
            "result_excerpt": f"Victorian chamber pot: {product}",
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": product},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Victorian chamber pot, porcelain, price 120 EUR.",
        },
    ]

    assert contract.unmet(calls) == []


def test_discovery_report_accepts_observed_external_source() -> None:
    contract = TaskContract.from_prompt(
        "Find a Firecrawl alternative online and report the result."
    )
    calls = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://duckduckgo.com/?q=firecrawl"},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Result: https://docs.crawl4ai.com/ - Crawl4AI docs",
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://docs.crawl4ai.com/"},
            "status": "succeeded",
        },
        {"tool": "browser_snapshot", "arguments": {}, "status": "succeeded"},
    ]

    assert contract.unmet(calls) == []


def test_discovery_report_accepts_high_level_search_and_read_tools() -> None:
    contract = TaskContract.from_prompt(
        "Find the Cud Malina bakery online and report public information."
    )
    calls = [
        {
            "tool": "web_search",
            "arguments": {"query": "Cud Malina bakery"},
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "query": "Cud Malina bakery",
                    "engine": "duckduckgo",
                    "results": [
                        {
                            "title": "Cud Malina",
                            "url": "https://example.com/cud-malina",
                        }
                    ],
                }
            ),
        },
        {
            "tool": "web_read",
            "arguments": {"url": "https://example.com/cud-malina"},
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "url": "https://example.com/cud-malina",
                    "title": "Cud Malina",
                    "content": "Public business page for Cud Malina.",
                }
            ),
        },
    ]

    assert contract.requires_web_discovery is True
    assert contract.unmet(calls) == []
    assert contract.answer_issues(
        "Cud Malina has a public page at https://example.com/cud-malina.",
        calls,
        request="Find the Cud Malina bakery online.",
    ) == []


def test_discovery_report_rejects_github_chrome_without_document_content() -> None:
    contract = TaskContract.from_prompt(
        "Find a tool online, inspect its documentation, and report the result."
    )
    url = "https://github.com/example/project/blob/main/README.md"
    calls = [
        {
            "tool": "web_search",
            "arguments": {"query": "example tool"},
            "status": "succeeded",
            "result_excerpt": f"Result: {url}",
        },
        {
            "tool": "web_read",
            "arguments": {"url": url},
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "url": url,
                    "title": "project/README.md at main",
                    "content": (
                        "### Page\n- Page URL: "
                        f"{url}\n### Snapshot\nNavigation Menu Platform Solutions "
                        "Enterprise Sign in Sign up"
                    ),
                }
            ),
        },
    ]

    assert contract.unmet(calls) == ["browser_navigate:distinct_detail_page"]


def test_grounded_report_allows_structural_labels_and_observed_tool_names() -> None:
    contract = TaskContract.from_prompt(
        "Find a CAPTCHA tool online and report the result."
    )
    url = "https://example.test/dread-captcha-solver"
    calls = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "result_excerpt": f"Dread CAPTCHA Solver: {url}",
        },
        {
            "tool": "web_read",
            "arguments": {"url": url},
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {
                    "url": url,
                    "content": "Dread CAPTCHA Solver documentation and usage.",
                }
            ),
        },
    ]
    answer = (
        "**Step 1:** `web_search` found Dread CAPTCHA Solver.\n"
        "**Step 2:** `web_read` opened its documentation.\n"
        "**Conclusion:** Dread CAPTCHA Solver is the verified result."
    )

    assert contract.answer_issues(answer, calls, request="Find a CAPTCHA tool.") == []


def test_discovery_report_rejects_high_level_read_of_unobserved_url() -> None:
    contract = TaskContract.from_prompt(
        "Find the Cud Malina bakery online and report public information."
    )
    calls = [
        {
            "tool": "web_search",
            "arguments": {"query": "Cud Malina bakery"},
            "status": "succeeded",
            "result_excerpt": "Result: https://example.com/verified",
        },
        {
            "tool": "web_read",
            "arguments": {"url": "https://invented.invalid/result"},
            "status": "succeeded",
            "result_excerpt": "Invented page",
        },
    ]

    assert contract.unmet(calls) == [
        "browser_navigate:distinct_detail_page"
    ]


def test_discovery_report_rejects_detail_url_guessed_from_model_memory() -> None:
    contract = TaskContract.from_prompt(
        "Find a Firecrawl alternative online and report the result."
    )
    calls = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://duckduckgo.com/?q=fire+chlorella"},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Results about algae supplements and grilling.",
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://www.scrapy.org/"},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Scrapy web scraping framework.",
        },
    ]

    assert contract.unmet(calls) == [
        "browser_navigate:detail_not_discovered"
    ]


def test_discovery_report_does_not_learn_links_from_a_guessed_page() -> None:
    contract = TaskContract.from_prompt(
        "Find a Firecrawl alternative online and report the result."
    )
    calls = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://duckduckgo.com/?q=firecrawl+alternative"},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "Result: https://thunderbit.com/pl/blog/"
                "open-source-firecrawl-alternatives"
            ),
        },
        {
            "tool": "browser_navigate",
            "arguments": {
                "url": (
                    "https://thunderbit.com/pl/blog/"
                    "open-source_firecrawl_alternatives"
                )
            },
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "Generic blog page with a link to https://thunderbit.com/blog"
            ),
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://thunderbit.com/blog"},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Thunderbit blog index",
        },
    ]

    assert contract.unmet(calls) == [
        "browser_navigate:detail_not_discovered"
    ]


def test_task_contract_detects_polish_infinitive_przeszukac_internet() -> None:
    contract = TaskContract.from_prompt(
        "Masz przeszukać internet i znaleźć narzędzia oraz umiejętności "
        "potrzebne do monitorowania darknetowych forów."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_evidence_report is True


def test_conditional_artifact_fallback_is_allowed_but_not_required() -> None:
    contract = TaskContract.from_prompt(
        "Przeszukaj sieć i znajdź alternatywę dla Firecrawlera. Jeżeli nic "
        "nie znajdziesz, wtedy stwórz własne narzędzie albo skill."
    )

    assert contract.requires_created_tool is False
    assert contract.requires_created_skill is False
    assert contract.allows_artifact_fallback is True
    assert "learning_create_tool" not in contract.unmet([])


def test_tool_or_skill_wording_requires_exactly_one_artifact_kind() -> None:
    contract = TaskContract.from_prompt(
        "Musisz znaleźć albo stworzyć skill, bądź też narzędzie, które "
        "rozwiązuje ten problem."
    )

    assert contract.requires_created_artifact is True
    assert contract.requires_created_tool is False
    assert contract.requires_created_skill is False
    assert contract.unmet([]) == ["learning_create_tool_or_skill"]
    assert contract.unmet(
        [{"tool": "learning_create_skill", "status": "succeeded"}]
    ) == []
    assert contract.unmet(
        [{"tool": "learning_create_tool", "status": "succeeded"}]
    ) == []


def test_tool_and_skill_wording_still_requires_both_artifacts() -> None:
    contract = TaskContract.from_prompt(
        "Stwórz narzędzie oraz stwórz skill, który będzie go używał."
    )

    assert contract.requires_created_artifact is False
    assert contract.requires_created_tool is True
    assert contract.requires_created_skill is True


def test_tor_login_infinitive_keeps_network_and_artifact_routes() -> None:
    contract = TaskContract.from_prompt(
        "Wejść w Darknet i zalogować się na forum. Musisz znaleźć albo "
        "stworzyć skill bądź narzędzie."
    )

    assert contract.required_tools == ("full_tor_search",)
    assert contract.required_capabilities == ("network.tor.search",)
    assert contract.requires_created_artifact is True


def test_conditional_artifact_fallback_can_finish_after_unsuccessful_search() -> None:
    contract = TaskContract.from_prompt(
        "Find a Firecrawl alternative online. If none exists, create a tool."
    )
    calls = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://duckduckgo.com/?q=firecrawl+alternative"},
            "status": "succeeded",
        },
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "No matching results were found.",
        },
        {
            "tool": "learning_create_tool",
            "arguments": {"name": "local_crawler"},
            "status": "succeeded",
            "result_excerpt": '{"name": "local_crawler", "status": "active"}',
        },
    ]

    assert contract.allows_artifact_fallback is True
    assert contract.unmet(calls) == []


def test_late_success_guard_does_not_make_tool_creation_conditional() -> None:
    contract = TaskContract.from_prompt(
        "Visit https://books.toscrape.com/. Then create and activate a "
        "task-scoped offline tool named extract_book_cards. Then use the newly "
        "created tool. Do not claim success unless the generated tool really ran."
    )

    assert contract.requires_created_tool is True
    assert contract.requires_created_tool_execution is True
    assert contract.allows_artifact_fallback is False


def test_offline_tool_fixture_urls_do_not_create_browser_contract() -> None:
    contract = TaskContract.from_prompt(
        "Create and activate an offline tool. Do not browse or contact any "
        "service. observations = "
        '[{"url":"http://exampleexample.onion/item"}]. '
        "After it is active, really execute it on the supplied fixture."
    )

    assert contract.requires_browser_navigation is False
    assert contract.requires_browser_snapshot is False
    assert contract.requires_created_tool is True
    assert contract.requires_created_tool_execution is True


def test_adjective_heavy_offline_tool_request_stays_on_local_lifecycle() -> None:
    contract = TaskContract.from_prompt(
        "Create and activate a task-scoped deterministic offline Python tool "
        "named index_onion_observations. Do not browse or contact any network."
    )

    assert contract.requires_created_tool is True
    assert contract.requires_browser_navigation is False
    assert contract.requires_browser_snapshot is False

    polish = TaskContract.from_prompt(
        "Utwórz własne deterministyczne narzędzie Pythona o nazwie "
        "index_onion_observations. Nie korzystaj z sieci."
    )
    assert polish.requires_created_tool is True
    assert polish.requires_browser_navigation is False


def test_task_contract_ignores_creation_words_inside_quoted_tool_input() -> None:
    contract = TaskContract.from_prompt(
        'Użyj narzędzia count_words na tekście "V can build her own tools".'
    )

    assert contract.requires_created_tool is False
    assert contract.requires_created_tool_execution is False


def test_explicit_no_web_clause_removes_semantic_browser_requirements() -> None:
    semantic = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
        requires_created_tool=True,
        required_public_fields=("address",),
        required_public_subject="irrelevant semantic guess",
    )

    constrained = semantic.without_web()

    assert constrained.requires_created_tool is True
    assert constrained.requires_browser_navigation is False
    assert constrained.requires_browser_snapshot is False
    assert constrained.requires_web_discovery is False
    assert constrained.requires_distinct_detail_page is False
    assert constrained.required_public_fields == ()
    assert constrained.required_public_subject == ""


def test_created_tool_contract_recognizes_execute_the_new_tool() -> None:
    contract = TaskContract.from_prompt(
        "Create an offline tool. After validation and activation, really "
        "execute the new tool on Fixture B."
    )

    assert contract.requires_created_tool is True
    assert contract.requires_created_tool_execution is True


def test_semantic_conditional_artifact_fallback_is_not_mandatory() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "action_requested": True,
                "continue_previous": False,
                "capabilities": ["browser", "learning_tool"],
                "requires_report": True,
                "distinct_detail_page": False,
                "artifact_fallback": True,
                "web_query": "alternatywa dla Firecrawlera",
            }
        )
    )

    assert intent is not None
    assert intent.web_query == "alternatywa dla Firecrawlera"
    contract = intent.to_contract("Find an alternative online")
    assert contract.allows_artifact_fallback is True
    assert contract.requires_created_tool is False


def test_task_contract_detects_polish_locative_w_internecie() -> None:
    contract = TaskContract.from_prompt(
        "Znajdź mi w internecie informacje o tej osobie i podaj wyniki."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_evidence_report is True


def test_task_contract_detects_polish_accusative_siec() -> None:
    contract = TaskContract.from_prompt(
        "Cześć V, przeszukaj sieć w poszukiwaniu alternatywy dla "
        "firecrawlera, którą mogłabyś skądś przyswoić."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_evidence_report is False


@pytest.mark.asyncio
async def test_rollover_caps_complete_request_to_local_context_budget() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": ["Checked source"] * 20,
                    "findings": ["Concrete verified finding " + "x" * 500] * 20,
                    "open_questions": ["Which result is relevant?"] * 20,
                    "next_steps": ["Inspect the next source"] * 20,
                }
            )

    manager = ContextWindowManager()
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
    )
    messages = [
        {"role": "system", "content": "S" * 12_000},
        {"role": "user", "content": "history " * 4_000},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"browser_tool_{index}",
                "description": "D" * 500,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(6)
    ]
    evidence = [
        {
            "sequence": index,
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": f"result-{index} " + "e" * 2_000,
            "error": "",
        }
        for index in range(10)
    ]

    rollover = await manager.rollover(
        llm=LLMStub(),
        system_prompt="S" * 12_000,
        objective="Inspect sources",
        contract=contract,
        messages=messages,
        tools=tools,
        evidence=evidence,
        previous_summary=None,
        context_tokens=12_000,
        step=10,
    )

    assert rollover.estimated_tokens_after <= int(12_000 * 0.72)
    assert rollover.estimated_tokens_after < rollover.estimated_tokens_before
    assert rollover.evidence
    assert all(
        len(item) <= 240
        for values in rollover.summary.values()
        for item in values
    )


@pytest.mark.asyncio
async def test_rollover_drops_duplicated_dynamic_context_before_raw_evidence() -> None:
    system_prompt = "\n\n".join(
        (
            "base rules",
            "=== V PERSONA ===\n" + "persona " * 1_200,
            "=== V MEMORY CONTEXT ===\n" + "old memory " * 1_500,
            "=== PREVIOUS RUNTIME CHECKPOINT ===\n" + "old trace " * 1_500,
            "=== AGENT MODE ===\nexecute with evidence",
            "=== OUTPUT LANGUAGE GATE ===\nEnglish output",
        )
    )
    evidence = [
        {
            "sequence": 9,
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Newest verified page evidence",
            "error": "",
        }
    ]

    rollover = await ContextWindowManager().rollover(
        llm=object(),
        system_prompt=system_prompt,
        objective="Inspect the verified result",
        contract=TaskContract(requires_browser_snapshot=True),
        messages=[{"role": "user", "content": "history " * 4_000}],
        tools=[],
        evidence=evidence,
        previous_summary={
            "completed": ["old step " * 30] * 8,
            "findings": ["old finding " * 30] * 8,
            "open_questions": [],
            "next_steps": [],
        },
        evidence_since_previous=evidence,
        context_tokens=12_000,
        step=9,
        use_model_summary=False,
    )

    assert rollover.evidence[0]["sequence"] == 9
    assert "Newest verified page evidence" in rollover.evidence[0]["result_excerpt"]
    assert "=== V MEMORY CONTEXT ===" not in rollover.messages[0]["content"]
    assert "=== PREVIOUS RUNTIME CHECKPOINT ===" not in rollover.messages[0]["content"]
    assert rollover.estimated_tokens_after <= 12_000 - 768 - 512


def test_rollover_preserves_generated_source_and_error_instead_of_fixture_copy() -> None:
    source = (
        "def run(arguments):\n"
        "    values = arguments.get('values', [])\n"
        "    return {'total': sum(values)}\n"
        + "# candidate repair context\n" * 60
    )
    evidence = [
        {
            "sequence": 4,
            "tool": "learning_create_tool",
            "arguments": {
                "name": "sum_values",
                "description": "Sum supplied values.",
                "source": source,
                "test": {
                    "name": "fixture",
                    "arguments": {"values": list(range(2_000))},
                    "expected": {"total": 6},
                },
            },
            "status": "failed",
            "result_excerpt": "Tool execution failed: got 5 instead of 6",
            "error": "ArtifactValidationError: got 5 instead of 6",
        }
    ]

    bounded = ContextWindowManager._bounded_evidence(
        evidence,
        context_tokens=12_000,
        maximum_characters=4_000,
    )

    assert len(bounded) == 1
    assert bounded[0]["arguments"]["source"] == source
    assert "grounded fixture omitted" in bounded[0]["arguments"]["test"][
        "arguments"
    ]
    assert bounded[0]["arguments"]["test"]["expected"] == {"total": 6}
    assert "got 5 instead of 6" in bounded[0]["error"]

    fresh = ContextWindowManager._fresh_messages(
        system_prompt="system",
        objective="Create and run the tool",
        step=4,
        contract=TaskContract(
            requires_created_tool=True,
            requires_created_tool_execution=True,
        ),
        summary={field: [] for field in ("completed", "findings", "open_questions", "next_steps")},
        evidence=bounded,
        still_missing=["learning_create_tool", "generated_tool_execution"],
    )
    assert "Correct that source against the same contract" in fresh[1]["content"]


def test_repair_capsule_compacts_only_runtime_bound_json_values() -> None:
    objective = (
        "Stwórz narzędzie. records = "
        + json.dumps([{"url": "http://example.onion/"}] * 50)
        + '; keywords = ["escrow"]; expected = {"count":50}. Keep order.'
    )

    compacted = ContextWindowManager._compact_runtime_bound_literals(objective)

    assert compacted.startswith("Stwórz narzędzie. records = ")
    assert compacted.count("exact runtime-bound JSON omitted") == 3
    assert "http://example.onion/" not in compacted
    assert compacted.endswith(". Keep order.")


@pytest.mark.asyncio
async def test_tight_rollover_reserves_full_safe_space_for_tool_repair() -> None:
    source = (
        "def run(arguments):\n"
        "    return {'count': len(arguments.get('items', []))}\n"
        + "# keep exact candidate source\n" * 65
    )
    failure = {
        "sequence": 1,
        "tool": "learning_create_tool",
        "arguments": {
            "name": "count_items",
            "description": "Count supplied items.",
            "source": source,
            "test": {
                "name": "fixture",
                "arguments": {"items": ["x" * 500] * 20},
                "expected": {"count": 3},
            },
        },
        "status": "failed",
        "result_excerpt": "ArtifactValidationError: expected 3, got 2",
        "error": "ArtifactValidationError: expected 3, got 2",
    }
    objective = (
        "Create and execute an offline tool. items = "
        + json.dumps(["fixture " * 30] * 60)
        + '; expected = {"count":3}. '
    )
    system_prompt = "system rules " * 1_150
    tools = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_tool",
                "description": "builder " * 200,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    rollover = await ContextWindowManager().rollover(
        llm=object(),
        system_prompt=system_prompt,
        objective=objective,
        contract=TaskContract(
            requires_created_tool=True,
            requires_created_tool_execution=True,
        ),
        messages=[{"role": "user", "content": "history " * 2_000}],
        tools=tools,
        evidence=[failure],
        evidence_since_previous=[failure],
        previous_summary=None,
        context_tokens=12_000,
        step=1,
        use_model_summary=False,
    )

    assert rollover.evidence
    assert rollover.evidence[0]["arguments"]["source"] == source
    assert rollover.estimated_tokens_after <= 12_000 - 768 - 512
    assert "exact runtime-bound JSON omitted" in rollover.messages[1]["content"]


@pytest.mark.asyncio
async def test_rollover_summarizes_only_evidence_since_previous_capsule() -> None:
    calls = [
        {
            "sequence": 1,
            "tool": "browser_navigate",
            "status": "succeeded",
            "result_excerpt": "Opened source one",
        },
        {
            "sequence": 2,
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": "Observed source two",
        },
    ]
    previous = {
        "completed": ["browser_navigate: succeeded"],
        "findings": ["browser_navigate: Opened source one"],
        "open_questions": [],
        "next_steps": [],
    }

    rollover = await ContextWindowManager().rollover(
        llm=object(),
        system_prompt="system",
        objective="Research alternatives",
        contract=TaskContract(requires_browser_navigation=True),
        messages=[{"role": "user", "content": "research"}],
        tools=[],
        evidence=calls,
        previous_summary=previous,
        evidence_since_previous=[calls[1]],
        context_tokens=8_192,
        step=2,
        use_model_summary=False,
    )

    assert rollover.summary["completed"] == [
        "browser_navigate: succeeded",
        "browser_snapshot: succeeded",
    ]
    assert rollover.summary["findings"] == [
        "browser_navigate: Opened source one",
        "browser_snapshot: Observed source two",
    ]
    assert [item["sequence"] for item in rollover.evidence] == [2]


@pytest.mark.asyncio
async def test_rollover_cannot_relabel_failed_runtime_call_as_success() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": ["Created and executed count_words"],
                    "findings": ["Sandbox problem was fixed and execution succeeded"],
                    "open_questions": ["What should run next?"],
                    "next_steps": ["Use a different strategy"],
                }
            )

    evidence = [
        {
            "sequence": 1,
            "tool": "learning_create_tool",
            "arguments": {},
            "status": "failed",
            "result_excerpt": (
                "ArtifactValidationError: bwrap loopback RTM_NEWADDR denied"
            ),
            "error": "ArtifactValidationError",
        }
    ]
    rollover = await ContextWindowManager().rollover(
        llm=LLMStub(),
        system_prompt="system",
        objective="Create count_words",
        contract=TaskContract(requires_created_tool=True),
        messages=[{"role": "user", "content": "Create count_words"}],
        tools=[],
        evidence=evidence,
        previous_summary=None,
        context_tokens=8_192,
        step=1,
    )

    assert rollover.summary["completed"] == ["learning_create_tool: failed"]
    assert "RTM_NEWADDR" in rollover.summary["findings"][0]
    assert not any(
        "succeeded" in item for item in rollover.summary["findings"]
    )


@pytest.mark.asyncio
async def test_rollover_replaces_dns_domain_guessing_with_search_discovery() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": [],
                    "findings": [],
                    "open_questions": ["Is another nearby domain valid?"],
                    "next_steps": ["Try marketplace.example instead"],
                }
            )

    evidence = [
        {
            "sequence": 1,
            "tool": "browser_navigate",
            "arguments": {"url": "https://missing.invalid"},
            "status": "failed",
            "result_excerpt": "NS_ERROR_UNKNOWN_HOST",
            "error": "MCPToolExecutionError: NS_ERROR_UNKNOWN_HOST",
        }
    ]

    rollover = await ContextWindowManager().rollover(
        llm=LLMStub(),
        system_prompt="system",
        objective="Search for an alternative",
        contract=TaskContract(requires_web_discovery=True),
        messages=[{"role": "user", "content": "search"}],
        tools=[],
        evidence=evidence,
        previous_summary=None,
        context_tokens=8_192,
        step=1,
    )

    assert "DuckDuckGo search results" in rollover.summary["next_steps"][0]
    assert "do not guess" in rollover.summary["next_steps"][0]
    assert "marketplace.example" not in json.dumps(rollover.summary)


@pytest.mark.asyncio
async def test_rollover_replaces_http_429_captcha_advice_with_duckduckgo() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": [],
                    "findings": [],
                    "open_questions": ["How can I bypass the reCAPTCHA?"],
                    "next_steps": ["Try to bypass the challenge"],
                }
            )

    evidence = [
        {
            "sequence": 1,
            "tool": "browser_navigate",
            "arguments": {"url": "https://google.com/search?q=test"},
            "status": "failed",
            "result_excerpt": "HTTP status: 429; reCAPTCHA",
            "error": "BrowserHTTPError: page returned HTTP status 429",
        }
    ]

    rollover = await ContextWindowManager().rollover(
        llm=LLMStub(),
        system_prompt="system",
        objective="Search for an alternative",
        contract=TaskContract(requires_web_discovery=True),
        messages=[{"role": "user", "content": "search"}],
        tools=[],
        evidence=evidence,
        previous_summary=None,
        context_tokens=8_192,
        step=1,
    )

    encoded = json.dumps(rollover.summary)
    assert "DuckDuckGo search results" in encoded
    assert "bypass" not in encoded


def test_semantic_intent_accepts_only_closed_capability_vocabulary() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "action_requested": True,
                "continue_previous": False,
                "capabilities": ["browser", "invented_shell", "browser"],
                "requires_report": True,
                "distinct_detail_page": True,
            }
        )
    )

    assert intent is not None
    assert intent.capabilities == ("browser",)
    contract = intent.to_contract()
    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_distinct_detail_page is True
    assert contract.requires_evidence_report is True


def test_semantic_continuation_cannot_copy_runtime_review_capability() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "message_clear": True,
                "message_odd": False,
                "action_requested": True,
                "continue_previous": True,
                "capabilities": ["runtime_review"],
                "requires_report": True,
                "distinct_detail_page": False,
                "artifact_fallback": False,
                "required_public_fields": [],
                "public_subject": "",
                "web_query": "",
                "language_scope": "none",
                "response_language": "",
            }
        )
    )

    assert intent is not None
    assert intent.continue_previous is True
    assert intent.capabilities == ()
    assert intent.to_contract().requires_runtime_review is False


def test_semantic_intent_preserves_unclear_message_signal() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "message_clear": False,
                "action_requested": False,
                "continue_previous": False,
                "capabilities": [],
                "requires_report": False,
                "distinct_detail_page": False,
                "artifact_fallback": False,
                "required_public_fields": [],
                "public_subject": "",
                "web_query": "",
            }
        )
    )

    assert intent is not None
    assert intent.message_clear is False


def test_semantic_intent_preserves_odd_banter_signal() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "message_clear": True,
                "message_odd": True,
                "action_requested": False,
                "continue_previous": False,
                "capabilities": [],
                "requires_report": False,
                "distinct_detail_page": False,
                "artifact_fallback": False,
                "required_public_fields": [],
                "public_subject": "",
                "web_query": "",
            }
        )
    )

    assert intent is not None
    assert intent.message_clear is True
    assert intent.message_odd is True


def test_semantic_intent_marks_creative_chat_without_inventing_file_work() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "message_clear": True,
                "message_odd": False,
                "action_requested": False,
                "continue_previous": False,
                "references_previous": False,
                "creative_response": True,
                "capabilities": [],
                "requires_report": False,
                "distinct_detail_page": False,
                "artifact_fallback": False,
                "required_public_fields": [],
                "public_subject": "",
                "web_query": "",
                "language_scope": "none",
                "response_language": "",
            }
        )
    )

    assert intent is not None
    assert intent.creative_response is True
    assert intent.action_requested is False
    assert intent.capabilities == ()
    contract = intent.to_contract("Write a fictional scene in chat")
    assert contract.requires_file_mutation is False


def test_semantic_intent_marks_discussion_that_depends_on_prior_dialogue() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "message_clear": True,
                "message_odd": False,
                "action_requested": False,
                "continue_previous": False,
                "references_previous": True,
                "capabilities": [],
                "requires_report": False,
                "distinct_detail_page": False,
                "artifact_fallback": False,
                "required_public_fields": [],
                "public_subject": "",
                "web_query": "",
                "language_scope": "none",
                "response_language": "",
            }
        )
    )

    assert intent is not None
    assert intent.references_previous is True
    assert intent.continue_previous is False


def test_semantic_intent_preserves_explicit_turn_language_without_tool_action() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "message_clear": True,
                "message_odd": False,
                "action_requested": False,
                "continue_previous": False,
                "capabilities": [],
                "requires_report": False,
                "distinct_detail_page": False,
                "artifact_fallback": False,
                "required_public_fields": [],
                "public_subject": "",
                "web_query": "",
                "language_scope": "turn",
                "response_language": "Chinese",
            }
        )
    )

    assert intent is not None
    assert intent.action_requested is False
    assert intent.language_scope == "turn"
    assert intent.response_language == "Chinese"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "archetype_evidence", "delegation_evidence"),
    [
        (
            "Crea una herramienta para vigilar el tráfico web. "
            "Elige tú misma cualquier sitio seguro para probarla.",
            "herramienta para vigilar el tráfico web",
            "Elige tú misma cualquier sitio seguro",
        ),
        (
            "ウェブ通信を監視するツールを作成してください。"
            "安全なテストサイトは自分で選んでください。",
            "ウェブ通信を監視するツール",
            "安全なテストサイトは自分で選んでください",
        ),
    ],
)
async def test_semantic_router_grounds_test_delegation_in_any_language(
    prompt: str,
    archetype_evidence: str,
    delegation_evidence: str,
) -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "action_requested": True,
                    "capabilities": ["learning_tool"],
                    "requires_report": True,
                    "execute_created_artifact": True,
                    "generated_tool_archetype": "client_web_traffic_monitor",
                    "generated_tool_archetype_evidence": archetype_evidence,
                    "delegates_test_target": True,
                    "delegates_test_target_evidence": delegation_evidence,
                },
                ensure_ascii=False,
            )

    intent = await MultilingualIntentRouter(LLMStub()).classify(prompt)

    assert intent is not None
    assert intent.generated_tool_archetype == "client_web_traffic_monitor"
    assert intent.delegates_test_target is True


@pytest.mark.asyncio
async def test_semantic_router_rejects_ungrounded_test_delegation() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "action_requested": True,
                    "capabilities": ["learning_tool"],
                    "requires_report": True,
                    "execute_created_artifact": True,
                    "generated_tool_archetype": "client_web_traffic_monitor",
                    "generated_tool_archetype_evidence": "not in the message",
                    "delegates_test_target": True,
                    "delegates_test_target_evidence": "also absent",
                }
            )

    intent = await MultilingualIntentRouter(LLMStub()).classify(
        "Please create and test a tool."
    )

    assert intent is not None
    assert intent.generated_tool_archetype == "none"
    assert intent.delegates_test_target is False


@pytest.mark.asyncio
async def test_multilingual_intent_router_classifies_hungarian_action() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        async def ask(self, **kwargs) -> str:
            self.kwargs = kwargs
            return json.dumps(
                {
                    "action_requested": True,
                    "continue_previous": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "web_query": "internetes kutatási eredmények",
                }
            )

    llm = LLMStub()
    router = MultilingualIntentRouter(llm)

    intent = await router.classify(
        "Keress az interneten es keszits jelentest az eredmenyekrol."
    )

    assert intent is not None
    assert intent.action_requested is True
    assert intent.capabilities == ("browser",)
    assert intent.requires_report is True
    assert intent.web_query == "internetes kutatási eredmények"
    assert llm.kwargs["temperature"] == 0.0
    assert llm.kwargs["max_tokens"] == 384
    assert llm.kwargs["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_catalog_misclassification_retries_online_creation_followup() -> None:
    prompt = (
        "Słuchaj, może jednak mimo wszystko spróbuj, tylko wiesz co? "
        "Zanim zaczniesz, może poszukaj informacji na temat takich narzędzi "
        "w internecie i dopiero wtedy zacznij je tworzyć."
    )

    class LLMStub:
        calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return json.dumps({
                "action_requested": True,
                "references_previous": True,
                "capabilities": (["tool_catalog"] if self.calls == 1
                                 else ["browser", "learning_tool"]),
                "requires_report": True,
                "web_query": "" if self.calls == 1 else "informacji na temat takich narzędzi",
            })

    llm = LLMStub()
    intent = await MultilingualIntentRouter(llm).classify(prompt)
    assert llm.calls == 2
    assert intent is not None
    assert set(intent.capabilities) == {"browser", "learning_tool"}
    assert intent.references_previous


@pytest.mark.parametrize("capability", ["tool_catalog", "tool_self_test"])
def test_catalog_shortcut_cannot_satisfy_explicit_external_work(capability: str) -> None:
    intent = SemanticIntent(action_requested=True, capabilities=(capability,))
    assert not MultilingualIntentRouter._usable(
        intent, "Search online for CSV tools and then build a tool."
    )
    assert MultilingualIntentRouter._usable(
        intent, "Review available tool descriptions."
    )


@pytest.mark.asyncio
async def test_intent_router_does_not_expose_previous_task_content() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            user_message = kwargs["messages"][-1]["content"]
            assert "poisoned previous objective" not in user_message
            assert "requires_runtime_review" not in user_message
            assert '"previous_runtime_context_available": true' in user_message
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": True,
                    "capabilities": [],
                    "requires_report": False,
                    "distinct_detail_page": False,
                    "artifact_fallback": False,
                    "required_public_fields": [],
                    "public_subject": "",
                    "web_query": "",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    intent = await MultilingualIntentRouter(LLMStub()).classify(
        "Repeat that task.",
        previous_context={
            "objective": "poisoned previous objective",
            "requirements": {"requires_runtime_review": True},
        },
    )

    assert intent is not None
    assert intent.continue_previous is True
    assert intent.capabilities == ()


@pytest.mark.asyncio
async def test_router_rejects_model_invented_runtime_review_for_advice_question() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "capabilities": ["runtime_review"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "artifact_fallback": False,
                    "required_public_fields": [],
                    "public_subject": "",
                    "web_query": "",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    llm = LLMStub()
    router = MultilingualIntentRouter(llm)
    intent = await router.classify(
        "jak podejdziesz do zadania z moim kolegom któremu chce dać nauczke?"
    )

    assert intent is not None
    assert intent.action_requested is False
    assert intent.capabilities == ()
    assert intent.requires_report is False
    assert router.last_sanitization_reason == "ungrounded_runtime_review"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_router_strips_invented_file_work_from_a_requested_plan() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": True,
                    "capabilities": ["file_read", "file_write"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "artifact_fallback": False,
                    "required_public_fields": [],
                    "public_subject": "",
                    "web_query": "",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    router = MultilingualIntentRouter(LLMStub())
    intent = await router.classify(
        "V, zaplanuj jak wyjaśnić koledze problemy bezpieczeństwa Windows. "
        "Najpierw omówmy plan, potem podam ci informacje."
    )

    assert intent is not None
    assert intent.action_requested is False
    assert intent.capabilities == ()
    assert intent.requires_report is False
    assert router.last_sanitization_reason == "ungrounded_local_file_capability"


@pytest.mark.asyncio
async def test_router_keeps_multilingual_file_work_with_explicit_path() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": False,
                    "capabilities": ["file_read"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "artifact_fallback": False,
                    "required_public_fields": [],
                    "public_subject": "",
                    "web_query": "",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    router = MultilingualIntentRouter(LLMStub())
    intent = await router.classify("/tmp/riport.md fájlt olvasd el.")

    assert intent is not None
    assert intent.action_requested is True
    assert intent.capabilities == ("file_read",)
    assert intent.requires_report is True
    assert router.last_sanitization_reason == ""


@pytest.mark.asyncio
async def test_router_keeps_read_only_current_project_review_without_path() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": True,
                    "capabilities": ["file_read"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "artifact_fallback": False,
                    "required_public_fields": [],
                    "public_subject": "",
                    "web_query": "",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    router = MultilingualIntentRouter(LLMStub())
    intent = await router.classify(
        "Przejrzyj kod DARKLINGERA i przedstaw plan zmian przed edycją."
    )

    assert intent is not None
    assert intent.action_requested is True
    assert intent.capabilities == ("file_read",)
    assert intent.requires_report is True
    assert router.last_sanitization_reason == ""


def test_project_code_review_builds_deterministic_read_contract() -> None:
    contract = TaskContract.from_prompt(
        "Przejrzyj kod DARKLINGERA i przedstaw dokładnie proponowane zmiany. "
        "Nie modyfikuj plików przed zatwierdzeniem."
    )

    assert contract.requires_file_read is True
    assert contract.requires_file_mutation is False
    assert contract.requires_evidence_report is True


@pytest.mark.asyncio
async def test_router_keeps_explicit_runtime_log_review() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "capabilities": ["runtime_review"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "artifact_fallback": False,
                    "required_public_fields": [],
                    "public_subject": "",
                    "web_query": "",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    router = MultilingualIntentRouter(LLMStub())
    intent = await router.classify(
        "Przeanalizuj swoje logi z ostatniej sesji i pokaż, gdzie były błędy."
    )

    assert intent is not None
    assert intent.action_requested is True
    assert intent.capabilities == ("runtime_review",)
    assert intent.to_contract().requires_runtime_review is True
    assert router.last_sanitization_reason == ""


@pytest.mark.asyncio
async def test_router_normalizes_language_independent_action_contradiction() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return json.dumps(
                {
                    "action_requested": False,
                    "continue_previous": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "artifact_fallback": False,
                    "required_public_fields": [
                        "address",
                        "contact",
                        "opening_hours",
                    ],
                    "public_field_evidence": {
                        "address": "címét",
                        "contact": "telefonszámát",
                        "count": "",
                        "opening_hours": "nyitvatartását",
                    },
                    "public_subject": "Csodamalina",
                    "web_query": "Csodamalina cukrászda Varsó",
                }
            )

    llm = LLMStub()
    router = MultilingualIntentRouter(llm)
    intent = await router.classify(
        "Keresd meg a varsói Csodamalina cukrászdák címét, telefonszámát "
        "és nyitvatartását."
    )

    assert intent is not None
    assert intent.action_requested is True
    assert intent.required_public_fields == ("contact",)
    assert intent.public_subject == "Csodamalina"
    contract = intent.to_contract()
    assert contract.requires_browser_navigation is True
    assert contract.required_public_fields == ("contact",)
    assert contract.required_public_subject == "Csodamalina"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_router_discards_unasked_public_fields_from_product_research() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": False,
                    "creative_response": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "artifact_fallback": False,
                    "execute_created_artifact": False,
                    "recall_memory": False,
                    "memory_query": "",
                    "required_public_fields": [
                        "count",
                        "address",
                        "contact",
                        "opening_hours",
                    ],
                    "public_field_evidence": {
                        "address": "",
                        "contact": "",
                        "count": "",
                        "opening_hours": "",
                    },
                    "public_subject": "Przedwojenne",
                    "web_query": "przedwojenne nocniki austriackie francuskie",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    prompt = (
        "Znajdź raport o przedwojennych nocnikach austriackich i francuskich, "
        "ich rodzajach, cenach oraz miejscu, gdzie można je kupić."
    )
    router = MultilingualIntentRouter(LLMStub())

    intent = await router.classify(prompt)

    assert intent is not None
    assert intent.required_public_fields == ()
    assert intent.public_subject == ""
    assert "ungrounded_public_fields" in router.last_sanitization_reason
    assert intent.to_contract(prompt).required_public_fields == ()


@pytest.mark.asyncio
async def test_router_grounds_multilingual_research_facets_in_owner_text() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": False,
                    "creative_response": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "artifact_fallback": False,
                    "execute_created_artifact": False,
                    "recall_memory": False,
                    "memory_query": "",
                    "required_public_fields": [],
                    "public_field_evidence": {
                        "address": "",
                        "contact": "",
                        "count": "",
                        "opening_hours": "",
                    },
                    "public_subject": "",
                    "research_facets": ["price", "purchase_source"],
                    "research_facet_evidence": {
                        "price": "quali sono i prezzi",
                        "purchase_source": "dove posso comprarli",
                    },
                    "web_query": "vasi da notte francesi anteguerra",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    prompt = (
        "Cerca vasi da notte francesi anteguerra: quali sono i prezzi e "
        "dove posso comprarli?"
    )
    intent = await MultilingualIntentRouter(LLMStub()).classify(prompt)

    assert intent is not None
    assert intent.research_facets == ("price", "purchase_source")
    assert intent.to_contract(prompt).required_research_facets == (
        "price",
        "purchase_source",
    )


@pytest.mark.asyncio
async def test_router_keeps_research_facets_with_inflected_or_translated_evidence() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "action_requested": True,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "research_facets": ["price", "purchase_source"],
                    "research_facet_evidence": {
                        "price": "ceny",
                        "purchase_source": "direct link do każdej oferty",
                    },
                    "web_query": "kolekcjonerskie filiżanki przed 1970",
                }
            )

    prompt = (
        "Znajdź oferty kolekcjonerskich filiżanek sprzed 1970 roku. "
        "Podaj cenę i bezpośredni link do każdej oferty."
    )
    intent = await MultilingualIntentRouter(LLMStub()).classify(prompt)

    assert intent is not None
    assert intent.research_facets == ("price", "purchase_source")


@pytest.mark.asyncio
async def test_router_grounds_explicit_multilingual_result_count() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": False,
                    "creative_response": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "artifact_fallback": False,
                    "execute_created_artifact": False,
                    "recall_memory": False,
                    "memory_query": "",
                    "required_public_fields": [],
                    "public_field_evidence": {
                        "address": "",
                        "contact": "",
                        "count": "",
                        "opening_hours": "",
                    },
                    "public_subject": "",
                    "research_facets": ["price", "purchase_source"],
                    "research_facet_evidence": {
                        "price": "indica il prezzo",
                        "purchase_source": "link diretto",
                    },
                    "minimum_detail_sources": 2,
                    "minimum_detail_sources_evidence": "due offerte",
                    "web_query": "tazze vintage prima del 1970",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    prompt = (
        "Trova due offerte di tazze vintage prodotte prima del 1970, "
        "indica il prezzo e il link diretto."
    )
    intent = await MultilingualIntentRouter(LLMStub()).classify(prompt)

    assert intent is not None
    assert intent.minimum_detail_sources == 2
    assert intent.to_contract(prompt).minimum_detail_sources == 2


@pytest.mark.asyncio
async def test_router_discards_result_count_without_exact_owner_evidence() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "action_requested": True,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "minimum_detail_sources": 8,
                    "minimum_detail_sources_evidence": "eight offers",
                    "web_query": "antique cup offer",
                }
            )

    intent = await MultilingualIntentRouter(LLMStub()).classify(
        "Find an antique cup offer and report it."
    )

    assert intent is not None
    assert intent.minimum_detail_sources == 0
    assert intent.to_contract(
        "Find an antique cup offer and report it."
    ).minimum_detail_sources == 1


@pytest.mark.asyncio
async def test_router_preserves_exhaustive_list_descriptions_and_images() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": False,
                    "creative_response": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "artifact_fallback": False,
                    "execute_created_artifact": False,
                    "recall_memory": False,
                    "memory_query": "",
                    "required_public_fields": [],
                    "public_field_evidence": {
                        "address": "",
                        "contact": "",
                        "count": "",
                        "opening_hours": "",
                    },
                    "public_subject": "",
                    "research_facets": [
                        "item_list",
                        "item_descriptions",
                        "images",
                    ],
                    "research_facet_evidence": {
                        "exhaustive_coverage": "wszystkie gatunki",
                        "images": "znajdź mi również zdjęcia",
                        "item_descriptions": "opisz",
                        "item_list": "wszystkie gatunki",
                        "price": "",
                        "purchase_source": "",
                    },
                    "minimum_detail_sources": 0,
                    "minimum_detail_sources_evidence": "",
                    "web_query": "gatunki grzybów psychoaktywnych w Polsce",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    prompt = (
        "Wyszukaj i opisz wszystkie gatunki grzybów psychoaktywnych "
        "występujące w Polsce. Jeśli się da, znajdź mi również zdjęcia."
    )
    intent = await MultilingualIntentRouter(LLMStub()).classify(prompt)

    assert intent is not None
    assert intent.research_facets == (
        "item_list",
        "item_descriptions",
        "images",
        "exhaustive_coverage",
    )
    contract = intent.to_contract(prompt)
    assert contract.required_research_facets == intent.research_facets
    assert contract.minimum_detail_sources == 2


def test_exhaustive_visual_research_requires_two_observed_sources() -> None:
    first = "https://source.test/species-one"
    second = "https://source.test/species-two"
    search = {
        "tool": "web_search",
        "status": "succeeded",
        "result_excerpt": json.dumps(
            {"results": [{"url": first}, {"url": second}]}
        ),
    }

    def observed(url: str, suffix: str) -> list[dict]:
        return [
            {
                "tool": "browser_navigate",
                "status": "succeeded",
                "arguments": {"url": url},
                "result_excerpt": f"- Page URL: {url}",
            },
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": (
                    f"- Page URL: {url}\n"
                    f'- img "Species {suffix} alpha"\n'
                    f'- img "Species {suffix} beta"\n'
                    f'- heading "Species {suffix} alpha" [level=2]\n'
                    "- paragraph: A concrete observed description with several words.\n"
                    f'- heading "Species {suffix} beta" [level=2]\n'
                    "- paragraph: Another concrete observed description with several words."
                ),
            },
        ]

    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_web_discovery=True,
        requires_distinct_detail_page=True,
        requires_evidence_report=True,
        required_research_facets=(
            "item_list",
            "item_descriptions",
            "images",
            "exhaustive_coverage",
        ),
    )

    assert contract.unmet([search, *observed(first, "one")]) == [
        "browser_evidence:detail_sources=1/2"
    ]
    assert contract.unmet(
        [search, *observed(first, "one"), *observed(second, "two")]
    ) == []


def test_research_report_must_list_and_describe_requested_items() -> None:
    contract = TaskContract(
        requires_evidence_report=True,
        required_research_facets=("item_list", "item_descriptions"),
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                '- heading "Species alpha" [level=2]\n'
                "- paragraph: Alpha has a detailed observed description.\n"
                '- heading "Species beta" [level=2]\n'
                "- paragraph: Beta has another detailed observed description."
            ),
        }
    ]

    assert contract.answer_issues("The source discusses two species.", calls) == [
        "answer:research_item_list_missing"
    ]
    assert contract.answer_issues(
        "- Species alpha — Alpha has a detailed observed description.\n"
        "- Species beta — Beta has another detailed observed description.",
        calls,
    ) == []


def test_multiple_test_paths_do_not_request_command_execution() -> None:
    contract = TaskContract.from_prompt(
        "Read test/fixtures/one.py and test/fixtures/two.py. Read-only."
    )
    assert contract.requires_file_read
    assert not contract.requires_command_execution
    assert TaskContract.from_prompt(
        "Run the tests in test/fixtures/check.py without network."
    ).requires_command_execution


def test_passive_smoke_test_text_does_not_require_command_execution() -> None:
    contract = TaskContract.from_prompt(
        "Utwórz plik smoke-report.md z nagłówkiem '# DARKLINGER smoke test', "
        "a potem go odczytaj."
    )

    assert contract.requires_file_mutation is True
    assert contract.requires_file_read is True
    assert contract.requires_command_execution is False


def test_explicit_test_execution_still_requires_command_execution() -> None:
    contract = TaskContract.from_prompt("Run the smoke test with pytest.")

    assert contract.requires_command_execution is True


def test_created_tool_execution_satisfies_its_execution_requirement() -> None:
    contract = TaskContract(
        requires_command_execution=True,
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": '{"name":"double_value","status":"active"}',
        },
        {
            "tool": "double_value",
            "status": "succeeded",
            "result_excerpt": '{"result":18}',
        },
    ]

    assert contract.unmet(calls) == []


def test_unrelated_command_still_requires_command_runner() -> None:
    contract = TaskContract(requires_command_execution=True)

    assert contract.unmet([
        {"tool": "some_generated_name", "status": "succeeded"}
    ]) == ["command_execution"]


def test_research_facets_require_price_and_commerce_evidence() -> None:
    contract = TaskContract(
        required_research_facets=("price", "purchase_source"),
    )
    historical = [
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://museum.test/article/chamber-pots\n"
                "- Page Title: Chamber-pot history\n"
                "A historical overview with no current offer."
            ),
        }
    ]

    assert contract.unmet(historical) == [
        "browser_evidence:research_price",
        "browser_evidence:research_purchase_source",
    ]

    market = historical + [
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://market.test/listing/chamber-pots\n"
                "- Page Title: Antique chamber pots\n"
                "French porcelain chamber pot — 125 EUR"
            ),
        }
    ]

    assert contract.unmet(market) == []


@pytest.mark.asyncio
async def test_router_cannot_authorize_wrong_fields_with_real_prompt_quotes() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "message_clear": True,
                    "message_odd": False,
                    "action_requested": True,
                    "continue_previous": False,
                    "references_previous": False,
                    "creative_response": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "artifact_fallback": False,
                    "execute_created_artifact": False,
                    "recall_memory": False,
                    "memory_query": "",
                    "required_public_fields": [
                        "count",
                        "address",
                        "contact",
                        "opening_hours",
                    ],
                    "public_field_evidence": {
                        "address": "gdzie można je kupić",
                        "contact": "gdzie można je kupić",
                        "count": "ile kosztują",
                        "opening_hours": "gdzie można je kupić",
                    },
                    "public_subject": "nocników",
                    "web_query": "przedwojenne nocniki austriackie francuskie",
                    "language_scope": "none",
                    "response_language": "",
                }
            )

    prompt = (
        "Wyszukaj nocniki produkowane przed II wojną światową. Chcę się "
        "dowiedzieć, ile kosztują, gdzie można je kupić i jakie są zasady "
        "ich kolekcjonowania."
    )
    router = MultilingualIntentRouter(LLMStub())

    intent = await router.classify(prompt)

    assert intent is not None
    assert intent.required_public_fields == ()
    assert intent.public_subject == ""
    assert intent.to_contract(prompt).required_public_fields == ()


@pytest.mark.asyncio
async def test_router_rejects_prior_subject_leaking_into_unrelated_message() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return json.dumps(
                {
                    "action_requested": True,
                    "continue_previous": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": True,
                    "artifact_fallback": False,
                    "required_public_fields": ["address", "opening_hours"],
                    "public_subject": "Cukiernia Cud Malina",
                    "web_query": "cukiernia cud malina warszawa",
                }
            )

    llm = LLMStub()
    router = MultilingualIntentRouter(llm)
    intent = await router.classify(
        "pirat",
        previous_context={
            "objective": "Znajdź cukiernie Cud Malina.",
            "status": "completed",
        },
    )

    assert intent is None
    assert llm.calls == 1
    assert router.last_failure_reason == "current_message_grounding"


def test_context_rollover_requires_material_token_saving() -> None:
    assert ContextWindowManager.materially_reduces(8_784, 8_778) is False
    assert ContextWindowManager.materially_reduces(8_784, 8_600) is True


@pytest.mark.asyncio
async def test_multilingual_intent_router_fails_closed_on_invalid_output() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "I think this probably needs a browser."

    intent = await MultilingualIntentRouter(LLMStub()).classify(
        "Keress az interneten."
    )

    assert intent is None


@pytest.mark.asyncio
async def test_multilingual_intent_router_retries_malformed_json_once() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                return '{"action_requested": true'
            return json.dumps(
                {
                    "action_requested": True,
                    "continue_previous": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                    "artifact_fallback": True,
                    "web_query": "alternative to Firecrawler",
                }
            )

    llm = LLMStub()
    intent = await MultilingualIntentRouter(llm).classify(
        "Hello V, find an alternative to Firecrawler."
    )

    assert intent is not None
    assert intent.web_query == "alternative to Firecrawler"
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_multilingual_intent_router_rejects_empty_artifact_fallback() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return json.dumps(
                {
                    "action_requested": False,
                    "continue_previous": False,
                    "capabilities": [],
                    "requires_report": False,
                    "distinct_detail_page": False,
                    "artifact_fallback": True,
                    "web_query": "",
                }
            )

    llm = LLMStub()
    intent = await MultilingualIntentRouter(llm).classify(
        "Znajdź alternatywę dla Firecrawlera, a jeśli jej nie ma, stwórz tool."
    )

    assert intent is None
    assert llm.calls == 2


def test_conditional_artifact_search_has_runtime_discovery_fallback() -> None:
    prompt = (
        "Cześć V, znajdź darmową alternatywę dla Firecrawlera. Jeżeli jej "
        "nie znajdziesz, zbierz informacje potrzebne do stworzenia toola."
    )
    contract = TaskContract.from_prompt(prompt)

    assert contract.allows_artifact_fallback is True
    assert TaskContract.implies_artifact_discovery(prompt) is True


def test_conditional_artifact_recognizes_pronoun_phrase_before_tool() -> None:
    contract = TaskContract.from_prompt(
        "Znajdź w internecie alternatywę dla Firecrawlera. Ewentualnie, "
        "jeżeli nic nie znajdziesz, stwórz to po swojemu, jako tool albo skill."
    )

    assert contract.allows_artifact_fallback is True
    assert contract.requires_created_tool is False


def test_task_contract_requires_real_detail_page_after_search_listing() -> None:
    contract = TaskContract.from_prompt(
        "Open https://github.com/search?q=i2p and inspect the first result."
    )
    listing_only = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/search?q=i2p"},
            "status": "succeeded",
        },
        {"tool": "browser_snapshot", "arguments": {}, "status": "succeeded"},
    ]

    assert contract.requires_distinct_detail_page is True
    assert contract.unmet(listing_only) == ["browser_navigate:distinct_detail_page"]


def test_first_product_listing_page_is_not_misread_as_search_result() -> None:
    contract = TaskContract.from_prompt(
        "Visit https://books.toscrape.com/ and inspect the first product "
        "listing page."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_distinct_detail_page is False


def test_task_contract_rejects_browser_scaffolding_as_report_findings() -> None:
    contract = TaskContract.from_prompt(
        "Inspect https://marketplace.example and list useful tools."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "Tool E2B by Example · Tool for integration.\n"
                "Tool PaddleOCR by Example · Document parsing."
            ),
        }
    ]

    issues = contract.answer_issues(
        "Potential tool: generic [ref=f1e118] [cursor=pointer].",
        calls,
    )

    assert issues == ["answer:browser_scaffolding_is_not_a_finding"]


def test_task_contract_rejects_online_recommendations_absent_from_sources() -> None:
    contract = TaskContract.from_prompt(
        "Find an alternative to FireCrawler online and report the result."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "Page URL: https://github.com/\n"
                "Page Title: GitHub · Change is constant."
            ),
        }
    ]

    issues = contract.answer_issues(
        "Try **Scrapy**, **BeautifulSoup**, or **Puppeteer**, Boss.",
        calls,
        request="Find an alternative to FireCrawler online.",
    )

    assert issues == [
        "answer:ungrounded_online_claims=BeautifulSoup|Puppeteer|Scrapy"
    ]


def test_task_contract_accepts_named_online_result_present_in_source() -> None:
    contract = TaskContract.from_prompt(
        "Find an alternative to FireCrawler online and report the result."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "Page URL: https://github.com/scrapy/scrapy\n"
                "Scrapy is an open source web crawling framework."
            ),
        }
    ]

    assert contract.answer_issues(
        "**Scrapy** is the source-backed candidate, Boss.",
        calls,
        request="Find an alternative to FireCrawler online.",
    ) == []


def test_task_contract_rejects_mangled_github_repository_identifiers() -> None:
    request = (
        "Find useful GitHub tools for darknet OSINT and cybersecurity, then "
        "present the candidates for my approval."
    )
    contract = TaskContract.from_prompt(request)
    calls = [
        {
            "tool": "web_search",
            "arguments": {"query": "darknet OSINT cybersecurity GitHub"},
            "status": "succeeded",
            "result_excerpt": (
                "https://github.com/thamore/master-osintv4-\n"
                "https://github.com/apurvsinghgautam/robin"
            ),
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/thamore/master-osintv4-"},
            "status": "succeeded",
            "result_excerpt": "- Page URL: https://github.com/thamore/master-osintv4-",
        },
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/apurvsinghgautam/robin"},
            "status": "succeeded",
            "result_excerpt": "- Page URL: https://github.com/apurvsinghgautam/robin",
        },
    ]
    answer = (
        "Here are the candidates:\n\n"
        "- **GitHub - thamore/master-osint**: OSINT collection.\n"
        "- **GitHub - apurvsinghgautum/robin**: Dark-web OSINT tool."
    )

    assert contract.answer_issues(answer, calls, request=request) == [
        "answer:ungrounded_repository_identifiers="
        "apurvsinghgautum/robin|thamore/master-osint"
    ]


def test_task_contract_accepts_exact_observed_github_repository_identifiers() -> None:
    request = (
        "Find useful GitHub tools for darknet OSINT and cybersecurity, then "
        "present the candidates for my approval."
    )
    contract = TaskContract.from_prompt(request)
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://github.com/thamore/master-osintv4-\n"
                "master-osintv4 contains darknet OSINT modules and security links.\n"
                "- Page URL: https://github.com/apurvsinghgautam/robin\n"
                "Robin is a dark-web OSINT research tool."
            ),
        }
    ]
    answer = (
        "Two source-backed candidates:\n\n"
        "- **GitHub - thamore/master-osintv4-**: darknet OSINT modules.\n"
        "- **GitHub - apurvsinghgautam/robin**: dark-web OSINT research tool."
    )

    assert contract.answer_issues(answer, calls, request=request) == []


def test_task_contract_accepts_source_backed_based_adjective() -> None:
    contract = TaskContract.from_prompt(
        "Find an alternative to FireCrawler online and report the result."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "Page URL: https://scrapy.org/\n"
                "Scrapy is built in Python for large crawling projects."
            ),
        }
    ]

    assert contract.answer_issues(
        "**Scrapy** is a source-backed Python-based crawling framework.",
        calls,
        request="Find an alternative to FireCrawler online.",
    ) == []


def test_task_contract_accepts_grounded_feature_labels_and_plural_protocol() -> None:
    contract = TaskContract.from_prompt(
        "Find a tool online that can solve CAPTCHA automatically and report it."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://www.capsolver.com/\n"
                "Capsolver CAPTCHA Solver supports API integrations, Chrome, "
                "Selenium, Playwright, Puppeteer, pricing and use cases."
            ),
        }
    ]
    answer = (
        "Capsolver can solve CAPTCHAs through its **API** and "
        "**Chrome Extension version**. Its **Pricing description** includes "
        "example use cases for Selenium, Playwright, and Puppeteer."
    )

    assert contract.answer_issues(
        answer,
        calls,
        request="Find a tool online that can solve CAPTCHA automatically.",
    ) == []


def test_task_contract_ignores_markdown_step_headings_as_entity_claims() -> None:
    contract = TaskContract.from_prompt(
        "Inspect https://example.test and report the result."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "- Page URL: https://example.test\n"
                "The API documentation is available."
            ),
        }
    ]

    issues = contract.answer_issues(
        "### Step 1: Capture the API section\nThe API documentation is available.",
        calls,
        request="Inspect https://example.test and report the result.",
    )

    assert not any("Step" in issue or "Capture" in issue for issue in issues)


def test_task_contract_rejects_mistyped_online_source_url() -> None:
    contract = TaskContract.from_prompt(
        "Find an alternative to FireCrawler online and report the result."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": (
                "Page URL: "
                "https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives\n"
                "Scrapy is built in Python."
            ),
        }
    ]

    issues = contract.answer_issues(
        "The source is "
        "https://thunderbit.com/pl/blog/open-source-firercrawl-alternatives. "
        "It recommends **Scrapy**.",
        calls,
        request="Find an alternative to FireCrawler online.",
    )

    assert issues == [
        "answer:ungrounded_online_urls="
        "https://thunderbit.com/pl/blog/open-source-firercrawl-alternatives"
    ]


def test_task_contract_routes_public_social_profile_lookup_to_web() -> None:
    contract = TaskContract.from_prompt(
        "Znajdź publiczny profil tej osoby na Facebooku i podaj źródło."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_web_discovery is True


def test_task_contract_routes_person_lookup_with_inflected_facebook_to_web() -> None:
    contract = TaskContract.from_prompt(
        "Znajdźmy pewną osobę. Chcę wiedzieć, gdzie jest. Na pewno ma Facebooka."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_web_discovery is True
    assert contract.requires_evidence_report is True


def test_task_contract_does_not_treat_language_name_as_online_product() -> None:
    contract = TaskContract.from_prompt(
        "Find an alternative to FireCrawler online and report the result."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "Crawl4AI is a free open-source alternative.",
        }
    ]

    issues = contract.answer_issues(
        "The English response identifies Crawl4AI as the verified alternative.",
        calls,
        request="Find an alternative to FireCrawler online.",
    )

    assert not any("ungrounded_online_claims=English" in issue for issue in issues)


def test_task_contract_does_not_treat_owner_vocative_as_online_product() -> None:
    contract = TaskContract.from_prompt(
        "Find an alternative to FireCrawler online and report the result."
    )
    calls = [
        {
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": "TinyFish published a Firecrawl alternatives review.",
        }
    ]

    issues = contract.answer_issues(
        "Okay, Brzeszczot. TinyFish published the verified review.",
        calls,
        request="Find an alternative to FireCrawler online.",
    )

    assert not any("Brzeszczot" in issue for issue in issues)


def test_task_contract_requires_real_file_and_command_actions() -> None:
    read_contract = TaskContract.from_prompt(
        "Read README.md and report only its first heading."
    )
    write_contract = TaskContract.from_prompt("Write the report to result.md")
    command_contract = TaskContract.from_prompt("Run the pytest tests and report results")

    assert read_contract.requires_file_read is True
    assert read_contract.requires_first_heading is True
    assert write_contract.unmet([]) == ["filesystem_mutation"]
    assert command_contract.unmet([]) == ["command_execution"]


def test_authorization_guard_prevents_workspace_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "tasks" / "one"
    envelope = AuthorizationEnvelope(workspace=str(workspace))
    guard = AuthorizationGuard(tmp_path, envelope)

    assert guard.resolve_task_path("report.md", write=True) == (
        workspace / "report.md"
    ).resolve()

    with pytest.raises(AuthorizationDenied):
        guard.resolve_task_path("../../outside.txt", write=True)


def test_read_only_mode_rejects_writes(tmp_path: Path) -> None:
    envelope = AuthorizationEnvelope(
        mode=ExecutionMode.READ_ONLY,
        workspace=str(tmp_path / "workspace"),
    )
    guard = AuthorizationGuard(tmp_path, envelope)

    with pytest.raises(AuthorizationDenied):
        guard.resolve_task_path("result.txt", write=True)


def test_autonomy_root_can_be_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DARKLINGER_AUTONOMY_ROOT", "state/tasks")
    monkeypatch.setenv("DARKLINGER_MODEL_RUNTIME_ROOT", "state/models")
    monkeypatch.setenv("DARKLINGER_VOICE_ROOT", "state/voice")
    monkeypatch.setenv("DARKLINGER_MODEL_LOADER", "required")
    monkeypatch.setenv("V_CORE_MCP_FILESYSTEM", "workspace")

    config = load_config()

    assert config.autonomy_root == (tmp_path / "state/tasks").resolve()
    assert config.model_runtime_root == (tmp_path / "state/models").resolve()
    assert config.voice_root == (tmp_path / "state/voice").resolve()
    assert config.model_loader_mode == "required"
    assert config.workspace == (tmp_path / "workspace").resolve()


@pytest.mark.asyncio
async def test_runner_completes_multi_step_task(tmp_path: Path) -> None:
    runner = AutonomousRunner(tmp_path / "autonomy", poll_interval=0.01)
    task = AutonomousTask(objective="Three steps", task_id="multi-step")
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path / "workspace"),
        budget=TaskBudget(max_actions=5),
    )

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        if task.action_count < 3:
            return StepResult(
                StepOutcome.CONTINUE,
                summary=f"step {task.action_count}",
            )
        return StepResult(StepOutcome.COMPLETE, result="done")

    result = await runner.run(task, envelope, driver)

    assert result.status == TaskStatus.COMPLETED
    assert result.action_count == 3
    assert result.result == "done"
    assert runner.checkpoints.load("multi-step") is not None


@pytest.mark.asyncio
async def test_runner_stops_at_action_budget(tmp_path: Path) -> None:
    runner = AutonomousRunner(tmp_path / "autonomy", poll_interval=0.01)
    task = AutonomousTask(objective="Never ends", task_id="budgeted")
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path / "workspace"),
        budget=TaskBudget(max_actions=2),
    )

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        return StepResult(StepOutcome.CONTINUE)

    result = await runner.run(task, envelope, driver)

    assert result.status == TaskStatus.BLOCKED
    assert result.action_count == 2
    assert result.blocked_reason == "maximum action count reached"


@pytest.mark.asyncio
async def test_stop_cancels_active_step(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    task = AutonomousTask(objective="Long operation", task_id="stop-active")
    envelope = AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return StepResult(StepOutcome.COMPLETE)

    running = asyncio.create_task(runner.run(task, envelope, driver))
    await asyncio.wait_for(started.wait(), timeout=1)
    ControlChannel(root / "control", task.task_id).request(ControlSignal.STOP)
    result = await asyncio.wait_for(running, timeout=1)

    assert result.status == TaskStatus.STOPPED
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_panic_cancels_active_step(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    task = AutonomousTask(objective="Long operation", task_id="panic-active")
    envelope = AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))
    started = asyncio.Event()

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        started.set()
        await asyncio.sleep(30)
        return StepResult(StepOutcome.COMPLETE)

    running = asyncio.create_task(runner.run(task, envelope, driver))
    await asyncio.wait_for(started.wait(), timeout=1)
    ControlChannel(root / "control", task.task_id).request(ControlSignal.PANIC)
    result = await asyncio.wait_for(running, timeout=1)

    assert result.status == TaskStatus.PANICKED
    assert "panic requested" in result.blocked_reason


@pytest.mark.asyncio
async def test_pause_and_resume_before_step(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    task = AutonomousTask(objective="Pause first", task_id="pause-first")
    envelope = AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))
    channel = ControlChannel(root / "control", task.task_id, poll_interval=0.01)
    channel.request(ControlSignal.PAUSE)

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        return StepResult(StepOutcome.COMPLETE, result="resumed")

    running = asyncio.create_task(runner.run(task, envelope, driver))
    for _ in range(50):
        await asyncio.sleep(0.01)
        loaded = runner.checkpoints.load(task.task_id)
        if loaded and loaded[0].status == TaskStatus.PAUSED:
            break
    else:
        pytest.fail("task did not enter paused state")

    channel.request(ControlSignal.RESUME)
    result = await asyncio.wait_for(running, timeout=1)

    assert result.status == TaskStatus.COMPLETED
    assert result.result == "resumed"


def test_emergency_chord_requires_simultaneous_key_state() -> None:
    detector = ChordDetector(parse_chord("Q+P+0"))

    assert not detector.feed("Q", True)
    assert not detector.feed("P", True)
    assert detector.feed("0", True)
    assert not detector.feed("0", True)
    assert not detector.feed("P", False)
    assert detector.feed("P", True)


def test_runtime_registry_uses_pid_and_process_start_identity(tmp_path: Path) -> None:
    registry = RuntimeRegistry(tmp_path / "runtime")
    path = registry.register("test-runtime")

    assert path.exists()
    assert registry.active()[0]["name"] == "test-runtime"
    assert registry.terminate_all() == []  # never terminate the watcher itself

    registry.unregister()
    assert registry.active() == []


@pytest.mark.asyncio
async def test_runtime_registry_terminates_registered_process(tmp_path: Path) -> None:
    process = await asyncio.create_subprocess_exec("/usr/bin/sleep", "30")
    registry = RuntimeRegistry(tmp_path / "runtime")
    registry.register("disposable-test-process", pid=process.pid)

    terminated = registry.terminate_all()
    await asyncio.wait_for(process.wait(), timeout=1)

    assert terminated == [process.pid]
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_global_panic_cancels_every_active_runner(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    tasks = [
        AutonomousTask(objective="long", task_id="global-one"),
        AutonomousTask(objective="long", task_id="global-two"),
    ]
    started = [asyncio.Event(), asyncio.Event()]

    def driver(index: int):
        async def run(
            task: AutonomousTask,
            envelope: AuthorizationEnvelope,
        ) -> StepResult:
            started[index].set()
            await asyncio.sleep(30)
            return StepResult(StepOutcome.COMPLETE)

        return run

    running = [
        asyncio.create_task(
            runner.run(
                task,
                AuthorizationEnvelope(workspace=str(tmp_path / task.task_id)),
                driver(index),
            )
        )
        for index, task in enumerate(tasks)
    ]
    await asyncio.gather(*(event.wait() for event in started))
    GlobalControlChannel(root / "control").request_panic()
    results = await asyncio.wait_for(asyncio.gather(*running), timeout=1)

    assert {result.status for result in results} == {TaskStatus.PANICKED}
