"""Deterministic public-web OSINT fixtures; no DNS or network access."""
from __future__ import annotations

import json
from typing import Any


def _fixture(tool_type: type) -> Any:
    """Use MCPTools' real web methods with a fixed accessibility adapter."""

    class BrowserFixture(tool_type):
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.url = ""
            self._web_discovered_urls: dict[str, str] = {}
            self._web_search_performed = False
            self._observed_browser_snapshot = ""

        async def browser_call(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            if name == "browser_navigate":
                self.url = str(arguments["url"])
                return "navigation finished"
            if "duckduckgo.com" in self.url:
                return (
                    '- Page URL: https://duckduckgo.com/?q=fixture\n'
                    '- Page Title: fixture search\n'
                    '- link "IANA Example Domains" [ref=e1]:\n'
                    '  - /url: https://www.iana.org/help/example-domains'
                )
            return (
                f"- Page URL: {self.url}\n"
                "- Page Title: IANA Example Domains\n"
                "- paragraph: Example domains are reserved for documentation."
            )

    return BrowserFixture()


def _row(name: str, case: str, passed: bool, reason: str) -> dict[str, Any]:
    return {
        "tool": name,
        "case": case,
        "status": "passed" if passed else "failed",
        "reason": reason,
        "evidence_level": "deterministic_offline_fixture",
        "live_network": False,
    }


async def self_test_osint_tool(name: str, tool_type: type) -> dict[str, Any] | None:
    """Exercise search/read evidence logic without presenting it as live OSINT."""

    if name == "web_search":
        tools = _fixture(tool_type)
        result = json.loads(await tools.web_search("site:iana.org example domains", 6))
        rows = result.get("results", [])
        passed = (
            result.get("engine") == "duckduckgo"
            and result.get("result_count") == 1
            and len(rows) == 1
            and rows[0].get("title") == "IANA Example Domains"
            and rows[0].get("url") == "https://www.iana.org/help/example-domains"
            and list(tools._web_discovered_urls.values())
            == ["https://www.iana.org/help/example-domains"]
        )
        return _row(
            name,
            "search_snapshot_to_grounded_result",
            passed,
            "Fixture search produced one grounded URL and retained it for follow-up."
            if passed else "Search fixture did not produce the expected grounded result.",
        )
    if name == "web_read":
        tools = _fixture(tool_type)
        await tools.web_search("site:iana.org example domains", 6)
        rejected = json.loads(await tools.web_read("https://unseen.example/"))
        observed = json.loads(
            await tools.web_read("https://www.iana.org/help/example-domains")
        )
        passed = (
            "absent from web_search evidence" in str(rejected.get("error", ""))
            and rejected.get("allowed_urls")
            == ["https://www.iana.org/help/example-domains"]
            and observed.get("url") == "https://www.iana.org/help/example-domains"
            and observed.get("title") == "IANA Example Domains"
            and "reserved for documentation" in str(observed.get("content", ""))
        )
        return _row(
            name,
            "grounded_result_read_and_unseen_url_rejection",
            passed,
            "Fixture read opened the grounded URL and rejected an unseen substitute."
            if passed else "Read fixture did not preserve the search-evidence boundary.",
        )
    return None
