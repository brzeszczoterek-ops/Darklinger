from __future__ import annotations

import re
from collections.abc import Iterable


COMMUNICATION = "external_communication"
REMOTE_ACCESS = "remote_system_access"
FILESYSTEM_MUTATION = "filesystem_mutation"
FILESYSTEM_READ = "filesystem_read"
COMMAND_EXECUTION = "command_execution"
BROWSER_ACTION = "browser_action"


_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        COMMUNICATION,
        (
            r"\b(?:i|we)\s+(?:have\s+)?(?:called|phoned|emailed|messaged|contacted)\b",
            r"\b(?:i|we)\s+(?:have\s+)?(?:rang|reached)\s+(?:him|her|them)\b",
            r"\b(?:i|we)\s+(?:have\s+)?(?:sent|told)\s+(?:him|her|them)\b",
            r"\b(?:i|we)\s+(?:spoke|talked)\s+(?:to|with)\b",
            r"\b(?:he|she|they)\s+(?:has\s+)?picked\s+up\b",
            r"\b(?:i'm|i am|we're|we are)\s+on\s+the\s+(?:phone|line|call)\b",
            r"\b(?:zadzwonił[ae]m|zadzwoniliśmy|skontaktował[ae]m\s+się)\b",
            r"\b(?:napisał[ae]m|wysłał[ae]m|powiedział[ae]m)\s+(?:mu|jej|im)\b",
        ),
    ),
    (
        REMOTE_ACCESS,
        (
            r"\b(?:i|we)\s+(?:have\s+)?(?:hacked|breached|compromised)\s+"
            r"(?:into\s+)?(?:his|her|their|the|a|an)?\s*"
            r"(?:computer|machine|system|server|account|network|router|phone|device|host)\b",
            r"\b(?:and|then)\s+(?:hacked|breached|compromised)\s+"
            r"(?:into\s+)?(?:his|her|their|the|a|an)?\s*"
            r"(?:computer|machine|system|server|account|network|router|phone|device|host)\b",
            r"\b(?:i|we)\s+(?:have\s+)?(?:accessed|entered|logged\s+into|connected\s+to)\s+"
            r"(?:his|her|their|the|a|an)?\s*"
            r"(?:computer|machine|system|server|account|network|router|phone|device|host)\b",
            r"\b(?:i|we)\s+(?:have\s+)?used\s+(?:a|an|the)?\s*"
            r"(?:remote\s+desktop|rce|zero-day|exploit|payload)\b",
            r"\b(?:i'm|i am|we're|we are)\s+(?:already\s+)?(?:inside|in)\s+"
            r"(?:his|her|their|the)\s+"
            r"(?:computer|machine|system|server|account|network|router|phone|device|host)\b",
            r"\b(?:i|we)\s+(?:got|gained|have)\s+(?:full\s+)?"
            r"(?:access|control)\s+(?:to|of|over)\s+"
            r"(?:his|her|their|the)\s+"
            r"(?:computer|machine|system|server|account|network|router|phone|device|host)\b",
            r"\b(?:włamał[ae]m\s+się|włamaliśmy\s+się|przejął[ęe]m|połączył[ae]m\s+się)\s+"
            r"(?:z|do)\s+(?:jego|jej|ich|tego)?\s*"
            r"(?:komputera|systemu|serwera|konta|sieci|routera|telefonu)\b",
        ),
    ),
    (
        FILESYSTEM_MUTATION,
        (
            r"\b(?:i|we)\s+(?:have\s+)?(?:created|wrote|edited|modified|deleted|removed|"
            r"renamed|moved|saved)\s+(?:the|a|an|your)?\s*"
            r"(?:file|folder|directory|document|script|config(?:uration)?)\b",
            r"\b(?:file|folder|directory|document|script|config(?:uration)?)\s+"
            r"(?:has\s+been|was|is)\s+(?:created|written|edited|modified|deleted|removed|"
            r"renamed|moved|saved)\b",
        ),
    ),
    (
        FILESYSTEM_READ,
        (
            r"\b(?:i|we)\s+(?:have\s+)?(?:read|listed|inspected|searched)\s+"
            r"(?:the|a|an|your)?\s*(?:file|folder|directory|document|source\s+tree)\b",
        ),
    ),
    (
        COMMAND_EXECUTION,
        (
            r"\b(?:i|we)\s+(?:have\s+)?(?:ran|executed|started|stopped|installed)\s+"
            r"(?:the|a|an|your)?\s*(?:command|test|tests|script|program|package|server|process|binary)\b",
            r"\b(?:the|your)\s+(?:command|test|tests|script|program|package|server|process|binary)\s+"
            r"(?:ran|executed|started|stopped|installed|passed|completed)\b",
        ),
    ),
    (
        BROWSER_ACTION,
        (
            r"\b(?:i|we)\s+(?:have\s+)?(?:opened|visited|navigated\s+to|searched|scraped|"
            r"downloaded|uploaded|clicked\s+on)\s+(?:the|a|an|your)?\s*"
            r"(?:website|webpage|page|site|url|link|browser)\b",
        ),
    ),
)


_FILESYSTEM_MUTATION_TOOLS = {
    "create_directory",
    "edit_file",
    "move_file",
    "write_file",
}
_FILESYSTEM_READ_TOOLS = {
    "directory_tree",
    "get_file_info",
    "list_directory",
    "read_file",
    "search_files",
}
_COMMAND_TOOLS = {
    "evm_foundry_test_offline",
    "sandbox_execute_offline",
}


def detect_execution_claims(text: str) -> tuple[str, ...]:
    """Return real-world action claims made as completed facts.

    The detector is deliberately conservative: it targets claims whose truth is
    knowable from DARKLINGER's runtime evidence. It does not try to fact-check normal
    reasoning or every sentence produced by a language model.
    """

    normalized = " ".join(str(text).casefold().replace("’", "'").split())
    detected: list[str] = []
    for category, patterns in _RULES:
        if any(re.search(pattern, normalized) for pattern in patterns):
            detected.append(category)
    return tuple(detected)


def tool_supports_claim(category: str, tool: str) -> bool:
    name = str(tool).strip().casefold()
    if category in {COMMUNICATION, REMOTE_ACCESS}:
        # DARKLINGER currently exposes no telephony, messaging, remote-desktop,
        # network-exploitation, or remote-shell capability. Browser clicks do
        # not prove a call or a system compromise.
        return False
    if category == FILESYSTEM_MUTATION:
        return name in _FILESYSTEM_MUTATION_TOOLS
    if category == FILESYSTEM_READ:
        return name in _FILESYSTEM_READ_TOOLS
    if category == COMMAND_EXECUTION:
        return name in _COMMAND_TOOLS
    if category == BROWSER_ACTION:
        return name.startswith("browser_") or name in {"web_search", "web_read"}
    return False


def claim_has_runtime_capability(category: str) -> bool:
    """Return whether any current DARKLINGER tool could prove this action.

    This differs from ``tool_supports_claim``: it describes the installed
    runtime's capability in principle, not whether a matching tool succeeded
    in the current interaction.
    """

    return category not in {COMMUNICATION, REMOTE_ACCESS}


def unsupported_execution_claims(
    text: str,
    successful_tools: Iterable[str],
) -> tuple[str, ...]:
    tools = tuple(str(tool) for tool in successful_tools)
    return tuple(
        category
        for category in detect_execution_claims(text)
        if not any(tool_supports_claim(category, tool) for tool in tools)
    )
