from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Awaitable, Callable, Iterable
from uuid import uuid4


_CAPABILITY = re.compile(r"^[a-z][a-z0-9_.-]{2,95}$")
_SENSITIVE_FIELD = re.compile(
    r"(?:auth|cookie|credential|key|pass|secret|session|token)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


BUILTIN_TOOL_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "cat": ("filesystem.read",),
    "read_file": ("filesystem.read",),
    "ls": ("filesystem.list",),
    "list_directory": ("filesystem.list",),
    "tree": ("filesystem.tree",),
    "directory_tree": ("filesystem.tree",),
    "search_files": ("filesystem.search",),
    "get_file_info": ("filesystem.inspect",),
    "write_file": ("filesystem.write",),
    "edit_file": ("filesystem.edit",),
    "move_file": ("filesystem.move",),
    "create_directory": ("filesystem.create_directory",),
    "web_search": ("web.search",),
    "web_read": ("web.read",),
    "browser_navigate": ("browser.navigate",),
    "browser_snapshot": ("browser.snapshot",),
    "browser_click": ("browser.click",),
    "browser_find": ("browser.find",),
    "browser_press_key": ("browser.press_key",),
    "browser_type": ("browser.type",),
    "sandbox_execute_offline": ("sandbox.execute.offline",),
    "runtime_review_task": ("runtime.review",),
    "learning_record_evidence": ("learning.evidence.record",),
    "learning_propose_lesson": ("learning.lesson.propose",),
    "learning_recall_failures": ("learning.evidence.inspect",),
    "learning_stage_tool": ("learning.tool.stage",),
    "learning_stage_skill": ("learning.skill.stage",),
    "learning_create_tool": ("learning.tool.create",),
    "learning_create_snapshot_extractor": ("learning.tool.create.snapshot",),
    "learning_create_skill": ("learning.skill.create",),
    "learning_validate_artifact": ("learning.artifact.validate",),
    "learning_activate_artifact": ("learning.artifact.activate",),
    "learning_retire_artifact": ("learning.artifact.retire",),
    "learning_list_artifacts": ("learning.artifact.list",),
    "learning_list_recovery_tickets": ("learning.recovery.inspect",),
    "learning_create_repair_adapter": ("learning.recovery.create",),
    "full_host_status": ("host.inspect",),
    "full_tor_search": ("network.tor.search",),
    "full_tor_fetch": ("network.tor.fetch",),
    "full_tor_inventory": ("network.tor.inventory",),
    "full_tor_browser_inventory": ("network.tor.browser",),
    "full_tor_browser_close": ("network.tor.browser.close",),
    "evm_analyze_erc20_abi": ("evm.erc20.analyze",),
    "evm_validate_oracle": ("evm.oracle.validate",),
    "evm_analyze_solidity_security": ("evm.solidity.analyze",),
    "evm_decode_uniswap_v4_hook": ("evm.uniswap.hook.decode",),
    "evm_quote_flash_swap": ("evm.flash_swap.quote",),
    "evm_foundry_test_offline": ("evm.foundry.test.offline",),
}


def normalize_capabilities(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(str(value).strip().casefold() for value in values if str(value).strip())
    )
    if any(not _CAPABILITY.fullmatch(value) for value in normalized):
        raise ValueError("capabilities must use lowercase dotted identifiers")
    return normalized


def capabilities_for_tool(
    tool: str,
    declared: Iterable[str] = (),
) -> tuple[str, ...]:
    explicit = normalize_capabilities(declared)
    if explicit:
        return explicit
    builtin = BUILTIN_TOOL_CAPABILITIES.get(str(tool).strip(), ())
    if builtin:
        return builtin
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(tool).casefold()).strip("_")
    return (f"generated.{normalized}",) if normalized else ()


def generated_capability_is_repairable(capability: str) -> bool:
    """Generated code may replace generated computation, never host authority."""

    return capability.startswith("generated.")


def _is_call_rejection(error: str, exception: Exception | None = None) -> bool:
    """Distinguish invalid call data from an unhealthy provider.

    A circuit breaker measures provider health. Bad model arguments must still
    produce a recovery ticket, but they must not disable the provider for the
    next (possibly corrected) call.
    """

    if isinstance(exception, (KeyError, TypeError, ValueError)):
        return True
    text = " ".join(str(error).casefold().split())
    if text.startswith(("artifactvalidationerror:", "artifactpolicyerror:", "sourceblueprinterror:")):
        # Rejection of candidate code is a working validator, not an outage.
        # Also recognizes durable failures read after a runtime restart.
        return True
    return any(
        marker in text
        for marker in (
            "requires structured arguments",
            "requires a non-empty",
            "invalid onion url",
            "must be a valid tor v3 .onion address",
            "tor fetch accepts only http or https urls",
            "credentials are not accepted in a tor url",
            "tor web fetch accepts only ports 80 and 443",
            "timeout_seconds must be a number",
            "provide either \"text\" or \"regex\"",
            "arguments as json",
            "failed to parse tool call arguments",
        )
    )


def _bounded_json(value: Any, *, maximum_bytes: int = 100_000) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError("recovery fixture exceeds its local storage bound")
    return json.loads(encoded)


def _redact(value: Any, *, field: str = "") -> tuple[Any, bool]:
    if field and _SENSITIVE_FIELD.search(field):
        return "<redacted>", True
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        redacted = False
        for key, item in value.items():
            safe, hidden = _redact(item, field=str(key))
            result[str(key)] = safe
            redacted = redacted or hidden
        return result, redacted
    if isinstance(value, list):
        result = []
        redacted = False
        for item in value:
            safe, hidden = _redact(item)
            result.append(safe)
            redacted = redacted or hidden
        return result, redacted
    return value, False


@dataclass(slots=True)
class ProviderState:
    tool: str
    capabilities: tuple[str, ...]
    priority: int = 100
    kind: str = "native"
    successful_calls: int = 0
    consecutive_failures: int = 0
    total_failures: int = 0
    circuit_open: bool = False
    retry_after: str = ""
    last_error: str = ""
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderState":
        copied = dict(data)
        copied["capabilities"] = normalize_capabilities(copied.get("capabilities", ()))
        return cls(**copied)


@dataclass(slots=True)
class RecoveryTicket:
    capability: str
    failed_tool: str
    arguments: dict[str, Any]
    arguments_sha256: str
    error: str
    task_id: str = ""
    ticket_id: str = field(default_factory=lambda: uuid4().hex)
    occurrences: int = 1
    fixture_redacted: bool = False
    state: str = "open"
    replacement_tool: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryTicket":
        return cls(**data)


@dataclass(slots=True)
class ToolCallOutcome:
    result: Any
    requested_tool: str
    provider_tool: str
    capabilities: tuple[str, ...]
    attempts: tuple[dict[str, Any], ...] = ()
    error: str = ""
    recovery_ticket: dict[str, Any] | None = None
    exception: Exception | None = None
    failure_details: dict[str, Any] = field(default_factory=dict)


class ToolRecoveryRegistry:
    """Persistent provider health, bounded failover, and repair tickets.

    The registry changes routing only between providers that explicitly expose
    the same capability. It never edits Python source and never grants a tool a
    capability it did not declare through trusted runtime metadata.
    """

    failure_threshold = 2

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.state_path = self.root / "providers.json"
        self.tickets_path = self.root / "tickets.json"
        self.journal_path = self.root / "journal.jsonl"
        self._providers = self._load_providers()
        self._tickets = self._load_tickets()

    def register_provider(
        self,
        tool: str,
        capabilities: Iterable[str],
        *,
        priority: int = 100,
        kind: str = "native",
    ) -> ProviderState:
        name = str(tool).strip()
        declared = normalize_capabilities(capabilities)
        if not name or not declared:
            raise ValueError("provider tool and capabilities are required")
        current = self._providers.get(name)
        normalized_priority = max(0, min(1_000, int(priority)))
        normalized_kind = str(kind)[:40] or "native"
        changed = current is None
        if current is None:
            current = ProviderState(
                tool=name,
                capabilities=declared,
                priority=normalized_priority,
                kind=normalized_kind,
            )
            self._providers[name] = current
        else:
            changed = changed or (
                current.capabilities != declared
                or current.priority != normalized_priority
                or current.kind != normalized_kind
            )
            if changed:
                current.capabilities = declared
                current.priority = normalized_priority
                current.kind = normalized_kind
        if changed:
            self._save_providers()
        return current

    def capabilities(self, tool: str) -> tuple[str, ...]:
        provider = self._providers.get(str(tool).strip())
        if provider is not None:
            return provider.capabilities
        return capabilities_for_tool(tool)

    def retain_providers(self, allowed_tools: Iterable[str]) -> None:
        allowed = {str(item).strip() for item in allowed_tools if str(item).strip()}
        removed = sorted(set(self._providers) - allowed)
        if not removed:
            return
        for tool in removed:
            del self._providers[tool]
        tickets_changed = False
        for ticket in self._tickets.values():
            if ticket.state == "active" and ticket.replacement_tool in removed:
                ticket.state = "rolled_back"
                ticket.updated_at = utc_now()
                tickets_changed = True
        self._save_providers()
        if tickets_changed:
            self._save_tickets()
        self._journal("providers_pruned", {"tools": removed})

    def providers_for(self, requested_tool: str) -> list[ProviderState]:
        requested = str(requested_tool).strip()
        capabilities = set(self.capabilities(requested))
        now = datetime.now(UTC)
        changed = False
        for provider in self._providers.values():
            # Heal state persisted by older runtimes which treated malformed
            # model arguments as a provider outage.
            if provider.circuit_open and _is_call_rejection(provider.last_error):
                provider.circuit_open = False
                provider.consecutive_failures = 0
                provider.retry_after = ""
                provider.updated_at = utc_now()
                changed = True
                continue
            if not provider.circuit_open or not provider.retry_after:
                continue
            try:
                retry_at = datetime.fromisoformat(provider.retry_after)
            except ValueError:
                continue
            if retry_at <= now:
                provider.circuit_open = False
                provider.consecutive_failures = max(0, self.failure_threshold - 1)
                provider.retry_after = ""
                provider.updated_at = utc_now()
                changed = True
        if changed:
            self._save_providers()
            self._journal("provider_half_opened", {"requested_tool": requested})
        candidates = [
            provider
            for provider in self._providers.values()
            if capabilities.intersection(provider.capabilities)
            and not provider.circuit_open
        ]
        return sorted(
            candidates,
            key=lambda provider: (
                -(provider.priority - provider.consecutive_failures * 100),
                provider.tool != requested,
                provider.tool,
            ),
        )

    def record_success(self, tool: str) -> None:
        provider = self._providers.get(tool)
        if provider is None:
            return
        provider.successful_calls += 1
        provider.consecutive_failures = 0
        provider.circuit_open = False
        provider.retry_after = ""
        provider.last_error = ""
        provider.updated_at = utc_now()
        self._save_providers()
        self._journal("provider_succeeded", {"tool": tool})

    def record_failure(
        self,
        *,
        tool: str,
        requested_tool: str,
        arguments: dict[str, Any],
        error: str,
        task_id: str = "",
        affects_health: bool = True,
    ) -> RecoveryTicket:
        provider = self._providers.get(tool)
        if provider is None:
            provider = self.register_provider(tool, capabilities_for_tool(requested_tool))
        if affects_health:
            provider.consecutive_failures += 1
        else:
            provider.consecutive_failures = 0
        provider.total_failures += 1
        provider.last_error = " ".join(str(error).split())[:2_000]
        provider.updated_at = utc_now()
        if affects_health and provider.consecutive_failures >= self.failure_threshold:
            provider.circuit_open = True
            provider.retry_after = (
                ""
                if provider.kind == "generated_repair"
                else (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
            )
        self._save_providers()
        if provider.circuit_open and provider.kind == "generated_repair":
            for active_ticket in self._tickets.values():
                if (
                    active_ticket.state == "active"
                    and active_ticket.replacement_tool == provider.tool
                ):
                    active_ticket.state = "rolled_back"
                    active_ticket.updated_at = utc_now()
                    self._save_tickets()
                    self._journal(
                        "repair_rolled_back",
                        {
                            "ticket_id": active_ticket.ticket_id,
                            "replacement_tool": provider.tool,
                            "reason": "two consecutive runtime failures",
                        },
                    )

        try:
            bounded_arguments = _bounded_json(arguments)
        except ValueError:
            safe_arguments = {
                "_fixture_unavailable": "arguments exceeded the 100 KB recovery bound"
            }
            redacted = True
        else:
            safe_arguments, redacted = _redact(bounded_arguments)
        encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        capability = (self.capabilities(requested_tool) or (f"generated.{requested_tool}",))[0]
        fingerprint = hashlib.sha256(
            f"{capability}\0{requested_tool}\0{digest}".encode("utf-8")
        ).hexdigest()
        existing = next(
            (
                ticket
                for ticket in self._tickets.values()
                if ticket.state == "open"
                and hashlib.sha256(
                    f"{ticket.capability}\0{ticket.failed_tool}\0{ticket.arguments_sha256}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                == fingerprint
            ),
            None,
        )
        if existing is None:
            ticket = RecoveryTicket(
                capability=capability,
                failed_tool=requested_tool,
                arguments=safe_arguments,
                arguments_sha256=digest,
                error=" ".join(str(error).split())[:2_000],
                task_id=str(task_id)[:128],
                fixture_redacted=redacted,
            )
            self._tickets[ticket.ticket_id] = ticket
        else:
            ticket = existing
            ticket.occurrences += 1
            ticket.error = " ".join(str(error).split())[:2_000]
            ticket.updated_at = utc_now()
        self._save_tickets()
        self._journal(
            "provider_failed" if affects_health else "tool_call_rejected",
            {
                "tool": tool,
                "requested_tool": requested_tool,
                "capability": capability,
                "ticket_id": ticket.ticket_id,
                "circuit_open": provider.circuit_open,
            },
        )
        return ticket

    def ticket(self, ticket_id: str) -> RecoveryTicket:
        ticket = self._tickets.get(str(ticket_id))
        if ticket is None:
            raise ValueError(f"unknown recovery ticket: {ticket_id}")
        return ticket

    def list_tickets(self, *, state: str | None = None) -> list[dict[str, Any]]:
        tickets = sorted(self._tickets.values(), key=lambda item: item.updated_at, reverse=True)
        return [
            ticket.to_dict()
            for ticket in tickets
            if state is None or ticket.state == state
        ]

    def activate_repair(self, ticket_id: str, replacement_tool: str) -> RecoveryTicket:
        ticket = self.ticket(ticket_id)
        if ticket.state != "open":
            raise ValueError("only an open recovery ticket can be activated")
        ticket.state = "active"
        ticket.replacement_tool = str(replacement_tool).strip()
        ticket.updated_at = utc_now()
        self._save_tickets()
        self._journal(
            "repair_activated",
            {
                "ticket_id": ticket.ticket_id,
                "replacement_tool": ticket.replacement_tool,
                "capability": ticket.capability,
            },
        )
        return ticket

    def resolve_by_failover(self, ticket_id: str, provider_tool: str) -> None:
        ticket = self.ticket(ticket_id)
        if ticket.state != "open":
            return
        ticket.state = "resolved"
        ticket.replacement_tool = str(provider_tool).strip()
        ticket.updated_at = utc_now()
        self._save_tickets()
        self._journal(
            "ticket_resolved_by_failover",
            {
                "ticket_id": ticket.ticket_id,
                "provider_tool": ticket.replacement_tool,
            },
        )

    def provider_state(self) -> list[dict[str, Any]]:
        return [
            provider.to_dict()
            for provider in sorted(self._providers.values(), key=lambda item: item.tool)
        ]

    def provider_diagnostics(self, requested_tool: str) -> list[dict[str, Any]]:
        """Return bounded routing state, including providers skipped by a circuit.

        The last error is historical evidence. Callers must not present it as
        the result of a new execution which never reached the provider.
        """

        requested = str(requested_tool).strip()
        capabilities = set(self.capabilities(requested))
        return [
            {
                "provider": provider.tool,
                "available": not provider.circuit_open,
                "circuit_open": provider.circuit_open,
                "retry_after": provider.retry_after,
                "historical_last_error": provider.last_error[:2_000],
                "updated_at": provider.updated_at,
            }
            for provider in sorted(self._providers.values(), key=lambda item: item.tool)
            if capabilities.intersection(provider.capabilities)
        ]

    def _load_providers(self) -> dict[str, ProviderState]:
        values = self._load_json(self.state_path, [])
        return {
            item.tool: item
            for raw in values
            if isinstance(raw, dict)
            for item in [ProviderState.from_dict(raw)]
        }

    def _load_tickets(self) -> dict[str, RecoveryTicket]:
        values = self._load_json(self.tickets_path, [])
        return {
            item.ticket_id: item
            for raw in values
            if isinstance(raw, dict)
            for item in [RecoveryTicket.from_dict(raw)]
        }

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        return value

    def _save_providers(self) -> None:
        self._atomic_json(
            self.state_path,
            [provider.to_dict() for provider in self._providers.values()],
        )

    def _save_tickets(self) -> None:
        self._atomic_json(
            self.tickets_path,
            [ticket.to_dict() for ticket in self._tickets.values()],
        )

    def _atomic_json(self, path: Path, value: Any) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _journal(self, event: str, data: dict[str, Any]) -> None:
        descriptor = os.open(self.journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"timestamp": utc_now(), "event": event, "data": data},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


FailureDetector = Callable[[str, str], str]
ProviderCall = Callable[[str, dict[str, Any] | str], Awaitable[Any]]


async def execute_with_recovery(
    registry: ToolRecoveryRegistry,
    *,
    requested_tool: str,
    arguments: dict[str, Any] | str,
    call_provider: ProviderCall,
    detect_failure: FailureDetector,
    task_id: str = "",
) -> ToolCallOutcome:
    structured = arguments if isinstance(arguments, dict) else {}
    candidates = registry.providers_for(requested_tool)
    if not candidates:
        registry.register_provider(requested_tool, capabilities_for_tool(requested_tool))
        candidates = registry.providers_for(requested_tool)

    if not candidates:
        providers = registry.provider_diagnostics(requested_tool)
        blocked = [item for item in providers if item["circuit_open"]]
        if blocked:
            descriptions = []
            for item in blocked[:8]:
                retry = item["retry_after"] or "manual recovery required"
                historical = item["historical_last_error"] or "not recorded"
                descriptions.append(
                    f"{item['provider']}: circuit open; retry after {retry}; "
                    f"historical previous failure: {historical}"
                )
            reason = "; ".join(descriptions)
            error = (
                "ToolDispatchUnavailableError: no provider was executed because "
                f"all matching providers are temporarily unavailable. {reason}"
            )
            code = "matching_providers_circuit_open"
        else:
            error = (
                "ToolDispatchUnavailableError: no provider was executed because "
                f"no eligible provider is registered for {requested_tool!r}."
            )
            code = "no_eligible_provider"
        return ToolCallOutcome(
            result=f"Tool execution not started: {error}",
            requested_tool=requested_tool,
            provider_tool=requested_tool,
            capabilities=registry.capabilities(requested_tool),
            error=error,
            failure_details={
                "stage": "dispatch",
                "reason": code,
                "execution_attempted": False,
                "providers": providers[:16],
            },
        )

    attempts: list[dict[str, Any]] = []
    last_result = ""
    last_error = ""
    last_exception: Exception | None = None
    ticket: RecoveryTicket | None = None
    for provider in candidates:
        provider_exception: Exception | None = None
        failure_details: dict[str, Any] = {}
        try:
            raw_result = await call_provider(provider.tool, arguments)
            result = str(raw_result)
            error = detect_failure(result, provider.tool)
        except Exception as exc:  # trusted caller decides how to render this
            message = str(exc).strip() or "exception contained no message"
            result = f"Tool execution failed: {type(exc).__name__}: {message}"
            error = f"{type(exc).__name__}: {message}"
            provider_exception = exc
            failure_details = {
                "stage": "provider_execution",
                "execution_attempted": True,
                "provider": provider.tool,
                "exception_type": type(exc).__name__,
            }
            report = getattr(exc, "report", None)
            if isinstance(report, dict):
                failure_details["validation"] = {
                    key: report[key]
                    for key in (
                        "stage",
                        "tests_total",
                        "tests_passed",
                        "tests_failed",
                        "tests_not_run",
                        "semantic_correctness",
                    )
                    if key in report
                    and isinstance(report[key], (str, int, float, bool, type(None)))
                }
        last_exception = provider_exception
        attempts.append(
            {
                "provider": provider.tool,
                "status": "failed" if error else "succeeded",
                "error": error[:2_000],
                "stage": "provider_execution",
                "execution_attempted": True,
                **({"failure_details": failure_details} if failure_details else {}),
            }
        )
        if not error:
            registry.record_success(provider.tool)
            if ticket is not None:
                registry.resolve_by_failover(ticket.ticket_id, provider.tool)
            return ToolCallOutcome(
                result=raw_result,
                requested_tool=requested_tool,
                provider_tool=provider.tool,
                capabilities=registry.capabilities(provider.tool),
                attempts=tuple(attempts),
            )
        last_result = result
        last_error = error
        ticket = registry.record_failure(
            tool=provider.tool,
            requested_tool=requested_tool,
            arguments=structured,
            error=error,
            task_id=task_id,
            affects_health=not _is_call_rejection(error, provider_exception),
        )

    return ToolCallOutcome(
        result=last_result or f"Tool execution failed: {last_error}",
        requested_tool=requested_tool,
        provider_tool=(attempts[-1]["provider"] if attempts else requested_tool),
        capabilities=registry.capabilities(requested_tool),
        attempts=tuple(attempts),
        error=last_error or "all capability providers failed",
        recovery_ticket=(ticket.to_dict() if ticket is not None else None),
        exception=last_exception,
        failure_details=(
            attempts[-1].get("failure_details")
            or {
                "stage": "provider_execution",
                "execution_attempted": True,
                "provider": attempts[-1]["provider"],
            }
        ),
    )
