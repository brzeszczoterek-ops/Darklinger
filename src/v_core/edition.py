from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Protocol, runtime_checkable

from .autonomy import AuthorizationEnvelope


class EditionUnavailable(RuntimeError):
    """Raised when a requested DARKLINGER edition is not installed."""


@dataclass(frozen=True, slots=True)
class Edition:
    name: str
    extension_module: str | None
    default_evm_profile: str
    default_learning_profile: str

    @property
    def is_full(self) -> bool:
        return self.name == "full"


PUBLIC_EDITION = Edition(
    name="public",
    extension_module=None,
    default_evm_profile="client",
    default_learning_profile="client",
)
FULL_EDITION = Edition(
    name="full",
    extension_module="v_full.extension",
    default_evm_profile="owner_lab",
    default_learning_profile="owner_lab",
)


@runtime_checkable
class EditionExtension(Protocol):
    name: str

    def configure_authorization(
        self,
        envelope: AuthorizationEnvelope,
        *,
        evm_profile: str,
        learning_profile: str,
    ) -> None: ...

    def bind_runtime(self, authorization: object, sandbox_backend: object | None) -> None: ...

    def tool_names(self) -> tuple[str, ...]: ...

    def tool_definitions(self) -> dict[str, tuple[str, dict[str, Any]]]: ...

    def handles_tool(self, name: str) -> bool: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> str: ...

    def ui_manifest(self) -> dict[str, Any] | None: ...

    def ui_status(self, context: dict[str, Any]) -> dict[str, Any] | None: ...


@dataclass(slots=True)
class PublicEditionExtension:
    name: str = "public"

    def configure_authorization(
        self,
        envelope: AuthorizationEnvelope,
        *,
        evm_profile: str,
        learning_profile: str,
    ) -> None:
        if evm_profile != "client" or learning_profile != "client":
            raise EditionUnavailable(
                "owner_lab capabilities require the private DARKLINGER-Full package"
            )

    def bind_runtime(self, authorization: object, sandbox_backend: object | None) -> None:
        return None

    def tool_names(self) -> tuple[str, ...]:
        return ()

    def tool_definitions(self) -> dict[str, tuple[str, dict[str, Any]]]:
        return {}

    def handles_tool(self, name: str) -> bool:
        return False

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> str:
        raise EditionUnavailable(f"tool {name!r} is unavailable in public DARKLINGER")

    def ui_manifest(self) -> dict[str, Any] | None:
        return None

    def ui_status(self, context: dict[str, Any]) -> dict[str, Any] | None:
        return None


def resolve_edition(requested: str | None = None) -> Edition:
    normalized = (requested or "auto").strip().casefold()
    if normalized not in {"auto", "public", "full"}:
        raise ValueError("DARKLINGER_EDITION must be 'auto', 'public', or 'full'")
    if normalized == "auto":
        normalized = "full" if find_spec("v_full") is not None else "public"
    if normalized == "public":
        return PUBLIC_EDITION
    if find_spec("v_full") is None:
        raise EditionUnavailable(
            "DARKLINGER_EDITION=full was requested, but the private v_full package "
            "is not installed"
        )
    return FULL_EDITION


def load_edition_extension(edition: Edition) -> EditionExtension:
    if edition.extension_module is None:
        return PublicEditionExtension()
    try:
        module = import_module(edition.extension_module)
    except ModuleNotFoundError as exc:
        if exc.name == "v_full" or (exc.name or "").startswith("v_full."):
            raise EditionUnavailable(
                "the private DARKLINGER-Full extension is unavailable"
            ) from exc
        raise
    factory = getattr(module, "create_extension", None)
    if not callable(factory):
        raise EditionUnavailable(
            f"edition extension {edition.extension_module!r} has no create_extension()"
        )
    extension = factory()
    if not isinstance(extension, EditionExtension):
        raise EditionUnavailable(
            f"edition extension {edition.extension_module!r} does not satisfy "
            "the DARKLINGER extension contract"
        )
    return extension


def launch_edition_monitor(
    edition: Edition,
    session: object | None,
) -> bool:
    if not edition.is_full or session is None:
        return False
    module = import_module("v_full.owner_monitor")
    launch = getattr(module, "launch_owner_monitor")
    return bool(launch(session))
