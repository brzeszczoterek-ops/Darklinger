from __future__ import annotations

from pathlib import Path

import pytest

from v_core.autonomy import AuthorizationEnvelope
from v_core.config import load_config
from v_core.edition import (
    EditionUnavailable,
    load_edition_extension,
    resolve_edition,
)


def test_public_edition_has_no_private_extension() -> None:
    edition = resolve_edition("public")
    extension = load_edition_extension(edition)

    assert edition.name == "public"
    assert extension.name == "public"
    assert extension.tool_names() == ()


def test_public_edition_fails_closed_on_owner_profiles() -> None:
    extension = load_edition_extension(resolve_edition("public"))

    with pytest.raises(EditionUnavailable, match="DARKLINGER-Full"):
        extension.configure_authorization(
            AuthorizationEnvelope(),
            evm_profile="owner_lab",
            learning_profile="client",
        )


def test_public_config_rejects_owner_lab_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DARKLINGER_EDITION", "public")
    monkeypatch.setenv("DARKLINGER_EVM_PROFILE", "owner_lab")

    with pytest.raises(ValueError, match="require DARKLINGER-Full"):
        load_config()
