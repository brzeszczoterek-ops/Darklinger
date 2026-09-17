from __future__ import annotations

from v_core.branding import env_value


def test_canonical_darklinger_setting_wins_over_legacy_alias() -> None:
    environment = {
        "DARKLINGER_UI_PORT": "9001",
        "PALADYN_UI_PORT": "8765",
    }

    assert env_value(
        "DARKLINGER_UI_PORT",
        environment=environment,
    ) == "9001"


def test_legacy_setting_remains_a_transition_alias() -> None:
    assert env_value(
        "DARKLINGER_UI_PORT",
        "8000",
        environment={"PALADYN_UI_PORT": "8765"},
    ) == "8765"


def test_unknown_setting_uses_default() -> None:
    assert env_value(
        "DARKLINGER_UI_PORT",
        "8765",
        environment={},
    ) == "8765"
