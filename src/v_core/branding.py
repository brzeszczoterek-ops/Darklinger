from __future__ import annotations

from collections.abc import Mapping
import os


PRODUCT_NAME = "Darklinger"
PRODUCT_NAME_UPPER = "DARKLINGER"
ENV_PREFIX = "DARKLINGER_"

# Keep one release cycle of configuration compatibility without treating the
# former name as the current product identity.
_LEGACY_ENV_PREFIX = "PALA" + "DYN_"


def env_value(
    name: str,
    default: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Read a canonical Darklinger setting, then its legacy alias.

    Canonical values always win when both forms are present. This lets existing
    installations start after the rename while every newly documented setting
    uses ``DARKLINGER_*``.
    """

    if not name.startswith(ENV_PREFIX):
        raise ValueError(f"canonical setting must start with {ENV_PREFIX}")
    values = os.environ if environment is None else environment
    if name in values:
        return values[name]
    legacy_name = _LEGACY_ENV_PREFIX + name.removeprefix(ENV_PREFIX)
    return values.get(legacy_name, default)


__all__ = ["ENV_PREFIX", "PRODUCT_NAME", "PRODUCT_NAME_UPPER", "env_value"]
