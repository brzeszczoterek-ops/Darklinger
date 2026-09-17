from __future__ import annotations

from dataclasses import asdict, fields
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from .state import RelationshipState


class RelationshipStorageError(RuntimeError):
    pass


class RelationshipStorage:

    def __init__(
        self,
        root: str | Path = "memory/relationship",
    ):
        self.root = Path(root)
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._protect(self.root, 0o700)

        self.path = self.root / "state.yaml"

    def save(
        self,
        state: RelationshipState,
    ) -> None:

        payload = {
            "schema_version": 1,
            "state": asdict(state),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".state-",
            suffix=".tmp",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        try:
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                yaml.safe_dump(
                    payload,
                    file,
                    allow_unicode=True,
                    sort_keys=False,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
            self._protect(self.path, 0o600)
            self._sync_directory()
        except Exception as error:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise RelationshipStorageError(
                f"Could not save relationship state: {error}"
            ) from error

    def load(self) -> RelationshipState:

        if not self.path.exists():
            return RelationshipState()

        try:
            with self.path.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}
        except (OSError, yaml.YAMLError) as error:
            raise RelationshipStorageError(
                f"Could not load relationship state: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise RelationshipStorageError("Relationship state must be a mapping")

        # Version 1 uses a wrapper. Flat files from DARKLINGER 1.0 are migrated
        # transparently on the next successful save.
        if "schema_version" in payload:
            if payload.get("schema_version") != 1:
                raise RelationshipStorageError(
                    f"Unsupported relationship schema: {payload.get('schema_version')!r}"
                )
            data: Any = payload.get("state")
        else:
            data = payload

        if not isinstance(data, dict):
            raise RelationshipStorageError("Relationship state payload must be a mapping")

        allowed = {item.name for item in fields(RelationshipState)}
        try:
            state = RelationshipState(
                **{key: value for key, value in data.items() if key in allowed}
            )
        except (TypeError, ValueError) as error:
            raise RelationshipStorageError(
                f"Invalid relationship state: {error}"
            ) from error

        self._protect(self.path, 0o600)
        return state

    @staticmethod
    def _protect(path: Path, mode: int) -> None:
        try:
            path.chmod(mode)
        except OSError:
            pass

    def _sync_directory(self) -> None:
        try:
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
