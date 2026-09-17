from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..branding import env_value
from typing import Sequence

from .audit import audit_learning_store
from .storage import LearningStore


def _root(value: str | None) -> Path:
    selected = value or env_value("DARKLINGER_LEARNING_ROOT", "learning")
    return Path(selected).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darklinger-learning",
        description="Inspect and verify DARKLINGER's evidence and artifact store.",
    )
    parser.add_argument("--root", help="learning root (default: DARKLINGER_LEARNING_ROOT)")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("verify", help="verify journals, records, and bundle digests")
    subcommands.add_parser("artifacts", help="list artifact lifecycle records")
    evidence = subcommands.add_parser("evidence", help="list recorded evidence")
    evidence.add_argument("--limit", type=int, default=20)
    subcommands.add_parser("lessons", help="list proposed and validated lessons")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root(args.root)
    if args.command == "verify":
        payload = audit_learning_store(root).to_dict()
    else:
        store = LearningStore(root)
        if args.command == "artifacts":
            payload = {"artifacts": [item.to_dict() for item in store.list_records()]}
        elif args.command == "lessons":
            payload = {"lessons": [item.to_dict() for item in store.list_lessons()]}
        else:
            limit = max(1, min(1_000, args.limit))
            payload = {
                "evidence": [item.to_dict() for item in store.list_evidence()[-limit:]]
            }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
