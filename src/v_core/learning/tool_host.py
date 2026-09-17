"""Trusted, minimal host for generated DARKLINGER tools.

This module is executed only inside the offline sandbox. Generated code is
loaded as data from a read-only mount and must expose ``run(arguments)``.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise RuntimeError("tool host requires source path and JSON arguments")
    source = Path(sys.argv[1])
    arguments = json.loads(sys.argv[2])
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be a JSON object")

    spec = importlib.util.spec_from_file_location("darklinger_generated_tool", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generated tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError("generated tool does not expose run(arguments)")
    result = run(arguments)
    if inspect.isawaitable(result):
        raise TypeError("generated tools must be synchronous")
    if not isinstance(result, dict):
        raise TypeError("generated tool output must be a JSON object")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
