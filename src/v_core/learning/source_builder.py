from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from ..generated_tool_contract import GeneratedToolContract


class SourceBlueprintError(ValueError):
    """The runtime could not build a grounded tool contract from source."""


@dataclass(frozen=True, slots=True)
class SourceBlueprint:
    name: str
    description: str
    arguments: dict[str, Any]
    expected: dict[str, Any] | None
    oracle: str
    tests: tuple[tuple[dict[str, Any], dict[str, Any]], ...] = ()
    final_arguments: dict[str, Any] | None = None


_EXPLICIT_NAME = re.compile(
    r"(?:\b(?:called|named|name)\b|\b(?:nazw(?:a|ij|any|ane)|o\s+nazwie)\b)"
    r"[\s:=-]*[`\"']?([a-z][a-z0-9_]{2,63})",
    re.IGNORECASE,
)
_SNAKE_NAME = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9]+(?:_[a-z0-9]+)+)(?![A-Za-z0-9_])")
_IGNORED_NAMES = {
    "learning_create_tool",
    "snapshot_text",
    "timeout_seconds",
    "input_schema",
    "output_schema",
    "test_arguments",
    "test_expected",
}


def json_assignments(text: str) -> dict[str, list[Any]]:
    """Extract exact ``name = JSON`` literals without interpreting prose."""

    decoder = json.JSONDecoder()
    assignments: dict[str, list[Any]] = {}
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    )
    for match in pattern.finditer(text):
        remainder = text[match.end() :]
        leading = len(remainder) - len(remainder.lstrip())
        try:
            value, _ = decoder.raw_decode(remainder[leading:])
        except (TypeError, json.JSONDecodeError):
            continue
        assignments.setdefault(match.group(1), []).append(value)
    return assignments


def source_argument_defaults(source: str) -> tuple[set[str], dict[str, Any]]:
    """Return argument keys used by ``run`` and literal ``dict.get`` defaults."""

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise SourceBlueprintError(f"generated tool has invalid Python: {error}") from error

    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run is None:
        raise SourceBlueprintError("generated tool must define run(arguments)")
    if not run.args.args:
        raise SourceBlueprintError("run must accept one arguments object")
    argument_name = run.args.args[0].arg
    fields: set[str] = set()
    defaults: dict[str, Any] = {}

    class _RunArgumentUsage(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == argument_name:
                key = _literal_string(node.slice)
                if key:
                    fields.add(key)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == argument_name
                and node.args
            ):
                key = _literal_string(node.args[0])
                if key:
                    fields.add(key)
                    if len(node.args) >= 2:
                        try:
                            default = ast.literal_eval(node.args[1])
                            json.dumps(default, allow_nan=False)
                        except (TypeError, ValueError):
                            pass
                        else:
                            defaults[key] = default
            self.generic_visit(node)

    usage = _RunArgumentUsage()
    for statement in run.body:
        usage.visit(statement)
    return fields, defaults


def repair_generated_source_argument_alias(
    source: str,
    generated_contract: GeneratedToolContract,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Repair one unambiguous model-written alias for a runtime-owned field.

    The frozen contract owns the wire names.  A small code model may still
    spell the sole input as camelCase, kebab-case, or natural-language text.
    When there is exactly one missing contract field and exactly one unknown
    source field, rewrite only string subscripts on ``run``'s arguments object.
    Every ambiguous case is left untouched for the normal validator to reject.
    """

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return source, ()

    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run is None or not run.args.args:
        return source, ()

    expected_fields = {
        str(name)
        for case in generated_contract.tests
        for name in case.arguments
    }
    expected_fields.update(str(name) for name in generated_contract.final_arguments)
    source_fields, _ = source_argument_defaults(source)
    missing = expected_fields - source_fields
    unknown = source_fields - expected_fields
    if len(missing) != 1 or len(unknown) != 1:
        return source, ()

    old_name = next(iter(unknown))
    new_name = next(iter(missing))
    argument_name = run.args.args[0].arg

    class _ArgumentAliasRepair(ast.NodeTransformer):
        replacements = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            if node is not run:
                return node
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            if node is not run:
                return node
            return self.generic_visit(node)

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
            return node

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if not (
                isinstance(node.value, ast.Name)
                and node.value.id == argument_name
                and _literal_string(node.slice) == old_name
            ):
                return node
            node.slice = ast.copy_location(ast.Constant(new_name), node.slice)
            self.replacements += 1
            return node

    repair = _ArgumentAliasRepair()
    repair.visit(run)
    if repair.replacements == 0:
        return source, ()

    ast.fix_missing_locations(tree)
    repaired = ast.unparse(tree)
    repaired_fields, _ = source_argument_defaults(repaired)
    if repaired_fields != expected_fields:
        return source, ()
    return repaired, ((old_name, new_name),)


def repair_generated_source_json_wrapper(
    source: str,
    generated_contract: GeneratedToolContract,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Insert one unambiguous runtime-observed JSON wrapper traversal.

    This repairs a common transport mistake where generated code parses a JSON
    input correctly but reads keys from the document root even though every
    frozen fixture places those keys under the same sole object wrapper.  It is
    data-shape driven, independent of the owner's language and the tool domain.
    Ambiguous shapes or source flows are left for normal validation to reject.
    """

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return source, ()
    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run is None or not run.args.args:
        return source, ()

    argument_name = run.args.args[0].arg
    contract_rows = [case.arguments for case in generated_contract.tests]
    contract_rows.append(generated_contract.final_arguments)
    wrapper_shapes: dict[str, tuple[str, tuple[dict[str, Any], ...]]] = {}
    expected_fields = {str(name) for row in contract_rows for name in row}
    for field in sorted(expected_fields):
        parsed_rows: list[dict[str, Any]] = []
        wrapper_name = ""
        valid = True
        for row in contract_rows:
            raw = row.get(field)
            if not isinstance(raw, str):
                valid = False
                break
            try:
                parsed = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                valid = False
                break
            if not isinstance(parsed, dict) or len(parsed) != 1:
                valid = False
                break
            current_wrapper, nested = next(iter(parsed.items()))
            if not isinstance(nested, dict):
                valid = False
                break
            if wrapper_name and current_wrapper != wrapper_name:
                valid = False
                break
            wrapper_name = str(current_wrapper)
            parsed_rows.append(parsed)
        if valid and wrapper_name and parsed_rows:
            wrapper_shapes[field] = (wrapper_name, tuple(parsed_rows))
    if not wrapper_shapes:
        return source, ()

    argument_aliases: dict[str, str] = {}
    parsed_targets: dict[str, tuple[ast.Assign, str]] = {}
    for statement in run.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            continue
        target_name = statement.targets[0].id
        if isinstance(statement.value, ast.Subscript) and isinstance(
            statement.value.value,
            ast.Name,
        ):
            if statement.value.value.id == argument_name:
                field = _literal_string(statement.value.slice)
                if field:
                    argument_aliases[target_name] = field
            continue
        call = statement.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "json"
            and call.func.attr == "loads"
            and len(call.args) == 1
        ):
            continue
        raw_input = call.args[0]
        field = ""
        if isinstance(raw_input, ast.Name):
            field = argument_aliases.get(raw_input.id, "")
        elif isinstance(raw_input, ast.Subscript) and isinstance(
            raw_input.value,
            ast.Name,
        ) and raw_input.value.id == argument_name:
            field = _literal_string(raw_input.slice) or ""
        if field:
            parsed_targets[target_name] = (statement, field)

    direct_keys: dict[str, set[str]] = {name: set() for name in parsed_targets}

    class _ParsedObjectAccess(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in direct_keys:
                key = _literal_string(node.slice)
                if key:
                    direct_keys[node.value.id].add(key)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in direct_keys
                and node.func.attr == "get"
                and node.args
            ):
                key = _literal_string(node.args[0])
                if key:
                    direct_keys[node.func.value.id].add(key)
            self.generic_visit(node)

    access = _ParsedObjectAccess()
    for statement in run.body:
        access.visit(statement)

    candidates: list[tuple[ast.Assign, str, str]] = []
    for target_name, (statement, field) in parsed_targets.items():
        shape = wrapper_shapes.get(field)
        keys = direct_keys.get(target_name, set())
        if shape is None or not keys:
            continue
        wrapper_name, parsed_rows = shape
        if all(
            all(key not in parsed and key in parsed[wrapper_name] for key in keys)
            for parsed in parsed_rows
        ):
            candidates.append((statement, field, wrapper_name))
    if len(candidates) != 1:
        return source, ()

    statement, field, wrapper_name = candidates[0]
    statement.value = ast.copy_location(
        ast.Subscript(
            value=statement.value,
            slice=ast.Constant(wrapper_name),
            ctx=ast.Load(),
        ),
        statement.value,
    )
    ast.fix_missing_locations(tree)
    repaired = ast.unparse(tree)
    repaired_fields, _ = source_argument_defaults(repaired)
    if repaired_fields != expected_fields:
        return source, ()
    return repaired, ((field, wrapper_name),)


def build_source_blueprint(
    *,
    objective: str,
    source: str,
    observed_snapshot: str = "",
    name_hint: str = "",
    description_hint: str = "",
    generated_contract: GeneratedToolContract | None = None,
) -> SourceBlueprint:
    """Build identity and a concrete test fixture from trusted runtime context.

    The model supplies source only. Names and fixture values come from the owner's
    immutable objective, an actual observed browser snapshot, or literal defaults
    in the generated function. No domain-specific patching is involved.
    """

    assignments = json_assignments(objective)
    fields, defaults = source_argument_defaults(source)
    if generated_contract is not None:
        contract_fields = set(generated_contract.tests[0].arguments)
        if fields != contract_fields:
            missing = sorted(contract_fields - fields)
            extra = sorted(fields - contract_fields)
            details = []
            if missing:
                details.append("ignored contract fields: " + ", ".join(missing))
            if extra:
                details.append("uncontracted source fields: " + ", ".join(extra))
            raise SourceBlueprintError(
                "generated source does not match the frozen runtime test contract ("
                + "; ".join(details)
                + ")"
            )
        name = _tool_name(
            objective,
            source,
            name_hint or generated_contract.name_hint,
        )
        description = (
            description_hint.strip()
            or generated_contract.description_hint.strip()
            or (
            f"Process bounded JSON input for the current PALADYN task with {name}."
            )
        )
        frozen_tests = tuple(
            (dict(case.arguments), dict(case.expected))
            for case in generated_contract.tests
        )
        return SourceBlueprint(
            name=name,
            description=description[:500],
            arguments=dict(frozen_tests[0][0]),
            expected=dict(frozen_tests[0][1]),
            oracle=generated_contract.provenance,
            tests=frozen_tests,
            final_arguments=dict(generated_contract.final_arguments),
        )
    objective_fields = {
        name
        for name in assignments
        if name
        not in {
            "expected",
            "test_expected",
            "version",
            "timeout_seconds",
        }
    }
    ignored_fixture_fields = sorted(objective_fields - fields)
    if ignored_fixture_fields:
        rendered = ", ".join(ignored_fixture_fields)
        raise SourceBlueprintError(
            "generated source ignores concrete objective fixture fields: "
            f"{rendered}. Read every fixture field from the arguments object; "
            "do not hard-code the expected result."
        )
    name = _tool_name(objective, source, name_hint)
    description = description_hint.strip() or (
        f"Process bounded JSON input for the current PALADYN task with {name}."
    )
    arguments: dict[str, Any] = {}
    missing: list[str] = []
    for field in sorted(fields):
        values = assignments.get(field)
        if values:
            arguments[field] = values[0]
        elif field == "snapshot_text" and observed_snapshot:
            arguments[field] = observed_snapshot
        elif field in defaults:
            arguments[field] = defaults[field]
        else:
            missing.append(field)
    if missing:
        rendered = ", ".join(missing)
        raise SourceBlueprintError(
            "PALADYN cannot derive a concrete test fixture for generated-tool "
            f"input fields: {rendered}. Supply exact JSON assignments in the "
            "objective or use literal defaults in source."
        )

    expected_values = assignments.get("expected")
    expected = (
        expected_values[0]
        if expected_values and isinstance(expected_values[0], dict)
        else None
    )
    return SourceBlueprint(
        name=name,
        description=description[:500],
        arguments=arguments,
        expected=expected,
        oracle=("owner_expected" if expected is not None else "deterministic_smoke"),
        tests=((arguments, expected),) if expected is not None else (),
    )


def schema_from_example(value: Any) -> dict[str, Any]:
    """Derive PALADYN's strict JSON-schema subset from one concrete value."""

    if isinstance(value, dict):
        properties = {
            str(name): schema_from_example(item) for name, item in value.items()
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        schemas = [schema_from_example(item) for item in value]
        return {
            "type": "array",
            "items": merge_example_schemas(schemas) if schemas else {"type": "null"},
        }
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if value is None:
        return {"type": "null"}
    return {"type": "string"}


def merge_example_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    if not schemas:
        return {"type": "null"}
    kinds = {str(schema.get("type", "")) for schema in schemas}
    if kinds <= {"integer", "number"}:
        return {"type": "number" if "number" in kinds else "integer"}
    if len(kinds) != 1:
        return schemas[0]
    kind = next(iter(kinds))
    if kind == "object":
        names = {
            name for schema in schemas for name in schema.get("properties", {})
        }
        properties: dict[str, Any] = {}
        for name in sorted(names):
            children = [
                schema["properties"][name]
                for schema in schemas
                if name in schema.get("properties", {})
            ]
            properties[name] = merge_example_schemas(children)
        required_sets = [set(schema.get("required", [])) for schema in schemas]
        required = sorted(set.intersection(*required_sets)) if required_sets else []
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    if kind == "array":
        return {
            "type": "array",
            "items": merge_example_schemas([schema["items"] for schema in schemas]),
        }
    return {"type": kind}


def _tool_name(objective: str, source: str, hint: str) -> str:
    candidates = [hint.strip()]
    explicit = _EXPLICIT_NAME.search(objective)
    if explicit:
        candidates.append(explicit.group(1))
    candidates.extend(
        match.group(1) for match in _SNAKE_NAME.finditer(objective)
    )
    for candidate in candidates:
        lowered = candidate.casefold()
        if (
            lowered not in _IGNORED_NAMES
            and re.fullmatch(r"[a-z][a-z0-9_]{2,63}", lowered)
        ):
            return lowered
    digest = hashlib.sha256((objective + "\0" + source).encode("utf-8")).hexdigest()
    return f"generated_{digest[:12]}"


def _literal_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Python <3.9 compatibility is harmless and keeps AST fixtures portable.
    if hasattr(ast, "Index") and isinstance(node, ast.Index):  # pragma: no cover
        return _literal_string(node.value)
    return ""
