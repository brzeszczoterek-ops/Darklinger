"""Runtime-owned semantic test contracts for source-only generated tools.

The language model may interpret the owner's wording, but it cannot authorize
invented fixtures.  Every value in an accepted contract must be recoverable
from an exact quote in the current owner message.  The resulting contract is
then frozen before a different model is asked to write code.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any

from .utils import parse_llm_json


class GeneratedToolContractError(ValueError):
    """Natural-language fixtures could not be frozen without guessing."""


@dataclass(frozen=True, slots=True)
class GeneratedToolTest:
    arguments: dict[str, Any]
    expected: dict[str, Any]
    evidence_quote: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arguments": deepcopy(self.arguments),
            "expected": deepcopy(self.expected),
            "evidence_quote": self.evidence_quote,
        }


@dataclass(frozen=True, slots=True)
class GeneratedToolContract:
    tests: tuple[GeneratedToolTest, ...]
    final_arguments: dict[str, Any]
    final_evidence_quote: str
    provenance: str = "owner_text_semantic_extraction"
    name_hint: str = ""
    description_hint: str = ""
    specification: str = ""
    archetype: str = ""

    def source_examples(self) -> dict[str, Any]:
        """Project code-writing inputs without copying execution/provenance data.

        Keep every independent oracle, but send each only once. The complete
        immutable contract (including final invocation and grounding quotes)
        remains runtime-owned and is still used for validation and persistence.
        """
        return {"tests": [
            {"arguments": deepcopy(case.arguments), "expected": deepcopy(case.expected)}
            for case in self.tests
        ]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tests": [case.to_dict() for case in self.tests],
            "final_arguments": deepcopy(self.final_arguments),
            "final_evidence_quote": self.final_evidence_quote,
            "provenance": self.provenance,
            "name_hint": self.name_hint,
            "description_hint": self.description_hint,
            "specification": self.specification,
            "archetype": self.archetype,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GeneratedToolContract":
        return cls(
            tests=tuple(
                GeneratedToolTest(
                    arguments=deepcopy(item["arguments"]),
                    expected=deepcopy(item["expected"]),
                    evidence_quote=str(item["evidence_quote"]),
                )
                for item in value.get("tests", [])
            ),
            final_arguments=deepcopy(value.get("final_arguments", {})),
            final_evidence_quote=str(value.get("final_evidence_quote", "")),
            provenance=str(value.get("provenance", "owner_text_semantic_extraction")),
            name_hint=str(value.get("name_hint", "")),
            description_hint=str(value.get("description_hint", "")),
            specification=str(value.get("specification", "")),
            archetype=str(value.get("archetype", "")),
        )


_CREATE_WEB_TRAFFIC_TOOL = re.compile(
    r"(?:\b(?:create|build|implement|generate)\w*\b.{0,180}\btool\w*\b|"
    r"\b(?:stworz|stwórz|utworz|utwórz|zbuduj|wygeneruj|zaimplementuj)\w*\b"
    r".{0,180}\b(?:narzedzi|narzędzi)\w*\b)",
    re.IGNORECASE | re.DOTALL,
)
_WEB_TRAFFIC = re.compile(
    r"(?:\b(?:monitor|observe|inspect|summari[sz]e)\w*\b.{0,100}"
    r"\b(?:http\s+requests?|network\s+traffic|traffic)\b|"
    r"\b(?:obserw|monitor|sprawdz)\w*\b.{0,100}\bruch\w*\b)",
    re.IGNORECASE | re.DOTALL,
)
_WEB_TARGET = re.compile(
    r"\b(?:web(?:site|page)?|site|page|stron\w*|witryn\w*|www)\b",
    re.IGNORECASE,
)
_DELEGATES_TEST_TARGET = re.compile(
    r"\b(?:any\s+(?:safe\s+)?(?:website|site|page)|pick\w*\s+(?:one|a\s+site)|"
    r"choose\w*\s+(?:one|a\s+site)|your(?:\s+own)?\s+choice|"
    r"jakiejkolwiek\s+stron\w*|dowoln\w*\s+stron\w*|"
    r"sam[ao]?\s+(?:wybierz|wybiera|podejm|podj[eę][łl])\w*|"
    r"(?:podejm|podj[eę][łl])\w*\s+(?:samodzieln\w*\s+)?decyzj\w*)\b",
    re.IGNORECASE,
)


def autonomous_web_traffic_contract(
    prompt: str,
    *,
    semantic_archetype: str = "",
    delegates_test_target: bool = False,
) -> GeneratedToolContract | None:
    """Return an independent safe oracle when Boss delegates the test target.

    Generated tools remain offline.  This archetype therefore analyzes supplied
    HAR data; the browser phase may visit the harmless target, but PALADYN never
    pretends that an accessibility snapshot exposes server-side visitor logs.
    The expected answers are runtime-owned and fixed before candidate source is
    generated, so failed approaches remain meaningful instead of self-grading.
    """

    text = str(prompt or "")
    lexical_match = all(
        pattern.search(text)
        for pattern in (
            _CREATE_WEB_TRAFFIC_TOOL,
            _WEB_TRAFFIC,
            _WEB_TARGET,
            _DELEGATES_TEST_TARGET,
        )
    )
    semantic_match = (
        semantic_archetype == "client_web_traffic_monitor"
        and delegates_test_target
    )
    if not lexical_match and not semantic_match:
        return None

    def har(rows: list[tuple[str, int, float]]) -> dict[str, str]:
        return {
            "har_json": json.dumps(
                {
                    "log": {
                        "version": "1.2",
                        "entries": [
                            {
                                "request": {"url": url, "method": "GET"},
                                "response": {"status": status},
                                "time": duration,
                            }
                            for url, status, duration in rows
                        ],
                    }
                },
                separators=(",", ":"),
            )
        }

    def expected(
        count: int,
        domains: list[str],
        errors: int,
        duration: float,
    ) -> dict[str, Any]:
        return {
            "requests_count": count,
            "domains": domains,
            "error_count": errors,
            "total_time_ms": duration,
        }

    cases = (
        GeneratedToolTest(
            har(
                [
                    ("https://example.com/", 200, 12.5),
                    ("https://example.com/app.js", 404, 7.5),
                    ("https://cdn.example.com/theme.css", 200, -1.0),
                ]
            ),
            expected(3, ["cdn.example.com", "example.com"], 1, 20.0),
            "runtime-owned independent HAR fixture: mixed responses",
        ),
        GeneratedToolTest(
            har([]),
            expected(0, [], 0, 0.0),
            "runtime-owned independent HAR fixture: empty trace",
        ),
        GeneratedToolTest(
            har([("https://example.com/redirect", 302, 4.0)]),
            expected(1, ["example.com"], 0, 4.0),
            "runtime-owned independent HAR fixture: redirect",
        ),
    )
    return GeneratedToolContract(
        tests=cases,
        final_arguments=har(
            [
                ("https://example.com/", 200, 8.0),
                ("https://example.com/favicon.ico", 503, 2.0),
            ]
        ),
        final_evidence_quote=(
            "runtime-selected bounded demonstration trace for https://example.com"
        ),
        provenance="runtime_independent_har_oracle",
        name_hint="har_traffic_summary",
        description_hint=(
            "Summarize client-visible HTTP request traffic from supplied HAR JSON."
        ),
        specification=(
            "Parse arguments['har_json'] as HAR 1.2 JSON and read log.entries. "
            "Return requests_count for all entries; domains as sorted unique "
            "lowercase request URL hostnames without ports; error_count for "
            "response status at least 400; and total_time_ms as the sum of "
            "nonnegative entry time values, ignoring negative durations. This "
            "tool analyzes supplied client-side request data only and never "
            "claims access to server-side visitor analytics."
        ),
        archetype="client_har_summary",
    )


_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "generated_tool_test_contract",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ready", "ambiguous", "absent"]},
                "tests": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "arguments": {"type": "object"},
                            "expected": {"type": "object"},
                            "evidence_quote": {"type": "string", "maxLength": 500},
                        },
                        "required": ["arguments", "expected", "evidence_quote"],
                        "additionalProperties": False,
                    },
                },
                "final_arguments": {"type": "object"},
                "final_evidence_quote": {"type": "string", "maxLength": 500},
                "reason": {"type": "string", "maxLength": 300},
            },
            "required": [
                "status", "tests", "final_arguments", "final_evidence_quote", "reason"
            ],
            "additionalProperties": False,
        },
    },
}


_SYSTEM_PROMPT = """
You extract a test contract for a newly requested software tool. Understand the
owner's message in any language, including mixed-language speech transcription.
Do not write code and do not solve the examples. Extract only examples whose
input AND expected output are explicitly stated by the owner, plus the separate
input on which the owner wants the finished tool invoked.

Return JSON only:
{"status":"ready|ambiguous|absent","tests":[{"arguments":{},"expected":{},
"evidence_quote":"exact current-message quote"}],"final_arguments":{},
"final_evidence_quote":"exact current-message quote","reason":""}

Use short stable snake_case field names consistently. A test quote must contain
every value placed in that test. The final quote must contain every final input
value. Quotes must be exact substrings of current_user_message. Never infer a
missing expected value, invent an example, copy an output produced by candidate
code, or turn the final invocation into a test. If relationships conflict or a
value cannot be assigned unambiguously, return ambiguous. If no explicit
input/output example exists, return absent. JSON only, no markdown.
""".strip()


def has_natural_fixture_candidate(prompt: str) -> bool:
    """Cheap language-neutral gate before spending a model turn.

    Three scalar literals are the minimum useful shape: one input/output pair
    plus a separate final invocation.  The semantic extractor decides their
    roles; this gate does not interpret words or relationships.
    """

    numbers = re.findall(r"(?<![\w.])-?(?:0|[1-9]\d*)(?:\.\d+)?(?![\w.])", prompt)
    quoted = re.findall(r"(?<!\\)(?:\"(?:\\.|[^\"\\])+\"|'(?:\\.|[^'\\])+')", prompt)
    return len(numbers) + len(quoted) >= 3


def _leaf_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaf_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_values(child)
    else:
        yield value


def _value_is_grounded(value: Any, quote: str) -> bool:
    if isinstance(value, str):
        return value.casefold() in quote.casefold()
    if value is None:
        token = "null"
    elif isinstance(value, bool):
        token = "true" if value else "false"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        token = json.dumps(value, allow_nan=False)
    else:
        return False
    return re.search(rf"(?<![\w.]){re.escape(token)}(?![\w.])", quote, re.IGNORECASE) is not None


def _grounded_object(value: Any, quote: str, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise GeneratedToolContractError(f"{label} must be a non-empty JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GeneratedToolContractError(f"{label} contains non-JSON data") from error
    if len(encoded.encode("utf-8")) > 50_000:
        raise GeneratedToolContractError(f"{label} is too large")
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", str(key)) for key in value):
        raise GeneratedToolContractError(f"{label} contains an invalid field name")
    if not all(_value_is_grounded(item, quote) for item in _leaf_values(value)):
        raise GeneratedToolContractError(f"{label} contains a value absent from its evidence quote")
    return deepcopy(value)


def parse_generated_tool_contract(response: str, prompt: str) -> GeneratedToolContract:
    payload = parse_llm_json(response, default={})
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        reason = str(payload.get("reason", "")) if isinstance(payload, dict) else ""
        raise GeneratedToolContractError(reason or "no unambiguous owner-specified test contract")
    raw_tests = payload.get("tests")
    if not isinstance(raw_tests, list) or not 1 <= len(raw_tests) <= 8:
        raise GeneratedToolContractError("the contract must contain between one and eight tests")

    tests: list[GeneratedToolTest] = []
    input_keys: tuple[str, ...] | None = None
    output_keys: tuple[str, ...] | None = None
    seen: dict[str, str] = {}
    for index, raw in enumerate(raw_tests, start=1):
        if not isinstance(raw, dict):
            raise GeneratedToolContractError(f"test {index} is not an object")
        quote = str(raw.get("evidence_quote", ""))
        if not quote or quote not in prompt:
            raise GeneratedToolContractError(f"test {index} quote is not exact owner text")
        arguments = _grounded_object(raw.get("arguments"), quote, label=f"test {index} arguments")
        expected = _grounded_object(raw.get("expected"), quote, label=f"test {index} expected")
        current_input_keys = tuple(sorted(arguments))
        current_output_keys = tuple(sorted(expected))
        if input_keys is None:
            input_keys, output_keys = current_input_keys, current_output_keys
        elif current_input_keys != input_keys or current_output_keys != output_keys:
            raise GeneratedToolContractError("test cases use inconsistent input or output fields")
        argument_digest = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        expected_digest = json.dumps(expected, sort_keys=True, ensure_ascii=False)
        previous = seen.setdefault(argument_digest, expected_digest)
        if previous != expected_digest:
            raise GeneratedToolContractError("the same input has conflicting expected outputs")
        tests.append(GeneratedToolTest(arguments, expected, quote))

    final_quote = str(payload.get("final_evidence_quote", ""))
    if not final_quote or final_quote not in prompt:
        raise GeneratedToolContractError("final invocation quote is not exact owner text")
    final_arguments = _grounded_object(
        payload.get("final_arguments"), final_quote, label="final arguments"
    )
    if (
        tuple(sorted(final_arguments)) != input_keys
        and input_keys is not None
        and len(input_keys) == 1
        and len(final_arguments) == 1
    ):
        # In ordinary speech the examples may say "a number gives ..." while
        # the final invocation names the same sole input ``n``.  With one field
        # there is no competing mapping to guess between, so canonicalize by
        # arity while retaining the separately grounded value.  Multi-field
        # aliases remain ambiguous and fail below.
        final_arguments = {input_keys[0]: next(iter(final_arguments.values()))}
    if tuple(sorted(final_arguments)) != input_keys:
        raise GeneratedToolContractError("final invocation fields differ from test input fields")
    return GeneratedToolContract(tuple(tests), final_arguments, final_quote)


async def extract_generated_tool_contract(llm: Any, prompt: str) -> GeneratedToolContract:
    explicit = explicit_generated_tool_contract(prompt)
    if explicit is not None:
        return explicit
    response = await llm.ask(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"current_user_message": prompt}, ensure_ascii=False),
            },
        ],
        max_tokens=512,
        temperature=0.0,
        response_format=_RESPONSE_FORMAT,
    )
    return parse_generated_tool_contract(str(response), prompt)


def explicit_generated_tool_contract(prompt: str) -> GeneratedToolContract | None:
    """Freeze an owner-supplied JSON contract without model transcription."""
    markers = list(re.finditer(r"(?<![\w.])tool_contract\s*=\s*", prompt))
    if not markers:
        return None
    if len(markers) != 1:
        raise GeneratedToolContractError("supply exactly one tool_contract")
    try:
        value, end = json.JSONDecoder().raw_decode(prompt[markers[0].end():])
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GeneratedToolContractError("tool_contract must be finite JSON") from error
    if not isinstance(value, dict) or len(encoded) > 100_000:
        raise GeneratedToolContractError("tool_contract must be a bounded object")
    cases, final = value.get("tests"), value.get("final_arguments")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise GeneratedToolContractError("tool_contract requires one to eight tests")
    quote = prompt[markers[0].end():markers[0].end() + end]
    tests = []
    input_keys = output_keys = None
    seen = {}
    for case in cases:
        if not isinstance(case, dict):
            raise GeneratedToolContractError("each test must be an object")
        arguments, expected = case.get("arguments"), case.get("expected")
        if not isinstance(arguments, dict) or not arguments or not isinstance(expected, dict) or not expected:
            raise GeneratedToolContractError("each test needs nonempty arguments and expected output")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key) for key in arguments):
            raise GeneratedToolContractError("invalid input field name")
        keys, outputs = tuple(sorted(arguments)), tuple(sorted(expected))
        if input_keys is not None and (keys != input_keys or outputs != output_keys):
            raise GeneratedToolContractError("tests use inconsistent input or output fields")
        input_keys, output_keys = keys, outputs
        digest = json.dumps(arguments, sort_keys=True)
        if digest in seen and seen[digest] != expected:
            raise GeneratedToolContractError("the same input has conflicting expected outputs")
        seen[digest] = expected
        tests.append(GeneratedToolTest(deepcopy(arguments), deepcopy(expected), quote))
    if not isinstance(final, dict) or tuple(sorted(final)) != input_keys:
        raise GeneratedToolContractError("final invocation fields differ from test inputs")
    return GeneratedToolContract(tuple(tests), deepcopy(final), quote, "owner_structured_contract")


def source_generation_objective(prompt: str, contract: GeneratedToolContract) -> str:
    """Elide only a verified duplicate JSON contract, never surrounding prose.

    Natural-language objectives and unfamiliar payloads are retained verbatim.
    This is structural JSON handling, not language-specific intent recognition.
    """
    try:
        if explicit_generated_tool_contract(prompt) != contract:
            return prompt
    except GeneratedToolContractError:
        return prompt
    marker = re.search(r"(?<![\w.])tool_contract\s*=\s*", prompt)
    if marker is None:
        return prompt
    value, end = json.JSONDecoder().raw_decode(prompt[marker.end():])
    # Unknown extension fields could carry semantics we do not project.
    if set(value) != {"tests", "final_arguments"} or any(
        set(case) != {"arguments", "expected"} for case in value["tests"]
    ):
        return prompt
    return (
        prompt[:marker.start()]
        + "[Frozen examples supplied separately; final invocation is runtime-owned.]"
        + prompt[marker.end() + end:]
    )
