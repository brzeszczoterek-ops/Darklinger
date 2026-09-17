from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import json
import re
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from ..inference import InferenceController, InferenceParameters
from .llm_config import LLMConfig, load_llm_config


_WORD = re.compile(r"[\w']+", re.UNICODE)
_TEXTUAL_TOOL_CALL = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>",
    re.DOTALL,
)


def repetition_start(text: str) -> int | None:
    """Return the character offset where a clear generation loop begins."""

    matches = list(_WORD.finditer(text.casefold()))
    words = [match.group(0) for match in matches]
    if len(words) < 6:
        return None

    # Two long repeated spans, three shorter phrases, or six repeated tokens are
    # strong enough evidence to stop. Longer patterns are checked first so the
    # first complete repeated block is retained rather than a tiny suffix.
    starts: list[int] = []
    for repetitions, minimum_block in ((2, 8), (3, 3), (6, 1)):
        maximum_block = min(96, len(words) // repetitions)
        for block_size in range(maximum_block, minimum_block - 1, -1):
            tail = words[-block_size:]
            if all(
                words[
                    -block_size * offset : -block_size * (offset - 1)
                    if offset > 1
                    else None
                ]
                == tail
                for offset in range(2, repetitions + 1)
            ):
                second_block = len(words) - block_size * (repetitions - 1)
                starts.append(matches[second_block].start())
                break
    return min(starts) if starts else None


def trim_repetition(text: str) -> str:
    start = repetition_start(text)
    if start is None:
        return text
    return text[:start].rstrip()


@dataclass(slots=True)
class LLMToolCall:
    """One model-requested action, decoded from the provider protocol."""

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    argument_error: str = ""
    raw_arguments: str = ""


@dataclass(slots=True)
class LLMResponse:
    """Provider-neutral assistant response used by DARKLINGER's executor."""

    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str = ""
    native_tools_enabled: bool = False


class LLM:

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self.config = load_llm_config()
        self.inference = InferenceController()
        self.client = self._new_client()
        # None means untested. A strict/older GGUF template may reject the
        # OpenAI tool schema; after one explicit provider rejection DARKLINGER
        # uses its documented textual compatibility protocol for that run.
        self._native_tools_supported: bool | None = None

    def set_edition(self, edition_name: str) -> None:
        """Bind request tuning to the installed Darklinger edition."""

        self.inference.edition_name = str(edition_name).strip().casefold()

    def _new_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self._api_key or os.getenv("V_CORE_API_KEY", "local"),
            timeout=float(os.getenv("V_CORE_TIMEOUT", "300")),
            # The OpenAI client retries transport timeouts by default. Against
            # one local llama.cpp slot that silently restarts the same long
            # generation multiple times and makes DARKLINGER look frozen. Agent
            # recovery is explicit and journalled, so transport retry stays off.
            max_retries=max(0, int(os.getenv("V_CORE_HTTP_RETRIES", "0"))),
        )

    async def reconfigure(self, config: LLMConfig | None = None) -> None:
        """Point all shared DARKLINGER components at a newly selected local model."""

        previous = self.client
        self.config = config or load_llm_config()
        self.client = self._new_client()
        self._native_tools_supported = None
        close = getattr(previous, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def respond(
        self,
        *,
        messages: list,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        artifact_generation: bool = False,
    ) -> LLMResponse:
        """Return prose and native tool calls without discarding either.

        ``ask`` intentionally remains a string API for chat, reflection, and
        older tests. Agent execution uses this richer boundary.
        """

        normalized = self._normalize_system_messages(messages)
        controller = getattr(self, "inference", None)
        tuning = (
            controller.parameters()
            if controller is not None
            else InferenceParameters(
                temperature=float(self.config.temperature),
                top_p=float(self.config.top_p),
            )
        )
        request: dict[str, Any] = {
            "model": self.config.model,
            "temperature": (
                tuning.temperature if temperature is None else float(temperature)
            ),
            "top_p": tuning.top_p,
            "presence_penalty": tuning.presence_penalty,
            "frequency_penalty": tuning.frequency_penalty,
            "seed": tuning.seed,
            "extra_body": {
                "top_k": tuning.top_k,
                "min_p": tuning.min_p,
                "typical_p": tuning.typical_p,
                "repeat_penalty": tuning.repeat_penalty,
            },
            "messages": normalized,
            "max_tokens": max_tokens or int(os.getenv("V_CORE_MAX_TOKENS", "512")),
        }
        tool_names = {
            str(item.get("function", {}).get("name", ""))
            for item in tools or []
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        if artifact_generation or tool_names & {
            "learning_create_tool",
            "learning_create_skill",
        }:
            # A generated artifact contains schemas, source, and deterministic
            # tests. On a local 2-4 token/s model that can legitimately take
            # several minutes even though an ordinary chat turn should still
            # fail quickly. The source-writing turn has no creation tool schema,
            # so its caller marks the phase explicitly; otherwise it would
            # inherit the short conversational HTTP timeout.
            request["timeout"] = max(
                float(os.getenv("V_CORE_TIMEOUT", "300")),
                float(os.getenv("V_CORE_ARTIFACT_TIMEOUT", "1200")),
            )
        native_requested = bool(tools) and self._native_tools_supported is not False
        if response_format is not None:
            request["response_format"] = response_format
        if native_requested:
            request["tools"] = tools
            request["tool_choice"] = tool_choice
            request["parallel_tool_calls"] = False

        try:
            response = await self.client.chat.completions.create(**request)
        except APIStatusError as error:
            # llama.cpp supports many chat templates. Some older templates
            # return a provider error when tools are present. Retry once as a
            # normal chat request; the agent prompt still exposes the legacy
            # JSON action protocol. Do not swallow ordinary provider failures.
            error_detail = f"{error} {getattr(error, 'body', '')}".casefold()
            template_rejection = error.status_code == 501 or any(
                marker in error_detail
                for marker in (
                    "function calling",
                    "jinja",
                    "template",
                    "tool choice",
                    "tool_choice",
                    "tools are not",
                    "tools unsupported",
                )
            )
            malformed_tool_arguments = error.status_code in {400, 422, 500} and any(
                marker in error_detail
                for marker in (
                    "failed to parse tool call arguments as json",
                    "parse tool call arguments",
                    "parse_error.101",
                    "missing closing quote",
                )
            )
            if (
                not native_requested
                or error.status_code not in {400, 422, 500, 501}
                or not (template_rejection or malformed_tool_arguments)
            ):
                raise

            # A malformed native call is rejected by llama.cpp before its
            # partial arguments reach DARKLINGER, so the normal argument validator
            # cannot repair it. Retry this turn once through the compact textual
            # compatibility protocol. Unlike a template rejection, this does not
            # disable native tools for later turns: the template supports tools;
            # this particular generation was merely malformed or truncated.
            if template_rejection:
                self._native_tools_supported = False
            request.pop("tools", None)
            request.pop("tool_choice", None)
            if malformed_tool_arguments and not template_rejection:
                request["messages"] = self._normalize_system_messages(
                    [
                        *normalized,
                        {
                            "role": "user",
                            "content": (
                                "Your previous native tool call was rejected because "
                                "its arguments were malformed or truncated JSON. Retry "
                                "the same next action now using exactly one compact "
                                "compatibility object: "
                                '{"tool":"<available_tool>","arguments":'
                                '{"<required_field>":"<schema_value>"}}. '
                                "Replace every placeholder. Output JSON only, include "
                                "every required field from the supplied schema, and "
                                "close every string and brace. Do not narrate the call."
                            ),
                        },
                    ]
                )
                request["temperature"] = min(float(tuning.temperature), 0.1)
            response = await self.client.chat.completions.create(**request)
            native_requested = False
        else:
            if native_requested:
                self._native_tools_supported = True

        choice = response.choices[0]
        message = choice.message
        content = trim_repetition(str(getattr(message, "content", "") or ""))
        decoded_calls: list[LLMToolCall] = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(call, "function", None)
            name = str(getattr(function, "name", "") or "").strip()
            raw = str(getattr(function, "arguments", "") or "")
            call_id = str(getattr(call, "id", "") or f"call_{index + 1}")
            arguments: dict[str, Any] = {}
            argument_error = ""
            try:
                parsed = json.loads(raw or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must decode to a JSON object")
                arguments = parsed
            except (json.JSONDecodeError, ValueError) as error:
                argument_error = f"{type(error).__name__}: {error}"
            decoded_calls.append(
                LLMToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    argument_error=argument_error,
                    raw_arguments=raw,
                )
            )

        if not decoded_calls and tool_names and content:
            compatibility_call = _decode_textual_tool_call(content, tool_names)
            if compatibility_call is not None:
                decoded_calls.append(compatibility_call)
                content = ""
                # The request still carried schemas, but llama.cpp surfaced the
                # model-family envelope as text instead of a provider tool_call.
                native_requested = False

        return LLMResponse(
            content=content,
            tool_calls=decoded_calls,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            native_tools_enabled=native_requested,
        )

    async def ask(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:

        if messages is None:

            messages = [
                {
                    "role": "system",
                    "content": self.config.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt or "",
                },
            ]

        response = await self.respond(
            messages=messages,
            max_tokens=max_tokens or int(os.getenv("V_CORE_MAX_TOKENS", "512")),
            temperature=temperature,
            response_format=response_format,
        )
        return response.content

    async def stream(
        self,
        *,
        messages: list,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        messages = self._normalize_system_messages(messages)
        controller = getattr(self, "inference", None)
        tuning = (
            controller.parameters()
            if controller is not None
            else InferenceParameters(
                temperature=float(self.config.temperature),
                top_p=float(self.config.top_p),
            )
        )
        response = await self.client.chat.completions.create(
            model=self.config.model,
            temperature=tuning.temperature,
            top_p=tuning.top_p,
            presence_penalty=tuning.presence_penalty,
            frequency_penalty=tuning.frequency_penalty,
            seed=tuning.seed,
            extra_body={
                "top_k": tuning.top_k,
                "min_p": tuning.min_p,
                "typical_p": tuning.typical_p,
                "repeat_penalty": tuning.repeat_penalty,
            },
            messages=messages,
            max_tokens=max_tokens or int(os.getenv("V_CORE_MAX_TOKENS", "512")),
            stream=True,
        )

        generated = ""
        async for chunk in response:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                candidate = generated + content
                loop_start = repetition_start(candidate)
                if loop_start is not None:
                    if loop_start > len(generated):
                        safe_tail = candidate[len(generated) : loop_start].rstrip()
                        if safe_tail:
                            yield safe_tail
                    break
                generated = candidate
                yield content

    @staticmethod
    def _normalize_system_messages(messages: list) -> list:
        """Keep exactly one system message at the beginning for GGUF templates."""

        system_parts: list[str] = []
        conversation: list[dict] = []
        for message in messages:
            copied = dict(message)
            if copied.get("role") == "system":
                content = str(copied.get("content", "")).strip()
                if content:
                    system_parts.append(content)
            else:
                if (
                    conversation
                    and copied.get("role") in {"user", "assistant"}
                    and conversation[-1].get("role") == copied.get("role")
                ):
                    previous = str(conversation[-1].get("content", ""))
                    current = str(copied.get("content", ""))
                    conversation[-1]["content"] = "\n\n".join(
                        part for part in (previous, current) if part
                    )
                else:
                    conversation.append(copied)

        if not system_parts:
            return conversation
        return [
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            },
            *conversation,
        ]


def _decode_textual_tool_call(
    text: str,
    allowed_names: set[str],
) -> LLMToolCall | None:
    """Promote one exact, allowed compatibility envelope to a runtime call."""

    stripped = text.strip()
    match = _TEXTUAL_TOOL_CALL.fullmatch(stripped)
    if match is not None:
        candidate = match.group(1)
    elif "<tool_call>" in stripped or "</tool_call>" in stripped:
        return None
    else:
        candidate = stripped
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) == {"name", "arguments"}:
        name = payload.get("name")
    elif set(payload) == {"tool", "arguments"}:
        name = payload.get("tool")
    else:
        return None
    arguments = payload.get("arguments")
    if (
        not isinstance(name, str)
        or name not in allowed_names
        or not isinstance(arguments, dict)
    ):
        return None
    raw = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return LLMToolCall(
        call_id="compatibility_call_1",
        name=name,
        arguments=arguments,
        raw_arguments=raw,
    )
