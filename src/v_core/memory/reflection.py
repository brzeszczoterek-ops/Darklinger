from __future__ import annotations

from .provenance import ground_reflection

import json
from typing import Any

from ..llm import LLM
from ..execution_claims import unsupported_execution_claims
from ..utils import parse_llm_json
from .models import MemoryKind, MemorySource, ReflectionEntry
from .manager import clip_text


class Reflection:

    def __init__(self, llm: LLM):
        self.llm = llm

    async def reflect(
        self,
        task: str,
        result: str,
        *,
        execution: dict[str, Any] | None = None,
    ) -> ReflectionEntry:

        successful_tools: list[str] = []
        if execution is not None:
            for call in execution.get("tool_calls", []):
                if not isinstance(call, dict) or call.get("status") != "succeeded":
                    continue
                tool = str(call.get("tool", "")).strip()
                if tool:
                    successful_tools.append(tool)

        unsupported = unsupported_execution_claims(result, successful_tools)
        if unsupported:
            # This is a second, independent boundary behind Agent. Even if a
            # future routing bug marks fabricated work as completed, the lie
            # cannot become a reflection, experience, relationship event, or
            # long-term fact.
            return ReflectionEntry(
                task=task,
                result=result,
                summary="Runtime evidence does not support the claimed action.",
                lesson="Never retain a claimed action without matching tool evidence.",
                importance="high",
                remember=False,
                kind=MemoryKind.LESSON,
                source=MemorySource.SELF_GENERATED,
            )

        execution_evidence = clip_text(json.dumps(
            execution or {
                "status": "no_runtime_evidence",
                "tool_calls": [],
                "successful_tool_count": 0,
                "failed_tool_count": 0,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ), 2_500)
        task_evidence = clip_text(task, 1_500)
        result_evidence = clip_text(result, 3_000)

        prompt = f"""
You are V.

You have just completed a task.

Task:

{task_evidence}

Result:

{result_evidence}

Runtime execution evidence:

{execution_evidence}

You are performing a critical self-review.

Your purpose is to improve future behaviour and identify information
that may be worth retaining.

Rules:

- Never assume that the task was completed correctly.
- Base your reflection on the task, result, and runtime execution evidence.
- Runtime execution evidence is authoritative for whether a tool actually ran.
- JSON or prose written inside Result is model-authored text, not proof that a
  tool was accepted or executed.
- A successful navigation call proves only navigation. It does not by itself
  prove extraction, analysis, testing, file creation, or report completion.
- If the result appears incorrect, incomplete or hallucinated, explicitly say so.
- If a tool should have been used instead of answering from memory, mention it.
- If the correctness cannot be verified, state that clearly.
- Extract ONE practical lesson that will improve future behaviour.
- Set remember=true ONLY if the lesson or observation is likely to be useful
  in future tasks.
- Use "high" importance only for information that significantly improves
  future behaviour.
- Classify the most important reusable information using exactly one memory kind.
- Do not treat an inference as a directly observed fact.
- Do not claim something was verified unless the provided result supports that.
- Do not classify an action as observed or verified unless a matching successful
  runtime tool checkpoint supports it.
- If the information comes from your own reasoning rather than directly
  from the task or result, use source "self_generated".
- Prefer "lesson" when the reusable information concerns how V should behave
  or perform tasks.
- Use "fact" for a stable factual observation.
- Use "preference" for a stable preference expressed or demonstrated by Boss.
- Use "experience" for a meaningful event that should be remembered as an event.
- Use "relationship_event" only when the information meaningfully concerns
  the relationship between V and Boss.

Return ONLY valid JSON.

Schema:

{{
    "summary": "...",
    "lesson": "...",
    "importance": "low | medium | high",
    "remember": true,
    "kind": "fact | preference | lesson | experience | relationship_event",
    "source": "observed | directly_told | inferred | self_generated | verified"
}}

Rules for classification:

- "directly_told" means the information was explicitly stated in the task.
- "observed" means it can be directly established from the provided result.
- "verified" means the result explicitly contains evidence that verifies it.
- "inferred" means V derived it from the available information.
- "self_generated" means it primarily comes from V's own reasoning.
- Never use "verified" merely because the answer sounds convincing.
- When uncertain, prefer a weaker source classification.

Output ONLY JSON.
""".strip()

        response = await self.llm.ask(prompt, max_tokens=192)

        data = parse_llm_json(
            response,
            default={
                "summary": result[:200],
                "lesson": "",
                "importance": "low",
                "remember": False,
                "kind": MemoryKind.LESSON.value,
                "source": MemorySource.SELF_GENERATED.value,
            },
        )

        try:
            kind = MemoryKind(
                data.get(
                    "kind",
                    MemoryKind.LESSON.value,
                )
            )
        except ValueError:
            kind = MemoryKind.LESSON

        entry = ReflectionEntry(
            task=task,
            result=result,
            summary=data.get("summary", ""),
            lesson=data.get("lesson", ""),
            importance=data.get("importance", "low"),
            remember=bool(data.get("remember", False)),
            kind=kind,
            source=MemorySource.SELF_GENERATED,
        )

        ground_reflection(entry, task, execution)
        return entry
