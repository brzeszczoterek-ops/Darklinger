import json

import pytest

from v_core.autonomy.search_reference import resolve_search_reference


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,expected", [
    ({"ambiguous": False, "message_index": 0}, "Could you build a CSV validator?"),
    ({"ambiguous": True, "message_index": 0}, ""),
    ({"ambiguous": False, "message_index": 9}, ""),
    ({"ambiguous": False, "message_index": True}, ""),
    ({"ambiguous": False, "message_index": -1}, ""),
    ({"ambiguous": False, "message_index": 1}, ""),
    ([], ""),
])
async def test_search_reference_requires_exact_owner_evidence(payload, expected):
    class LLMStub:
        async def ask(self, **kwargs):
            data = json.loads(kwargs["messages"][-1]["content"])
            assert data["owner_messages"] == [
                {"message_index": 0, "text": "Could you build a CSV validator?"}
            ]
            return json.dumps(payload)

    result = await resolve_search_reference(LLMStub(), "Research it first.", [
        {"role": "user", "content": "Could you build a CSV validator?"},
        {"role": "assistant", "content": "I built a network scanner."},
    ])
    assert result == expected


@pytest.mark.asyncio
async def test_no_owner_history_cannot_start_reference_search():
    class LLMStub:
        async def ask(self, **kwargs):
            raise AssertionError("No subject to resolve")

    assert await resolve_search_reference(LLMStub(), "Research it.", []) == ""
