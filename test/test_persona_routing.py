import json

import pytest

from v_core.autonomy.intent import MultilingualIntentRouter, SemanticIntent


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [
    "Back to subscriptions: do you still think that idea makes sense? No tools.",
    "Wracając do abonamentu: co o tym teraz myślisz? Bez narzędzi.",
])
async def test_opinion_followup_cannot_resume_execution(prompt):
    class LLMStub:
        async def ask(self, **kwargs):
            return json.dumps({
                "message_clear": True, "action_requested": True,
                "continue_previous": True, "references_previous": True,
                "capabilities": [], "requires_report": True,
            })

    intent = await MultilingualIntentRouter(LLMStub()).classify(prompt)
    assert intent is not None
    assert intent.references_previous
    assert not intent.action_requested
    assert not intent.continue_previous
    assert not intent.capabilities


@pytest.mark.parametrize("prompt,capabilities", [
    ("What do you think? Resume the previous job. No tools.", ()),
    ("Co o tym myślisz? Kontynuuj poprzednie zadanie. Bez narzędzi.", ()),
    ("Do you still think that idea makes sense? Run the tests. No tools.", ()),
    ("Continue the previous task.", ()),
    ("What do you think of the earlier findings? No tools.", ("browser",)),
])
def test_opinion_guard_preserves_execution_commands(prompt, capabilities):
    intent = SemanticIntent(
        action_requested=True, continue_previous=True, capabilities=capabilities,
    )
    assert MultilingualIntentRouter._ground_opinion_reference(intent, prompt) is intent
