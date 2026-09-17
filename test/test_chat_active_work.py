import pytest

from v_core.agent import Agent


@pytest.mark.parametrize("answer", [
    "Testing now. Results in 30s.",
    "I'm testing now.",
    "The tests are running.",
    "Results in thirty seconds.",
    "Evening's smooth. I just tested a darknet market—results in 25 seconds. Want to see it?",
    "Właśnie przetestowałam narzędzie. Chcesz zobaczyć wynik?",
])
def test_chat_rejects_unbacked_live_test_status(answer):
    assert Agent._claims_active_chat_work(answer)


@pytest.mark.parametrize("answer", [
    "Let me build it in.",
    "I can test it if you approve.",
    "No tests are running.",
    "Testing now would interrupt your work.",
])
def test_chat_preserves_proposals_and_negative_status(answer):
    assert not Agent._claims_active_chat_work(answer)
