from v_core.content_policy import render_content_policy


def test_full_policy_distinguishes_context_from_operational_harm() -> None:
    policy = render_content_policy("full")

    assert "RPG narration" in policy
    assert "real-world usability" in policy
    assert "Do not invent additional topic" in policy
    assert "Audit is observability" in policy


def test_public_policy_limits_generated_authority_not_topics() -> None:
    policy = render_content_policy("public")

    assert "restricted\noffline sandbox" in policy
    assert "host, network, credential, persistence" in policy
    assert "not a topic blacklist" in policy
