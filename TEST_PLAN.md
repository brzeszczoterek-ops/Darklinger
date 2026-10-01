# Darklinger Public validation

Run `python -m pytest -q` from this exported tree after installing the development dependencies. Tests must import this tree's `src`, not another checkout.

Required checks include matching package/runtime versions, rejection of incomplete generation, tool argument validation, evidence-based final reports, conversation history and context rollover, and unavailable-extension rejection.

Unit tests and fixture tests are not proof of live model behavior. Before release, test long conversations, a multi-step tool task, output-budget exhaustion, and provider failure on a selected local model. Verify that the UI does not present partial output as success.

Publication and deployment are separate steps after these checks.
