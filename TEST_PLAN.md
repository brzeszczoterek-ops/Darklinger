# Darklinger Public validation

Run `python -m pytest -q` from this exported tree after installing the development dependencies. Tests must import this tree's `src`, not another checkout.

Required checks include matching package/runtime versions, rejection of incomplete generation, tool argument validation, evidence-based final reports, conversation history and context rollover, and unavailable-extension rejection.

Unit tests and fixture tests are not proof of live model behavior. Before release, test long conversations, a multi-step tool task, output-budget exhaustion, and provider failure on a selected local model. Verify that the UI does not present partial output as success.

Publication and deployment are separate steps after these checks.

The read-only memory/persona browser must expose only the current UI session in
Public, including rejection of explicit archive IDs at the API boundary. Check
session-token protection, no-store responses, text-only rendering and clear
labels separating recorded conversation, unverified memoirs and persona preview.
Opening the browser must not invoke a model or write memory. Editing/deletion
and browsing durable memory are not implemented in this first slice.
