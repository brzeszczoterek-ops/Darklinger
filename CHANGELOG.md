# Darklinger Public changelog

## 3.10 — 2026-10-04

- Align the runtime version with package metadata.
- Detect token-limited answers and retry eligible agent turns once with a larger output budget. A repeated truncation is reported as incomplete rather than successful.
- Add session memoir generation, speech settings, and outcome-based model routing evidence.
- Restrict distribution to explicitly listed files and Public-specific documentation.
- Add a read-only viewer for the current session's conversation, memoir and persona configuration. Previous sessions are not exposed by the Public browser API.
- Fix descriptive "test" / "testowej" mentions incorrectly forcing command execution.

Memoirs remain unverified model interpretations. When generation or grounding fails, the runtime preserves an explicit fallback and transcript. This source release does not guarantee universal model compatibility; memory/persona editing is not included.
