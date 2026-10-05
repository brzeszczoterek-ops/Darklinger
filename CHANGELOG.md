# Darklinger Public changelog

## 3.11 — 2026-10-05

- Add local JPEG/PNG/WebP observation, OCR, PDF page reading and timestamped
  speech-file transcription, with authenticated ATTACH MEDIA and PERCEPTION
  controls in the English UI.
- Observe browser viewport pixels separately from DOM text. Preserve browser
  navigation prerequisites in read-only tasks.
- Preserve plain transcripts and single-page visual-reading observations from
  actual processor receipts. Conflicting or incomplete source data cannot be
  converted into a complete literal report by a model rewrite.
- Reject below-threshold push-to-talk recordings before speech recognition.
  F2 voice chat remains half duplex; continuous interruptible live chat is not
  part of this release.
- Show supervised tool progress and model qualification questions/responses.
- Strengthen grounded reports, memory provenance and literal identifiers, and
  include the green/black dragon branding assets.

The image/audio recognizers remain fallible. Environmental-sound recognition
is unavailable by default after a failed tone check. See `docs/perception.md`
for setup and the precise live-validation boundaries.

## 3.10 — 2026-10-04

- Align the runtime version with package metadata.
- Detect token-limited answers and retry eligible agent turns once with a larger output budget. A repeated truncation is reported as incomplete rather than successful.
- Add session memoir generation, speech settings, and outcome-based model routing evidence.
- Restrict distribution to explicitly listed files and Public-specific documentation.
- Add a read-only viewer for the current session's conversation, memoir and persona configuration. Previous sessions are not exposed by the Public browser API.
- Fix descriptive "test" / "testowej" mentions incorrectly forcing command execution.

Memoirs remain unverified model interpretations. When generation or grounding fails, the runtime preserves an explicit fallback and transcript. This source release does not guarantee universal model compatibility; memory/persona editing is not included.
