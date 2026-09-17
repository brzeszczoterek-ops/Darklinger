# Local tool checks — bounded coverage

## Implemented

- Metadata review remains separate from functional checks.
- The semantic router can select `tool_self_test` for functional health checks.
- Prepared local fixtures cover `read_file`, `write_file`, `list_directory`,
  `create_directory`, `edit_file`, `move_file`, `search_files`, `directory_tree`,
  and `get_file_info`.
- Calls bypass automatic recovery. Generated or edition-specific overrides of
  those names are skipped rather than executed as trusted built-ins.
- Each executed check has a time limit and a runtime trace entry. Success is
  checked against fixture data, not the model's assertion of success.
- Other tools are reported as `not_tested`, with no live/network fallback.
- `web_search` and `web_read` have deterministic browser-snapshot fixtures.
  They verify result parsing, remembered search evidence, rejection of an unseen
  substitute URL, and reading the selected result. They do not contact DNS or
  the public Internet and therefore report `live_network: false`.
- An explicit tool name narrows the selection. Recognized read-only requests
  skip all four mutating providers. Temporary fixture preparation itself writes
  files, even for read-only provider checks.
- Temporary files are removed after completion, errors, or cancellation.
- Directory listing verification requires the exact fixture entry, accepting
  plain lines and MCP `[FILE]` text blocks. Merely mentioning the filename,
  returning a similarly named file, extra entries, or the wrong type does not
  pass. Normalization receives a copy, so in-place path changes cannot alter
  the expected arguments used by the fixture boundary check.
- Every tool uses its own temporary directory. Nested argument copies protect
  edit fixtures from in-place rewriting, while case identifiers state what was
  actually checked. The metadata case checks UTF-8 byte size and file type, not
  every timestamp or permission. Move/edit checks inspect the resulting files,
  rather than trusting success messages. Fixture setup failures are untested,
  execution failures are failed, and cancellation propagates with cleanup.

This is **temporary workspace isolation, not an OS sandbox**. These checks do
not establish correctness for all inputs, exercise every failure mode, audit
implementations, or certify generated tools. They do not repair tools.

## Failed action grounding

After semantic classification fails, an action verb alone must not authorize a
model-generated completion report. If the resulting contract has no execution
route after inheritance and owner constraints, PALADYN returns a clarification
and explicitly states that no tools, repairs, or tests ran.

Regression coverage includes the previously observed Polish voice transcript
with real lexical contract extraction, plus structural tests with other wording.
This is a guard on an invalid execution state, not a list of forbidden phrases.

## Verification on 2026-09-08

- Full suite: 809 passed.
- Agent integration: three recorded fixture calls; unsupported tool skipped;
  temporary fixture cleanup verified.
- Live UI run: automatic routing switched from Mythos to AgenticQwen. The
  `tool_self_test` path executed three filesystem checks successfully and
  reported 41 other tools as untested. The runtime checkpoint records exactly
  three successful calls, not a model-only report.
- Latest fixture runner was also exercised directly against the actual local
  MCP filesystem server: read, write, and listing passed. No network tools ran.
- A live semantic-classification probe of the previous malformed repair request
  reproduced `current_message_grounding`, with no accepted capabilities.
- An isolated agent run using the live local model then returned the expected
  clarification and explicitly denied performing repairs/tests. Tool access
  was disabled in that rejection probe; no tool discovery was attempted.

### Follow-up: result verification and fixture boundaries

- Before the correction, six new regression cases failed: five misleading
  directory results were accepted and an in-place argument rewrite escaped
  the fixture argument comparison. These were local synthetic providers.
- After the correction: 20 focused tests and the full suite of 818 tests pass.
- A fresh run against the installed MCP filesystem server, restricted to a
  temporary directory, passed read, write, and directory-listing checks with
  the corrected runner. Fixture cleanup was verified. No model, network tools,
  or user-data operations were used in that follow-up.
- This follow-up did not repeat the live model routing test or publish changes.
- Publication preparation: the public export includes this scope document;
  its full test suite passed 767 tests. The seven edition/export boundary
  checks also passed. These results do not expand functional tool coverage.

## Still outstanding

Universal claim-to-source verification, contradictory/stale source handling,
functional fixtures for other tools, and robustness across models/languages are
not established by these checks. Version 3.9 packages the bounded improvements
documented here; it does not claim that every tool works or that the broader
verification milestone is complete.

## Expanded live provider checks

All nine local fixture cases passed against the installed MCP filesystem server
in a fresh temporary directory. Cleanup was verified. This tests the actual
provider and wrapper, not just a stub, but does not repeat model routing.
After the expansion, the complete suites passed: **842 Full / 791 Public**.
The integration test also checks that the nine-tool report has nine recorded
executions. Its intent router is a stub, not a fresh live-model qualification.

### Opt-in public network checks

`scripts/check_network_tools.py` is a separate developer probe. It requires
`--allow-public-network` and `--mcp-cli PATH` pointing to an already installed
Playwright MCP entrypoint, with the project installed or `PYTHONPATH=src`.
It starts a headless, isolated Firefox profile, closes it afterward, and does
not install software or attach to an existing browser session.

Five real network cases passed on 2026-09-08:

1. `web_read` returned the expected title, URL, and documentation content from
   `https://example.com/`.
2. A separate `browser_snapshot` matched that page.
3. Reading the reserved `.invalid` test domain raised a controlled browser error.
4. `web_search` returned six URLs including an official IANA page.
5. `web_read` opened an IANA URL found by that search and returned relevant content.

The fixed-target probe was repeated on 2026-09-16 after integrating the same
search/read evidence boundary with automatic functional checks. All five live
cases passed again. The complete suites then passed **1035 Full / 962 Public**.

Offline regression tests additionally reject missing content, empty search
results, false success for unavailable pages, and timeouts; cancellation is
not swallowed. Offline browser fixtures do not count as live network evidence.

The same evidence boundary now applies to automatic functional self-tests:
`web_search` may establish a candidate URL, while `web_read` must use that
recorded candidate and retrieve page content before PALADYN can present it as
observed. Passing the deterministic fixtures proves only this parser and
grounding path. Current reachability still requires the separate opt-in live
probe.

These checks do not cover accounts, authentication, form submissions, Tor,
generated tools, or scanning. The automatic `tool_self_test` path remains local;
network checks are never an implicit fallback. External failures need diagnosis
and do not alone prove a tool defect. Passing this probe is not a promise about
arbitrary sites, rate limits, or future availability. Version 3.9 publication
was approved for this bounded package; the larger milestone remains deferred.
