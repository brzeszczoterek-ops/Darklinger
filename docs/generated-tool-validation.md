# Generated tool validation: evidence and limits

The validator executes supplied examples in the existing offline sandbox. This
change improves the evidence report; it does not loosen source policy, enable
network access, or expand activation permissions.

## Prototype activation gate

Source-only trials without independently supplied expectations still run in the
offline sandbox, but their output-derived smoke/sensitivity cases qualify only a
prototype. Its record stays `validated` (technical checks), with
`qualification: prototype_only` and `activation_eligible: false`. It is not
activated, listed as executable, or usable by name. Manual activation and restart
do not bypass this gate. Legacy active bundles bearing the reserved runtime smoke
test markers are also hidden from execution; their historical records remain.

The agent ends this attempt with an incomplete-task report and records the
qualification failure. It does not repeatedly generate code to satisfy technical
checks alone. This gate does not determine arbitrary task correctness: supplied
expectations still need independent justification, and passing finite examples
does not certify unseen inputs. No new networking or CAPTCHA functionality is added.

`test/test_prototype_activation.py` exercises real offline sandbox execution and
the agent loop with a scripted model, including an input-dependent fake file
inspection report, restart/direct-call blocking, an explicit constant `pi`
contract, and stopping without another generation. This is not a live LLM trial.

Prototype feedback uses a concise English V-style base message and the existing
response-language preference/override path, not the language of the incoming
prompt alone. This presentation step has no tools or prior conversation and is
bounded to 15 seconds. On rendering failure it retains the grounded English
message and records the requested/fallback languages. It cannot change the
artifact state or resume execution. Existing claim checks reject detectable
invented actions, but do not prove arbitrary multilingual paraphrase fidelity.
`test/test_prototype_voice.py` checks configured Polish, the English default,
fallback, cancellation, and an unsupported action with scripted responses.

## Persisted result

### Failed creation circuit breaker

The agent counts consecutive failures without **verified progress**, independently
of a non-renewable budget of eight creation requests and 600 seconds from the
first creation request. Tool, skill, snapshot-extractor and repair-adapter creation
share the total budget. Execution-ledger history survives context rollovers and
batch continuation; changing a model does not erase it. The deadline is enforced
around subsequent model responses and tool calls; presentation may take up to
another 15 seconds. Cancellation requests stop local coroutines; this does not
prove termination of an arbitrary external server process.

Progress renewal is deliberately narrower than arbitrary generated tools: only
source-only, import-free local candidates with input fixtures and expected JSON
supplied in the immutable user objective are eligible. The MCP executor provides
a single-use source/contract-bound receipt from actual offline Bubblewrap
validation. Model-written tool output, case names, changed source defaults,
model-supplied manifests/tests, and self-derived expectations do not qualify.
The source builder currently supplies one independent test in this path.

Reaching a new highest stage for that same contract (fixture validation,
execution, comparison, passing) resets the stagnation counter, not total attempts.
Regressing and then recovering to an old high-water mark is not new progress.
Changing the input, expected output or thereby derived schemas never resets it.

At three stagnant failures, a previously unseen normalized AST can receive one
probe; a new executor-observed model identity can receive one additional probe.
Each allowance is one-shot across the whole task, including progress resets.
Comments, formatting, docstrings and changed string wording are normalized away.
Structural novelty is a heuristic for admitting a probe, **not proof** that the
change fixes the cause. It can miss useful changes or admit an unhelpful change;
the actual test decides whether progress occurred. No model switch is initiated
by this accounting layer. New candidates without comparable trusted receipts
retain the conservative three-failure stop.

The terminal report marks the task incomplete and preserves failure evidence.
No new operation is admitted after budget exhaustion, including within a batch.
Root-cause diagnosis, universal clarification, independent oracle discovery and
automatic lesson promotion are not implemented by this change. Permissions,
activation qualification and sandbox restrictions are unchanged.
`test/test_creation_failure_budget.py` and `test/test_creation_progress.py` use
scripted models and the real local creation handler, including resumed ledgers,
arithmetic validation progress, bounded probes and deadline cancellation. These
are not live-model generation trials.

Source-draft validation is part of that same accounting even before the builder
is called. The trace records it as `runtime_validate_generated_tool_source`, not
as a fabricated provider execution. Repeated identical valid fenced modules are
deduplicated and accepted; two distinct valid modules remain ambiguous and are
rejected. The source phase allows up to 1536 generated tokens and tells the model
to implement only the reusable `run(arguments)` operation. Tests, activation and
the final invocation remain runtime-owned phases. Source generation receives an
isolated phase prompt and bounded runtime rejection feedback, not stale assistant
status claims from the conversation.

An empty source response is recorded as a rejected draft, with the provider's
finish reason when available, and consumes the same repair budget. It cannot
skip failure accounting or be reported as an exhausted 32-step batch. Scripted
regressions cover both recovery after an empty answer and stopping after three
empty drafts without invoking a builder.

Source-only requests explicitly select the artifact HTTP timeout, although they
expose no callable tool schema. The first awaited operation has a 600-second
executor bound. Once a creation attempt is recorded, later operations use the
remaining shared budget described above. This prevents the ordinary conversation
HTTP timeout from aborting a slow source response while still bounding execution.

The source parser can remove standalone bare identifiers preceding the module
(observed on a live model as a stray `e`). It preserves lines with other
statements and identifiers inside the module. Duplicate `run` definitions,
including asynchronous variants, are rejected rather than recovered by selecting
the last definition. Syntax-valid rejected modules cannot use prose recovery to
discard earlier code. Source policy and sandbox validation still run afterward.

Each test retains its index, name, status (`passed`, `failed`, or `not_run`),
whether execution was attempted, and the stage reached:

- `fixture_validation`: input and expected-output schema checks;
- `execution`: the sandbox invocation and output-schema checks;
- `comparison`: comparison against the expected result.

If a test fails, earlier successful comparisons stay in the persisted report,
and remaining cases are explicitly unrun. Static-policy rejection marks all
cases unrun. Failure still rejects the artifact and prevents its activation.
An already active prior version is not replaced by a rejected revision.

Expected/actual value digests are comparison receipts, not semantic proof or
digital signatures. They avoid duplicating entire example payloads in the
report. Execution errors have no actual-value receipt. The existing bounded
error diagnostic may still contain a value mismatch.

## Interpretation

Case names do not prove who supplied the expected answer. Merely naming a test
`owner-specified semantic oracle` no longer grants that validation strength.
Input-sensitivity labeling additionally requires distinct executed inputs and
distinct observed outputs. Neither result establishes domain correctness for
unseen inputs. Reports explicitly mark broader semantic correctness as not
independently established, and the deterministic completion summary retains
that limitation.

Input variation is not a universal requirement for useful tools. An explicit
contract may intentionally return a constant: the regression suite validates,
activates, and executes a no-input tool returning the literal text `pi`.
This does not certify endless execution or audio output, and does not change
the source-only builder's requirements for inferred tests.

## Verification

`test/test_generated_validation_report.py` uses harmless integer arithmetic in
the actual offline Bubblewrap sandbox. It covers a complete stage/validate/
activate/execute cycle, a wrong expected answer after a passing case, runtime
failure, timeout, static rejection, invalid expected-output schema, preserving
the prior active revision, and misleading case names. It also verifies that
temporary execution directories are cleaned up.

These are supplied-source tests, not a fresh end-to-end qualification of V's
code generation, task routing, or proposed examples. Independent test-oracle
provenance, coverage of generated skills beyond trigger matching, and domain
correctness remain separate work. The existing isolation is unchanged.

On 2026-09-09 an isolated live AgenticQwen trial created `podwoj_liczbe` from a
source-only response, checked the owner-supplied fixture `n = 2` against
`{"wynik": 4}` in Bubblewrap, activated the artifact, and then executed the
active tool with `n = 9`, producing `{"wynik": 18}`. The task completed after
that generated-tool call without the formerly redundant sandbox command. A
separate natural-language-only trial stopped after three failed drafts without
activation; it confirms bounded failure behavior, not universal extraction of
spoken test fixtures.

## Natural-language test contracts

DARKLINGER now separates semantic fixture extraction from source generation. When
Boss supplies explicit input/output examples and a separate final invocation in
ordinary language, a semantic pass normalizes them into a runtime-owned
contract. The parser is not tied to Polish or English keywords. It accepts only
JSON values grounded in exact spans of the current message, requires consistent
input and output fields, rejects conflicting outputs for the same input, and
stops for clarification when the contract is absent or ambiguous.

The frozen contract is not shown to the source model as editable test data. The
source model receives the required input/output field names and read-only
examples. DARKLINGER
runs every frozen example independently in Bubblewrap and activation requires
all comparisons to pass. Only afterward does the runtime invoke the active tool
with the separately frozen final arguments. Candidate output can never become a
replacement oracle for these cases.

`test/test_generated_tool_contract.py` covers Polish, English, mixed-language
wording, invented values, contradictory examples, multiple real sandbox cases,
and the full agent path from semantic extraction through activation and final
execution. These tests establish bounded fixture grounding, not universal
natural-language understanding; unclear speech remains a clarification case.

The 2026-09-09 live AgenticQwen regression used only natural Polish wording:
`dla 2 ma dać 4`, `dla 7 ma dać 14`, followed by the separate invocation
`n = 9`. The extractor grounded both examples, safely canonicalized the sole
input alias (`liczba` versus `n`), and froze two tests. Bubblewrap reported two
passed comparisons, DARKLINGER activated `podwoj_liczbe`, and the runtime-bound
final call returned `{"wynik": 18}`. One malformed source draft was rejected
before the successful draft; no test oracle was taken from candidate output.

## Explicit contracts and a reusable local example

Expert callers may include one bounded `tool_contract=<JSON>` object in the
owner objective. DARKLINGER freezes its one-to-eight input/output cases directly,
without spending a model turn to reinterpret escaped data. The contract is
immutable, requires consistent field names and a separately supplied final
input, and rejects conflicting or non-finite values. Embedded example URLs are
treated as data, so they cannot accidentally route an offline task to browser
navigation.

For online creation, the owner need not predict dynamic numeric results. DARKLINGER
still requires an observation boundary (a safe URL and the meaning of “traffic”)
because page loads, HTTP requests, and server-side visitor counts are different
measurements. If that boundary is absent, it pauses once for that information
before code generation; it does not spend creation attempts or ask for a made-up
expected number.

`scripts/verify_generated_har_tool.py` is a harmless end-to-end trial using
synthetic HAR-shaped JSON strings. The live local model wrote `har_summary`
without a supplied implementation; Bubblewrap passed three fixed cases,
executed a separate final case, then a holdout and a fresh runtime reload. The
verified holdout result was 3 requests, domains `other.example` and
`unseen.example`, 1 error, and 2.0 ms total. The trial artifacts and evidence
remain in its printed temporary directory; this does not grant network access
to generated tools.

On 2026-09-16 the current parser was also checked by replaying the actual draft
that previously failed with a leading `e`. Three fixed sandbox cases passed;
the separate final case returned 2 requests, 1 error and 10.0 ms, and the unseen
case returned the holdout result above both before and after runtime reload.
Artifacts: `/tmp/darklinger-har-parser-replay-vikj75f8`. This is a replay of captured
model output, distinct from a fresh model generation. The final automated suites
passed 1012 Full and 943 freshly exported Public tests.

The fresh local AgenticQwen run on 2026-09-16 also passed. Its trace
`interactive-fcb7f03d174a47dcbc8046fa03dddd9b` records a truncated first draft
(`finish_reason: length`), a successful `learning_create_tool`, and a successful
`har_summary` invocation before completion. All three fixed cases passed in
Bubblewrap; the independent holdout and fresh-runtime reload matched their
expected results. Evidence is under `/tmp/darklinger-har-trial-8dn8ncti`, including
`report.json` and the checkpoint/journal. The run lasted 603 seconds, from
02:42:50 to 02:52:54 UTC. This server was manually launched with default reasoning
and four slots, unlike DARKLINGER's default reasoning-off/single-slot profile. It
must not be used as the application-profile speed baseline. This
trial exercises explicit synthetic HAR inputs and expectations, not live website
traffic capture, general natural-language contract discovery, or every tool.

## Source-context performance (2026-09-16)

Source generation no longer repeats the full contract inside every grounding
quote. The contract projection contains each fixed input/expected-output example
once, without the final invocation or provenance copies. The runtime retains the full contract,
all independent tests, and final arguments unchanged. An embedded explicit JSON
contract is elided from the source objective only when it exactly matches the
frozen contract and contains no unknown extension fields. Surrounding prose,
natural-language objectives, unknown data, runtime operation specifications and
bounded rejection feedback are preserved. This is structural projection, not
language-specific intent matching or a reduced validation policy.

The HAR script now records per-completion wall time, provider token counts and
timings when available, cache reuse, and total task time. `--cold-context`
disables prompt-cache reuse for the trial only; production caching is unchanged.
Measurements do not record reasoning text or issue extra generations. The trial
still requires actual creation/execution receipts, all three frozen examples,
the separate final result, an unseen holdout and successful runtime reload.

The measured server used AgenticQwen-8B.Q8_0, DARKLINGER's `build_server_command`
with one slot, reasoning off, balanced repetition penalties, Q8 KV cache, 99 GPU
layers and context 16000 (server-rounded to 16128). Temperature was 0 and the
source cap remained 1536. The model was already loaded for each measured task;
startup time is excluded. No network tool or remote model was involved.

| Trial | Input tokens | Output tokens | Cache tokens | Prompt processing | Generation | Entire task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before, empty cache | 3976 | 283 | 0 | 14.65 s | 57.06 s | 72.81 s |
| Before, repeated with cache disabled | 3976 | 283 | 0 | 14.15 s | 58.74 s | 74.18 s |
| After, cache disabled | 893 | 290 | 0 | 2.95 s | 45.62 s | 49.70 s |
| After, warm prefix | 893 | 290 | 544 | 1.33 s | 46.88 s | 49.57 s |

All these trials passed, without repair attempts. Evidence directories are
`/tmp/darklinger-har-trial-uz6ucim7` (before),
`/tmp/darklinger-har-trial-i28ng9m5` (after, cold), and
`/tmp/darklinger-har-trial-_gt7fm4t` (after, warm); each contains `report.json` and
execution traces. The repeated baseline is `/tmp/darklinger-har-trial-3841uytv`:
only `_generated_source_messages` was restored in the test process from commit
`da98778`; the working tree, server settings and trial checks were unchanged.
These temporary files are local evidence, not release assets.
The cold comparison reduced input tokens by about 78% and task time by about
32%. This is one bounded explicit-contract workload on one local model, not a
promise of the same speedup for routing, arbitrary natural-language tasks,
every model or every tool. Generation still dominates the elapsed time.
