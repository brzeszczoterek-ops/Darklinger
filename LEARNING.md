# DARKLINGER Learning, Skills, and Generated Tools

DARKLINGER learning is an evidence-driven runtime mechanism. It is not model
fine-tuning or an LLM writing claims into its own system prompt. The owner build
may run privileged generated code, but that code remains a separately audited
artifact below DARKLINGER's protected core and emergency controls.

## Lifecycle

```text
task result
    |
    v
recorded evidence -- rejected if unsupported or malformed
    |
    v
candidate lesson -- remains a hypothesis without independent support
    |
    v
validated lesson -- trusted regression checks for this exact proposal
    |
    v
tool or skill bundle in quarantine
    |
    +--> static policy and schema checks
    +--> offline sandbox tests (tools)
    +--> deterministic trigger tests (skills)
    |
    v
validated artifact
    |
    v
explicit capability gate
    |
    v
active, versioned artifact -- retired on replacement or repeated failure
```

Every evidence and artifact transition is appended to a private hash-chained
journal. Artifact source, manifest, scope, and status are checked against that
journal before use. Bundles are immutable: a change requires a new semantic
version.

## What counts as learning

DARKLINGER records the origin of each observation:

- `user_correction`: Boss explicitly corrected a result;
- `tool_result`: a tool produced relevant evidence;
- `test_result`: a deterministic test or verifier succeeded or failed;
- `task_runtime`: Full Autonomous encountered a real runtime outcome;
- `self_review`: V proposed an interpretation that is not independently proven.

An autonomous exception is verifiable evidence that the step failed. It is not
automatically proof of the root cause. Likewise, task completion is not stored
as a successful learning event unless the step supplies `metadata.verified=true`.
That flag is accepted only on the trusted runtime path: JSON proposed by an LLM
cannot declare itself verified. Model-proposed observations are downgraded to
unverified `self_review` evidence with confidence capped at `0.50`.

When V identifies an explicit correction from Boss, the framework binds the
evidence to its own interaction ID and stores Boss's raw message instead of a
model-authored replacement. Significant whitespace and line breaks are
preserved up to the evidence-size bound. The correction remains unverified
until supported by a real test or runtime verifier.

A proposal requires evidence of failure, regression, or correction, but remains
`candidate` regardless of how many failures support it. Repeated observations
are not proof that the proposed remedy works; average evidence confidence is
not confidence in its cause or remedy.

The internal `record_lesson_regression` API accepts results from a trusted
evaluator, not from an LLM tool. That evaluator must actually test the proposed
behavior and supply independently established expectations. The runtime compares
canonical JSON receipts and binds them to the exact hypothesis, trigger, and
action. Only passing latest results for **all recorded test IDs** validate that
proposal. A subsequent failure returns it to candidate; passing another test
does not hide the unresolved failure. Repeating a receipt ID does not increase
counters or replace history. The counters count recorded regression attempts,
not production usage. Validation covers these tests, not universal correctness.

Legacy validated labels without matching regression receipts cannot justify
new persistent artifacts through a lesson link. This does not automatically
retire existing artifacts or change owner-profile permissions.

### Recalling failures

`learning_recall_failures` reads recorded tool failures from the current
workspace only. Supply `tool`, optional `arguments` for an exact digest match,
and `limit` (1–20). Without arguments, results are explicitly only same-tool
matches. Repetitions are grouped with an occurrence count, not promoted into
stronger evidence. Raw arguments are not returned. Error text is untrusted
historical data and can contain private or externally supplied text.

Failure evidence also records whether execution reached a provider and at which
stage it stopped. A dispatch rejected because every matching provider has an
open circuit is therefore not counted or described as a new provider execution.
Its previous error remains explicitly historical. Actual validation exceptions
retain a bounded report stage and aggregate case counts, not generated source or
raw test payloads.

Legacy unscoped events and unverified model claims are excluded. A match does
not compare tool versions or current environment state and does not establish
the cause. The tool is available for runtime reviews; it does not run repairs,
authorize retries, or inject proposed actions into the system prompt.

This is a persistent diagnostic/evaluation foundation, not a completed
autonomous learning loop. Automatic selection of a changed approach and
independent general-purpose regression generation are not implemented here.

## Generated tools

The first supported generated-tool format is a small offline Python tool. A
tool must expose a synchronous function:

```python
def run(arguments):
    return {"result": arguments["value"] * 2}
```

For the ordinary autonomous creation path, the model supplies **only that Python
source**. It does not author the manifest, schemas, tests, activation request,
execution request, or success report. DARKLINGER derives those control-plane
objects from the immutable owner request and the source AST, then owns every
remaining lifecycle transition. The complete manifest still defines immutable
versioning, JSON input/output schemas, scope, linked lessons, limits, and exact
test cases. DARKLINGER then:

1. validates the schema and reserved name boundary;
2. parses the Python AST and applies either the client restricted-code policy or
   the owner-approved privileged-code policy;
3. mounts the trusted host and generated source read-only into Bubblewrap;
4. runs every case with no network, no host home, no inherited environment,
   process/memory/CPU limits, output limits, a wall-clock timeout, and per-file
   and total-workspace disk limits;
5. validates the returned JSON against the declared output schema;
6. checks the bundle digest again immediately before activation.

When expected answers were obtained from the candidate itself, even successful
determinism and input-sensitivity checks stop at a non-executable prototype.
Technical status `validated` is not permission to activate it. Independent
expected-result verification is still required; see
`docs/generated-tool-validation.md`. The agent reports the incomplete task and
does not automatically retry this qualification gate.

After activation, DARKLINGER binds the final owner fixture to the validated input
schema and invokes the new tool itself. A successful creation call is therefore
not enough to satisfy a request that also asked to use the tool. The final
completion message is built from the artifact record and verified execution
result, not from a model claim.

If Boss supplied an exact `expected = {...}` value, it becomes the semantic
oracle for the corresponding quarantine fixture. Without an expected result,
DARKLINGER may run the candidate twice and accept only byte-equivalent JSON as a
determinism smoke test. That smoke test proves repeatability and schema
compatibility, not that the program implements an unstated domain rule.

Generated tools cannot replace built-in DARKLINGER, filesystem, browser, learning,
or EVM tool names. In `client`, generated source uses an allowlist of non-I/O
standard-library modules and rejects dynamic execution, subprocesses, and direct
file operations. In `owner_lab`, the pre-authorized
`owner:privileged_generated_code` capability permits arbitrary Python imports,
`open`, subprocesses, dynamic imports, and `eval`/`exec`/`compile` inside the
isolated sandbox. The owner policy controls containment and evidence, not the
subject matter or purpose of the tool.

Both profiles currently execute generated Python without network access and
without the host home, credentials, sockets, or DARKLINGER's protected state.
Network acquisition is composed through registered browser/web tools and skills;
the generated tool receives the resulting data for local processing. This keeps
autonomous creation non-interactive without silently turning generated code into
an unrestricted host process.

For deterministic tool creation, an owner request may provide repeated JSON
assignments such as `records = [...]`, `keywords = [...]`, and
`expected = {...}`. DARKLINGER reads those literals directly from the immutable task
request instead of asking a local model to copy them. The first occurrence of an
input is used by the quarantine test; after activation, the last occurrence is
bound to the real execution call. This is schema-driven and independent of the
tool name, subject, and language surrounding the JSON. It prevents long URLs,
nested records, and expected values from drifting during model generation.
Candidate source must consume the concrete fixture fields exposed by its
`run(arguments)` interface. A program that ignores them and merely hard-codes
the demonstrated expected result is rejected before staging.

When a generated candidate fails, a context rollover preserves its source,
expected output, and validator error while omitting the duplicate fixture body
already present in the objective. The next iteration therefore repairs a concrete
candidate instead of reconstructing one from a shortened summary. Passing the
lifecycle still depends on the selected model's coding ability; repeated invalid
source is rejected and checkpointed, never activated as a successful tool.

An active tool is automatically retired after three execution failures. If it
replaced an earlier active version, DARKLINGER rolls back to that directly
superseded, already validated bundle. Invalid caller input is rejected by the
schema and does not count as a tool failure.
Generated source is capped at 200 KB, a manifest at 2 MB, each invocation at
64 KB of JSON arguments, and a tool at 100 exact test cases. Schemas are bounded
in depth and field count; non-finite and non-JSON values are rejected.

## Generated skills

A skill is a versioned operational playbook, not arbitrary code. Its manifest
contains:

- description and trigger phrases;
- ordered workflow steps;
- required registered tools;
- positive and negative trigger tests;
- task or persistent scope;
- optional links to learned lessons.

Validation rejects missing tools, failed trigger cases, and attempts to modify
V's persona, constitution, permissions, credentials, or emergency controls.
Only matching active skills are rendered into the next agent prompt. They are
delimited as workflow data and explicitly remain below the constitution and
runtime policy.

A skill is capped at 20 triggers, 20 steps, 64 required tools, 100 trigger test
cases, and a 500 KB manifest. It must include at least one matching and one
non-matching test.

## Scope and owner profiles

`task` artifacts are bound to a SHA-256-derived identity of the authorized task
workspace. They remain available after a restart of that same workspace but are
invisible to other workspaces.

`persistent` artifacts can be reused across tasks. In `client` they require:

- a validated linked lesson;
- `owner:create_persistent_artifacts` to stage;
- `owner:activate_persistent_artifacts` to activate;
- both capabilities in the normal and owner-approved capability sets.

`DARKLINGER_LEARNING_PROFILE=owner_lab` pre-authorizes those two promotion
capabilities plus `owner:privileged_generated_code`. This allows Full Autonomous
to create privileged task or persistent artifacts without interrupting Boss for
every individual operation. A persistent owner artifact may be promoted without
a previously validated lesson, but still requires quarantine tests, exact schema
validation, digest verification, auditing, sandbox execution, and automatic
retirement after repeated failures. The `client` profile retains the lesson and
restricted-source requirements.

## Operator audit

```bash
darklinger-learning verify
darklinger-learning artifacts
darklinger-learning evidence --limit 50
darklinger-learning lessons
```

`verify` fails closed if an evidence/audit journal, artifact identity, status,
manifest, or source digest has been modified. The learning root defaults to
`learning` and can be changed with `DARKLINGER_LEARNING_ROOT` or `--root`.

## Protected core

Generated artifacts live outside the DARKLINGER source tree. The persona,
constitution, relationship policy, learning policy, trusted generated-tool
host, authorization envelope, and kill-switch implementation are protected
paths. Learning may add evidence, lessons, tools, and skills; it may not redefine
the mechanism that decides whether those artifacts are trusted.
