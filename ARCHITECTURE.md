# V-Core Architecture

## Overview

V-Core is a local autonomous AI agent framework.

Its purpose is not to be another chatbot.

Its purpose is to execute real-world tasks by combining:

- LLM
- Memory
- Tools
- Planning
- MCP

The LLM is only one component of the system.

The intelligence comes from the cooperation of all modules.

---

# High-Level Architecture

```
                User
                  │
                  ▼
              Agent Core
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
    LLM        Memory      Tool Dispatcher
                  │              │
                  ▼              ▼
           Memory Engine      MCP Tools
                  │              │
        ┌─────────┼───────┐      ▼
        ▼         ▼       ▼   MCP Client
   Reflection Experience Knowledge │
                  │                ▼
                  └────────── Filesystem
```

---

# Responsibilities

## Agent

Coordinates the entire system.

The Agent never performs work directly.

It decides what should happen next.

---

## LLM

Responsible only for language generation and reasoning.

It should never replace tools.

It should never fabricate external information.

---

## Session

Stores short-term conversation history.

Used to preserve conversational context.

---

## Memory Engine

Coordinates long-term learning.

Pipeline:

Task
↓

Reflection
↓

Experience
↓

Knowledge

---

## Reflection

Evaluates completed work.

Finds mistakes.

Extracts lessons.

---

## Experience

Determines whether a lesson is useful.

Ranks importance.

Removes repetition.

---

## Knowledge

Stores durable long-term knowledge.

Knowledge should represent principles, not events.

---

## Tool Dispatcher

Decides whether a tool should be used.

Routes requests to MCP.

Never executes tools directly.

---

## MCP Tools

High-level wrapper around available tools.

Provides a stable API for the Agent.

---

## MCP Client

Low-level communication layer.

Responsible only for communication with MCP servers.

---

## Interactive execution evidence

Every substantive interactive agent task receives a runtime-generated task ID.
DARKLINGER writes an atomic mode-0600 checkpoint and an append-only mode-0600 JSONL
journal under the autonomy root. Tool start, success, and failure events come
from the runtime around the actual `tools.call()` boundary; a model cannot create
them by printing prose or JSON.

The agent accepts one valid trailing tool object even when a local model violates
the protocol by preceding it with prose or a Markdown fence. The complete
candidate is buffered until classification, so internal JSON is never visible.
A candidate that promises future or background work without requesting a tool is
rejected inside the loop. The model must issue the real action immediately or
state truthfully that the work was not performed.

Runtime evidence is passed separately into reflection. Model-authored result text
is explicitly non-authoritative for execution, and `observed` or `verified`
provenance is deterministically downgraded when no successful tool checkpoint
exists.

Website targets are a stricter evidence class. Full URLs and bare domains paired
with browsing intent are routed through deterministic `browser_navigate` and
`browser_snapshot` calls. A website task cannot complete without both successful
checkpoints. MCP error results raise at the runtime boundary, and oversized
accessibility snapshots are bounded against the selected model's context while
retaining their beginning and end.

---

# Design Principles

- Single responsibility.
- Tools before guessing.
- Memory before repetition.
- Modular architecture.
- Local-first.
- LLM is a reasoning engine, not a database.
- Every module should be independently replaceable.

---

# Persona Architecture

V is implemented as separate layers so that speaking style cannot silently
replace judgment:

- `IdentityKernel` defines who V is and what she values.
- `Constitution` defines truth, user alignment, autonomy, risk judgment, and
  the narrow conditions in which V should object to a requested action.
- `VoiceProfile` defines V's direct, rebellious, contemporary, and naturally
  profane conversational register.
- `PersonaRuntime` combines those layers with the real relationship state and
  supplies few-shot anchors for smaller local models.

Every user-visible response must pass through the persona layer. This includes
normal chat, browser research, tool output, and error reporting. Untrusted tool
content remains data and cannot redefine V or her constitution.

## Local speech boundary

`SpeechRuntime` is an optional local I/O adapter around the existing agent. It
does not bypass the persona, language, memory, or tool layers: Whisper output is
submitted through the same `VCore.ask()` path as keyboard input, and Piper only
receives the final user-visible answer after the output-language boundary.

The terminal exposes toggle push-to-talk, one-turn, and continuous half-duplex
modes. The default push-to-talk hotkey is a GNU Readline binding local to the
focused terminal: the first press starts a bounded PipeWire recorder and the
second press stops, transcribes, and submits the utterance through `VCore.ask()`.
This avoids privileged, system-wide `/dev/input` capture. PipeWire capture in
silence-delimited modes is monitored by a deterministic PCM voice-activity
detector with start, silence, and total-duration bounds. Whisper.cpp performs
multilingual STT. Piper renders
the selected local voice, an argument-array SoX profile applies optional
texturing, and PipeWire plays the result. No shell command is constructed and no
network speech service is part of this path.

## Owner diagnostics boundary

The optional owner monitor is a separate read-only process. A managed
`llama-server` exposes Prometheus metrics and slot state on the same enforced
loopback listener used by V. The monitor combines those values with parsed
completed-response timings from the private server log and local Jetson
`tegrastats`. It receives the model PID, port, profile metadata, and log path as
an argument array; it cannot change model properties or invoke agent tools.

The feature is disabled unless `DARKLINGER_OWNER_MONITOR` is explicitly enabled.
The private owner launcher may enable it while public/client configurations
leave it off. The window terminates when the model PID no longer exists.

Each monitor process owns one session ID derived from the unique llama.cpp log
timestamp and model PID. It appends `session_start`, bounded periodic `sample`,
and `session_end` events to a mode-0600 JSONL journal in the private
`monitor_sessions` directory. A monitor never reads historical journals, so a
new window cannot aggregate or display previous-session counters.

## Relationship state

Relationship development is evidence-driven rather than a free-form roleplay
claim. The memory pipeline first reflects on an interaction, creates an
experience, and subjects it to the durable-memory confidence threshold. A
rejected experience cannot reach `RelationshipUpdater`.

For accepted experiences, the updater works on a copy of the current state,
clamps and confidence-scales numeric deltas, restricts shared-history events to
meaningful experience kinds, and accepts a form of address only when a reliable
`directly_told` preference contains that form in its evidence. The candidate is
atomically persisted before it replaces the in-memory state. Legacy flat YAML
loads remain supported; new writes use a versioned private state file.

`PersonaRuntime` converts the numeric state into `new`, `familiar`,
`established`, or `close` and includes the underlying scores and remembered
evidence. This makes the change visible to the model without permitting it to
perform intimacy unsupported by memory.

---

# Full Autonomous Control Plane

Full Autonomous is a runtime execution mode, not a system-prompt instruction.

Each task owns:

- an `AuthorizationEnvelope` describing pre-authorized capabilities;
- runtime, action, failure, disk, and external-cost budgets;
- an atomic JSON checkpoint;
- an append-only JSONL action journal;
- a control channel for `PAUSE`, `RESUME`, `STOP`, and `PANIC`;
- a task workspace separated from protected DARKLINGER state.

The autonomous runner checks control signals during active steps and cancels
in-flight work on `STOP` or `PANIC`. The LLM does not own this mechanism and
cannot grant itself new capabilities. The owner may change the envelope, while
worker models can only propose actions within it.

The emergency control plane also has a latched global PANIC. A separate Linux
input-event watcher detects a real simultaneous key chord and then:

1. writes the global panic latch consumed by every runner;
2. cancels all active task steps;
3. terminates registered DARKLINGER processes after validating both PID and Linux
   process start identity, preventing stale-PID termination.

The latch survives the stopped process and requires an explicit owner reset.

---

# Evidence-Driven Learning Plane

Learning is implemented outside the LLM as a state machine with six separate
objects: evidence, lesson, quarantined artifact, validation report, active
artifact, and retirement record. The separation prevents a model-generated
reflection from becoming executable authority.

Evidence and lifecycle events use append-only SHA-256 hash-chain journals.
Artifact records are checked against the journal, while manifests and source
are checked against their immutable bundle digest. Task artifacts carry a
workspace-derived scope key; persistent artifacts require a validated lesson
and double owner approval.

Generated Python tools run through a trusted host in the offline Bubblewrap
backend. Generated skills remain declarative workflows and are rendered below
the persona constitution only when their deterministic trigger tests match the
current request. Neither artifact type can modify its own validator, the
authorization model, persona, or emergency controls.

The autonomous generated-tool boundary is deliberately asymmetric. The LLM is
a source-code synthesizer only. DARKLINGER extracts the owner fixtures, derives the
tool identity and strict schemas, constructs the tests and manifest, validates
and activates the immutable bundle, binds the post-activation invocation, and
renders the evidence-backed result. During source synthesis the model receives
no callable tool definitions, so a malformed function-call envelope cannot
become lifecycle authority. Explicit owner instructions such as `offline` or
`do not browse` are enforced again after semantic routing and remove web tools
from the task contract.

Autonomous runtime exceptions are captured as failure evidence. Completion is
captured as success evidence only when a verifier explicitly marks it verified.
Learning capture failure is journaled but never allowed to crash or falsely
complete the primary task.

---

# Local Model Loader

The interactive entry point resolves the model before constructing any LLM,
memory, or agent component. It discovers local files by extension and GGUF
header, ignores auxiliary multimodal projections and later split shards, then
loads a versioned per-path profile from private runtime state.

DARKLINGER launches `llama-server` as an argument array without a shell. The model
path, alias, loopback host, and port are controlled fields; remote model flags,
public binding, server-side tools, model presets, and API-key overrides cannot
be injected through profile extras. Inherited `LLAMA_ARG_*` variables are
removed. The loader requires successful `/health` and `/v1/models` responses
before applying the selected alias and endpoint to V's OpenAI-compatible client.
The child process owns a private log and is terminated as a process group during
normal shutdown, cancellation, failed startup, or initialization failure.
K and V cache data types are first-class validated profile fields rather than
unstructured extra arguments. Legacy saved `--cache-type-k` and
`--cache-type-v` entries are migrated when loaded.
Reasoning mode is likewise a validated `off`, `on`, or `auto` profile field,
defaults to `off`, and migrates legacy `--reasoning` extra arguments.
Anti-repetition is a validated `off`, `balanced`, or `strong` profile field.
Balanced and strong profiles map to controlled llama.cpp repeat-penalty and DRY
sampler arguments; profile extras cannot override them.

An automatic routing pool may contain at most three local models. Each candidate
requires a current qualification card bound to both the sampled GGUF identity
and its behaviour-affecting profile. The runtime classifies the owner task into
a fixed capability class, applies code-owned weights, and unloads the active
model before starting the selected specialist. Agent and memory components share
one reconfigurable LLM client, so a switch cannot leave an old component talking
to a terminated server. Background reflection is cancelled at the turn boundary
before unloading. Startup failure follows the ranked qualified fallback order,
and routing journals store a prompt digest rather than owner text.

Short chat, explicit tool-result, and research generation use guarded token
streaming; routine conversation also uses a compact persona prompt. Multi-step
agent candidates are fully buffered until DARKLINGER distinguishes a tool request
from the final answer. DARKLINGER never streams internal tool-call JSON. A
deterministic repeated-span guard
terminates clear two-block, three-phrase, or runaway-token loops. Non-streamed
results pass through the same trimming rule. Recent session turns are selected
newest-first under a context-derived character budget before being restored to
chronological order. Persistent reflection is skipped for
greetings and runs as cancellable background work for substantive interactions;
a new user request always takes priority over unfinished reflection.

Before every OpenAI-compatible request, DARKLINGER normalizes chat history to one
leading `system` message followed by alternating `user` and `assistant` turns.
The model-specific Jinja template remains the responsibility of `llama-server`
and the GGUF metadata. This portable role boundary supports strict templates
without hardcoding prompt tokens for an individual model.

---

# EVM Capability Boundary

EVM support is divided by runtime profiles, not by persona promises:

| Capability | Client | Owner lab | Live chain |
|---|---:|---:|---:|
| ERC-20 interface analysis | yes | yes | read-only |
| Oracle validation | yes | yes | read-only |
| Security-wrapper lint | yes | yes | source only |
| Uniswap v4 hook simulation | no | yes | no |
| Flash-swap simulation | no | yes | no |
| State fork / arbitrary harness | no | pre-authorizable | local only |
| Wallet signing / broadcasting | no | no by default | separate grant |

Capabilities prefixed with `owner:` need to be present in both the task's normal
capability set and its owner-approved set. This prevents a worker model from
turning an ordinary capability request into an advanced EVM action.

Live-chain owner operations use two independent checks:

- the runtime envelope must contain an explicitly owner-approved live
  observation, simulation, signing, or broadcasting capability;
- the out-of-process bridge must receive a short-lived `LiveActionGrant` bound
  to chain ID, action type, targets, selectors, value, and confirmation.

Observation never implies signing. A grant contains policy only and never a
private key. The general agent process will not hold the signer.

The separate `darklinger-live` process currently implements only:

- pending-block and Geth txpool observation;
- transaction lookup;
- `eth_call` simulation and `eth_estimateGas`;
- exact chain-ID verification, response limits, and RPC error validation.

Its RPC method allowlist does not contain `eth_sendRawTransaction`, wallet,
personal, admin, or signing methods. A future signer will therefore be a new
process and capability, not an extension silently enabled inside this client.

Foundry compilation and testing use a dedicated `FoundrySandboxRunner`. `forge`
and `solc` are mounted read-only, network isolation remains enabled, compiler
autodetection/download is disabled, and an explicit solc binary is pinned.

# Sandbox Boundary

Generated and third-party code runs through an external Bubblewrap backend:

- new user, mount, PID, IPC, UTS, cgroup, and network namespaces;
- all Linux capabilities dropped and host environment cleared;
- read-only `/usr`, private `/proc`, `/dev`, `/tmp`, and `/home`;
- one writable task workspace and no visibility of DARKLINGER memory, `.env`, SSH,
  wallets, or the Docker socket;
- address-space, CPU, process-count, file-size, open-file, total-workspace,
  wall-clock, and output limits;
- process-group termination on timeout, output overflow, STOP, or PANIC.

The initial backend only accepts `OFFLINE`. Local-testnet, fetch-then-offline,
and allowlisted-proxy policies are declared but rejected until dedicated
enforcement backends are implemented. Bubblewrap is a process isolation layer,
not a VM boundary; untrusted native code should eventually use the planned
rootless-container or microVM backend.

---

# Future

The architecture is designed to support multiple agents.

Example:

agents/

    V/
    Atlas/
    Nova/
    Ghost/

Each agent will define:

- personality
- behavior
- constitution
- prompts
- memory profile

without changing the V-Core engine.
