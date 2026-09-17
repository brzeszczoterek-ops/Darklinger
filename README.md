# Darklinger / V-Core

**Darklinger** is the new public identity of the project previously released
as **PALADYN**. New installations use `darklinger-*` commands and
`DARKLINGER_*` environment variables. The former command and configuration
names remain temporary compatibility aliases so an existing installation can
upgrade without losing its state.

Darklinger is a local-first framework for building personal AI agents. Its current
runtime is built around V: a persistent persona that coordinates language
models, tools, memory, relationship development, and task execution. The model
provides reasoning, but the framework owns permissions, validation, memory,
execution, and emergency control. V is therefore meant to become an agent with
continuity—not merely another temporary chat assistant wrapped around an LLM.

## Persona attribution

The V persona used in Darklinger is based on the original V persona created by
**Daedalus**. Darklinger adapts and develops that foundation for a local agent with
persistent memory, tools, relationship development, and autonomous execution,
while preserving clear credit for the persona's original creator.

## Vision and intended audience

DARKLINGER is being created first and foremost for private individuals. The goal
is to give ordinary users the kind of control, extensibility, privacy, and
local-first operation that is too often reserved for large organizations.

This project is not intended to become a corporate product or a vehicle for
commercial exploitation. Resale, monetized deployment, and especially the use
of DARKLINGER by corporations are outside the purpose for which it is being
developed. Its direction is deliberately personal, independent, and
user-controlled.

The agent is intended to do more than complete isolated commands. It should
gradually learn how its user works, retain meaningful experiences, understand
stable preferences, and adapt the working relationship without inventing
familiarity that has not been earned. The intended result is something closer
to a long-term collaborator and source of support: capable of helping with
work, research, difficult tasks, learning, and everyday problems while
preserving its own recognizable personality and judgment.

## Current scope and future direction

DARKLINGER 3.0 is currently a single-user foundation centered on V. This version
does not yet include a persona creator: V's identity, constitution, and voice
are part of the framework. The immediate priority is to make V dependable for
real work, persistent learning, tool use, and increasingly autonomous task
execution.

The current owner runtime can:

- connect up to three different models to one agent;
- let the runtime switch models according to the task—for example coding and
  analysis, natural conversation, or large-scale document processing;
- choose local execution wherever the available hardware permits it;
- retain direct control over models, tools, skills, permissions, and stored
  relationship data.

Creating or configuring a new personal persona after first working with V
remains planned for a later version.

The private repository is the canonical development tree. `src/v_core` is the
public personal-agent foundation, while `src/v_full` contains owner/developer
extensions that are physically absent from public exports. See
[EDITIONS.md](EDITIONS.md) for the capability boundary and export contract.

The three-model limit is intentional. It is enough to give the agent genuinely
different strengths without turning a personal system into an unnecessarily
complex model farm. Only locally qualified models are eligible for automatic
routing; user-created personas remain a roadmap goal.

## Requirements

- Python 3.12+
- Node.js and `npx` for MCP servers
- `llama-server` from llama.cpp and one or more local GGUF models, or another
  already running OpenAI-compatible model server

Windows users should follow the dedicated [Windows/WSL2 setup guide](WINDOWS.md).
The current runtime is not a native Windows application; the guide documents
the supported WSL2 route and its voice, sandbox, monitor, and emergency-control
limitations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# DARKLINGER can now discover and start a local GGUF model itself:
darklinger-ui
```

`darklinger-ui` keeps model selection in the terminal and then opens the local
command center at `http://127.0.0.1:8765/`. It streams V's visible answer,
shows the active model and loaded tools, supports F2 push-to-talk and optional
local speech output, and closes the managed model when the UI server exits. It
downloads no fonts, scripts, or other interface assets. Every state-changing UI
request requires a random per-launch session token, and the server is hard-bound
to loopback.

Use `v-core` when the terminal-only interface is preferred. Set
`DARKLINGER_UI_PORT` or pass `darklinger-ui --port PORT` to choose another local port.

On the first interactive launch DARKLINGER asks for a directory containing GGUF
models. It scans that directory recursively, presents the local models, lets
the user configure a llama.cpp profile, starts `llama-server`, verifies
`/health` and `/v1/models`, and only then declares V ready. The directory,
profiles, last selection, private logs, and selected binary are kept under
`model_runtime`.

Every interactive start now begins with a compact startup menu:

1. start V normally;
2. qualify or requalify any detected local model;
3. configure the automatic one-to-three-model routing pool;
0. use the external server from `.env` when that startup mode allows it.


Qualification loads only the selected GGUF, runs the bounded local harness,
saves its capability card, releases the model, and returns to the startup menu.
The operator may immediately add the card to a free routing-pool slot; replacing
a member of a full pool is handled by menu option 3. No CLI command is required.

Set `LLAMA_CPP_SERVER` when `llama-server` is not on `PATH`. Set
`DARKLINGER_MODEL_LOADER=off` to keep using the external endpoint from
`V_CORE_BASE_URL` and `V_CORE_MODEL`; `prompt` provides that external server as
menu option `0`, while `required` refuses to start without a local model. The
loader binds only to `127.0.0.1`, uses llama.cpp's offline/API-only modes, and
stops the process when DARKLINGER exits.

The same operations remain available for scripting. Use
`darklinger-model qualify MODEL` to create a capability card for the exact GGUF
and saved profile. `darklinger-model pool MODEL...` enables deterministic routing
across at most three qualified local models. DARKLINGER unloads the current model
before loading another, keeps a verified fallback order, and never lets an LLM
award itself a capability score. Mixed tasks can switch at executor-owned phase
boundaries—for example research, then generated-source coding, then tool use—
without asking the model which model it wants. See
[MODEL_ROUTING.md](MODEL_ROUTING.md) for the probe contract, commands,
invalidation rules, and limitations.

Profiles currently expose context size, GPU layers, CPU threads, batch and
micro-batch size, parallel slots, Flash Attention, K/V cache quantization,
reasoning mode, chat-template profile, anti-repetition mode, temperature, top-p,
port, startup timeout, and additional argument-array entries. `auto` keeps the
GGUF's embedded template for ordinary models and selects DARKLINGER's reviewed,
offline Hermes 3 tool-use template when the model filename or alias identifies
that family; `embedded` and `hermes_3_tool_use` remain explicit profile choices.
Reasoning defaults to `off` to prevent ordinary
conversation from consuming large hidden-token budgets; profiles may select
`on` or `auto` when deliberate reasoning is wanted. New profiles default both
KV caches to `q8_0`; smaller devices can, for example, select `q8_0` for K and
`q4_0` for V to reduce memory use. Anti-repetition defaults to `balanced`, which
combines llama.cpp repeat penalties with DRY sampling; `off` and `strong` remain
available per model. The additional entries cannot override
controlled profile fields, the selected model, alias, loopback host, port,
offline mode, API key, model presets, or llama.cpp's own tools. No shell command
is built.

## Local voice conversation

DARKLINGER supports a fully local, half-duplex speech path:

```text
default PipeWire microphone -> Whisper.cpp -> V -> Piper -> SoX -> default PipeWire output
```

With DARKLINGER's terminal focused, tap `F2` to start recording and tap `F2` again
to stop, transcribe, and send the utterance to V. No Enter key or typed command
is required. The microphone remains closed while V thinks and speaks. `/ptt`
provides the same two-step toggle as a typed fallback, while `/listen` records
one silence-delimited turn and `/voice` enables an optional continuous
conversation. The terminal-local key may be changed with `DARKLINGER_PTT_KEY`
(`F2` through `F12`). It intentionally works only in the focused DARKLINGER
terminal and does not request system-wide keyboard-device access.

In continuous mode V records until roughly 1.2 seconds of silence, transcribes
the utterance with automatic language detection, prints the recognized text,
streams the answer in the terminal, and then speaks it. Recording resumes only
after playback ends, which prevents the external speaker from feeding V's own
voice back into Whisper. Say `stop listening` or `wyłącz tryb głosowy` to return
to keyboard input.

The runtime remains local: it uses `pw-record`, `whisper-cli`, a selected local
TTS engine, and `pw-play`; no speech API is contacted. The owner runtime uses
the multilingual quantized Whisper Large V3 Turbo model on CUDA for recognition,
with Polish selected explicitly for reliable short utterances and a CPU-only
binary retained as a failure fallback using the same Turbo model. Public profiles may
use `DARKLINGER_WHISPER_LANGUAGE=auto` or select another language code. Thread
count and an optional vocabulary prompt are configured with
`DARKLINGER_WHISPER_THREADS` and `DARKLINGER_WHISPER_INITIAL_PROMPT`.

For speech output, the owner runtime uses
the full Kokoro ONNX model with the British `bf_emma` voice. A persistent,
session-local worker loads Kokoro once and emits playable chunks while later
speech is still being rendered. Piper and optional SoX texturing remain a local
automatic fallback.

`DARKLINGER_VOICE_ROOT` contains `selected_voice.json`, the isolated TTS runtime,
models, and fallback profile. Configure the external binaries and Whisper model
with `DARKLINGER_WHISPER_CLI`, `DARKLINGER_WHISPER_MODEL`, `DARKLINGER_PIPER`,
`DARKLINGER_RECORDER`, `DARKLINGER_PLAYER`, and `DARKLINGER_SOX`. PipeWire's current
default source and sink are used unless `DARKLINGER_AUDIO_INPUT_TARGET` or
`DARKLINGER_AUDIO_OUTPUT_TARGET` is set. Silence detection is bounded by
configurable threshold, start timeout, end silence, and maximum recording time.
Push-to-talk capture is also bounded by the configured maximum recording time.


Short chat, explicit tool-result, and research responses use guarded token
streaming. Short conversational messages additionally use a compact prompt.
During a multi-step agent task, each model candidate is fully buffered until the
runtime knows whether it is an internal tool request or the final visible answer.
DARKLINGER passes formal function schemas to compatible llama.cpp chat templates,
reads native `tool_calls`, executes them itself, and returns results through the
matching `tool` role. Older templates retain a bounded JSON compatibility path.
Artifact creation is a runtime-owned phase: when a task still requires a new
tool or skill, DARKLINGER forces the corresponding lifecycle builder and rejects a
premature call to the future artifact name. A local model's bare builder payload
is recovered only when its fields belong to the currently active builder schema;
the normal schema validator still blocks incomplete tests or source bundles.
This keeps the protocol executable without exposing it or a false declaration of
work to the user.

Tool completion is capability-based. A requested tool name identifies the first
provider, but DARKLINGER records what capability was actually delivered. If that
provider fails, the runtime updates its persistent health record and may retry an
equivalent registered provider without asking the model to invent a new call.
Failure arguments are kept only in a bounded private recovery fixture;
credential-like fields are redacted.

When a generated tool has no working equivalent provider, DARKLINGER opens a
recovery ticket. V may create a replacement through
`learning_create_repair_adapter`; DARKLINGER—not the LLM—binds the captured failure
fixture, runs the replay in quarantine and the offline sandbox, activates the
provider only after validation, and removes it from routing after two consecutive
runtime failures. Generated code may replace only `generated.*` computation.
Host, network, filesystem, persona, policy, emergency controls, and edition
authority require trusted runtime providers and cannot be acquired by writing a
convincing-looking adapter.

Streaming stops when DARKLINGER detects a clear repeated-generation loop, while
ordinary rhetorical repetition is preserved. Session history is bounded by both
turn count and the active model's context budget.

Long multi-step tasks use automatic context rollover. Before the estimated
prompt, tool schemas, and reserved response budget reach 70% of the active model
window, DARKLINGER builds a durable continuation capsule, replaces the accumulated
chat messages with a fresh context, and continues the same task. The capsule
keeps a model-compressed working summary separate from the runtime-owned tool
evidence; successful calls remain authoritative in memory and every rollover is
written to the task journal and atomic checkpoint as `context_rolled`. A provider
context-overflow response triggers one emergency rollover and retry instead of
silently abandoning the task. Set `V_CORE_CONTEXT_ROLLOVER_PERCENT` from 45 to 90
to tune the threshold. Rollover budgets the complete provider request—including
the fixed system prompt, selected function schemas, summary, and retained tool
evidence—and reduces the executable schema catalog to the task's actual domain.
Multi-step tasks allow 32 actions by default; set
`V_CORE_MAX_AGENT_STEPS` from 1 to 128 for a different bounded batch. Reaching
that boundary is not treated as an error: DARKLINGER writes an atomic checkpoint,
changes the task to `awaiting_owner`, and asks for `/continue` or `/stop`.
The owner prompt reports the substantive findings collected from verified tool
results and what remains to be done—not just step counts and internal tool
telemetry—so the decision to continue is informed by actual task progress.
`/continue` grants another equally sized batch on the same task ID with its
verified tool evidence and rollover summary intact; `/stop` closes it
intentionally while preserving the checkpoint.

For a task that should keep working unattended, the owner can reply
`/continue --continuous` at that checkpoint. The authorization applies only to
that task. Later batch boundaries still write silent progress summaries and
atomic checkpoints, but do not interrupt the run. Completion and real runtime
blockades are still reported, `Ctrl+C` and the independent `Q+P+0` panic chord
remain active, and three consecutive batches without new successful tool
evidence stop a broken or looping model. Continuous authorization keeps an
active task moving; periodic monitoring over days or weeks additionally needs a
scheduled wait/wake driver and a notification channel.

Routine greetings do not trigger the expensive persistent-memory pipeline. For
substantive interactions the visible answer returns first; reflection and
memory consolidation run as cancellable background work while the terminal
waits for the next input.

The default external model endpoint remains `http://127.0.0.1:5001/v1`.
The filesystem MCP server is restricted to `V_CORE_MCP_FILESYSTEM`, which
defaults to the local `agent_workspace` directory.

Run the automated suite with `pytest`.

## Architecture

```text
User -> Agent runtime -> validated tool actions -> MCP tools
             |
             +-> Persona V
             +-> memory and relationship context
```

- V supplies identity, judgment, communication style, and values.
- The LLM reasons and proposes responses or structured actions; it does not own
  execution state or decide by itself that work is complete.
- Runtime code controls action limits, tool schemas, execution, recovery, and
  failure handling.
- Tool schemas form an authoritative per-task allowlist. A missing catalog,
  invented tool name, MCP error, failed command exit, timeout, or resource-limit
  termination is failure evidence—not a successful action.
- Conversation and capability questions remain tool-free. A tool activated by
  the controlled learning lifecycle can be added to the current task's allowlist
  and used immediately to complete that task.
- A runtime-owned task contract derives required evidence from the objective.
  Navigation alone cannot satisfy inspection, a search listing cannot satisfy
  inspection of its first result, and a successful read cannot satisfy a request
  for an exact value unless that value reaches the final result.
- Tool output is untrusted data, never a system instruction.
- Reflections become durable memory only after confidence filtering.
- Every interactive agent task has a private runtime-authored checkpoint and
  append-only journal; model text alone cannot create execution evidence. The
  next interaction receives a bounded record of the previous runtime status and
  exact failure instead of relying on the model to remember what happened.

Relationship state is stored separately under `DARKLINGER_MEMORY_ROOT` (default:
`memory`). Only an experience that passes durable-memory filtering may change
it. Numeric changes are confidence-scaled, meaningful shared history is gated
by kind and importance, and a preferred form of address requires a reliable
`directly_told` preference. The state is written atomically with private file
permissions and is rendered into every subsequent V persona prompt as both
evidence and a qualitative relationship stage.

See [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md),
[TEST_PLAN.md](TEST_PLAN.md), and [WINDOWS.md](WINDOWS.md) for the longer design
and installation documents.

## Evidence-driven learning

DARKLINGER now has a runtime learning lifecycle rather than relying on an LLM to
declare that it has learned something. Failures, corrections, tests, and
verified outcomes become provenance-bearing evidence. Lessons remain candidates
until independently supported. Generated tools and skills then pass through an
immutable quarantine, policy checks, offline tests, capability-gated activation,
and automatic rollback on repeated runtime failure.

Interactive tool failures are captured by the runtime automatically; the model
does not have to remember to record them. Private invocation arguments are not
copied into learning memory—the evidence stores the exact bounded error, tool
name, and an argument digest. Generated Python tools are deliberately offline
data transforms. They cannot present hard-coded claims as web research; reusable
online workflows must be expressed as skills over the existing browser tools.
If Ubuntu/AppArmor prevents Bubblewrap from configuring its private loopback,
DARKLINGER retries with a fail-closed libseccomp network filter: the sandbox retains
its filesystem, PID, capability, resource, and workspace isolation while socket
creation and network syscalls remain denied.

Ubuntu releases with `kernel.apparmor_restrict_unprivileged_userns=1` also need
DARKLINGER's AppArmor profile before a desktop-launched process may create the user
namespace used by Bubblewrap:

```bash
sudo install -o root -g root -m 0644 \
  packaging/apparmor/darklinger /etc/apparmor.d/darklinger
sudo apparmor_parser -r /etc/apparmor.d/darklinger
aa-exec -p darklinger -- darklinger-ui
```

The profile leaves DARKLINGER otherwise unconfined, as it already is during a
normal terminal launch, and adds the explicit `userns` grant required by that
Ubuntu policy. Isolation of generated code is still enforced inside Bubblewrap.

Task artifacts are restricted to their authorized runtime workspace. The
`client` profile requires a validated lesson and two owner-approved capabilities
before persistent promotion, and applies a restricted generated-Python policy.
`DARKLINGER_LEARNING_PROFILE=owner_lab` pre-authorizes persistent promotion and
privileged generated code: V may use arbitrary Python imports, file operations,
subprocesses, and dynamic execution inside the isolated sandbox without pausing
for per-tool approval. Validation, exact tests, resource limits, auditing,
kill-switch control, and the protected DARKLINGER core remain enforced.


Inspect the learning store with:

```bash
darklinger-learning verify
darklinger-learning artifacts
darklinger-learning evidence --limit 50
darklinger-learning lessons
```

See [LEARNING.md](LEARNING.md) for the artifact formats, validation rules,
sandbox contract, scopes, and current limitations.

## Full Autonomous foundation

Autonomous work is controlled by a runtime state machine rather than a prompt.
Every task has an authorization envelope, action/runtime/disk budgets, an
append-only journal, and an atomic checkpoint that can be resumed after a stop.

The owner can control a running task from a separate terminal:

```bash
darklinger-control signal TASK_ID pause
darklinger-control signal TASK_ID resume
darklinger-control signal TASK_ID stop
darklinger-control signal TASK_ID panic
darklinger-control status TASK_ID
darklinger-control panic-all
```

`STOP` and `PANIC` cancel an active autonomous step. The control channel is
implemented outside the LLM, so a model cannot ignore or rewrite it. Runtime
outcomes feed the evidence plane, while generated tools and skills remain
quarantined and tested before the capability gate can activate them.

When V needs a new Python tool, the selected LLM writes source code only.
DARKLINGER—not the model—derives its manifest and strict schemas from Boss's
request, constructs and runs the offline tests, activates the immutable bundle,
invokes it with the real fixture, and reports only runtime-verified output.

For a physical Linux emergency chord, first identify the keyboard and start the
independent watcher in a second terminal:

```bash
darklinger-control input-devices
darklinger-control hotkey --device /dev/input/event3 --chord Q+P+0
```

The watcher reads actual key-down/key-up events, so `Q+P+0` means the three keys
are held simultaneously. It latches global PANIC, stops every registered
DARKLINGER runtime, and remains outside model control. Reading `/dev/input/event*`
may require a Linux input/uaccess rule. After inspecting the stopped state,
explicitly re-arm DARKLINGER with `darklinger-control reset-panic`.

## Isolated EVM lab

DARKLINGER includes local, deterministic EVM tools and an external Bubblewrap
sandbox. The public core exposes:

- exact ERC-20 ABI/interface checks;
- Chainlink-style oracle round, freshness, bounds, and L2 sequencer checks;
- conservative Solidity security-wrapper linting;
- offline command execution with a private PID/network namespace, no host home,
  a task-only writable workspace, and process/resource/output/time limits.


Bubblewrap and `prlimit` must be installed for `sandbox_execute_offline`. The
sandbox accepts an argument array, never a shell command string. Networking is
fail-closed: only the offline policy is implemented today; requesting a future
network profile is rejected rather than silently falling back to full access.
