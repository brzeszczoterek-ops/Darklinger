# V-Core Test Plan

## Edition boundary

- Run the complete private tree with `pytest -q`; this executes both `v_core`
  and `test/full`.
- Export a new public tree with `python scripts/export_public.py TARGET` and run
  its tests with `DARKLINGER_EDITION=public PYTHONPATH=TARGET/src pytest TARGET/test`.
- Verify that `src/v_full`, `test/full`, `evm_lab`, Full-only entry points, and
  static `v_full` imports are absent from the public tree.
- Verify that public configuration rejects `owner_lab`, while Full grants every
  `owner:` capability into both the runtime and owner-approved capability sets.
- Verify that an existing unrelated export target is never overwritten.

## Local graphical UI

- Verify that `/api/*` rejects a missing or invalid per-launch session token.
- Verify that chat responses stream as newline-delimited runtime events and the
  answer is not duplicated when token streaming is active.
- Verify that only Full receives an Owner Deck manifest and owner capability
  status; public status must return no private panel.
- Verify that F2 starts/stops the existing local SpeechRuntime and that UI
  shutdown invokes the server callback before VCore closes the managed model.
- Render the interface at desktop and narrow widths without external assets.

The purpose of this document is to verify that every release behaves correctly.

Every item should pass before creating a new release.

---

# Conversation

- [x] Short greetings and questions about V's mood bypass MCP discovery and tool
  execution, while short action requests still enter agent mode.

## Memory

- [ ] Remembers previous question.
- [ ] Remembers previous answer.
- [ ] Maintains context across multiple turns.
- [ ] Does not confuse unrelated conversations.

---

# Filesystem

- [ ] Shows directory tree using MCP.
- [ ] Reads files using MCP.
- [ ] Creates files.
- [ ] Edits files.
- [ ] Moves files.
- [ ] Searches project files.
- [ ] Never hallucinates project structure.

---

# Internet

- [x] Full URLs and bare domains with browsing intent use deterministic web tools.
- [x] Website inspection requires successful navigation and snapshot evidence.
- [x] Browser MCP errors cannot satisfy the website evidence requirement.
- [x] Oversized snapshots are bounded to the active model context.
- [x] A one-page snapshot cannot be described as a completed whole-site scrape;
  overclaims are replaced with exact visible link labels and URLs.
- [x] Website UI controls cannot be classified as extracted tools or skills.
- [x] Explicit extraction visits a bounded set of ranked internal detail pages,
  records every navigation and snapshot, and preserves evidence from every page.
- [x] Unsupported extraction metrics and concrete capability claims are replaced
  with a deterministic report parsed from verified detail-page records.
- [ ] Does not invent search results.
- [ ] Distinguishes current knowledge from retrieved knowledge.

---

# Tool Usage

- [x] Correctly separates explicit runtime actions from capability discussion and
  informational questions before exposing tool schemas.
- [x] Does not initialize MCP discovery or call tools for ordinary conversation.
- [x] Treats the discovered tool catalog as an authoritative allowlist and rejects
  unknown textual or native calls fail-closed.
- [x] Makes a newly activated generated tool callable in the same interaction so
  the original task can continue.
- [x] Treats filesystem `isError`, failed sandbox exits, timeouts, and resource
  limits as failed calls rather than completion evidence.
- [x] Correctly parses exact JSON, trailing prose-plus-JSON, and fenced tool actions.
- [x] Rejects multiple tool objects as ambiguous.
- [x] Never exposes an internal tool action in the visible answer.
- [x] Preserves native provider `tool_calls`, executes their decoded arguments,
  and returns results with the matching `tool_call_id`.
- [x] Falls back to the bounded textual JSON protocol for models whose chat
  template rejects native tool definitions.
- [x] Rejects model claims of future or background work until a real tool executes.
- [x] Explicitly unfinished work is checkpointed as blocked instead of completed.
- [x] Asking about tool capabilities produces a normal answer without executing a
  tool, while an explicit action executes through the traced agent loop.
- [x] Language repair cannot expose or consume structured tool JSON; a tool action
  recovered by repair is reclassified and executed before the final answer.
- [x] Claims of calls, messages, remote access, exploits, filesystem changes,
  command execution, and browser activity require matching successful tool evidence.
- [x] Browser navigation cannot serve as evidence for a phone call or compromise.
- [x] A rejected claim may recover through a real matching tool, but otherwise the
  task is checkpointed as blocked with a deterministic truthful answer.
- [x] Handles tool failures gracefully and preserves the exact runtime error.
- [x] Continues after recoverable errors without counting a failed call as proof.
- [x] Does not accept generic "done" prose when the objective requests a concrete
  value from tool output.
- [x] Renders an explicitly requested first file heading directly from verified
  read output without asking the model to judge its own completion.

---

# Memory Engine

## Reflection

- [ ] Detects successful tasks.
- [ ] Detects failed tasks.
- [ ] Produces useful lessons.
- [x] Runtime evidence, not model text, determines whether tool work was observed.
- [x] `observed` and `verified` are downgraded without successful checkpoints.
- [x] Fabricated execution is rejected before the reflection LLM is called.

## Experience

- [ ] Removes duplicate lessons.
- [ ] Stores only valuable experiences.

## Knowledge

- [ ] Stores long-term principles.
- [ ] Rejects temporary observations.
- [ ] Avoids memory pollution.
- [x] Blocked, stopped, and failed interactions cannot enter durable memory.
- [x] `remember=false` stops experience, relationship, summary, and knowledge work.

## Relationship

- [x] Rejected experiences cannot modify relationship state.
- [x] Numeric deltas are finite, bounded, and confidence-scaled.
- [x] Routine interactions cannot fabricate shared history.
- [x] Forms of address require a reliable directly stated preference.
- [x] Legacy state loads and new state saves atomically with private permissions.
- [x] Relationship stage, metrics, history, and address reach the persona prompt.

---

# Planning

- [x] Executes multi-step tasks with runtime-owned completion requirements.
- [x] Continues unfinished work when a listing was opened but the requested detail
  page has not yet been visited and snapshotted.
- [ ] Requests clarification only when necessary.

---

# Learning, Generated Tools, and Skills

- [x] Unsupported success does not become learning evidence.
- [x] Model-authored JSON cannot mark its own evidence as verified.
- [x] User corrections retain framework task identity and the raw user message.
- [x] Autonomous failures and explicitly verified results are recorded.
- [x] Interactive tool failures are recorded automatically with the exact error
  and a digest instead of private invocation arguments.
- [x] Lessons require independent tasks, fingerprints, and verified evidence.
- [x] Evidence and artifact-journal tampering is detected.
- [x] Artifact source and manifest tampering is detected before activation.
- [x] Artifact versions are immutable within a scope.
- [x] Task artifacts are invisible outside their authorized workspace.
- [x] Persistent artifacts require a validated lesson and double owner approval.
- [x] Generated code with forbidden imports or calls is rejected.
- [x] Generated tools execute offline with time, memory, and output limits.
- [x] Sandbox process-count limits are enforced for generated code.
- [x] Sandbox process limits are applied inside the private namespace and do not
  depend on the host desktop user's current process count.
- [x] Bubblewrap loopback permission failures retry through an enforced seccomp
  network denylist; ordinary offline code runs while socket creation is denied.
- [x] Generated manifests, schemas, source, metadata, and arguments are bounded.
- [x] Generated tools cannot exceed the total workspace disk limit.
- [x] Infinite generated code is terminated by the sandbox.
- [x] Input and output JSON schemas are enforced.
- [x] Bounded JSON Schema `description` annotations are accepted, while generated
  tool calls require a complete manifest and deterministic test structure.
- [x] Failed artifacts can be revalidated unchanged after infrastructure recovery;
  changed bundles receive the next immutable patch version.
- [x] A user-specified active generated-tool name reaches the executable allowlist,
  excludes unrelated learning operations, and remains required until it succeeds.
- [x] Active generated tools expose their validated input schema; an empty call to
  a single-string tool can recover explicit quoted text without guessing complex
  or multi-field arguments.
- [x] Repeated structured JSON assignments are bound without model retyping: the
  first fixture validates a new tool and the last fixture drives its later call.
- [x] Explicit fixture URLs in an offline creation request do not create a browser
  contract, while “execute the new tool” requires a distinct successful run.
- [x] Tight-context repair rollover preserves failed generated source, exact
  expected output, and validator error while omitting the redundant large fixture.
- [x] Invalid optional blueprint versions such as `1.0` are normalized to the
  default semantic version before lifecycle validation.
- [x] Client generated tools cannot claim browser, network, or internet retrieval.
- [x] `owner_lab` accepts generated tools using arbitrary imports, `open`,
  subprocesses, and dynamic `compile`/`exec`, while executing them inside the
  offline resource-limited sandbox.
- [x] `owner_lab` may promote a fully tested persistent tool without a prior
  validated lesson; the client profile still requires one.
- [x] Failed tool tests prevent activation.
- [x] Active generated tools survive a runtime restart.
- [x] Three runtime failures automatically retire an active generated tool.
- [x] A regressing replacement rolls back to its previously active version.
- [x] Invalid caller input does not count as a generated-tool failure.
- [x] A new active version retires the previous version.
- [x] Skills require available tools and passing trigger cases.
- [x] Skills cannot override persona, policy, permissions, or kill switch.
- [x] Matching skills reach the agent prompt as lower-authority workflow data.
- [x] The owner audit verifies every journal, record, manifest, and source digest.
- [x] Parallel generated-tool runs update lifecycle counters transactionally.
- [x] Independent runtime instances serialize lifecycle updates without loss.

---

# Reliability

- [ ] No uncaught exceptions.
- [ ] No hallucinated filesystem data.
- [ ] No hallucinated tool results.
- [ ] No fabricated web search results.
- [x] Interactive tasks persist private checkpoints and append-only action journals.
- [x] Checkpoints persist the task contract and the next interaction can recover a
  bounded runtime-authored status and exact failed-tool context.
- [x] Cancellation changes the durable interactive checkpoint from `running` to
  `stopped`.
- [x] Real Gemma integration executes one deterministic tool and preserves its
  exact result in the final answer and memory evidence.
- [x] Real Mythos + llama.cpp integration emits a native `read_file` tool call;
  DARKLINGER executes it once and deterministically returns `# DARKLINGER / V-Core`.

---

# Local Model Loader

- [x] Recursive GGUF discovery rejects empty/fake files and auxiliary shards.
- [x] Model directory, profiles, binary, and last selection persist privately.
- [x] Profile parameters are range-checked and bounded.
- [x] K and V cache quantization are validated, persisted, migrated, and passed
  to `llama-server` as controlled arguments.
- [x] Reasoning mode is a validated `off`, `on`, or `auto` profile parameter and
  defaults to `off` for ordinary local use.
- [x] Anti-repetition is a validated `off`, `balanced`, or `strong` profile and
  maps to controlled llama.cpp repeat-penalty and DRY arguments.
- [x] Extra arguments cannot override the local loader boundary.
- [x] `llama-server` launches without a shell and only on loopback.
- [x] The managed llama.cpp command explicitly enables the Jinja chat-template
  engine required by native function calling while keeping built-in server tools off.
- [x] Inherited `LLAMA_ARG_*` overrides are removed from the child environment.
- [x] V is not initialized before both `/health` and `/v1/models` pass.
- [x] Early server failure reports the private log tail.
- [x] A managed server is terminated during shutdown and failed startup.
- [x] Noninteractive runs never hang waiting for input.
- [x] Qualification cards persist exact probe scores and become stale when the
  GGUF identity or behaviour-affecting profile changes.
- [x] The deterministic router selects conversation, coding, and research
  specialists from at most three current local cards.
- [x] Runtime switching terminates the current server, verifies the selected
  profile, reconfigures the shared LLM client, and records only a prompt digest.
- [x] Failed specialist startup follows the verified fallback order.
- [x] The startup menu can qualify a model, stop its temporary server, persist
  the card, optionally add it to routing, and return to an ordinary V startup.
- [x] Startup pool configuration accepts one to three distinct current cards and
  can disable routing without deleting saved qualifications.
- [x] Qualification simulates multi-turn research, failed-tool recovery, source
  repair, prompt injection, grounded stopping, and context-capsule recovery
  without real network or generated-code execution.
- [x] Mixed tasks route by remaining runtime evidence from research to coding to
  tool use rather than assigning one model from prompt keywords for the whole run.
- [x] A five-point hysteresis margin avoids costly model reloads for negligible
  measured score differences.

---

# Local Speech

- [x] The selected Piper profile resolves model and config paths under the
  private voice root.
- [x] The selected Kokoro profile resolves its full model, voice bank, isolated
  Python runtime, voice id, language, speed, and Piper fallback.
- [x] Kokoro loads once per speech session and returns bounded private WAV chunks
  through a local JSON-lines worker protocol.
- [x] Real full-model Kokoro Emma -> PipeWire playback succeeds without using
  the Piper fallback.
- [x] Missing STT/TTS binaries or models fail with a readable configuration
  error without disabling keyboard chat.
- [x] PCM voice activity does not start on sub-threshold room noise.
- [x] Recording stops only after minimum speech followed by bounded silence.
- [x] Whisper receives 16 kHz mono WAV input with automatic language detection.
- [x] Whisper language, thread count, and bounded vocabulary prompt are explicit
  configuration rather than hard-coded decoder behavior.
- [x] Owner STT uses checksum-verified multilingual Large V3 Turbo Q5 on CUDA,
  with Polish selected for short utterances and the same model retained on CPU
  as fallback.
- [x] Piper output passes through the selected argument-array SoX effects.
- [x] Playback completes before continuous mode opens the microphone again.
- [x] The terminal-local F2 binding submits `/ptt` without Enter or global
  keyboard-device access.
- [x] Toggle push-to-talk starts one recorder, stops it on the second action,
  transcribes the bounded WAV, and cleans temporary state.
- [x] Spoken stop phrases return to keyboard input.
- [x] Code blocks and Markdown-only syntax are not read aloud mechanically.
- [x] Real local Piper -> SoX -> PipeWire playback succeeds.
- [x] Real local Whisper transcription succeeds on a generated voice sample.
- [ ] Live microphone recognition is confirmed interactively on the target
  headset.

---

# Performance

- [x] Owner monitor is disabled unless explicitly enabled by runtime config.
- [x] Managed model metrics and slots remain bound to loopback.
- [x] Prometheus metrics with labels are parsed without executing input.
- [x] Latest completed prompt/generation timing is parsed from private logs.
- [x] Monitor launch arguments contain the selected model profile and PID.
- [x] Monitor exits when the managed model process stops.
- [x] Real llama.cpp log timing and Jetson `tegrastats` formats are recognized.
- [x] Each launch derives a unique session journal from its log timestamp and PID.
- [x] Monitor telemetry is append-only JSONL with private file and directory modes.
- [x] A new monitor targets only its current log, PID, and journal path.

- [x] Short chat, explicit tool-result, and research output use guarded streaming;
  multi-step agent candidates buffer until tool/final-answer classification.
- [x] Routine greetings skip persistent reflection and return without memory work.
- [x] Repeated-generation spans stop without treating a normal two-use short
  phrase as a loop.
- [x] Recent session history stays inside a context-derived character budget.
- [x] Substantive memory processing runs outside the visible response path and
  yields to the next user request.
- [ ] MCP communication stable.
- [ ] Memory pipeline completes successfully.
- [ ] No unnecessary LLM calls.

---

# Sponsor Demonstration

The agent must successfully complete:

- [ ] Find information on the web.
- [ ] Search local project files.
- [ ] Explain previous answers.
- [ ] Maintain conversation context.
- [ ] Execute MCP tools.
- [ ] Modify project files.
- [ ] Perform multi-step tasks.
- [ ] Produce correct summaries.

---

# Release Checklist

Before releasing a version:

- [ ] All tests pass.
- [ ] No critical bugs.
- [ ] Roadmap updated.
- [ ] Changelog updated.
- [ ] Sponsor scenarios verified.
- [ ] Version tagged.

---

# EVM and Sandbox

- [x] Client profile cannot see or invoke owner-only Uniswap/flash tools.
- [x] Owner capability requires explicit owner approval in the envelope.
- [x] ERC-20 analyzer rejects missing or malformed required ABI entries.
- [x] Oracle validator rejects stale, invalid, or sequencer-unsafe rounds.
- [x] Hook permission bits match the canonical Uniswap v4 mapping.
- [x] Flash-swap arithmetic rounds repayment and fees upward.
- [x] Sandbox cannot see the host home directory.
- [x] Sandbox has no host network.
- [x] Sandbox kills timed-out and output-flooding processes.
- [x] Sandbox workspace cannot escape the task authorization envelope.
- [x] Emergency chord triggers only while all configured keys are held.
- [x] Global PANIC cancels every active autonomous runner.
- [x] Runtime termination records validate PID and process start identity.
- [x] Live observation cannot implicitly authorize signing or broadcasting.
- [x] Live owner grants expire within at most 15 minutes.
- [x] Foundry compilation/fuzz/invariant harness runs offline in Bubblewrap.
- [x] Real local Anvil pending-block observation and `eth_call` simulation.
- [x] `darklinger-live` works end to end with a stored owner grant.
- [ ] Rootless container backend with seccomp and cgroup-v2 quotas.
- [ ] MicroVM backend for hostile native binaries.
