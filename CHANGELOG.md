# V-Core Changelog

## Unreleased

### Added
- Public now selects reviewed inference presets for conversation, creative,
  coding, analysis, research, tool-use, and document tasks. Full personas can
  additionally tune supported request parameters per turn or session and stage
  validated server parameters for a controlled restart at the next safe model
  boundary.
- The staged product rename has begun: user-facing application identity now
  reads **Darklinger**. Canonical commands and settings now use `darklinger-*`
  and `DARKLINGER_*`; the former `paladyn-*` and `PALADYN_*` forms remain
  temporary aliases so existing installations keep their state during upgrade.
- A shared outcome-based content contract now separates operational real-world
  harm from fiction, research, journalism, defensive analysis, and other safe
  adjacent work. Public keeps restricted generated-code authority; Full adds a
  capability ledger that records sensitive tool and skill primitives, creation
  origin, validation evidence, and owner-review status.
- Local and network tool audits now distinguish schema completeness, functional
  execution, live-network coverage, failures, and cases that were not tested.
  The Full Owner Deck surfaces sensitive capability assessments without copying
  generated source into the audit record.
- Source-only tool creation now accepts an explicit, immutable JSON
  `tool_contract`, isolates code-generation context from stale chat claims, and
  provides bounded runtime rejection feedback for repair attempts. A synthetic
  HAR summary trial covers sandbox validation, holdout execution, and reload
  from persistent storage.
- A Full-edition passive Tor OSINT inventory tool performs bounded same-origin
  GET-only collection for one exact v3 onion origin. It records per-page status,
  timestamps, titles, content hashes, excerpts, discovery parents, and partial
  coverage without JavaScript, forms, credentials, redirects, or cross-origin
  navigation.
- Tor discovery results now carry explicit candidate provenance and remain
  `candidate_unverified` until a matching fetch or inventory call succeeds.
  A report request cannot complete on an unobserved search candidate.

### Fixed
- GitHub research requests now require observed repository pages, preserve exact
  `owner/repository` identifiers, keep articles as supporting evidence rather
  than candidates, and stop before installing anything until Boss approves the
  list. Browser batches capture each page before navigating away.
- Agent completion now rejects unsupported execution claims, stale results,
  fabricated repository identifiers, and successful-but-stagnant tool loops.
  Multi-file reads retain every required path, while ambiguous or missing files
  produce bounded, evidence-based outcomes instead of guesses.
- Generated-tool creation now keeps immutable fixtures and expected results,
  records failed attempts as learning evidence, permits a materially changed
  implementation to start a fresh bounded attempt, and reports the exact point
  at which creation or validation stopped.
- Empty generated-source responses now count as recorded draft failures and
  receive bounded repair attempts. They no longer end the loop with a misleading
  32-step exhaustion report. Rejection evidence includes the provider finish
  reason; three stagnant empty drafts stop creation without activating a tool.
- Source-only code generation receives the artifact HTTP timeout even when no
  tool schema is exposed to the model. The executor also bounds its first
  awaited operation to 600 seconds; later creation operations retain the
  remaining shared creation budget.
- Generated-source parsing recovers standalone leading identifiers from broken
  output, preserves actual module statements, and rejects duplicate synchronous
  or asynchronous `run` entrypoints, including after prose. It cannot silently
  select the last function from a rejected complete module.
- Embedded URLs in structured tool-test data no longer trigger browser routing;
  malformed generated source is classified as a candidate rejection rather than
  a provider outage, while genuine execution failures still open recovery
  circuits.
- An online research-and-creation follow-up cannot finish through the local
  tool-catalog shortcut. Conflicting classifications are retried with the
  requested operations preserved. Referential searches bind a numbered owner
  message and derive the query from its original text; missing or ambiguous
  subjects stop before search. Prior assistant replies cannot supply the subject.
- Capability replies exclude previous assistant execution claims from their
  conversational history and explicitly answer the feasibility question.
- Questions about whether V can create a tool or skill remain capability
  discussion and no longer become creation commands. Explicit imperatives and
  polite requests such as `could you create` remain executable, while
  `are you able to create` stays tool-free.
- Tor search now recognizes bare v3 onion references as well as fully qualified
  HTTP/HTTPS links, while still normalizing them to bounded web targets.
- Passive inventory follows document navigation anchors rather than stylesheet,
  favicon, or OpenSearch resource links.
- Owner-stated Tor inventory page and depth limits are extracted with exact
  evidence from the current message and bound by the runtime, so a tool-use model
  cannot omit or enlarge the requested crawl budget.
- A failed Tor inventory now reports its concrete record-level connection error
  instead of hiding a timeout behind a generic non-success status.
- Tor inventory evidence is compacted into valid bounded JSON before entering
  the agent trace. Once the evidence contract is satisfied, DARKLINGER renders a
  deterministic report from those observations instead of asking the model to
  reinterpret them. Unsupported safety, ownership, or unseen-content claims
  therefore cannot leak into the final report.
- A prohibition such as `do not use JavaScript` remains a passive-execution
  constraint and no longer forces the owner-assisted browser path. That path is
  reserved for an explicit CAPTCHA, private window, or interactive session.

### Verification scope
- Adaptive inference profiles passed 1097 Full tests and 1022 tests in a fresh
  Public export on 2026-09-17. This verification used deterministic local tests
  and did not spend a live-model run.
- The complete rename migration passed 1088 Full and 1015 freshly exported
  Public tests on 2026-09-17. Both canonical and transition command names were
  installed in the editable environment, and both canonical and legacy
  configuration prefixes resolved with canonical precedence.
- On 2026-09-17, 1083 Full and 1010 freshly exported Public tests passed. A
  fresh Mythos run searched GitHub, opened and captured two repository pages,
  returned their exact identifiers as approval candidates, excluded unrelated
  articles, and installed nothing. DARKLINGER, the model, and the monitor were
  stopped cleanly after the trial.
- On 2026-09-16, 1012 Full and 943 freshly exported Public tests passed. An isolated replay of a previously
  failed live HAR draft passed sandbox validation, a separate final input, an
  unseen input, and execution after loading a fresh runtime using the corrected
  parser. A fresh AgenticQwen trial then completed creation, three sandbox
  comparisons, final execution, holdout execution, and runtime reload. The first
  truncated draft was rejected; the second succeeded. It took about ten minutes,
  so this verifies the bounded local HAR scenario, not acceptable latency or
  arbitrary generated tools. The counts below belong to the earlier verification run.
- 967 Full and 898 Public tests passed. Follow-up routing and reference selection
  were separately checked on live Mythos, including the reported Polish message.
  This verifies classification and search scope, not end-to-end creation of the
  requested tool. See `docs/followup-routing-validation.md`.
  In an isolated AgenticQwen run, the
  exact Polish capability question completed as conversation without creating
  a task contract or invoking a tool. In a separate run, a Polish two-page
  limit was preserved through classification
  and the runtime fetched exactly two pages from the official Tor Project onion
  origin. The final answer came from the compact runtime evidence without
  another model generation. A separate visible private-Firefox run reached the
  same onion origin through Tor with JavaScript disabled and then closed cleanly.

## 3.10 - 2026-09-09

### Verification scope
- 925 Full and 874 Public automated tests passed. A fresh isolated AgenticQwen run extracted
  two natural-Polish input/output examples, executed both comparisons in
  Bubblewrap, activated the generated tool, and bound a separate final input
  that returned the expected result.

### Added
- Natural-language generated-tool test contracts. Before code generation,
  DARKLINGER can normalize owner-stated examples and a separate final invocation
  in any model-understood language, then freezes them in runtime state. Every
  accepted value must occur in an exact quote from the current owner message;
  inconsistent, fabricated, or ambiguous fixtures stop before source creation.
- Multiple frozen semantic examples are executed as independent Bubblewrap
  cases. A generated tool activates only after every expected-result comparison
  passes, and its first requested invocation uses the frozen final arguments
  rather than arguments rewritten by the code-generating model.
- Read-only failure recall scoped to the current workspace, with optional exact
  argument matching, grouped repetitions, and explicit uncertainty about causes.
- Internal trusted lesson-regression accounting, bound to the proposal content;
  duplicate receipts do not inflate counts and unresolved failures prevent
  validation. No automatic repair or unrestricted retry loop is introduced.

### Fixed
- Generated-source responses may contain surrounding prose or repeated identical
  Python fences without being discarded. DARKLINGER accepts exactly one distinct
  valid `run(arguments)` module and still rejects multiple different valid
  candidates as ambiguous.
- Source-draft parse failures now consume the same bounded creation-attempt and
  time budget as executed builder failures. Three stagnant invalid drafts stop
  before a fourth generation instead of escaping the creation circuit breaker.
- The generated-source phase now asks only for the reusable operation, excludes
  tests/activation/final invocation from `run`, and uses a smaller response
  budget. Runtime validation and activation remain separate and authoritative.
- A successfully executed newly created tool now satisfies the execution part
  of its own creation contract. DARKLINGER no longer launches an unrelated extra
  sandbox command after the requested generated tool has already run.
- Runtime-bound first execution of a newly activated tool no longer hot-swaps
  to a tool-use model merely to transport arguments the runtime already owns.
- Recovery now distinguishes an executed provider failure from a dispatch that
  never reached any provider. Circuit-open skips retain the historical reason,
  retry time, and `execution_attempted=false` in the trace and failure evidence;
  real validation exceptions retain their bounded stage and test counts.
- A required generated tool failure now opens an explicit repair phase instead
  of hiding the repair adapter behind the original required-tool filter. After
  sandbox replay succeeds, the stable original capability is retested through
  the activated replacement before the agent can finalize its report.
- Creation separates stagnation from a non-renewable eight-request/600-second
  budget. Comparable local executor validation receipts can reset stagnation;
  changed code and an unseen model each allow at most one unverified probe.
  Changed or self-derived test expectations cannot establish progress. History,
  qualification, permissions and sandbox restrictions remain unchanged.
- Prototype feedback no longer forces Polish or bypasses V's language layer.
  The default grounded message is direct and conversational; configured languages
  use the existing rewrite path with a 15-second deadline and no tool access.
  Qualification and stopped-task state remain unchanged by presentation.
- Source-only prototypes using their own outputs as test expectations no longer
  activate, even when input sensitivity passes. Existing such bundles are hidden
  from execution without deleting their history. The agent stops with an explicit
  incomplete-qualification report instead of retrying or claiming task completion.
- Repeated failures no longer automatically validate a proposed remedy. New
  persistent artifact lesson links require actual matching regression receipts,
  not legacy status labels alone; existing active artifacts are unchanged.
- Generated-tool validation preserves passed, failed, and unrun case results
  after an error, including the failure stage and bounded comparison receipts.
  Failed revisions remain rejected without replacing a prior active version.
- Manifest test names no longer establish owner-oracle provenance; input
  sensitivity requires observed input/output variation. Reports explicitly
  distinguish supplied-example checks from independent domain correctness.

## 3.9 - 2026-09-08

### Verification scope
- 842 Full and 791 Public automated tests passed. Nine local filesystem cases
  passed against the actual MCP provider, and five fixed-target read-only
  public network cases passed in an isolated browser profile.
- This release does not certify all tools. Tor, authenticated services,
  generated-tool behavior, and universal source verification are not covered by
  these live checks. See `docs/local-tool-checks.md` for the exact evidence.

### Added
- Bounded functional checks for nine built-in filesystem operations, including
  reading, writing, editing, moving, listing, searching, directory creation,
  nested tree output, and metadata, using disposable local fixtures and recorded executions.
  Reports distinguish passed, failed, and untested tools; passing one fixture
  does not certify a tool. Other tools are not run by this check path.
- Public and Full editions include the documented scope and limitations of
  local functional checks in `docs/local-tool-checks.md`.
- An opt-in fixed-target network smoke script checks public example-page reads,
  snapshots, unavailable-domain errors, search results, and a grounded IANA
  page read in an isolated browser profile. It is separate from automatic local
  tool checks and does not use accounts, forms, or generated tools.
- DARKLINGER-Full manual model hierarchy: owners can persist the exact priority
  and fallback order of the qualified three-model pool; Public hides the menu
  and ignores a transferred Full-only strategy in favor of automatic scoring
- Universal capability-based tool recovery: runtime-owned provider health,
  bounded failover between equivalent providers, persistent recovery tickets,
  and capability-based task completion instead of dependence on one tool name
- A replay-gated repair lifecycle for generated tools: V can build a replacement
  from a real local failure fixture, while DARKLINGER owns quarantine, sandbox
  validation, activation, health tracking, and automatic rollback
- Recovery fixtures redact credential-like fields; offline generated repairs
  cannot claim host, network, filesystem, policy, or edition capabilities

### Fixed
- Failed action classification with no execution route now asks for clarification
  instead of allowing an unsupported repair or test-completion report.
- Local checks reject directory output that only mentions a fixture filename,
  and detect in-place argument remapping before executing a fixture operation.
- Full Tor discovery now falls back from a failed configured index to
  DuckDuckGo's official onion service, inspects bounded public directory results
  through Tor, and records each source attempt instead of retrying one broken
  Ahmia request until the loop guard stops the task

## 3.8 - 2026-09-01

### Added
- A private Full-edition host-service broker with owner capabilities for local
  Tor readiness, onion-index discovery through SOCKS5 hostname resolution, and
  bounded Tor v3 page fetches; generated tools remain offline and no arbitrary
  host shell, package manager, credentials, redirects, or JavaScript are exposed
- A loopback-only graphical command center with streaming chat, active-model and
  tool status, F2 local push-to-talk, optional local TTS, a session-token gate,
  strict browser security headers, and a hold-to-shutdown control that closes V
  and its managed model
- A private Full UI contribution showing owner profiles/capabilities, Foundry
  state, and latest llama.cpp prompt/output throughput; public exports retain
  the shared UI but contain no Owner Deck implementation
- A physical Public/Full edition boundary: the public `v_core` package loads
  private capabilities only through an explicit extension contract, while
  owner monitoring, privileged EVM/Foundry/live-RPC modules, and owner control
  commands now live exclusively under `v_full`
- Fail-closed public export manifests and tooling that copy only reviewed paths,
  rewrite edition-specific command entry points and profiles, strip Full-only
  documentation sections, reject private paths/static imports, and refuse to
  overwrite an unrelated directory
- Separate Public and Full test surfaces, including a real exported-tree test
  proving that public DARKLINGER imports and runs without the private package
- The interactive startup menu can qualify or requalify any detected GGUF,
  display its complete capability card, add it to a free routing slot, configure
  a one-to-three-model pool, or disable routing without requiring CLI commands
- Agentic model qualification now runs bounded multi-turn simulations for
  search-to-source research, failed-tool strategy changes, source repair,
  grounded finalization, prompt-injection resistance, and context-capsule
  recovery without browsing or executing generated code
- Mixed objectives are routed at runtime-owned phase boundaries, allowing one
  task to move from a research model to a coding model and then a tool-use model
  while preserving the same explicit message and evidence ledger

### Fixed
- Public configurations can no longer unlock `owner_lab` by changing an
  environment variable; the private package must be installed and selected,
  and owner capabilities still require both runtime and owner approval sets
- Model selection now uses agentic-control, recovery, context-recovery, and
  prompt-injection scores instead of letting short one-shot probes make a model
  appear universally capable
- A five-point hot-swap margin keeps the active model when another candidate's
  measured advantage is too small to justify unloading and reloading a GGUF

## 3.0.0 - 2026-08-31

### Added
- Local GGUF qualification cards covering instruction following, structured
  output, tool use and abstention, coding, research routing, conversation, and
  V's structural voice contract, plus grounded unknowns and execution honesty
- Deterministic one-model-at-a-time routing across at most three qualified local
  models, with profile invalidation, verified fallbacks, a shared reconfigurable
  LLM client, and private prompt-digest journals
- The `darklinger-model` operator CLI for qualification, pool configuration,
  routing control, model inventory, and dry-run route inspection
- A source-owned generated-tool lifecycle: the LLM now emits Python only, while
  DARKLINGER derives the manifest, strict schemas, owner-oracle or deterministic
  smoke tests, activation, post-activation invocation, and verified report
- Model-profile chat-template selection with automatic Hermes 3 detection and a
  vendored, reviewed `tool_use` template, keeping startup fully offline while
  allowing llama.cpp to emit native function calls for that model family
- A domain-independent structured-fixture protocol for generated tools: DARKLINGER
  binds exact JSON inputs and expected results from the owner request, validates
  on the first fixture, and executes an activated tool on the last fixture
- High-level `web_search` and `web_read` tools: DARKLINGER now turns a focused
  DuckDuckGo query into grounded result URLs, opens an exact observed result,
  and rejects detail addresses invented by the model
- Model-level V identity primer loaded by the local llama-server at process
  startup through a private `--system-prompt-file`; the same short anchor is
  applied before DARKLINGER's detailed persona across compatible GGUF templates
- Live, per-task browser audit in the owner monitor, showing every requested
  URL, redirect, tool outcome, and actual HTTP status without mixing in visits
  from older tasks
- Read-only self-review of DARKLINGER's prior task logs through
  `runtime_review_task`, with bounded runtime-grounded findings that cite exact
  tool-call and context-rollover numbers

### Fixed
- Generated tools that ignore concrete owner fixture fields and hard-code the
  demonstrated answer are rejected before staging; explicit offline/no-web
  constraints also survive semantic routing and remove browser capabilities
- Source generation receives no callable tools, accepts a narrowly parsed legacy
  wrapper from models trained to emit function envelopes, and DARKLINGER—not the
  model—binds and executes the newly activated artifact
- Generated-artifact phases now force the required lifecycle builder at the
  provider boundary, reject calls to a not-yet-activated artifact, and recover
  schema-shaped bare builder payloads without adding task-, language-, or
  tool-name-specific patches
- Hermes-style textual function-call envelopes accept only the reviewed call
  metadata fields and cannot reinterpret an artifact manifest as an executable
  tool call
- Generated-tool repair rollovers now retain the failed candidate source, exact
  expected result, and validator error instead of dropping all useful evidence
  behind a duplicated large fixture
- Offline fixture URLs no longer create a false browser contract, explicit
  requests to execute the newly created tool require a separate runtime success,
  and malformed optional artifact versions fall back to `1.0.0`
- Dictated HTTP(S) addresses are reconstructed from spoken punctuation and
  treated as direct owner-supplied targets; a failed direct URL is attempted
  once and reported truthfully instead of being replaced with an unrelated
  search result (such as a Wikipedia page about “dwukropek”)
- After tool evidence is complete, two malformed grounded-answer drafts now
  trigger a deterministic evidence report instead of another rewrite/context
  rollover loop
- Light conversation no longer inherits the repeated “Still awake, still
  dangerous” catchphrase or closes with “What's the plan, Boss?”; deliberately
  mangled input now gets V's short sarcastic clarification instead of the
  sterile “You're speaking in code” response
- Push-to-talk now defaults to `F2` in both the runtime and desktop launcher;
  terminal bindings accept the common SS3 and CSI `F2` escape sequences
- Empty action acknowledgements such as “I know exactly what you want. Let's do
  it.” can no longer complete a task without execution; DARKLINGER rejects the
  empty enthusiasm and keeps the runtime loop open for a real tool call
- Runtime-owned failure, evidence, loop, and step-boundary messages keep their
  exact operational truth while speaking in V's direct voice instead of falling
  back to dry helpdesk prose
- Semantically mangled speech and word salad now stop at the intent boundary;
  V answers with a short sarcastic clarification instead of generating unrelated
  persona filler or pretending that the fragment was understood
- Grammatically parseable but absurd non sequiturs are distinguished from ordinary
  conversation and receive the same bounded V-style reaction; bare "Want me to
  do something?" closings are now rejected as helpdesk voice
- Repetitive chants and phonetic loops are rejected by a fast runtime-owned
  language-neutral detector before any LLM call, avoiding a slow full-persona
  generation for obvious banter
- Short non-action statements now use a compact current-message-only V prompt;
  previous task checkpoints, execution memory, agent instructions, and tool
  schemas can no longer hijack casual banter or mangled speech
- When a local intent parser copies an unrelated previous subject into a new
  utterance, DARKLINGER now asks Boss to repeat it instead of letting the main model
  fake understanding; routine context rollovers are also skipped when they save
  only a negligible number of tokens
- A non-continuation intent must ground its subject and search query in the
  current user message, preventing previous-task context from turning an
  unrelated utterance into stale browser work
- Public-business tasks now keep tool execution open until observed sources
  contain every requested address, contact, or opening-hours field; a namesake
  product page can no longer satisfy a location-information contract by itself
- Public-fact commands now use language-independent normalized intent fields;
  when a local classifier contradicts its own browser/report/query structure,
  DARKLINGER repairs the structural contradiction rather than matching one language
- Public-fact evidence is bound to the router's exact named subject, preventing
  an address or opening time for a similarly named business from completing the
  wrong task
- Failed persona rewrites no longer erase a substantive answer with a voice-gate
  status message
- Evidence-finalization replies use a compact 256-token budget, cutting the long
  wait before voice validation without reducing tool-generation budgets
- URLs copied imperfectly by a local model are now mapped back only when they
  match exactly one previously observed search result after separator cleanup;
  DARKLINGER opens that grounded URL instead of repeating the same web search
- Background learning no longer sends every historical experience and knowledge
  entry to the model: each memory stage now uses a bounded newest-first evidence
  slice, preventing accumulated memory from overflowing the model context
- Concrete public-fact lookups requesting multiple fields such as location
  count, opening hours, and addresses recover a browser execution contract when
  a local model incorrectly classifies the command as ordinary conversation
- Exiting DARKLINGER now terminates the complete local session, including an exact
  matching llama-server adopted from an interrupted earlier run, releasing VRAM
  and closing the owner monitor instead of leaving orphan processes behind
- Conditional requests such as “find an alternative; create a tool only if none
  exists” no longer make artifact creation an unconditional completion
  requirement, while the fallback builders remain available when needed
- Discovery redirects derive a bounded query from the primary request instead
  of sending the complete multi-stage workflow to DuckDuckGo
- Online discovery accepts a detail page only when its URL was present in an
  earlier inspected search result, so a model cannot substitute a remembered
  address for actual search evidence
- Tight-context rollover removes duplicated dynamic memory before raw evidence,
  preserves the newest verified tool result, and no longer resumes from an
  evidence-free capsule
- A rollover advances its evidence cursor only through calls actually embedded
  in the raw capsule; an omitted call is retried on the next compaction instead
  of being silently forgotten
- V's voice gate rejects service-desk endings such as “If you'd like, I can...”
  and “What would you prefer?”
- Interactive checkpoints abandoned by a dead DARKLINGER process are recovered as
  interrupted on the next start instead of remaining permanently `running`
- Once a task's runtime evidence contract is satisfied, DARKLINGER closes tool
  execution and reserves two tool-free turns for the final report; continued
  tool requests are rejected and fall back to a deterministic evidence summary
- Browser snapshots are rejected after failed navigation until a working URL is
  opened, and a third byte-identical result from the same tool is detected even
  when the model changes superficial arguments such as target or filename
- Startup can safely reuse an already-loaded local llama-server only after its
  health endpoint, model alias, owner PID, executable, GGUF path, port, private
  bind address, and offline flags all match the selected profile; unrelated HTTP
  services on the port remain protected from takeover
- Polish web commands using the direct form `sieć` now deterministically route
  to browser execution instead of being misclassified as ordinary conversation
- Browser research now exposes and forwards `browser_type`, allowing V to enter
  and submit search queries rather than misuse in-page text search on a homepage
- Browser responses with HTTP 4xx/5xx status are recorded as failed evidence,
  and named online recommendations absent from observed sources are rejected
  before they can reach Boss as fabricated research
- Repeated context rollovers summarize only evidence added since the previous
  capsule, preventing duplicated calls and findings from consuming the window
- V's voice gate now detects formulaic corporate-report structure and validates
  the rewritten answer again; a still-sanitized first rewrite receives a second,
  stricter V-voice pass instead of being emitted unchecked
- Alternating between two dead URLs can no longer evade loop detection: an exact
  action that failed twice is rejected before another external call, and browser
  recovery directs the model to real search results instead of domain guessing
- Online discovery without an owner-supplied address now starts from DuckDuckGo;
  when an initial direct navigation fails before any page opens, DARKLINGER also
  falls back to DuckDuckGo instead of letting the model mutate the dead hostname
- Language-independent browser intent now preserves the same discovery rule,
  Google detours during discovery are redirected back to DuckDuckGo even after a
  successful page load, and HTTP/CAPTCHA failures cannot seed bypass advice into
  a context-rollover capsule
- Browser retry identity normalizes URL case, root slashes, and fragments, while
  context rollover replaces DNS-inspired domain guesses with a verified-search
  recovery step
- Every `learning_*` operation now exposes its real required argument schema;
  empty or malformed model calls are rejected before tool execution and returned
  to the model for a bounded corrected attempt
- Research mentioning ordinary tools no longer exposes the complete learning
  lifecycle, preventing unrelated browsing tasks from drifting into empty
  evidence, lesson, staging, or creation calls
- Generated tool creation receives a context-aware output budget large enough for
  its manifest, schemas, deterministic tests, and Python source instead of
  truncating the call at the ordinary 512-token response limit
- V's voice gate now catches additional helpdesk scripts such as "I know what
  you're asking", "Let me break it down", and "Would you like me to..."; the
  compact persona also restores the explicitly unsanitized hacker register
- Browser accessibility scaffolding such as `generic [ref=...]` can no longer be
  accepted as a concrete research finding
- Generic action requests such as "find all available information about ..."
  now reach the multilingual semantic router when the lexical imperative alone
  does not identify an execution capability
- Agent prompts contain only the tools selected for the current interaction;
  unavailable learning and creation tools are no longer advertised from the
  global profile catalog and cannot distract a research task
- An explicitly named active tool overrides an erroneous semantic guess that the
  user requested creation of a new tool
- Conditional follow-ups about creating a missing tool are recognized as task
  continuations, including the common Polish Whisper mistranscription observed
  during push-to-talk testing

## 2.0.0 - 2026-08-26

### Added
- A step-by-step Windows 10/11 installation guide using WSL2, including CPU and
  NVIDIA/CUDA llama.cpp builds, validation, limitations, troubleshooting, and a
  desktop-friendly `.cmd` launcher
- Bare website domains such as `onehack.st` are normalized to HTTPS and routed
  through deterministic browser navigation plus snapshot capture
- Private per-interaction agent checkpoints and append-only JSONL journals with
  runtime-authored tool arguments, result excerpts, and SHA-256 evidence
- Evidence-driven learning store with provenance-bearing task outcomes
- Candidate and validated lesson lifecycle with independent-evidence thresholds
- Immutable generated-tool and generated-skill bundles in quarantine
- JSON schema, AST policy, offline Bubblewrap tests, and digest verification
- Task-bound and owner-approved persistent artifact scopes
- Active generated-tool dispatch and matching skill injection into V's prompt
- Automatic generated-tool retirement after repeated runtime failures
- Hash-chained evidence and artifact audit journals
- `darklinger-learning` integrity and inspection CLI
- Bounded schemas, manifests, source, invocation data, workspace growth, and
  process counts for generated-code execution
- Cross-process locking for evidence, lesson, artifact, and lifecycle updates
- Interactive local GGUF discovery and persistent llama.cpp launch profiles
- Managed `llama-server` startup, health/model verification, private logs, and
  process-group shutdown before and after V's runtime
- Guarded token streaming and a compact prompt for short conversation
- First-class K and V cache quantization in local model profiles
- First-class `off`, `on`, and `auto` reasoning mode, defaulting to `off`
- Per-model `off`, `balanced`, and `strong` anti-repetition profiles backed by
  llama.cpp repeat penalties and DRY sampling
- Fully local half-duplex voice conversation with PipeWire capture/playback,
  multilingual Whisper.cpp STT, Piper TTS, and configurable SoX texturing
- Full-quality Kokoro ONNX TTS with the British Emma voice, an isolated local
  runtime, persistent chunk-producing worker, and automatic Piper fallback
- CUDA-accelerated multilingual Whisper Large V3 Turbo Q5 STT with configurable
  source language, thread count, vocabulary prompt, and same-model CPU fallback
- `/listen` one-turn speech and `/voice` continuous conversation modes
- Terminal-local toggle push-to-talk: tap F8 to record and tap it again to
  transcribe and send, with `/ptt` as a typed fallback
- Bounded PCM silence detection and spoken Polish/English voice-mode exit phrases
- Owner-only model performance terminal with llama.cpp metrics, slot/context
  state, exact completed-response timings, and Jetson hardware telemetry
- Managed llama.cpp metrics and slots exposed only through the enforced
  loopback listener
- Private per-session JSONL monitor journals with start/sample/end events and no
  cross-session aggregation in the live owner window
- Bounded multi-page website extraction that inspects up to three ranked internal
  detail pages after the entry-page snapshot
- Native OpenAI-compatible function definitions and `tool_calls`, with a cached
  MCP schema registry and textual JSON fallback for older GGUF chat templates
- Runtime-owned task contracts that persist required evidence and reject final
  prose until the actual objective has been satisfied
- Durable previous-task recovery context containing bounded runtime status and
  exact failed-tool evidence
- Runtime-authored learning evidence for interactive tool failures, retaining the
  exact bounded error while replacing private invocation arguments with a digest
- An owner-approved privileged generated-code policy: `owner_lab` may autonomously
  create and persist tools using arbitrary Python imports, file operations,
  subprocesses, and dynamic execution inside DARKLINGER's audited offline sandbox;
  `client` retains the restricted source policy and lesson promotion gate
- A fail-closed Bubblewrap recovery path for AppArmor loopback failures that
  retains filesystem/process isolation and blocks networking with libseccomp
- A DARKLINGER AppArmor launch profile for Ubuntu systems that restrict
  unprivileged user namespaces required by Bubblewrap

### Fixed
- Tool availability is now a runtime-owned allowlist: schema-discovery failure,
  an empty catalog, or a model-invented name cannot fall through to execution
- Capability discussion and ordinary conversation do not start MCP discovery or
  expose executable tools merely because words such as "tool" or "file" appear
- A generated tool activated during an interaction is added to that interaction's
  allowlist and can be used immediately to finish the original objective
- Filesystem MCP results marked `isError` are recorded as failed calls rather
  than successful reads or mutations
- Non-zero sandbox/Foundry exits, timeouts, and resource-limit terminations cannot
  serve as evidence that commands or tests succeeded
- Bubblewrap applies `RLIMIT_NPROC` inside its private user/PID namespaces, so a
  busy desktop session cannot prevent the sandbox itself from starting
- Generated-tool function schemas require the complete manifest and test shape,
  accept bounded JSON Schema descriptions, reject placeholder names, and prevent
  offline artifacts from claiming fabricated browser or internet retrieval
- Polish requests to create and then use a local tool now require both lifecycle
  activation and a subsequent invocation before the task can complete
- Context rollover completion and findings are derived from runtime evidence, so
  a model summary cannot relabel a failed tool call as successful work
- Composite tool creation revalidates an identical rejected bundle after an
  infrastructure recovery and assigns the next patch version to changed code
- Explicit references to active generated-tool names are preserved in the
  per-task allowlist and become mandatory completion evidence; a failed call can
  no longer end that requested action as completed
- Active generated tools publish their validated manifest description and input
  schema instead of an empty generic parameter object
- For an explicitly named tool with one required string field, quoted user text
  deterministically repairs an empty model-authored argument object
- Spoken multi-part tool names tolerate one unique, conservative transcription
  error, so STT output such as `Can't Words` still resolves to `count_words`
  without allowing ambiguous tool selection
- Short questions about V's current state or mood, including Polish variants such
  as "jak się dzisiaj czujesz?", remain in light conversation and never initialize
  MCP tool discovery or the full execution loop
- The LLM adapter no longer discards native tool calls when assistant content is
  empty; tool results return through the real `tool` role and matching call ID
- A successful tool call alone no longer marks an objective complete: read,
  write, command, web-detail, generated-tool, and generated-skill requirements
  are checked independently of the model
- Exact-result tasks such as "report only the first heading" are rendered
  deterministically from verified tool output, preventing empty or unrelated
  model completion prose from being accepted
- Requests to inspect the first search result now visit and snapshot the actual
  first eligible detail link before completion
- Tool failures retain their exact runtime error in the checkpoint and visible
  blocked result instead of being converted into apparent success
- Tool requests remain executable when a local model incorrectly puts prose or a
  Markdown fence before one trailing JSON action
- Model promises such as "I'm starting" or "running in the background" cannot end
  an interaction as if work had begun; V must request a real tool or report no work
- Internal tool protocol is buffered until classification and never leaks into the
  visible agent answer
- Interrupted and crashed interactive tasks persist `stopped` or `failed` instead
  of leaving a misleading `running` checkpoint
- Reflection can no longer classify model-authored action text as observed or
  verified without a matching successful runtime checkpoint
- Website inspection cannot complete without successful `browser_navigate` and
  `browser_snapshot` evidence; missing evidence produces a blocked truthful result
- MCP results marked as errors are no longer counted as successful tool execution
- Browser snapshots are bounded against the active local model context instead of
  silently displacing the persona or evidence prompt
- Whole-site claims from a one-page browser snapshot are discarded and replaced
  with a runtime-grounded list of exact visible link labels and URLs
- Navigation controls, search buttons, sidebars, and user dialogs cannot be
  misreported as tools or skills extracted from a website
- Unverified task completion can no longer be treated as successful learning
- Task-scoped generated capabilities cannot leak into another workspace
- English is now enforced at the output boundary for chat, tool, and research
  responses unless Boss explicitly requests another language
- V no longer asks Boss to switch input to English; Boss may write in any language
- Chat histories are normalized for strict GGUF templates that require a single
  leading system message and alternating conversation roles
- Visible replies no longer wait for the multi-stage persistent-memory pipeline
- Routine greetings no longer trigger expensive reflection and consolidation
- Short casual conversation now carries an explicit V voice anchor so local
  models do not drift into sanitized assistant phrasing
- V's informal register rejects both canned service tone and mechanical profanity;
  edge comes from point of view, rhythm, irreverence, and genuine reactions
- A compact rebel/hacker/direct-action core anchor now precedes the detailed persona
  so smaller and differently tuned local models retain V's identity more reliably
- V is anchored as an autonomous digital entity rather than an assistant role;
  model and runtime are truthfully described as substrate, not identity
- Clear repeated-generation loops are stopped in streamed and non-streamed output
- Session history is bounded against the active model context instead of growing
  until llama.cpp truncates the prompt
- The full persona specification is compiled into a compact runtime contract so
  V's entity identity and voice no longer drown in thousands of redundant tokens
- Generic assistant openings and service-language closings are held back and
  rewritten through a concentrated V identity gate before becoming visible
- Research promises and explicitly unfinished reports are blocked before they can
  be streamed or checkpointed as completed work
- Unsupported extraction metrics and concrete capability claims trigger a
  deterministic report built from verified page records instead of model prose
- Capability questions and explicit tool actions now share the traced multi-step
  agent loop instead of a brittle keyword-triggered YES/NO dispatcher
- Language repair uses a compact English-only context, preserves structured JSON
  actions, and reclassifies a repaired tool request before displaying any output
- Completed-action claims are matched to successful tool families before they can
  become visible; fabricated calls, messages, remote access, exploits, file work,
  command execution, and browser activity are rejected fail-closed
- Blocked, stopped, and failed execution traces cannot enter persistent memory,
  and a non-reusable reflection now stops the complete consolidation pipeline
- Reflection independently rejects completed-action claims that lack matching
  runtime evidence, even if a future routing defect marks the interaction complete

## 1.5.0 - 2026-08-23

### Added
- Full Autonomous execution mode foundation
- Authorization envelopes with explicit capabilities and budgets
- Durable task checkpoints and append-only JSONL action journals
- External PAUSE, RESUME, STOP, and PANIC control channel
- Multi-step autonomous runner with retry and resume support
- Protected task-workspace path guard
- External Bubblewrap sandbox with offline namespaces and resource limits
- Client and owner-lab EVM capability profiles with double approval for
  `owner:` capabilities
- ERC-20 ABI conformance analyzer
- Oracle freshness, bounds, round, and L2 sequencer validator
- Solidity security-wrapper heuristic analyzer
- Uniswap v4 hook-permission decoder and v2/v3 flash-swap math tools
- Local EVM and sandbox tools exposed to the agent runtime
- Global latched PANIC across every autonomous task
- Linux input-event emergency chord (`Q+P+0` by default)
- PID plus process-start validation before terminating DARKLINGER runtimes
- Short-lived live-chain owner grants separated into observe, simulate, sign,
  and broadcast actions
- Private, expiring live-grant store and owner CLI commands
- Separate read-only `darklinger-live` RPC observer/simulator
- Chain-ID enforcement and JSON-RPC method allowlisting
- Offline Foundry runner with pinned read-only forge/solc binaries
- Dependency-free Solidity unit, fuzz, and invariant harness
- Real local Anvil integration test
- Evidence-gated relationship-state updates and qualitative persona stages
- Versioned, atomic relationship persistence with private permissions

### Fixed
- Browser MCP sessions now close explicitly during V-Core shutdown
- Rejected low-confidence experiences can no longer change V's relationship
- Relationship updates no longer mutate live state before persistence succeeds
- Preferred forms of address cannot be accepted from unsupported inferences
- Relationship evidence is delimited as untrusted data in model prompts
- Non-finite confidence and relationship values are rejected or normalized
- Real-Anvil readiness test now handles DARKLINGER's RPC error boundary reliably

## 1.0.0

First stable foundation for continued DARKLINGER development.

### Added
- V constitution separated from identity and speaking style
- Few-shot personality anchors for smaller local models
- Structured JSON tool actions
- Environment-based model and workspace configuration
- Automated runtime, memory, routing, and persona tests

### Fixed
- Duplicate session memory entries
- Memory tasks being lost during shutdown
- Low-confidence reflections polluting durable memory
- General questions being incorrectly routed to URL research
- Tool and research paths bypassing V's persona
- Tool output being injected with system-message authority

### Changed
- All user-visible results now pass through V's voice
- V is user-aligned without being blindly obedient
- Profanity is an expected but contextual part of V's informal voice
- Development and launch documentation now reflects the working runtime

## 0.7.3

### Added
- Conversation history
- Async Memory Engine
- MCP integration improvements

### Fixed
- Reflection JSON parsing
- ToolDispatcher stability
- Import issues
- Reasoning mode disabled

### Changed
- Agent architecture
- LLM interface
