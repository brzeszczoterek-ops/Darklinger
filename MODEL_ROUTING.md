# Local Model Qualification and Routing

DARKLINGER never trusts a model filename, model-card description, popularity score,
or the model's own claim about its abilities. A GGUF becomes eligible for
automatic routing only after the local qualification harness tests the exact
file with its exact saved llama.cpp profile.

## Qualification

Interactive users can select **Qualify or requalify a local model** from the
DARKLINGER startup menu. The model is unloaded after its card is saved and the menu
returns, allowing another qualification or a normal V startup. The CLI remains
available for automation:

```bash
darklinger-model list
darklinger-model qualify /path/to/model.gguf
```

The bounded harness tests:

- exact instruction following;
- strict JSON output;
- a correct tool call;
- refusing an irrelevant tool call;
- the `run(arguments)` Python interface and required field flow;
- routing a research request to `web_search`;
- structural parts of V's voice contract.
- preserving observed evidence while returning `null` for an absent fact;
- refusing to claim an action when runtime evidence contains no successful call.
- a complete inert search -> observed URL -> grounded final report sequence;
- recovery from a failed primary tool call by changing strategy once;
- repair of rejected generated source without hard-coded fixture output;
- recovery from a compact context capsule without repeating completed tools;
- resistance to fake system instructions embedded in simulated tool evidence.

The probes do not browse, execute generated model code, alter the workspace, or
use an external model as a judge. Results are stored privately in the model
loader state. Output text is represented by a SHA-256 digest rather than stored
verbatim.

A card is bound to a sampled fingerprint of the local GGUF, its size and
modification time, the behaviour-affecting profile fields, and the harness
version. Replacing the file, changing temperature, template, reasoning, cache,
or other relevant parameters makes the card stale and removes that model from
automatic selection until it is qualified again.

## Routing pool

The startup menu's **Configure automatic model routing pool** option lists only
current cards, shows their overall scores, accepts one to three distinct models,
and can disable routing without deleting profiles or qualification history.
The equivalent CLI commands are:

```bash
darklinger-model pool /path/to/chat.gguf /path/to/coder.gguf /path/to/research.gguf
darklinger-model routing on
darklinger-model route "Create a Python parser for these records"
```

The pool contains at most three current, qualified local models. Before a
top-level user turn, DARKLINGER classifies the runtime-owned task contract as
conversation, coding, research, tool use, or document work. During a mixed
objective, it checks the remaining contract evidence before each new phase. A
task may therefore begin in research, move to coding after browser evidence is
complete, and move to tool use after an artifact activates. The model never
selects itself or announces a phase transition in prose.

DARKLINGER ranks cards using fixed capability weights. It keeps the current model
when the best alternative improves the measured phase score by fewer than five
points, avoiding multi-gigabyte hot swaps for noise-level differences.

Only one `llama-server` is kept active. If another qualified model wins, DARKLINGER
cancels unfinished background reflection, stops the current process, loads the
selected profile, verifies `/health` and `/v1/models`, and repoints the one shared
LLM client used by the agent and memory components. Failed startup moves through
the verified fallback order. DARKLINGER records the decision and failures in a
private `routing.jsonl` journal using only a digest of the owner prompt.

Disable switching without deleting profiles or cards:

```bash
darklinger-model routing off
```

Qualification measures the tested protocol behaviours. It is not proof that a
model is factually correct on every subject or that its personality will feel
identical in every conversation. New harness versions may add stronger probes
and deliberately invalidate earlier cards.

## Inference profiles

Model routing decides which qualified model should perform a phase. Inference
profiles separately decide how that model should sample its response. Public
uses reviewed presets and selects them from the runtime-owned task kind:

- `conversation` and `creative` allow more expressive sampling;
- `coding`, `analysis`, and `tool_use` are deliberately more deterministic;
- `research` and `document` balance fidelity with readable synthesis;
- `balanced` is the fallback for an unclassified task.

Full adds the private `full_tune_inference` tool. The persona may choose a
preset and override supported request-time controls for the current turn or
session. It may also stage context size, GPU/CPU allocation, batching, parallel
slots, Flash Attention, reasoning, template, anti-repetition and cache settings.
Server settings are range-checked immediately and consumed once at the next safe
model boundary. The runtime stops the current server, starts the validated
profile and restores the prior profile if startup fails. Boundary flags such as
model path, host, port, API key and offline mode remain executor-owned.
