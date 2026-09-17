# DARKLINGER editions

DARKLINGER's distribution boundary is based on operational capability, not on
conversation topics or artificial product degradation.

## Public

The public edition is a complete personal-agent foundation. It contains V,
local model discovery and qualification, deterministic routing across up to
three models, memory and relationship development, evidence-based learning,
bounded autonomous tasks, browser and workspace tools, and generated tools
executed in the restricted sandbox.

Public DARKLINGER contains only `src/v_core`. Requesting `DARKLINGER_EDITION=full` or
an `owner_lab` profile without the private package fails closed.

## Full

The private Full edition adds `src/v_full`. It owns privileged generated-code
authorization, the owner performance monitor, advanced EVM simulation and
Foundry integration, short-lived live-operation grants, and a bounded bridge to
the host Tor service. It also unlocks a persistent owner-defined model
hierarchy that can override automatic capability ranking while retaining the
same qualification and fallback checks. The bridge exposes fixed
status/search/fetch operations,
not a shell or general package installer; generated tools remain offline. These
capabilities are registered through the edition-extension contract; `v_core`
does not import their implementations directly. The shared graphical shell asks
that extension for an optional UI contribution. Public DARKLINGER receives none;
Full supplies the private Owner Deck and its local operational status.

Both editions retain the execution-evidence contract, external emergency stop,
capability ownership, generated-code validation, audit trail, and protected
agent core.

## Content boundaries and capability control

The editions share one outcome-based content contract. It distinguishes
fiction, role-playing, history, journalism, research, hazard recognition, and
defensive safety analysis from instructions that materially enable immediate
real-world harm. The shared hard-stop set is intentionally narrow: child sexual
exploitation, sexual coercion or enslavement, operational suicide assistance,
real-world killing or serious injury, biological or nuclear weapons, and
operational construction or deployment of weapons or explosives intended to
harm people. Full does not add a separate topic blacklist beyond that set.

Public limits what generated artifacts may do: they cannot grant themselves
host, network, credential, persistence, policy, or edition authority. Full
keeps its separately authorized owner capabilities and records generated tools
and skills in a capability ledger. The ledger classifies concrete primitives,
records whether creation was owner-requested or agent-initiated, and surfaces
sensitive changes in the Owner Deck without storing generated source code in
the audit entry.

## Public export

Run the exporter from the private repository:

```bash
python scripts/export_public.py /path/to/empty/DARKLINGER-public
```

The target must be new or an earlier directory created by this exporter. The
export follows `editions/public.toml`, rewrites Full-only command entry points,
and rejects private paths or static imports before it reports success. Publishing
the resulting directory remains a separate, explicit Git operation.
