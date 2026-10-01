"""Runtime-owned outcome memory. No prompts, answers or model self-ratings.

SQLite transactions keep observations, incumbent state and decision audit durable.
Only deterministic verification contributes to quality; explicit user feedback is
an independent veto. Latency/cost can only break ties between proven configurations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import median
from typing import Any, Mapping
from uuid import uuid4

SCHEMA_VERSION = 1
POLICY_VERSION = 1
TASK_KINDS = {"conversation", "coding", "code_review", "research", "tool_use", "document"}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def task_context(prompt: str) -> str:
    from ..autonomy.task_contract import TaskContract
    contract = TaskContract.from_prompt(prompt).to_dict()
    # Only structural contract flags and a coarse input-size bucket. Never text,
    # paths, URLs, prompt hashes, tool arguments or inferred personal attributes.
    flags = {k: v for k, v in contract.items() if type(v) is bool}
    return fingerprint({"version": 1, "flags": flags,
                        "size_bucket": min(16, len(prompt).bit_length() // 3)})


@dataclass(frozen=True, slots=True)
class OutcomePolicy:
    min_observations: int = 5
    window: int = 20
    minimum_success_rate: float = 0.8
    promotion_margin: float = 0.1
    efficiency_margin: float = 0.2
    rollback_failures: int = 2
    cooldown_observations: int = 5
    max_router_adjustment: int = 10

    def __post_init__(self) -> None:
        for name in ("min_observations", "window", "rollback_failures", "cooldown_observations", "max_router_adjustment"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 1000:
                raise ValueError(f"invalid policy {name}")
        if self.max_router_adjustment > 10:
            raise ValueError("router adjustment must not exceed ten points")
        if self.min_observations > self.window or self.rollback_failures > self.window:
            raise ValueError("policy window too small")
        for name in ("minimum_success_rate", "promotion_margin", "efficiency_margin"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"invalid policy {name}")


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    task_id: str
    task_kind: str
    context: str
    edition: str
    model_fingerprint: str
    server_profile_fingerprint: str
    profile: str
    request: dict[str, Any]
    request_context: str
    signal: str = "runtime"
    outcome: str = "unverified"
    verified: bool = False
    failure_domain: str = "none"
    latency_ms: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    currency: str | None = None
    observation_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        from ..inference import InferenceParameters, PUBLIC_INFERENCE_PROFILES
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported outcome schema")
        for name in ("task_id", "observation_id", "created_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise ValueError(f"invalid {name}")
        if self.task_kind not in TASK_KINDS or self.edition not in {"public", "full"}:
            raise ValueError("invalid task kind or edition")
        for name in ("context", "model_fingerprint", "server_profile_fingerprint", "request_context"):
            if not _digest(getattr(self, name)):
                raise ValueError(f"invalid {name}")
        if self.profile not in PUBLIC_INFERENCE_PROFILES or not isinstance(self.request, dict):
            raise ValueError("invalid inference profile")
        if set(self.request) != set(asdict(InferenceParameters())):
            raise ValueError("sampling snapshot must contain exactly the supported parameters")
        InferenceParameters(**self.request)
        if type(self.verified) is not bool:
            raise ValueError("verified must be boolean")
        if self.failure_domain not in {"none", "model", "tool", "environment", "user", "cancelled"}:
            raise ValueError("invalid failure domain")
        if self.signal == "runtime":
            if self.outcome not in {"success", "failure", "unverified"}:
                raise ValueError("invalid runtime outcome")
            if self.verified != (self.outcome != "unverified"):
                raise ValueError("inconsistent verification")
            if self.outcome == "success" and self.failure_domain != "none":
                raise ValueError("success cannot have a failure domain")
            if self.outcome == "failure" and self.failure_domain != "model":
                raise ValueError("only proven model failures are quality observations")
        elif self.signal == "user_feedback":
            if self.outcome not in {"positive", "negative"} or not self.verified:
                raise ValueError("invalid explicit feedback")
        else:
            raise ValueError("invalid signal")
        for name in ("latency_ms", "total_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid measured metric")
        if self.cost is not None:
            if type(self.cost) not in {int, float} or not math.isfinite(self.cost) or self.cost < 0:
                raise ValueError("invalid measured cost")
            if self.currency not in {"USD", "EUR", "PLN"}:
                raise ValueError("measured cost needs a supported currency")
        elif self.currency is not None:
            raise ValueError("currency without measured cost")

    @property
    def config_fingerprint(self) -> str:
        return fingerprint([self.profile, self.request, self.request_context])

    @property
    def scope(self) -> str:
        return fingerprint([self.task_kind, self.context, self.edition,
                            self.model_fingerprint, self.server_profile_fingerprint])


@dataclass(frozen=True, slots=True)
class ConfigurationRecommendation:
    profile: str
    request: dict[str, Any]
    reason: str
    rolled_back: bool = False
    decision_id: int | None = None
    policy_version: int = POLICY_VERSION
    allowed: bool = True


class InferenceOutcomeStore:
    def __init__(self, root: Path, policy: OutcomePolicy | None = None) -> None:
        self.root = Path(root)
        self.path = self.root / "inference-outcomes.sqlite3"
        self.error = ""
        self.policy = policy or OutcomePolicy()
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            policy_path = self.root / "outcome-policy.json"
            if policy is None and policy_path.exists():
                self.policy = OutcomePolicy(**json.loads(policy_path.read_text()))
            with self._connection() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in {0, SCHEMA_VERSION}:
                    raise ValueError("unsupported outcome database schema")
                db.execute("CREATE TABLE IF NOT EXISTS outcomes (id INTEGER PRIMARY KEY, task TEXT, config TEXT, signal TEXT, scope TEXT, data TEXT, UNIQUE(task,config,signal,scope))")
                db.execute("CREATE INDEX IF NOT EXISTS outcome_scope ON outcomes(scope,id)")
                db.execute("CREATE TABLE IF NOT EXISTS states (scope TEXT PRIMARY KEY, data TEXT)")
                db.execute("CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, scope TEXT, data TEXT)")
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error, ValueError, TypeError) as error:
            self.error = f"outcome memory disabled: {type(error).__name__}"
        self.policy_id = fingerprint([POLICY_VERSION, asdict(self.policy)])

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=2)
        try:
            with db:
                yield db
        finally:
            db.close()

    def append(self, outcome: InferenceOutcome) -> bool:
        if self.error:
            return False
        try:
            # Validate again: frozen dataclasses still contain mutable mappings.
            checked = InferenceOutcome(**asdict(outcome))
            with self._connection() as db:
                cursor = db.execute("INSERT OR IGNORE INTO outcomes(task,config,signal,scope,data) VALUES(?,?,?,?,?)",
                                    (checked.task_id, checked.config_fingerprint, checked.signal, checked.scope,
                                     json.dumps(asdict(checked), allow_nan=False)))
                return cursor.rowcount == 1
        except (sqlite3.Error, OSError, ValueError, TypeError):
            self.error = "outcome write failed; learning disabled"
            return False

    def _rows(self, db, scope: str | None = None, task_id: str | None = None):
        clauses, args = [], []
        if scope is not None:
            clauses.append("scope=?")
            args.append(scope)
        if task_id is not None:
            clauses.append("task=?")
            args.append(task_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = db.execute("SELECT id,task,config,signal,scope,data FROM outcomes" + where + " ORDER BY id DESC LIMIT 20000", args).fetchall()
        result = []
        for sequence, task, config, signal, stored_scope, data in reversed(rows):
            # Invalid evidence disables adaptive selection, rather than silently
            # deleting negative observations and promoting the remaining successes.
            item = InferenceOutcome(**json.loads(data))
            if (item.task_id, item.config_fingerprint, item.signal, item.scope) != (task, config, signal, stored_scope):
                raise ValueError("outcome identity index mismatch")
            result.append((sequence, item))
        return result

    def load(self) -> list[InferenceOutcome]:
        if self.error:
            return []
        try:
            with self._connection() as db:
                return [row for _, row in self._rows(db)]
        except (sqlite3.Error, OSError, ValueError, TypeError):
            self.error = "invalid outcome history; learning disabled"
            return []

    def record_feedback(self, task_id: str, positive: bool) -> InferenceOutcome:
        if type(positive) is not bool or self.error:
            raise ValueError(self.error or "feedback must be explicit")
        try:
            with self._connection() as db:
                rows = [o for _, o in self._rows(db, task_id=task_id) if o.signal == "runtime"]
            # A whole-task rating cannot safely blame any single model in a mixed run.
            if len(rows) != 1:
                raise ValueError("feedback requires a task with one attributable configuration")
            previous = rows[0]
            feedback = replace(previous, signal="user_feedback", outcome="positive" if positive else "negative",
                               verified=True, failure_domain=previous.failure_domain, latency_ms=None, total_tokens=None,
                               cost=None, currency=None, observation_id=uuid4().hex)
            if not self.append(feedback):
                raise ValueError(self.error or "feedback already recorded for this task")
            return feedback
        except sqlite3.Error as error:
            raise ValueError("feedback storage unavailable") from error

    def _stats(self, items):
        runtime = [o for o in items if o.signal == "runtime" and o.verified][-self.policy.window:]
        feedback = [o for o in items if o.signal == "user_feedback" and o.failure_domain in {"none", "model"}][-self.policy.window:]
        rate = sum(o.outcome == "success" for o in runtime) / len(runtime) if runtime else 0.0
        tail = runtime[-self.policy.rollback_failures:]
        regressed = (len(tail) == self.policy.rollback_failures and all(o.outcome == "failure" for o in tail))
        # Positive ratings never turn unverified prose or failed work into success.
        veto = any(o.outcome == "negative" for o in feedback)
        reliable = len(runtime) >= self.policy.min_observations
        metrics = [o for o in runtime if o.outcome == "success"]
        latency = [o.latency_ms for o in metrics if o.latency_ms is not None]
        costs = [o.cost for o in metrics if o.cost is not None]
        currencies = {o.currency for o in metrics if o.cost is not None}
        return {"n": len(runtime), "rate": rate, "regressed": regressed,
                "eligible": reliable and rate >= self.policy.minimum_success_rate and not veto and not regressed,
                "veto": veto, "feedback": len(feedback),
                "latency": median(latency) if len(latency) >= self.policy.min_observations else None,
                "cost": median(costs) if len(costs) >= self.policy.min_observations and len(currencies) == 1 else None,
                "currency": next(iter(currencies)) if len(currencies) == 1 else None}

    def recommend(self, *, task_kind: str, context: str, model_fingerprint: str,
                  server_profile_fingerprint: str, edition: str, default_profile: str,
                  default_request: Mapping[str, Any], request_context: str) -> ConfigurationRecommendation:
        fallback = ConfigurationRecommendation(default_profile, dict(default_request),
                                               self.error or "qualified default: insufficient verified evidence")
        if self.error:
            return fallback
        try:
            prototype = InferenceOutcome(task_id="decision", task_kind=task_kind, context=context,
                edition=edition, model_fingerprint=model_fingerprint, server_profile_fingerprint=server_profile_fingerprint,
                profile=default_profile, request=dict(default_request), request_context=request_context)
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                rows = self._rows(db, scope=prototype.scope)
                groups: dict[str, list[InferenceOutcome]] = {}
                for _, item in rows:
                    from ..inference import PUBLIC_INFERENCE_PROFILES
                    preset_ok = edition == "full" or item.request == asdict(PUBLIC_INFERENCE_PROFILES[item.profile])
                    if item.request_context == request_context and preset_ok:
                        groups.setdefault(item.config_fingerprint, []).append(item)
                stats = {key: self._stats(items) for key, items in groups.items()}
                state_key = fingerprint([prototype.scope, request_context, self.policy_id])
                saved = db.execute("SELECT data FROM states WHERE scope=?", (state_key,)).fetchone()
                state = json.loads(saved[0]) if saved else {}
                if not isinstance(state, dict) or set(state) - {"incumbent", "blocked", "changed_at"}:
                    raise ValueError("invalid adaptive state")
                if type(state.get("changed_at", 0)) is not int or state.get("changed_at", 0) < 0:
                    raise ValueError("invalid decision cooldown")
                blocked = set(state.get("blocked", []))
                incumbent = state.get("incumbent", prototype.config_fingerprint)
                if not all(_digest(k) for k in (*blocked, incumbent)):
                    raise ValueError("invalid incumbent fingerprint")
                for key, stat in stats.items():
                    if stat["regressed"] or stat["veto"] or (stat["n"] >= self.policy.min_observations and stat["rate"] < self.policy.minimum_success_rate):
                        blocked.add(key)
                quality_count = sum(o.signal == "runtime" and o.verified for items in groups.values() for o in items)
                rolled_back = incumbent in blocked
                reason = fallback.reason
                old = incumbent
                eligible = [key for key, stat in stats.items() if stat["eligible"] and key not in blocked]
                eligible.sort(key=lambda key: (-stats[key]["rate"], key))
                if rolled_back or incumbent not in groups:
                    incumbent = prototype.config_fingerprint
                    if eligible:
                        incumbent = eligible[0]
                        reason = "selected proven configuration: minimum verified observations and quality gate met"
                    if rolled_back:
                        reason = "rollback: verified failures or explicit negative feedback; configuration quarantined"
                current = stats.get(incumbent)
                cooldown = quality_count - int(state.get("changed_at", 0)) < self.policy.cooldown_observations
                if not rolled_back and not cooldown:
                    for key in eligible:
                        if key == incumbent:
                            continue
                        candidate = stats[key]
                        better_quality = current is None or not current["eligible"] or candidate["rate"] >= current["rate"] + self.policy.promotion_margin
                        # Efficiency requires equal measured quality and no feedback disadvantage.
                        better_efficiency = False
                        if current and candidate["rate"] == current["rate"] and candidate["feedback"] >= current["feedback"]:
                            for metric in ("latency", "cost"):
                                a, b = candidate[metric], current[metric]
                                same_unit = metric != "cost" or candidate["currency"] == current["currency"]
                                if same_unit and a is not None and b is not None and a < b * (1 - self.policy.efficiency_margin):
                                    better_efficiency = True
                        if better_quality or better_efficiency:
                            incumbent, current = key, candidate
                            reason = "promoted: verified quality threshold and margin" if better_quality else "promoted: equal verified quality, measured efficiency margin"
                            break
                allowed = incumbent not in blocked
                chosen = groups[incumbent][-1] if incumbent in groups and allowed else prototype
                if not allowed:
                    reason = "no safe fallback: default configuration is quarantined; select another profile/model or review outcome memory"
                if incumbent == old and saved and not rolled_back and allowed:
                    reason = "kept incumbent: hysteresis / no reliable improvement"
                state = {"incumbent": incumbent, "blocked": sorted(blocked),
                         "changed_at": quality_count if incumbent != old else int(state.get("changed_at", 0))}
                db.execute("INSERT OR REPLACE INTO states VALUES(?,?)", (state_key, json.dumps(state)))
                audit = {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION,
                         "policy": asdict(self.policy), "policy_id": self.policy_id,
                         "selected": chosen.config_fingerprint, "reason": reason,
                         "profile": chosen.profile, "request": chosen.request,
                         "task_kind": task_kind, "context": context, "edition": edition,
                         "model_fingerprint": model_fingerprint,
                         "server_profile_fingerprint": server_profile_fingerprint,
                         "request_context": request_context,
                         "created_at": datetime.now(UTC).isoformat(),
                         "rolled_back": rolled_back, "allowed": allowed, "statistics": stats, "blocked": sorted(blocked)}
                decision_id = db.execute("INSERT INTO decisions(scope,data) VALUES(?,?)", (state_key, json.dumps(audit))).lastrowid
                return ConfigurationRecommendation(chosen.profile, dict(chosen.request), reason, rolled_back, decision_id, allowed=allowed)
        except (sqlite3.Error, OSError, ValueError, TypeError, KeyError):
            self.error = "invalid outcome history or decision state; learning disabled"
            return replace(fallback, reason=self.error)

    def model_adjustment(self, *, task_kind: str, context: str, model_fingerprint: str,
                         server_profile_fingerprint: str, edition: str) -> int:
        """Conservative quality-only routing adjustment; never unlocks qualification."""
        scope = fingerprint([task_kind, context, edition, model_fingerprint, server_profile_fingerprint])
        if self.error:
            return 0
        try:
            with self._connection() as db:
                groups = {}
                from ..inference import PUBLIC_INFERENCE_PROFILES
                for _, item in self._rows(db, scope=scope):
                    if edition == "public" and item.request != asdict(PUBLIC_INFERENCE_PROFILES[item.profile]):
                        continue
                    groups.setdefault(item.config_fingerprint, []).append(item)
                stats = [self._stats(items) for items in groups.values()]
                reliable = [s for s in stats if s["n"] >= self.policy.min_observations]
                eligible = [
                    s
                    for s in reliable
                    if not s["regressed"]
                    and not s["veto"]
                    and s["rate"] >= self.policy.minimum_success_rate
                ]
                # Quarantine a failed request configuration locally.  Do not
                # demote the whole model while another sufficiently sampled,
                # reliable configuration for the same task remains healthy.
                if eligible:
                    best_rate = max(s["rate"] for s in eligible)
                    return round(
                        (best_rate - self.policy.minimum_success_rate)
                        * self.policy.max_router_adjustment
                    )
                if any(s["regressed"] or s["veto"] for s in stats):
                    return -self.policy.max_router_adjustment
                return (
                    -self.policy.max_router_adjustment
                    if reliable
                    else 0
                )
        except (sqlite3.Error, OSError, ValueError, TypeError):
            self.error = "invalid outcome history; learning disabled"
            return 0

    def inspect(self) -> dict[str, Any]:
        records = self.load()
        decision = None
        if not self.error:
            try:
                with self._connection() as db:
                    row = db.execute("SELECT id,data FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
                    if row:
                        decision = {"id": row[0], **json.loads(row[1])}
            except (sqlite3.Error, ValueError, TypeError):
                self.error = "invalid decision audit; learning disabled"
        return {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION,
                "policy": asdict(self.policy), "error": self.error, "observations": len(records),
                "last_decision": decision,
                "last_task_id": records[-1].task_id if records else None}
