"""
NELP / Work Passport: scoring engine + fraud checks, exactly as in the technical
proposal v1.0 (sections 8 and 9). PURE FUNCTIONS, NO I/O, so backend code, the
data generator and the tests can all call the same thing.

Contract (section 14):
    assess(rider_id, jobs, ctx, signup=None) -> {total, band, status, label, components[], flags[], facts}
    score(jobs, flags)          -> same, when flags are already known

All thresholds below are hand-set priors from the proposal, not fitted values.
They are FROZEN before held-out data is generated (proposal section 12).
Lines marked ASSUMPTION are ones the proposal does not specify.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import networkx as nx

# ----------------------------------------------------------------- score (sec 8)
W_DIVERSITY, W_SPREAD, W_REPEAT = 45, 35, 20
FULL_EFFECTIVE_CUSTOMERS = 30
FULL_ACTIVE_WEEKS = 12
FULL_REPEAT_RATE = 0.30
REPEAT_MIN_CUSTOMERS = 10
MIN_CONFIRMED_JOBS = 5          # below this: "Not enough evidence yet"
HIGH_CAP = 20                   # high-severity flag caps the score
MEDIUM_DEDUCTION = 15           # medium-severity flag deducts points (floor 0)
BAND_BUILDING_FROM, BAND_STRONG_FROM = 40, 70

# ----------------------------------------------------------------- fraud (sec 9)
B1_MAX_JOBS_PER_HOUR = 10       # more than this in any 60 min -> medium
B2_SHARE, B2_MIN_JOBS = 0.40, 20   # >=40% of confirmed jobs (min 20) in one 24h -> high
D1_CUSTOMERS_PER_DEVICE = 4     # one device confirms for >=4 customer numbers -> medium
D2_MIN_SELF_CONFIRMED = 3       # ASSUMPTION: proposal gives no count for self-attestation
R1_MIN_CYCLE, R1_MAX_CYCLE = 2, 4
R2_POOL_MAX, R2_SHARE, R2_MIN_RIDERS = 8, 0.90, 2
# A1 is NOT in the proposal: an addition taken from the XGBoost review model's "account age /
# signup -> first activity" features. Needs team agreement before the threshold freeze.
A1_JOBS, A1_WINDOW_HOURS = 20, 48   # >=20 confirmed jobs within 48h of sign-up -> medium


@dataclass
class Job:
    rider_id: str
    customer_id: str                 # customer phone number (never shown publicly)
    logged_at: datetime
    status: str = "CONFIRMED"        # PENDING | CONFIRMED | EXPIRED | DISPUTED
    confirmed_at: datetime | None = None
    confirm_device: str = ""         # browser device ID on the confirmation page
    rider_device: str = ""           # device the rider was logged in from

    @property
    def confirmed(self) -> bool:
        return self.status == "CONFIRMED"

    @property
    def when(self) -> datetime:
        return self.confirmed_at or self.logged_at


@dataclass
class Context:
    """Cross-rider facts needed by the fraud checks. Build once per scoring run."""
    device_customers: dict = field(default_factory=dict)   # device -> {customers}
    ring_cycles: dict = field(default_factory=dict)         # rider -> longest cycle len
    pool_hits: dict = field(default_factory=dict)           # rider -> {pool_size, share, riders}


# ------------------------------------------------------------------ score
def _band(total):
    if total >= BAND_STRONG_FROM:
        return "Strong"
    return "Building" if total >= BAND_BUILDING_FROM else "Low"


def score(jobs, flags=None):
    flags = flags or []
    conf = [j for j in jobs if j.confirmed]            # unconfirmed NEVER count
    n = len(conf)
    facts = {"logged_jobs": len(jobs), "confirmed_jobs": n}
    if n < MIN_CONFIRMED_JOBS:
        return {"total": None, "raw_total": None, "band": "Not enough evidence yet",
                "status": "insufficient", "label": "Not enough evidence yet",
                "components": [], "flags": flags, "facts": facts}

    counts = Counter(j.customer_id for j in conf)
    eff = 1 / sum((c / n) ** 2 for c in counts.values())          # effective customers
    weeks = len({j.when.isocalendar()[:2] for j in conf})
    days = defaultdict(set)
    for j in conf:
        days[j.customer_id].add(j.when.date())
    repeat_rate = sum(1 for d in days.values() if len(d) >= 2) / len(counts)
    top3 = sum(c for _, c in counts.most_common(3))

    diversity = W_DIVERSITY * min(1, eff / FULL_EFFECTIVE_CUSTOMERS)
    spread = W_SPREAD * min(1, weeks / FULL_ACTIVE_WEEKS)
    repeat = (W_REPEAT * min(1, repeat_rate / FULL_REPEAT_RATE)
              * min(1, len(counts) / REPEAT_MIN_CUSTOMERS))
    raw = diversity + spread + repeat

    total, status, label = raw, "ok", None
    if any(f["severity"] == "high" for f in flags):
        total, status, label = min(raw, HIGH_CAP), "flagged", "Flagged, under review"
    elif any(f["severity"] == "medium" for f in flags):
        total, status, label = max(0.0, raw - MEDIUM_DEDUCTION), "caution", "Caution"

    facts.update(distinct_customers=len(counts), effective_customers=round(eff, 1),
                 active_weeks=weeks, repeat_rate_pct=round(100 * repeat_rate),
                 top3_jobs=top3, top3_share_pct=round(100 * top3 / n))
    comp = lambda name, pts, mx, detail: {"name": name, "points": round(pts, 1), "max": mx, "detail": detail}
    return {
        "total": round(total, 1), "raw_total": round(raw, 1), "band": _band(total),
        "status": status, "label": label,
        "components": [
            comp("counterparty_diversity", diversity, W_DIVERSITY,
                 f"{round(eff, 1)} effective customers (full marks at {FULL_EFFECTIVE_CUSTOMERS})"),
            comp("time_spread", spread, W_SPREAD,
                 f"{weeks} active weeks (full marks at {FULL_ACTIVE_WEEKS})"),
            comp("repeat_customers", repeat, W_REPEAT,
                 f"{round(100 * repeat_rate)}% of customers returned on another day"),
        ],
        "flags": flags, "facts": facts,
    }


# ------------------------------------------------------------------ fraud checks
def _flag(code, check, severity, reason, **evidence):
    return {"code": code, "check": check, "severity": severity, "reason": reason, "evidence": evidence}


def burst_flags(conf):
    """B1: >10 confirmed jobs in any 60 min (medium). B2: >=40% of jobs in one 24h (high)."""
    ts = sorted(j.when for j in conf)
    out = []

    def peak(window):
        j, best = 0, 1 if ts else 0
        for i in range(len(ts)):
            while ts[i] - ts[j] > window:
                j += 1
            best = max(best, i - j + 1)
        return best

    hour = peak(timedelta(minutes=60))
    if hour > B1_MAX_JOBS_PER_HOUR:
        out.append(_flag("B1", "burst", "medium",
                         f"{hour} confirmed jobs were logged within one hour", peak_jobs_in_hour=hour))
    if len(ts) >= B2_MIN_JOBS:
        day = peak(timedelta(hours=24))
        if day / len(ts) >= B2_SHARE:
            out.append(_flag("B2", "burst", "high",
                             f"{day} of {len(ts)} confirmed jobs fall inside one 24-hour window",
                             jobs_in_24h=day, confirmed_jobs=len(ts)))
    return out


def account_age_flags(conf, signup):
    """A1: >=20 confirmed jobs within 48 hours of the rider account being created (medium)."""
    if signup is None:
        return []
    k = sum(1 for j in conf if j.when - signup <= timedelta(hours=A1_WINDOW_HOURS))
    if k >= A1_JOBS:
        return [_flag("A1", "account_age", "medium",
                      f"{k} confirmed jobs were recorded within {A1_WINDOW_HOURS} hours of sign-up",
                      jobs_in_first_48h=k)]
    return []


def device_flags(rider_id, conf, ctx):
    """D1: one device confirmed for >=4 customer numbers (medium).
    D2: confirmations made from the rider's own device (high)."""
    out = []
    devs = {j.confirm_device for j in conf if j.confirm_device}
    worst = max((len(ctx.device_customers.get(d, ())) for d in devs), default=0)
    if worst >= D1_CUSTOMERS_PER_DEVICE:
        out.append(_flag("D1", "device", "medium",
                         f"one device confirmed jobs for {worst} different customer numbers",
                         customers_on_one_device=worst))
    selfc = sum(1 for j in conf if j.confirm_device and j.confirm_device == j.rider_device)
    if selfc >= D2_MIN_SELF_CONFIRMED:
        out.append(_flag("D2", "device", "high",
                         f"{selfc} jobs were confirmed from the rider's own device",
                         self_confirmed_jobs=selfc))
    return out


def ring_flags(rider_id, ctx):
    out = []
    if rider_id in ctx.ring_cycles:
        k = ctx.ring_cycles[rider_id]
        out.append(_flag("R1", "ring", "high",
                         f"confirmations form a closed loop of {k} rider accounts", cycle_length=k))
    if rider_id in ctx.pool_hits:
        h = ctx.pool_hits[rider_id]
        out.append(_flag("R2", "ring", "high",
                         f"a pool of {h['pool_size']} numbers accounts for {h['share_pct']}% of "
                         f"confirmed jobs for {h['riders']} riders", **h))
    return out


def fraud_flags(rider_id, jobs, ctx, signup=None):
    conf = [j for j in jobs if j.confirmed]
    return (burst_flags(conf) + account_age_flags(conf, signup)
            + device_flags(rider_id, conf, ctx) + ring_flags(rider_id, ctx))


# ------------------------------------------------------------------ cross-rider context
def build_context(jobs_by_rider, rider_phones):
    """jobs_by_rider: {rider_id: [Job]}; rider_phones: {rider_id: phone}."""
    ctx = Context()
    phone_to_rider = {p: r for r, p in rider_phones.items()}

    dev = defaultdict(set)
    g = nx.DiGraph()
    counters = {}
    for rid, jobs in jobs_by_rider.items():
        conf = [j for j in jobs if j.confirmed]
        counters[rid] = Counter(j.customer_id for j in conf)
        for j in conf:
            if j.confirm_device:
                dev[j.confirm_device].add(j.customer_id)
            src = phone_to_rider.get(j.customer_id)
            if src and src != rid:
                g.add_edge(src, rid)                 # confirmer account -> rider account
    ctx.device_customers = dev

    # R1: directed cycles of length 2-4 among registered accounts
    for cyc in nx.simple_cycles(g, length_bound=R1_MAX_CYCLE):
        if len(cyc) >= R1_MIN_CYCLE:
            for r in cyc:
                ctx.ring_cycles[r] = max(ctx.ring_cycles.get(r, 0), len(cyc))

    # R2: a pool of <=8 numbers covering >=90% of confirmed jobs for 2+ riders
    pools = {}
    for rid, c in counters.items():
        n = sum(c.values())
        if not n:
            continue
        acc, pool = 0, []
        for phone, k in c.most_common(R2_POOL_MAX):
            pool.append(phone); acc += k
            if acc / n >= R2_SHARE:
                pools[rid] = frozenset(pool)
                break
    for rid, pool in pools.items():
        covered = [r for r, c in counters.items()
                   if sum(c.values()) and sum(c[p] for p in pool) / sum(c.values()) >= R2_SHARE]
        if len(covered) >= R2_MIN_RIDERS:
            for r in covered:
                ctx.pool_hits[r] = {"pool_size": len(pool), "share_pct": round(100 * R2_SHARE),
                                    "riders": len(covered)}
    return ctx


# ------------------------------------------------------------------ one-call API
def assess(rider_id, jobs, ctx=None, signup=None):
    """signup = rider.created_at (datetime) or None; only the A1 check uses it."""
    ctx = ctx or Context()
    result = score(jobs, fraud_flags(rider_id, jobs, ctx, signup))
    result["rider_id"] = rider_id
    return result


def explanation_payload(a):
    """The ONLY thing Claude sees (proposal section 11). Numbers stay deterministic.
    Everything Claude may quote is a field here; percentages are precomputed integers."""
    f = a["facts"]
    return {
        "max_score": 100, "top_n": 3, "min_confirmed_jobs": MIN_CONFIRMED_JOBS,
        "score": a["total"], "score_before_flags": a["raw_total"],
        "band": a["band"], "label": a["label"],
        "confirmed_jobs": f["confirmed_jobs"], "logged_jobs": f["logged_jobs"],
        "distinct_customers": f.get("distinct_customers"),
        "effective_customers": f.get("effective_customers"),
        "top3_jobs": f.get("top3_jobs"), "top3_share_pct": f.get("top3_share_pct"),
        "active_weeks": f.get("active_weeks"), "repeat_rate_pct": f.get("repeat_rate_pct"),
        "components": {c["name"]: c["points"] for c in a["components"]},
        "flags": [{"code": x["code"], "severity": x["severity"], "reason": x["reason"]}
                  for x in a["flags"]],
    }
