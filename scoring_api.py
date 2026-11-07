"""
NELP scoring service (FastAPI). Stateless: send jobs in, get score + flags + explanation out.
No database, no SMS, no login. Those belong to the site. This is the Data/AI part.

Run standalone:   uvicorn scoring_api:app --host 0.0.0.0 --port 8000
Or mount in the team's site:   from scoring_api import router; app.include_router(router)
Interactive docs for teammates:  http://localhost:8000/docs

Environment variables (set them as secrets on the host, never commit them):
  API_KEY      if set, every /api call must send header  X-API-Key: <value>
  ADMIN_KEY    if set, /api/attack-demo also needs       X-Admin-Key: <value>
  OPENAI_API_KEY, ANTHROPIC_API_KEY   used by llm_tools (offline fallbacks if missing)
  CORS_ORIGINS comma-separated origins; only needed if a separate frontend calls this
If API_KEY / ADMIN_KEY are NOT set the endpoints are open: fine on a laptop, not when deployed.

Timestamps: send ISO 8601, ideally UTC ("2026-10-31T09:30:00Z"). They are converted to naive UTC.
"""
from __future__ import annotations

import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import llm_tools
import work_passport as wp
from work_passport import Job, assess, build_context, explanation_payload

Status = Literal["PENDING", "CONFIRMED", "EXPIRED", "DISPUTED"]
Lang = Literal["en", "sw"]


# ------------------------------------------------------------------ schemas
class JobIn(BaseModel):
    customer_id: str = Field(..., description="customer phone number in one normalised format, as text")
    logged_at: datetime
    status: Status = "CONFIRMED"
    confirmed_at: Optional[datetime] = None
    confirm_device: str = Field("", description="device ID from the confirmation page")
    rider_device: str = Field("", description="device ID the rider was logged in from")


class RiderIn(BaseModel):
    rider_id: str
    phone: str = Field("", description="rider's own phone (lets ring checks match confirmers to riders)")
    signup_at: Optional[datetime] = Field(None, description="rider.created_at; enables the A1 check")
    jobs: list[JobIn]


class ScoreRequest(BaseModel):
    rider: RiderIn
    peers: list[RiderIn] = Field(default_factory=list,
                                 description="OTHER riders, for the ring and shared-device checks. "
                                             "Omit for a stand-alone score.")
    explain: bool = False
    lang: Lang = "en"


class BatchRequest(BaseModel):
    riders: list[RiderIn] = Field(..., max_length=500)
    explain: bool = False
    lang: Lang = "en"


class ExplainRequest(BaseModel):
    assessment: dict = Field(..., description="the `assessment` object returned by /api/score")
    lang: Lang = "en"


class AttackRequest(BaseModel):
    level: Literal["naive", "careful"] = "naive"
    n_jobs: int = Field(200, ge=50, le=300)
    lang: Lang = "en"
    refresh: bool = Field(False, description="skip the cached fake and call OpenAI again")


# ------------------------------------------------------------------ helpers
def _naive(dt):
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _jobs(r: RiderIn):
    return [Job(r.rider_id, j.customer_id, _naive(j.logged_at), j.status, _naive(j.confirmed_at),
                j.confirm_device, j.rider_device) for j in r.jobs]


def _explained(a, lang):
    text, source = llm_tools.explain(a, lang)
    return {"text": text, "source": source}


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_key(x_api_key: str = Header(default="")):
    key = os.getenv("API_KEY")
    if key and not _same(x_api_key, key):
        raise HTTPException(401, "missing or wrong X-API-Key")


def require_admin(x_admin_key: str = Header(default="")):
    key = os.getenv("ADMIN_KEY")
    if key and not _same(x_admin_key, key):
        raise HTTPException(401, "missing or wrong X-Admin-Key")


router = APIRouter(prefix="/api", dependencies=[Depends(require_key)])


# ------------------------------------------------------------------ endpoints
@router.post("/score", summary="Score one rider (optionally against peers for ring/device checks)")
def score_rider(req: ScoreRequest):
    riders = [req.rider] + [p for p in req.peers if p.rider_id != req.rider.rider_id]
    ctx = build_context({r.rider_id: _jobs(r) for r in riders},
                        {r.rider_id: r.phone for r in riders if r.phone})
    a = assess(req.rider.rider_id, _jobs(req.rider), ctx, _naive(req.rider.signup_at))
    return {"assessment": a, "explanation": _explained(a, req.lang) if req.explain else None}


@router.post("/score/batch", summary="Score many riders together (full cross-rider fraud checks)")
def score_batch(req: BatchRequest):
    ids = [r.rider_id for r in req.riders]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "duplicate rider_id in batch")
    ctx = build_context({r.rider_id: _jobs(r) for r in req.riders},
                        {r.rider_id: r.phone for r in req.riders if r.phone})
    out = {}
    for r in req.riders:
        a = assess(r.rider_id, _jobs(r), ctx, _naive(r.signup_at))
        out[r.rider_id] = {"assessment": a, "explanation": _explained(a, req.lang) if req.explain else None}
    return {"results": out}


@router.post("/explain", summary="Plain-language explanation of an assessment (Claude, with template fallback)")
def explain_assessment(req: ExplainRequest):
    try:
        explanation_payload(req.assessment)
    except (KeyError, TypeError):
        raise HTTPException(422, "assessment must be the object returned by /api/score")
    return _explained(req.assessment, req.lang)


@router.get("/config", summary="Weights, caps, bands and fraud thresholds (for the profile page)")
def config():
    return {
        "max_score": 100,
        "weights": {"counterparty_diversity": wp.W_DIVERSITY, "time_spread": wp.W_SPREAD,
                    "repeat_customers": wp.W_REPEAT},
        "full_marks_at": {"effective_customers": wp.FULL_EFFECTIVE_CUSTOMERS,
                          "active_weeks": wp.FULL_ACTIVE_WEEKS, "repeat_rate": wp.FULL_REPEAT_RATE},
        "min_confirmed_jobs": wp.MIN_CONFIRMED_JOBS,
        "high_flag_cap": wp.HIGH_CAP, "medium_flag_deduction": wp.MEDIUM_DEDUCTION,
        "bands": {"Low": f"0-{wp.BAND_BUILDING_FROM - 1}",
                  "Building": f"{wp.BAND_BUILDING_FROM}-{wp.BAND_STRONG_FROM - 1}",
                  "Strong": f"{wp.BAND_STRONG_FROM}-100"},
        "fraud_checks": {
            "B1": f"more than {wp.B1_MAX_JOBS_PER_HOUR} confirmed jobs in 60 min (medium)",
            "B2": f">={int(wp.B2_SHARE * 100)}% of confirmed jobs in 24h, min {wp.B2_MIN_JOBS} jobs (high)",
            "A1": f">={wp.A1_JOBS} confirmed jobs within {wp.A1_WINDOW_HOURS}h of sign-up (medium)",
            "D1": f"one device confirms for >={wp.D1_CUSTOMERS_PER_DEVICE} customer numbers (medium)",
            "D2": f">={wp.D2_MIN_SELF_CONFIRMED} jobs confirmed from the rider's own device (high)",
            "R1": f"closed loop of {wp.R1_MIN_CYCLE}-{wp.R1_MAX_CYCLE} rider accounts (high)",
            "R2": f"pool of <={wp.R2_POOL_MAX} numbers covers >={int(wp.R2_SHARE * 100)}% of jobs "
                  f"for {wp.R2_MIN_RIDERS}+ riders (high)"},
    }


_ATTACK_CACHE = {}


@router.post("/attack-demo", dependencies=[Depends(require_admin)],
             summary="ADMIN: have OpenAI fabricate a rider history, score it and explain why it scores low")
def attack_demo(req: AttackRequest):
    key = (req.level, req.n_jobs)
    if req.refresh or key not in _ATTACK_CACHE:
        rid = f"FAKE_{req.level.upper()}"
        jobs, source = llm_tools.generate_fake_history(rid, req.level, req.n_jobs)
        _ATTACK_CACHE[key] = (rid, jobs, source)
    rid, jobs, source = _ATTACK_CACHE[key]
    first = min(j.logged_at for j in jobs)
    # same assumption as the data generator: a lazy fabricator signs up just before inserting jobs,
    # a careful one ages the account first
    signup = first - (timedelta(hours=2) if req.level == "naive" else timedelta(days=30))
    ctx = build_context({rid: jobs}, {rid: f"phone_{rid}"})
    a = assess(rid, jobs, ctx, signup)
    return {"rider_id": rid, "jobs_source": source, "level": req.level, "jobs_generated": len(jobs),
            "assessment": a, "explanation": _explained(a, req.lang)}


@router.get("/demo/genuine", summary="A seeded genuine full-time rider, scored (the 'real rider scores high' demo)")
def demo_genuine(seed: int = 3, lang: Lang = "en"):
    from generate_synthetic_data import Gen          # lazy: pulls in pandas only when used
    g = Gen(seed)
    g.genuine("DEMO_GENUINE", "full_time")
    jobs = [Job("DEMO_GENUINE", x["customer_phone"], x["logged_at"], x["status"],
                x["confirmed_at"] or None, x["confirm_device"], x["rider_device"]) for x in g.rows]
    signup = g.riders[0]["signup_at"]
    ctx = build_context({"DEMO_GENUINE": jobs}, {"DEMO_GENUINE": g.riders[0]["phone"]})
    a = assess("DEMO_GENUINE", jobs, ctx, signup)
    return {"rider_id": "DEMO_GENUINE", "jobs_logged": len(jobs), "assessment": a,
            "explanation": _explained(a, lang)}


# ------------------------------------------------------------------ app
app = FastAPI(title="NELP scoring service", version="1.0",
              description="Rule-based trust score, fraud checks and explanations for Nelp riders.")
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/health", summary="Liveness check for deploy and uptime monitoring")
def health():
    return {"status": "ok"}
