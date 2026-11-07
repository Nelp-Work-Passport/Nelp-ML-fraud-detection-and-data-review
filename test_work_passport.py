"""Tests for the scoring + fraud rules. Run from the project folder:  pytest -q"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from work_passport import Job, assess, build_context, explanation_payload, score

T0 = datetime(2026, 5, 4, 9, 0)           # a Monday


def job(rider, cust, when, status="CONFIRMED", dev="", rdev=""):
    return Job(rider, cust, when, status, when + timedelta(minutes=5), dev or f"d_{cust}", rdev)


def spread_jobs(rider, plan, weeks, dev=None):
    """plan: list of (customer, n_jobs). Jobs go on different DAYS cycling through `weeks` weeks."""
    out, i = [], 0
    for cust, k in plan:
        for _ in range(k):
            w, d = i % weeks, (i // weeks) % 5
            out.append(job(rider, cust, T0 + timedelta(weeks=w, days=d, hours=i % 7), dev=dev))
            i += 1
    return out


# ---- proposal section 8 worked examples -------------------------------------------
def test_example_a_established_genuine():
    plan = [(f"r{i}", 3) for i in range(17)] + [("r17", 2)] + [(f"s{i}", 1) for i in range(67)]
    jobs = spread_jobs("A", plan, weeks=10)
    assert len(jobs) == 120 and len(plan) == 85
    r = assess("A", jobs)
    comps = {c["name"]: c["points"] for c in r["components"]}
    assert comps["counterparty_diversity"] == 45.0
    assert comps["time_spread"] == pytest.approx(29.2, abs=0.1)
    assert comps["repeat_customers"] == pytest.approx(14.1, abs=0.1)
    assert round(r["total"]) == 88 and r["band"] == "Strong"


def test_example_b_new_genuine():
    plan = [(f"r{i}", 2) for i in range(5)] + [(f"s{i}", 1) for i in range(15)]
    jobs = spread_jobs("B", plan, weeks=4)
    comps = {c["name"]: c["points"] for c in assess("B", jobs)["components"]}
    assert comps["counterparty_diversity"] == pytest.approx(26.8, abs=0.1)
    assert comps["time_spread"] == pytest.approx(11.7, abs=0.1)
    assert comps["repeat_customers"] == pytest.approx(16.7, abs=0.1)
    assert round(assess("B", jobs)["total"]) == 55


def test_example_c_fake_three_customers():
    # 200 jobs, 3 customers, one week, two days, most inside one 24h window -> B2 high
    jobs = []
    for i in range(200):
        day = 1 if i < 120 else 2
        jobs.append(job("C", f"c{i % 3}", T0 + timedelta(days=day, minutes=5 * (i % 120)), dev=f"d_c{i % 3}"))
    r = assess("C", jobs)
    assert {f["code"] for f in r["flags"]} >= {"B2"}
    assert r["status"] == "flagged" and r["label"] == "Flagged, under review"
    assert round(r["total"]) == 13 and r["band"] == "Low"
    assert r["raw_total"] == pytest.approx(13.4, abs=0.2)


# ---- gates and rules --------------------------------------------------------------
def test_unconfirmed_never_counts():
    jobs = spread_jobs("X", [(f"c{i}", 1) for i in range(10)], weeks=5)
    base = assess("X", jobs)["total"]
    extra = [job("X", f"u{i}", T0 + timedelta(days=i), status=s)
             for i, s in enumerate(["PENDING", "EXPIRED", "DISPUTED"] * 10)]
    assert assess("X", jobs + extra)["total"] == base


def test_minimum_evidence_gate():
    r = assess("N", spread_jobs("N", [(f"c{i}", 1) for i in range(4)], weeks=2))
    assert r["total"] is None and r["band"] == "Not enough evidence yet" and r["status"] == "insufficient"


def test_high_flag_caps_at_20():
    jobs = [job("H", f"c{i % 40}", T0 + timedelta(weeks=i % 12, days=i % 5), dev="rider_phone", rdev="rider_phone")
            for i in range(120)]
    r = assess("H", jobs)                    # self-attestation on its own device -> D2 high
    assert any(f["code"] == "D2" for f in r["flags"])
    assert r["total"] == 20 and r["raw_total"] > 20


def test_medium_flag_deducts_15():
    # 12 jobs within one hour -> B1 medium; fewer than 20 jobs so B2 cannot fire
    jobs = [job("M", f"c{i}", T0 + timedelta(minutes=4 * i)) for i in range(12)]
    r = assess("M", jobs)
    assert [f["code"] for f in r["flags"]] == ["B1"] and r["status"] == "caution"
    assert r["total"] == pytest.approx(max(0, r["raw_total"] - 15), abs=0.1)


def test_ring_cycle_detected():
    phones = {"A": "p_A", "B": "p_B", "C": "p_C"}
    by = {"A": [job("A", "p_C", T0 + timedelta(days=i)) for i in range(8)],
          "B": [job("B", "p_A", T0 + timedelta(days=i)) for i in range(8)],
          "C": [job("C", "p_B", T0 + timedelta(days=i)) for i in range(8)]}
    ctx = build_context(by, phones)
    for rid in by:
        r = assess(rid, by[rid], ctx)
        assert any(f["code"] == "R1" for f in r["flags"]), rid


def test_shared_small_pool_across_riders():
    pool = [f"pool{i}" for i in range(6)]
    by = {rid: [job(rid, pool[i % 6], T0 + timedelta(days=i)) for i in range(30)] for rid in ("P", "Q")}
    ctx = build_context(by, {"P": "pp", "Q": "pq"})
    assert all(any(f["code"] == "R2" for f in assess(r, by[r], ctx)["flags"]) for r in by)


def test_one_device_many_customers():
    jobs = [job("D", f"c{i}", T0 + timedelta(days=i), dev="shared") for i in range(6)]
    ctx = build_context({"D": jobs}, {"D": "pd"})
    assert any(f["code"] == "D1" for f in assess("D", jobs, ctx)["flags"])


def test_payload_contains_every_number_the_explainer_may_use():
    jobs = spread_jobs("E", [(f"c{i}", 2) for i in range(12)], weeks=6)
    p = explanation_payload(assess("E", jobs))
    for k in ("score", "confirmed_jobs", "distinct_customers", "top3_jobs", "active_weeks", "max_score"):
        assert p[k] is not None


def test_a1_many_jobs_right_after_signup():
    signup = T0
    jobs = [job("Z", f"c{i}", T0 + timedelta(hours=1 + i)) for i in range(22)]      # 22 jobs in ~23h
    jobs += [job("Z", f"d{i}", T0 + timedelta(weeks=1 + i % 6, days=i % 5)) for i in range(60)]   # 22 of 82: under B2's 40%
    r = assess("Z", jobs, signup=signup)
    assert any(f["code"] == "A1" for f in r["flags"]) and r["status"] == "caution"
    assert not any(f["code"] == "A1" for f in assess("Z", jobs, signup=T0 - timedelta(days=30))["flags"])
    assert not any(f["code"] == "A1" for f in assess("Z", jobs)["flags"])             # no signup -> skipped
