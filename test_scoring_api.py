"""API tests. Run from the project folder:  pytest -q"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import scoring_api

T0 = datetime(2026, 5, 4, 9, 0)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("API_KEY", "ADMIN_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)          # offline stub + template, no network
    scoring_api._ATTACK_CACHE.clear()


client = TestClient(scoring_api.app)


def rider(rid, n_customers=12, jobs_each=2, phone=""):
    jobs = [{"customer_id": f"c{rid}{i}", "logged_at": (T0 + timedelta(weeks=(i * jobs_each + k) % 8, days=k)).isoformat(),
             "status": "CONFIRMED", "confirm_device": f"dev_{rid}{i}"}
            for i in range(n_customers) for k in range(jobs_each)]
    return {"rider_id": rid, "phone": phone, "jobs": jobs}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score_one_rider():
    r = client.post("/api/score", json={"rider": rider("A")})
    assert r.status_code == 200
    a = r.json()["assessment"]
    assert a["total"] is not None and a["band"] in ("Low", "Building", "Strong")
    assert [c["name"] for c in a["components"]] == ["counterparty_diversity", "time_spread", "repeat_customers"]
    assert r.json()["explanation"] is None


def test_score_with_explanation_uses_template_offline():
    r = client.post("/api/score", json={"rider": rider("A"), "explain": True, "lang": "sw"})
    e = r.json()["explanation"]
    assert e["source"] == "template" and "Alama" in e["text"]


def test_not_enough_evidence():
    r = client.post("/api/score", json={"rider": rider("N", n_customers=2, jobs_each=1)})
    assert r.json()["assessment"]["band"] == "Not enough evidence yet"


def test_ring_detected_with_peers():
    a, b = rider("RA", phone="p_a"), rider("RB", phone="p_b")
    a["jobs"] = [{**j, "customer_id": "p_b"} for j in a["jobs"]]     # B confirms A's jobs
    b["jobs"] = [{**j, "customer_id": "p_a"} for j in b["jobs"]]     # A confirms B's jobs
    r = client.post("/api/score", json={"rider": a, "peers": [b]})
    assert "R1" in [f["code"] for f in r.json()["assessment"]["flags"]]


def test_batch_and_duplicate_ids():
    ok = client.post("/api/score/batch", json={"riders": [rider("A"), rider("B")]})
    assert set(ok.json()["results"]) == {"A", "B"}
    dup = client.post("/api/score/batch", json={"riders": [rider("A"), rider("A")]})
    assert dup.status_code == 422


def test_explain_endpoint_round_trip_and_bad_input():
    a = client.post("/api/score", json={"rider": rider("A")}).json()["assessment"]
    assert client.post("/api/explain", json={"assessment": a}).json()["text"]
    assert client.post("/api/explain", json={"assessment": {"nonsense": 1}}).status_code == 422


def test_timezone_aware_and_naive_timestamps_mix():
    r = rider("T")
    r["jobs"][0]["logged_at"] = "2026-05-04T06:00:00Z"
    assert client.post("/api/score", json={"rider": r}).status_code == 200


def test_validation_errors_are_422():
    assert client.post("/api/score", json={"rider": {"rider_id": "X", "jobs": [{"customer_id": "c"}]}}).status_code == 422
    bad = rider("B"); bad["jobs"][0]["status"] = "DONE"
    assert client.post("/api/score", json={"rider": bad}).status_code == 422


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/api/score", json={"rider": rider("A")}).status_code == 401
    ok = client.post("/api/score", json={"rider": rider("A")}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200                    # health stays open


def test_attack_demo_scores_fake_low_and_needs_admin_key(monkeypatch):
    r = client.post("/api/attack-demo", json={"level": "naive"})
    d = r.json()
    assert d["jobs_source"] == "offline-stub" and d["jobs_generated"] == 200
    assert d["assessment"]["band"] == "Low" and d["assessment"]["status"] == "flagged"
    assert d["explanation"]["text"]
    monkeypatch.setenv("ADMIN_KEY", "adm")
    assert client.post("/api/attack-demo", json={}).status_code == 401
    assert client.post("/api/attack-demo", json={}, headers={"X-Admin-Key": "adm"}).status_code == 200


def test_demo_genuine_scores_high():
    d = client.get("/api/demo/genuine").json()
    assert d["assessment"]["band"] == "Strong" and d["assessment"]["flags"] == []


def test_config_lists_thresholds():
    c = client.get("/api/config").json()
    assert c["weights"] == {"counterparty_diversity": 45, "time_spread": 35, "repeat_customers": 20}
    assert set(c["fraud_checks"]) == {"B1", "B2", "A1", "D1", "D2", "R1", "R2"}
