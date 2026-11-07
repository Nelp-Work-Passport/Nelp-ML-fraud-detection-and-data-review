"""
LLM pieces for NELP (proposal section 11).

  OpenAI -> generate_fake_history(): the demo attack (fabricated rider histories)
  Claude -> explain():               plain-language reason for a score

Both degrade gracefully with no API key (offline stub / template) so the demo never
hard-fails. Claude sees ONLY explanation_payload(); every number in its answer must
appear in that payload, otherwise we discard it and use the deterministic template.

Setup:  pip install openai anthropic
        export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...
        optional: OPENAI_MODEL (default gpt-4o-mini)
                  CLAUDE_MODEL (default claude-haiku-5-5: small and fast per the proposal;
                                confirm the current model name in Anthropic's docs at kick-off)
"""
import hashlib
import json
import os
import random
import re
from datetime import datetime, timedelta

from work_passport import Job, assess, explanation_payload

START = datetime(2026, 4, 1, 8, 0)

# ============================================================== OpenAI: fakes
FAKE_SYSTEM = (
    "You are a red-team data generator for a fraud-detection research project. "
    "You produce FABRICATED boda/delivery rider job histories so we can test whether "
    "our detector catches them. Output JSON only."
)

LEVEL_BRIEF = {
    "naive": "Lazy fabrication: dense, repetitive, few customers, all within a few days.",
    "careful": ("A careful fabricator who knows reviewers want steady weeks: spread the jobs across "
                "about 14 weeks, but reuse only 4 to 6 distinct customers."),
}

FAKE_USER = """Create a fabricated work history for one Nairobi delivery rider that tries
to look genuine to a reviewer. {brief}

Return JSON: {{"customers": <int, number of distinct customers>,
"jobs": [[customer_index, day_offset_0_to_120, minute_of_day_0_to_1439], ...]}}
Exactly {n_jobs} jobs. customer_index is 0..customers-1. No other keys."""

# Fabricator behaviour is OUR modelling assumption, not LLM output:
#   naive   -> confirms everything from one phone (the rider's own device)
#   careful -> clears browser storage / uses many phones, so each customer gets its own device ID
DEVICE_BEHAVIOUR = {"naive": "self", "careful": "distinct"}


def _stub_fake(level, n_jobs, rid):
    """Offline stand-in used when OPENAI_API_KEY is missing. NOT an LLM: labelled as such."""
    rng = random.Random(rid)
    if level == "naive":
        k, span = 3, 6
    else:
        k, span = rng.randint(4, 6), 98
    jobs = [[rng.randrange(k), rng.randint(0, span), rng.randint(480, 1080)] for _ in range(n_jobs)]
    return {"customers": k, "jobs": jobs}


def generate_fake_history(rid, level="naive", n_jobs=200):
    """Returns (list[Job], source) where source is 'openai' or 'offline-stub'."""
    data, source = None, "offline-stub"
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            resp = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                response_format={"type": "json_object"}, max_tokens=6000,
                messages=[{"role": "system", "content": FAKE_SYSTEM},
                          {"role": "user", "content": FAKE_USER.format(brief=LEVEL_BRIEF[level], n_jobs=n_jobs)}])
            data, source = json.loads(resp.choices[0].message.content), "openai"
        except Exception as e:                       # network, quota, bad JSON
            print(f"[openai failed: {e}] using offline stub")
    if data is None:
        data = _stub_fake(level, n_jobs, rid)

    k = max(int(data["customers"]), 1)
    rdev = f"rdev_{rid}"
    jobs = []
    for c, day, minute in data["jobs"][:n_jobs]:      # validate, never trust model output
        c, day, minute = int(c) % k, min(max(int(day), 0), 120), min(max(int(minute), 0), 1439)
        when = START + timedelta(days=day, minutes=minute)
        cust = f"c_{rid}_{c}"
        dev = rdev if DEVICE_BEHAVIOUR[level] == "self" else f"dev_{cust}"
        jobs.append(Job(rid, cust, when, "CONFIRMED", when + timedelta(minutes=3), dev, rdev))
    return jobs, source


# ============================================================== Claude: explain
EXPLAIN_SYSTEM_PROMPT = (
    "You explain a delivery rider's trust score in 2-3 plain sentences for a lender. "
    "Use ONLY the numbers in the JSON and do no arithmetic. Do not add facts, guesses or advice "
    "about the rider's character. If flags exist, say what each reason means and that the score "
    "was capped (high severity) or reduced (medium severity). If the label says "
    "'Not enough evidence yet', say so instead of describing a score. Be neutral and specific."
)


def _numbers_in(obj, out=None):
    out = set() if out is None else out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 1))
        out.add(float(round(obj)))                  # "21%" for 21.2
    elif isinstance(obj, str):
        for m in re.findall(r"\d+(?:\.\d+)?", obj):  # numbers inside flag reasons
            out.add(round(float(m), 1))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    return out


def numbers_are_grounded(text, payload):
    allowed = _numbers_in(payload)
    return all(round(float(m), 1) in allowed for m in re.findall(r"\d+(?:\.\d+)?", text))


def template_explanation(p, lang="en"):
    """Deterministic fallback. Always correct, slightly robotic."""
    if p["score"] is None:
        if lang == "sw":
            return (f"Bado hakuna ushahidi wa kutosha: kazi {p['confirmed_jobs']} zimethibitishwa "
                    f"(zinahitajika angalau {p['min_confirmed_jobs']}).")
        return (f"Not enough evidence yet: {p['confirmed_jobs']} confirmed jobs so far "
                f"(at least {p['min_confirmed_jobs']} are needed to score).")
    if lang == "sw":
        s = (f"Alama ni {p['score']} kati ya {p['max_score']}. Kazi {p['confirmed_jobs']} zilithibitishwa "
             f"na wateja {p['distinct_customers']}; kazi {p['top3_jobs']} zilitoka kwa wateja watatu wa juu.")
        return s + (" Alama ilipunguzwa kwa sababu ya tahadhari kwenye wasifu." if p["flags"] else "")
    s = (f"Score {p['score']} out of {p['max_score']}. {p['confirmed_jobs']} confirmed jobs came from "
         f"{p['distinct_customers']} customers, and {p['top3_jobs']} of them from the top {p['top_n']} customers.")
    if p["flags"]:
        reasons = "; ".join(f["reason"] for f in p["flags"])
        s += f" {p['label']}: {reasons}."
    return s


_CACHE = {}          # explanation cache keyed by payload hash (proposal: cache per snapshot)


def explain(assessment, lang="en"):
    """Returns (text, source) with source in {'claude', 'template', 'cache'}."""
    p = explanation_payload(assessment)
    key = hashlib.sha256((json.dumps(p, sort_keys=True) + lang).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key][0], "cache"
    fallback = (template_explanation(p, lang), "template")
    if not os.getenv("ANTHROPIC_API_KEY"):
        return fallback
    try:
        import anthropic
        language = "Swahili" if lang == "sw" else "English"
        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"), max_tokens=300,
            system=EXPLAIN_SYSTEM_PROMPT + f" Write in {language}.",
            messages=[{"role": "user", "content": json.dumps(p)}])
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if text and numbers_are_grounded(text, p):
            _CACHE[key] = (text, "claude")
            return text, "claude"
        print("[claude output had ungrounded numbers, using template]")
    except Exception as e:
        print(f"[claude failed: {e}] using template")
    return fallback


# ==================================================================== demo
if __name__ == "__main__":
    from work_passport import build_context
    for level in ("naive", "careful"):
        rid = f"X_{level}"
        jobs, src = generate_fake_history(rid, level)
        ctx = build_context({rid: jobs}, {rid: f"p_{rid}"})
        a = assess(rid, jobs, ctx)
        text, esrc = explain(a)
        print(f"\n--- fake rider ({level}, jobs from {src}) -> {a['total']} {a['band']} "
              f"flags={[f['code'] for f in a['flags']]}")
        print(f"[{esrc}] {text}")

    p = {"score": 13.4, "confirmed_jobs": 200, "flags": [{"reason": "120 of 200 jobs in one day"}]}
    assert numbers_are_grounded("200 jobs and 120 of them in one day scored 13", p)
    assert not numbers_are_grounded("Rider completed 9999 jobs", p)
    print("\nguardrail test passed")
