"""
Validation (proposal section 12).

  - Rules are FROZEN (constants in work_passport.py); nothing is fitted.
  - A rider is predicted FAKE if the profile is Flagged (any high-severity flag) or in the Low band.
    "Not enough evidence yet" is not predicted fake.
  - Baselines are given their best shot: the count threshold that maximises their own F1 on the
    DEV set. (1) confirmed job count, (2) logged job count including unconfirmed jobs.
  - Everything is reported on the HELD-OUT set. Adaptive attackers are reported separately.

Run:  python generate_synthetic_data.py && python evaluate_work_passport.py
"""
import json
from collections import Counter

import pandas as pd

from work_passport import Job, assess, build_context


def load(d):
    riders = pd.read_csv(f"{d}/riders.csv", dtype={"phone": str}, parse_dates=["signup_at"]).fillna({"gen_source": ""})
    jobs = pd.read_csv(f"{d}/jobs.csv", dtype={"customer_phone": str}).fillna({"confirm_device": "", "rider_device": ""})
    for col in ("logged_at", "confirmed_at"):      # mixed formats and blanks -> datetime / NaT
        jobs[col] = pd.to_datetime(jobs[col], format="mixed", errors="coerce")
    by = {r: [] for r in riders.rider_id}
    for x in jobs.itertuples():
        by[x.rider_id].append(Job(
            x.rider_id, x.customer_phone, x.logged_at.to_pydatetime(), x.status,
            None if pd.isna(x.confirmed_at) else x.confirmed_at.to_pydatetime(),
            x.confirm_device, x.rider_device))
    return riders, by


def run(d):
    riders, by = load(d)
    ctx = build_context(by, dict(zip(riders.rider_id, riders.phone)))
    signup = {r: pd.Timestamp(t).to_pydatetime() for r, t in zip(riders.rider_id, riders.signup_at)}
    res = {r: assess(r, j, ctx, signup[r]) for r, j in by.items()}
    return riders, by, res


def predicted_fake(a):
    return a["status"] == "flagged" or a["band"] == "Low"


def confusion(pred, truth, ids):
    tp = sum(1 for r in ids if truth[r] and pred[r])
    fp = sum(1 for r in ids if not truth[r] and pred[r])
    fn = sum(1 for r in ids if truth[r] and not pred[r])
    tn = sum(1 for r in ids if not truth[r] and not pred[r])
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=round(p, 2), recall=round(rc, 2), F1=round(f1, 2))


def best_count_threshold(counts, truth, ids):
    """Fake if count < t. Pick t maximising F1 on these ids."""
    cands = sorted(set(counts[r] for r in ids)) + [max(counts[r] for r in ids) + 1]
    return max(cands, key=lambda t: confusion({r: counts[r] < t for r in ids}, truth, ids)["F1"])


def main(base="data"):
    dev_riders, dev_by, dev_res = run(f"{base}/dev")
    riders, by, res = run(f"{base}/heldout")
    truth = dict(zip(riders.rider_id, riders.is_fake))
    cohort = dict(zip(riders.rider_id, riders.cohort))
    arche = dict(zip(riders.rider_id, riders.archetype))
    head = [r for r in riders.rider_id if cohort[r] != "adaptive"]       # 30 genuine + 30 fake
    adaptive = [r for r in riders.rider_id if cohort[r] == "adaptive"]

    # --- baselines: threshold tuned on DEV, applied to HELD-OUT
    dev_truth = dict(zip(dev_riders.rider_id, dev_riders.is_fake))
    dev_head = [r for r in dev_riders.rider_id if dev_riders.set_index("rider_id").cohort[r] != "adaptive"]
    conf_n = lambda by_, r: sum(j.confirmed for j in by_[r])
    log_n = lambda by_, r: len(by_[r])
    t_conf = best_count_threshold({r: conf_n(dev_by, r) for r in dev_head}, dev_truth, dev_head)
    t_log = best_count_threshold({r: log_n(dev_by, r) for r in dev_head}, dev_truth, dev_head)

    rules_pred = {r: predicted_fake(res[r]) for r in riders.rider_id}
    b1_pred = {r: conf_n(by, r) < t_conf for r in riders.rider_id}
    b2_pred = {r: log_n(by, r) < t_log for r in riders.rider_id}

    out = {"headline_riders": len(head), "genuine": sum(1 for r in head if not truth[r]),
           "fake": sum(1 for r in head if truth[r]),
           "rules": confusion(rules_pred, truth, head),
           "baseline_confirmed_count": {"genuine_if_count>=": t_conf, **confusion(b1_pred, truth, head)},
           "baseline_logged_count": {"genuine_if_count>=": t_log, **confusion(b2_pred, truth, head)}}

    print(f"HELD-OUT: {out['genuine']} genuine + {out['fake']} fake (adaptive attackers reported separately)\n")
    for name, key in [("RULES (frozen)", "rules"),
                      (f"BASELINE 1: confirmed jobs >= {t_conf}", "baseline_confirmed_count"),
                      (f"BASELINE 2: logged jobs incl. unconfirmed >= {t_log}", "baseline_logged_count")]:
        m = out[key]
        print(f"{name:<52} precision {m['precision']:.2f}  recall {m['recall']:.2f}  F1 {m['F1']:.2f}   "
              f"[TP {m['TP']}  FP {m['FP']}  FN {m['FN']}  TN {m['TN']}]")

    print("\nRecall per fake cohort (counts of fakes caught):")
    rows = []
    for c in ["llm", "ring", "burst", "adaptive"]:
        ids = [r for r in riders.rider_id if cohort[r] == c]
        rows.append(dict(cohort=c, riders=len(ids), rules=f"{sum(rules_pred[r] for r in ids)}/{len(ids)}",
                         baseline_confirmed=f"{sum(b1_pred[r] for r in ids)}/{len(ids)}",
                         baseline_logged=f"{sum(b2_pred[r] for r in ids)}/{len(ids)}"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nGenuine riders wrongly flagged, by archetype (false positives / riders):")
    for k in ["full_time", "part_timer", "new_rider", "office_regular"]:
        ids = [r for r in riders.rider_id if arche[r] == k]
        scores = pd.Series([res[r]["total"] if res[r]["total"] is not None else float("nan") for r in ids])
        print(f"  {k:<15} {sum(rules_pred[r] for r in ids)}/{len(ids)}   median score {scores.median():.1f}")

    print("\nWhich fraud rules fired, per cohort:")
    for c in ["genuine", "llm", "ring", "burst", "adaptive"]:
        ids = [r for r in riders.rider_id if cohort[r] == c]
        cnt = Counter(f["code"] for r in ids for f in res[r]["flags"])
        print(f"  {c:<9} {dict(sorted(cnt.items())) or '{}'}")

    print("\nFAKES THE RULES MISSED (failure-case list for the slide):")
    missed = [r for r in riders.rider_id if truth[r] and not rules_pred[r]]
    for r in missed:
        a = res[r]
        print(f"  {r:<8} {arche[r]:<13} score {a['total']} {a['band']:<9} "
              f"customers {a['facts'].get('distinct_customers')}  weeks {a['facts'].get('active_weeks')}")
    out["missed_fakes"] = missed
    out["adaptive"] = confusion(rules_pred, truth, adaptive + [r for r in head if not truth[r]])

    json_path = f"{base}/validation_heldout.json"
    json.dump(out, open(json_path, "w"), indent=2)
    print(f"\nSaved {json_path}")
    print("Both genuine and fake data are synthetic and share the authors' assumptions: this shows the "
          "pipeline works, not real-world accuracy.")


if __name__ == "__main__":
    main()
