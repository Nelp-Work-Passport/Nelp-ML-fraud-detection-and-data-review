"""
Synthetic data for NELP validation (proposal section 12). Writes two independent sets:

  data/dev/      seed 7    sanity-check the rules and tune the BASELINES here
  data/heldout/  seed 202  generated after the thresholds are frozen; report numbers from here
                (seed 101 was used before rule A1 was added and had already been seen: retired)

Each set has riders.csv and jobs.csv:
  riders: rider_id, phone, archetype, cohort, is_fake, gen_source, signup_at
  jobs:   job_id, rider_id, customer_phone, logged_at, status, confirmed_at, confirm_device,
          rider_device, job_type, amount_kes, payment_method, is_synthetic

Cohorts per set: 30 genuine | 10 LLM-written fakes | 10 ring-based | 10 burst-based |
5 adaptive attackers (stretch, reported separately as honest failure cases).

EVERY PARAMETER IS AN ASSUMPTION, NOT A MEASUREMENT. Replace them with numbers from
your rider interviews, and label them as estimates in the pitch.
LLM-written fakes come from OpenAI when OPENAI_API_KEY is set; otherwise from an offline
stub, and riders.csv records which in `gen_source`. Do not call them LLM-written if the stub ran.

Run:  python generate_synthetic_data.py [--out data]
"""
import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd

import llm_tools

WINDOW_END = datetime(2026, 10, 31, 18, 0)
JOB_TYPES = ["parcel", "food", "grocery", "document"]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.seed = seed
        self.phones, self.rows, self.riders = set(), [], []
        self.cust_device = {}
        self._shared = None                 # [device, size] of a shared-device group being filled
        self.global_pool = [self.customer() for _ in range(250)]   # customers who use several riders

    # ----------------------------------------------------------- primitives
    def _phone(self):
        while True:
            p = "07" + "".join(self.r.choice("0123456789") for _ in range(8))
            if p not in self.phones:
                self.phones.add(p)
                return p

    def customer(self):
        """New customer phone. ~12% share a device in groups of 2-3 (family phones)."""
        p = self._phone()
        if self.r.random() < 0.12:
            if self._shared and self._shared[1] < 3:
                self._shared[1] += 1
            else:
                self._shared = [f"shared_{p}", 1]
            self.cust_device[p] = self._shared[0]
        else:
            self.cust_device[p] = f"dev_{p}"
        return p

    def peak_time(self, day):
        hour = self.r.choices([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                              [9, 10, 6, 4, 4, 7, 6, 4, 4, 5, 8, 10, 8, 4])[0]
        return day.replace(hour=hour, minute=self.r.randint(0, 59), second=self.r.randint(0, 59))

    def add_rider(self, rid, archetype, cohort, signup, gen_source=""):
        phone = self._phone()
        self.riders.append(dict(rider_id=rid, phone=phone, archetype=archetype, cohort=cohort,
                                is_fake=int(cohort != "genuine"), gen_source=gen_source,
                                signup_at=signup))
        return phone

    def add_job(self, rid, cust, when, status, conf_dev, rider_dev):
        confirmed_at = ""
        if status == "CONFIRMED":
            confirmed_at = when + timedelta(minutes=min(2880, self.r.expovariate(1 / 90)))
        self.rows.append(dict(
            rider_id=rid, customer_phone=cust, logged_at=when, status=status, confirmed_at=confirmed_at,
            confirm_device=conf_dev if status == "CONFIRMED" else "", rider_device=rider_dev,
            job_type=self.r.choice(JOB_TYPES),
            amount_kes=int(self.r.choice([150, 200, 250, 300, 400, 500, 700]) * self.r.uniform(0.9, 1.3)),
            payment_method=self.r.choice(["mpesa", "cash"]), is_synthetic=1))

    # ----------------------------------------------------------- genuine
    KINDS = {   # weeks, days/wk, jobs/day, customer pool, regulars, P(regular), P(confirm)
        "full_time":      dict(w=(8, 22),  d=(4, 6), pj=(2, 5), pool=(60, 160), reg=6, preg=0.25, pc=(0.45, 0.75)),
        "part_timer":     dict(w=(10, 20), d=(2, 3), pj=(2, 3), pool=(25, 70),  reg=4, preg=0.25, pc=(0.45, 0.75)),
        "new_rider":      dict(w=(3, 4),   d=(5, 6), pj=(3, 4), pool=(15, 40),  reg=3, preg=0.20, pc=(0.45, 0.75)),
        "office_regular": dict(w=(10, 22), d=(4, 5), pj=(2, 3), pool=(4, 5),    reg=4, preg=0.95, pc=(0.50, 0.80)),
    }

    def genuine(self, rid, kind):
        k, r = self.KINDS[kind], self.r
        rdev = f"rdev_{rid}"
        weeks = r.randint(*k["w"])
        start = WINDOW_END - timedelta(weeks=weeks) - timedelta(days=r.randint(0, 10))
        self.add_rider(rid, kind, "genuine", start - timedelta(days=r.uniform(0, 14)))
        pool = [self.customer() for _ in range(r.randint(*k["pool"]))]
        regulars, pc = pool[:k["reg"]], r.uniform(*k["pc"])
        for w in range(weeks):
            for d in r.sample(range(7), r.randint(*k["d"])):
                day = start + timedelta(weeks=w, days=d)
                if day > WINDOW_END:
                    continue
                for _ in range(r.randint(*k["pj"])):
                    if kind != "office_regular" and r.random() < 0.10:
                        c = r.choice(self.global_pool)             # customer shared with other riders
                    else:
                        c = r.choice(regulars) if r.random() < k["preg"] else r.choice(pool)
                    when = self.peak_time(day)
                    if r.random() < pc:
                        status = "CONFIRMED"
                    elif when > WINDOW_END - timedelta(hours=48):
                        status = "PENDING"
                    else:
                        status = "EXPIRED" if r.random() < 0.93 else "DISPUTED"
                    dev = rdev if r.random() < 0.005 else self.cust_device[c]   # customer borrowed rider's phone
                    self.add_job(rid, c, when, status, dev, rdev)

    # ----------------------------------------------------------- fakes
    def llm_fake(self, rid, level):
        jobs, src = llm_tools.generate_fake_history(rid, level)
        first = min(j.logged_at for j in jobs)
        # ASSUMPTION: a lazy fabricator creates the account just before inserting the history;
        # a careful one ages the account for about a month first.
        signup = first - (timedelta(hours=self.r.uniform(0.5, 6)) if level == "naive"
                          else timedelta(days=self.r.uniform(25, 40)))
        self.add_rider(rid, f"llm_{level}", "llm", signup, src)
        for j in jobs:
            self.rows.append(dict(
                rider_id=rid, customer_phone=j.customer_id, logged_at=j.logged_at, status=j.status,
                confirmed_at=j.confirmed_at, confirm_device=j.confirm_device, rider_device=j.rider_device,
                job_type=self.r.choice(JOB_TYPES), amount_kes=self.r.choice([200, 300, 500]),
                payment_method="cash", is_synthetic=1))

    def burst_fake(self, rid):
        """~200 jobs inside 24-48 hours from a few numbers, confirmed from 1-3 phones."""
        rdev, r = f"rdev_{rid}", self.r
        nums = [self._phone() for _ in range(r.randint(5, 12))]
        devs = [f"fab_{rid}_{i}" for i in range(r.randint(1, 3))]
        t0 = WINDOW_END - timedelta(days=r.randint(5, 60))
        self.add_rider(rid, "burst", "burst", t0 - timedelta(hours=r.uniform(0.5, 3)))
        span = r.randint(24, 48) * 60
        for _ in range(r.randint(180, 220)):
            self.add_job(rid, r.choice(nums), t0 + timedelta(minutes=r.randint(0, span)),
                         "CONFIRMED", r.choice(devs), rdev)

    def cycle_ring(self, ids):
        """Members confirm each other's jobs from their own phones, over 10-14 weeks."""
        weeks = self.r.randint(10, 14)
        made = WINDOW_END - timedelta(days=weeks * 7 + self.r.uniform(0, 2))   # created together
        phones = {rid: self.add_rider(rid, "ring_cycle", "ring", made) for rid in ids}
        for rid in ids:
            others = [p for k, p in phones.items() if k != rid]
            for _ in range(self.r.randint(60, 110)):
                day = WINDOW_END - timedelta(days=self.r.randint(0, weeks * 7))
                c = self.r.choice(others)
                self.add_job(rid, c, self.peak_time(day), "CONFIRMED", f"rdev_{c}", f"rdev_{rid}")

    def pool_ring(self, ids):
        """Several fake riders share one small pool of numbers, confirmed from 2 shared phones."""
        pool = [self._phone() for _ in range(6)]
        devs = [f"farm_{ids[0]}_{i}" for i in range(2)]
        made = WINDOW_END - timedelta(days=100 + self.r.uniform(0, 2))
        for rid in ids:
            self.add_rider(rid, "ring_pool", "ring", made)
            for _ in range(self.r.randint(100, 150)):
                day = WINDOW_END - timedelta(days=self.r.randint(0, 100))
                self.add_job(rid, self.r.choice(pool), self.peak_time(day), "CONFIRMED",
                             self.r.choice(devs), f"rdev_{rid}")

    def adaptive(self, rid):
        """Built to evade: many numbers, slow drip over months, a different device each."""
        r = self.r
        nums = [self._phone() for _ in range(r.randint(30, 60))]
        weeks = r.randint(16, 24)
        # ASSUMPTION: adaptive attackers age their accounts 20-60 days before starting
        self.add_rider(rid, "adaptive", "adaptive", WINDOW_END - timedelta(days=weeks * 7 + r.uniform(20, 60)))
        for _ in range(r.randint(100, 200)):
            day = WINDOW_END - timedelta(days=r.randint(0, weeks * 7))
            c = r.choice(nums)
            self.add_job(rid, c, self.peak_time(day), "CONFIRMED", f"dev_{c}", f"rdev_{rid}")


def build(seed):
    g = Gen(seed)
    n = 0
    for kind, count in [("full_time", 18), ("part_timer", 6), ("new_rider", 3), ("office_regular", 3)]:
        for _ in range(count):
            n += 1
            g.genuine(f"GEN{n:03d}", kind)
    for i in range(10):
        g.llm_fake(f"LLM{i:03d}", "naive" if i < 5 else "careful")
    g.cycle_ring([f"RNG{j:03d}" for j in range(4)])                 # 4 ring-cycle riders
    g.pool_ring([f"RNG{j:03d}" for j in range(4, 7)])               # 3 + 3 pool-ring riders
    g.pool_ring([f"RNG{j:03d}" for j in range(7, 10)])
    for i in range(10):
        g.burst_fake(f"BST{i:03d}")
    for i in range(5):
        g.adaptive(f"ADP{i:03d}")
    jobs = pd.DataFrame(g.rows).sort_values("logged_at").reset_index(drop=True)
    jobs.insert(0, "job_id", range(1, len(jobs) + 1))
    return pd.DataFrame(g.riders), jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    for name, seed in [("dev", 7), ("heldout", 202)]:
        riders, jobs = build(seed)
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        riders.to_csv(f"{d}/riders.csv", index=False)
        jobs.to_csv(f"{d}/jobs.csv", index=False)
        print(f"{name}: {len(riders)} riders ({riders.is_fake.sum()} fake), {len(jobs)} jobs")
        print("  " + ", ".join(f"{k}={v}" for k, v in riders.cohort.value_counts().items()),
              "| llm source:", riders[riders.cohort == "llm"].gen_source.iloc[0])
