"""
Review fraud detection with XGBoost.

Features (one row per review):
  1. account_age_days, hours_signup_to_first_review
  2. reviews_per_day, burstiness (max reviews by this user in any 24h / total)
  3. rating habits: user_avg_rating, user_rating_std, extreme_rating_share
  4. mpesa_verified
  5. text: review_len, exclam_count, caps_share
  6. near_duplicate_score (max TF-IDF cosine similarity to any other review)
  7. device_shared_users, ip_shared_users

IMPORTANT: build_features() is the ONE function used for both training and
serving, so the model sees identical features in production.

Run:  python review_fraud_xgb.py
"""
import random
import re
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

RNG = random.Random(42)
NP = np.random.default_rng(42)

FEATURES = [
    "account_age_days", "hours_signup_to_first_review",
    "reviews_per_day", "burstiness",
    "user_avg_rating", "user_rating_std", "extreme_rating_share",
    "mpesa_verified",
    "review_len", "exclam_count", "caps_share",
    "near_duplicate_score",
    "device_shared_users", "ip_shared_users",
]


# --------------------------------------------------------------------------
# 1. Feature engineering (shared by training and the Flask API)
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """df columns: review_id, user_id, business_id, rating, text, created_at,
    signup_at, mpesa_verified, device_id, ip. Returns df with FEATURES added."""
    df = df.copy()
    now = now or df["created_at"].max()

    # --- account age and signup -> first review
    df["account_age_days"] = (now - df["signup_at"]).dt.total_seconds() / 86400
    first = df.groupby("user_id")["created_at"].transform("min")
    df["hours_signup_to_first_review"] = (first - df["signup_at"]).dt.total_seconds() / 3600

    # --- reviews per day and burstiness
    g = df.groupby("user_id")
    n_reviews = g["review_id"].transform("count")
    df["reviews_per_day"] = n_reviews / df["account_age_days"].clip(lower=1)

    def max_in_24h(ts):
        ts = np.sort(ts.values.astype("datetime64[s]").astype(np.int64))
        j, best = 0, 1
        for i in range(len(ts)):
            while ts[i] - ts[j] > 86400:
                j += 1
            best = max(best, i - j + 1)
        return best

    peak = g["created_at"].transform(max_in_24h)
    df["burstiness"] = peak / n_reviews

    # --- rating habits
    df["user_avg_rating"] = g["rating"].transform("mean")
    df["user_rating_std"] = g["rating"].transform("std").fillna(0)
    df["extreme_rating_share"] = df.groupby("user_id")["rating"].transform(
        lambda r: ((r == 1) | (r == 5)).mean())

    # --- text style
    df["review_len"] = df["text"].str.len()
    df["exclam_count"] = df["text"].str.count("!")
    letters = df["text"].str.count(r"[A-Za-z]").clip(lower=1)
    df["caps_share"] = df["text"].str.count(r"[A-Z]") / letters

    # --- near duplicates (TF-IDF cosine; char n-grams cope with Swahili/Sheng)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(df["text"].str.lower())
    sim = cosine_similarity(X, dense_output=False).tolil()
    sim.setdiag(0)
    df["near_duplicate_score"] = np.asarray(sim.tocsr().max(axis=1).todense()).ravel()

    # --- device / IP sharing (distinct users per device / ip)
    df["device_shared_users"] = df.groupby("device_id")["user_id"].transform("nunique")
    df["ip_shared_users"] = df.groupby("ip")["user_id"].transform("nunique")

    df["mpesa_verified"] = df["mpesa_verified"].astype(int)
    return df


# --------------------------------------------------------------------------
# 2. Synthetic data (pipeline test only. Fakes here are YOUR assumptions)
# --------------------------------------------------------------------------
GOOD = [
    "Fundi alikuja on time, kazi ilikuwa safi. Price was fair.",
    "Good food, the chapati was soft. Service a bit slow on Saturday.",
    "Mama fua did a great job but came late. Will use again.",
    "Delivery ilifika fast, packaging was okay.",
    "Nilikuwa na tatizo la maji, plumber fixed it same day. Recommended.",
    "Average place. Mutura ni ya kawaida, nothing special.",
    "Mechanic found the real problem, wasn't trying to upsell. Good.",
    "Gas refill was quick and the guy was honest about the price.",
]
SPAM_TEMPLATES = [
    "BEST SERVICE EVER!!! Highly recommended!!! Call now!!!",
    "AMAZING!!! Number one in Nairobi, 5 stars!!!",
    "Very good very good very good. Best best best. Visit today!",
    "Worst place ever!!! Do not go, terrible, go to {rival} instead!!!",
]


def _mutate(t):  # tiny edits so duplicates aren't byte-identical
    return t + RNG.choice(["", " ", ".", " Thanks", "!"])


def make_dataset(n_genuine=400, n_fraud=60, n_businesses=80):
    start = datetime(2026, 1, 1)
    end = datetime(2026, 10, 31)
    span = (end - start).days
    rows, rid = [], 0
    labels = {}

    def add(uid, bid, rating, text, ts, signup, mpesa, dev, ip, fake):
        nonlocal rid
        rid += 1
        rows.append(dict(review_id=rid, user_id=uid, business_id=bid, rating=rating,
                         text=text, created_at=ts, signup_at=signup,
                         mpesa_verified=mpesa, device_id=dev, ip=ip, is_fake=fake))

    # genuine users
    for u in range(n_genuine):
        uid = f"g{u}"
        signup = start + timedelta(days=RNG.randint(0, span - 60))
        dev, ip = f"dev_{uid}", f"ip_{RNG.randint(0, 5000)}"
        t = signup + timedelta(hours=RNG.uniform(6, 24 * 20))
        for _ in range(max(1, int(NP.poisson(3)))):
            t += timedelta(days=float(NP.exponential(12)), hours=RNG.uniform(0, 8))
            if t > end:
                break
            rating = int(np.clip(round(NP.normal(3.8, 1.0)), 1, 5))
            add(uid, RNG.randrange(n_businesses), rating,
                _mutate(RNG.choice(GOOD)), t, signup,
                RNG.random() < 0.35, dev, ip, 0)

    # fraud users: three patterns
    for u in range(n_fraud):
        uid = f"f{u}"
        kind = ["burst", "farm", "paid"][u % 3]
        signup = start + timedelta(days=RNG.randint(100, span - 10))
        biz = RNG.randrange(n_businesses)
        if kind == "burst":          # new account, many 5-stars in hours
            dev, ip = f"dev_{uid}", f"ip_{uid}"
            t = signup + timedelta(minutes=RNG.randint(5, 120))
            for _ in range(RNG.randint(6, 15)):
                t += timedelta(minutes=RNG.randint(1, 20))
                add(uid, biz, 5, _mutate(RNG.choice(SPAM_TEMPLATES)).replace("{rival}", "X"),
                    t, signup, False, dev, ip, 1)
        elif kind == "farm":         # many accounts on one device/IP
            dev, ip = f"farm_{u // 9}", f"farm_ip_{u // 9}"
            t = signup + timedelta(hours=RNG.uniform(1, 30))
            for _ in range(RNG.randint(2, 5)):
                t += timedelta(hours=RNG.uniform(1, 40))
                add(uid, biz, RNG.choice([1, 5]),
                    _mutate(RNG.choice(SPAM_TEMPLATES)).replace("{rival}", "Y"),
                    t, signup, False, dev, ip, 1)
        else:                        # paid fluent reviews, patient, hard case
            dev, ip = f"dev_{uid}", f"ip_{RNG.randint(0, 5000)}"
            t = signup + timedelta(days=RNG.uniform(2, 10))
            for _ in range(RNG.randint(2, 4)):
                t += timedelta(days=RNG.uniform(1, 9))
                add(uid, biz, 5, _mutate(RNG.choice(GOOD)), t, signup,
                    RNG.random() < 0.1, dev, ip, 1)

    df = pd.DataFrame(rows)
    df = df[df["created_at"] <= end].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 3. Train and evaluate
# --------------------------------------------------------------------------
def main():
    raw = make_dataset()
    df = build_features(raw)
    print(f"{len(df)} reviews, fake share = {df.is_fake.mean():.1%}")

    # split BY USER so one user's reviews never sit in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(gss.split(df, df.is_fake, groups=df.user_id))
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(tr.is_fake == 0).sum() / max((tr.is_fake == 1).sum(), 1),
        eval_metric="aucpr", random_state=42,
    )
    model.fit(tr[FEATURES], tr.is_fake)

    probs = model.predict_proba(te[FEATURES])[:, 1]
    print("PR-AUC:", round(average_precision_score(te.is_fake, probs), 3))

    # pick the threshold that gives >= 90% precision, report recall there
    p, r, th = precision_recall_curve(te.is_fake, probs)
    ok = np.where(p[:-1] >= 0.90)[0]
    thr = th[ok[0]] if len(ok) else 0.5
    print(f"threshold for ~90% precision: {thr:.2f}")
    print(classification_report(te.is_fake, probs >= thr, target_names=["genuine", "fake"]))

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importance:\n", imp.round(3).to_string())

    joblib.dump({"model": model, "features": FEATURES, "version": "fraud_xgb_v1",
                 "publish_below": 0.30, "hold_above": 0.80}, "fraud_xgb_v1.joblib")
    print("\nSaved fraud_xgb_v1.joblib")


if __name__ == "__main__":
    main()
