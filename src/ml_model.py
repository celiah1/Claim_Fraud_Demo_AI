"""ML risk model: learns fraud patterns from labelled claims and scores every claim 0-100.

Label source (in order of preference):
  1. Investigator decisions (output/labels.csv) once both classes have >= MIN_LABELS_PER_CLASS claims
  2. Otherwise, the synthetic ground truth (planted_fraud). Demo only, since real data has no such column.

Validation is time-based: train on earlier service dates, test on later ones (no random-split leakage).

    python src/ml_model.py        # train, evaluate, save models/risk_model.joblib + metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

import detect

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "risk_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
MIN_LABELS_PER_CLASS = 50
SPLIT_DATE = "2026-09-01"     # train before, test on/after
ML_FLAG = 80                  # ml_score at which a claim is worth a look

FEATURES = {
    "amount_vs_code_median": "Billed amount ÷ median for the same CPT code",
    "units": "Units billed",
    "submit_lag_days": "Days between service and submission",
    "provider_day_hours": "Provider's billed hours that day",
    "provider_day_claims": "Provider's claims that day",
    "provider_day_hours_vs_peers": "Provider-day hours ÷ specialty median",
    "provider_top_level_share": "Provider's share of office visits billed 99215",
    "provider_top_level_z": "That share vs specialty peers (std devs)",
    "is_99215": "Claim is a top-level office visit",
    "near_match_claims": "Other claims: same member, provider, code within ±1 day",
    "exact_match_claims": "Other claims: identical member, provider, code, date, amount",
    "member_claims_90d": "Member's claims in the period",
}


def build_features(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    minutes = detect.load_reference()
    df["minutes"] = df.cpt_code.map(minutes).fillna(15) * df.units
    f = pd.DataFrame(index=df.index)
    f["amount_vs_code_median"] = df.billed_amount / df.groupby("cpt_code").billed_amount.transform("median")
    f["units"] = df.units
    f["submit_lag_days"] = (df.submitted_date - df.service_date).dt.days

    day = df.groupby(["provider_id", "service_date"])
    f["provider_day_hours"] = day.minutes.transform("sum") / 60
    f["provider_day_claims"] = day.claim_id.transform("count")
    pd_hours = df.assign(h=f.provider_day_hours).groupby(["specialty"]).h.transform("median")
    f["provider_day_hours_vs_peers"] = f.provider_day_hours / pd_hours

    em = df[df.cpt_code.isin(detect.EM_CODES)]
    share = em.groupby("provider_id").cpt_code.apply(lambda c: (c == "99215").mean())
    f["provider_top_level_share"] = df.provider_id.map(share).fillna(0)
    spec = df[["provider_id", "specialty"]].drop_duplicates().set_index("provider_id").specialty
    s = pd.DataFrame({"share": share, "spec": spec.reindex(share.index)})
    s["z"] = (s.share - s.groupby("spec").share.transform("median")) / s.groupby("spec").share.transform("std").fillna(0.05).replace(0, 0.05)
    f["provider_top_level_z"] = df.provider_id.map(s.z).fillna(0)
    f["is_99215"] = (df.cpt_code == "99215").astype(int)

    key = ["member_id", "provider_id", "cpt_code"]
    exact_key = key + ["service_date", "billed_amount"]
    f["exact_match_claims"] = df.groupby(exact_key).claim_id.transform("count") - 1
    near = np.zeros(len(df))
    for _, g in df.groupby(key):
        if len(g) < 2:
            continue
        d = g.service_date.values.astype("datetime64[D]").astype(int)
        cnt = (np.abs(d[:, None] - d[None, :]) <= 1).sum(1) - 1
        near[df.index.get_indexer(g.index)] = cnt
    f["near_match_claims"] = near
    f["member_claims_90d"] = df.groupby("member_id").claim_id.transform("count")
    return f[list(FEATURES)]


def get_labels(scored: pd.DataFrame) -> tuple[pd.Series, str]:
    lab_file = ROOT / "output" / "labels.csv"
    if lab_file.exists():
        lab = pd.read_csv(lab_file, usecols=["claim_id", "label"]).drop_duplicates("claim_id")
        counts = lab.label.value_counts()
        if counts.get(0, 0) >= MIN_LABELS_PER_CLASS and counts.get(1, 0) >= MIN_LABELS_PER_CLASS:
            y = scored.claim_id.map(dict(zip(lab.claim_id, lab.label)))
            return y, f"investigator decisions ({len(lab):,} labelled claims)"
    if "planted_fraud" in scored:
        return scored.planted_fraud.astype(int), "synthetic ground truth (demo; not enough investigator labels yet)"
    raise ValueError("No labels available: close more cases in the app first.")


def train(scored: pd.DataFrame) -> dict:
    X = build_features(scored)
    y, source = get_labels(scored)
    known = y.notna()
    train_m = known & (scored.service_date < SPLIT_DATE)
    test_m = known & (scored.service_date >= SPLIT_DATE)

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           class_weight="balanced", random_state=0)
    model.fit(X[train_m], y[train_m].astype(int))

    p = model.predict_proba(X[test_m])[:, 1]
    yt = y[test_m].astype(int)
    top = np.argsort(-p)[:100]
    imp = permutation_importance(model, X[test_m], yt, scoring="average_precision", n_repeats=5, random_state=0)
    metrics = {
        "label_source": source,
        "train_claims": int(train_m.sum()), "test_claims": int(test_m.sum()),
        "test_fraud_rate": float(yt.mean()),
        "roc_auc": float(roc_auc_score(yt, p)),
        "avg_precision": float(average_precision_score(yt, p)),
        "precision_at_80": float(precision_score(yt, p >= ML_FLAG / 100, zero_division=0)),
        "recall_at_80": float(recall_score(yt, p >= ML_FLAG / 100, zero_division=0)),
        "precision_top100": float(yt.iloc[top].mean()),
        "importance": sorted([{"feature": c, "description": FEATURES[c], "importance": float(v)}
                              for c, v in zip(X.columns, imp.importances_mean)],
                             key=lambda r: -r["importance"]),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def score(scored: pd.DataFrame) -> pd.Series | None:
    """ML risk 0-100 for every claim, or None if no model has been trained."""
    if not MODEL_FILE.exists():
        return None
    model = joblib.load(MODEL_FILE)
    return pd.Series((model.predict_proba(build_features(scored))[:, 1] * 100).round(), index=scored.index)


def load_metrics() -> dict | None:
    return json.loads(METRICS_FILE.read_text(encoding="utf-8")) if METRICS_FILE.exists() else None


if __name__ == "__main__":
    claims = pd.read_csv(ROOT / "data" / "claims.csv", dtype={"cpt_code": str})
    s = detect.run_detection(claims)["scored"]
    m = train(s)
    print(f"Labels: {m['label_source']}")
    print(f"Train {m['train_claims']:,} / test {m['test_claims']:,} claims (time split at {SPLIT_DATE})")
    for k in ["roc_auc", "avg_precision", "precision_at_80", "recall_at_80", "precision_top100"]:
        print(f"  {k:18s} {m[k]:.3f}")
    print("Top features:")
    for r in m["importance"][:6]:
        print(f"  {r['feature']:28s} {r['importance']:.3f}")
    ml = score(s)
    s["ml_score"] = ml
    only = s[(s.ml_score >= ML_FLAG) & ~s.flagged]
    print(f"ML-only flags (score >= {ML_FLAG}, missed by rules): {len(only)} "
          f"({int(only.planted_fraud.sum()) if 'planted_fraud' in only else '?'} planted fraud)")
