"""Rule-based fraud detection: duplicate billing + impossible days.

No LLM calls here, so this step has zero token cost.
Output: one row per claim with risk_score (0-100), risk_level, reasons, case_id.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Thresholds (tune here)
NEAR_DUP_DAYS = 1           # service dates within +/- N days
NEAR_DUP_AMOUNT_PCT = 0.10  # billed amounts within 10%
IMPOSSIBLE_HOURS_MED = 12   # billed hours/provider/day -> Medium
IMPOSSIBLE_HOURS_HIGH = 16  # -> High
EM_CODES = ["99213", "99214", "99215"]  # established-patient office visits, low -> high
UPCODE_MIN_VISITS = 20      # provider needs this many E/M visits to be assessed
UPCODE_EXCESS = 0.25        # top-level share must exceed peer median by 25 points...
UPCODE_Z = 2.0              # ...and be this many peer std-devs above it


def load_reference() -> dict[str, int]:
    ref = pd.read_csv(ROOT / "data" / "cpt_reference.csv", dtype={"cpt_code": str})
    return dict(zip(ref.cpt_code, ref.minutes))


def prepare(claims: pd.DataFrame) -> pd.DataFrame:
    df = claims.copy()
    df["cpt_code"] = df.cpt_code.astype(str)
    df["service_date"] = pd.to_datetime(df.service_date)
    df["submitted_date"] = pd.to_datetime(df.submitted_date)
    return df


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows flagged as exact or near duplicates (the later-submitted copy)."""
    out = []
    key = ["member_id", "provider_id", "cpt_code", "service_date", "billed_amount"]
    ordered = df.sort_values(["submitted_date", "claim_id"])

    # Exact: identical key, every occurrence after the first
    exact_mask = ordered.duplicated(key, keep="first")
    firsts = ordered[~ordered.duplicated(key, keep="first")].set_index(key)["claim_id"]
    for _, r in ordered[exact_mask].iterrows():
        orig = firsts.loc[tuple(r[k] for k in key)]
        out.append({"claim_id": r.claim_id, "rule": "duplicate", "score": 92,
                    "reason": f"Exact duplicate of {orig} (same member, provider, code, date, amount)",
                    "related_claim": orig})
    exact_ids = set(ordered[exact_mask].claim_id)

    # Near: same member/provider/code, date within N days, amount within X%, not exact
    g = ordered[~ordered.claim_id.isin(exact_ids)]
    for _, grp in g.groupby(["member_id", "provider_id", "cpt_code"]):
        if len(grp) < 2:
            continue
        rows = grp.to_dict("records")
        for i in range(1, len(rows)):
            for j in range(i):
                a, b = rows[j], rows[i]  # b submitted later
                days = abs((b["service_date"] - a["service_date"]).days)
                pct = abs(b["billed_amount"] - a["billed_amount"]) / max(a["billed_amount"], 1)
                if days <= NEAR_DUP_DAYS and pct <= NEAR_DUP_AMOUNT_PCT:
                    out.append({"claim_id": b["claim_id"], "rule": "near_duplicate", "score": 68,
                                "reason": f"Near-duplicate of {a['claim_id']} ({days}d apart, {pct:.0%} amount diff)",
                                "related_claim": a["claim_id"]})
                    break
    return pd.DataFrame(out, columns=["claim_id", "rule", "score", "reason", "related_claim"])


def find_impossible_days(df: pd.DataFrame, minutes: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag every claim on provider-days whose billed time exceeds the thresholds."""
    d = df.assign(minutes=df.cpt_code.map(minutes).fillna(15) * df.units)
    day = (d.groupby(["provider_id", "provider_name", "specialty", "service_date"])
             .agg(billed_hours=("minutes", lambda m: m.sum() / 60), claims=("claim_id", "count"),
                  billed_amount=("billed_amount", "sum"))
             .reset_index())
    # Peer benchmark: median daily hours across providers in the same specialty
    day["peer_median_hours"] = day.groupby("specialty").billed_hours.transform("median")
    flagged = day[day.billed_hours > IMPOSSIBLE_HOURS_MED].copy()
    flagged["score"] = np.clip(60 + (flagged.billed_hours - IMPOSSIBLE_HOURS_MED) * 3, 60, 97).round()
    flagged.loc[flagged.billed_hours > IMPOSSIBLE_HOURS_HIGH, "score"] = flagged.score.clip(lower=85)

    hits = d.merge(flagged[["provider_id", "service_date", "billed_hours", "score"]],
                   on=["provider_id", "service_date"])
    out = pd.DataFrame({
        "claim_id": hits.claim_id, "rule": "impossible_day", "score": hits.score,
        "reason": [f"Provider billed {h:.1f} hrs on {dt:%d %b %Y}" for h, dt in zip(hits.billed_hours, hits.service_date)],
        "related_claim": "",
    })
    return out, flagged


def find_upcoding(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Peer comparison: providers billing the top E/M level far more often than specialty peers."""
    em = df[df.cpt_code.isin(EM_CODES)]
    prov = (em.groupby(["provider_id", "provider_name", "specialty"])
              .agg(em_visits=("claim_id", "count"), top_level=("cpt_code", lambda c: (c == "99215").sum()))
              .reset_index())
    prov = prov[prov.em_visits >= UPCODE_MIN_VISITS].copy()
    prov["top_share"] = prov.top_level / prov.em_visits
    # Leave-one-out peer stats so an outlier doesn't inflate its own benchmark
    stats = []
    for _, r in prov.iterrows():
        peers = prov[(prov.specialty == r.specialty) & (prov.provider_id != r.provider_id)].top_share
        stats.append((peers.median(), peers.std(ddof=0) or 0.05, len(peers)))
    prov["peer_median_share"], prov["peer_std"], prov["peers"] = zip(*stats) if stats else ([], [], [])
    prov["z"] = (prov.top_share - prov.peer_median_share) / prov.peer_std
    flagged = prov[(prov.peers >= 3) & (prov.top_share - prov.peer_median_share > UPCODE_EXCESS)
                   & (prov.z > UPCODE_Z)].copy()
    flagged["score"] = np.clip(55 + flagged.z * 5, 55, 88).round()

    hits = em[em.cpt_code == "99215"].merge(flagged[["provider_id", "top_share", "peer_median_share", "score"]],
                                            on="provider_id")
    out = pd.DataFrame({
        "claim_id": hits.claim_id, "rule": "upcoding", "score": hits.score,
        "reason": [f"Provider bills 99215 on {s:.0%} of office visits vs peer median {m:.0%}"
                   for s, m in zip(hits.top_share, hits.peer_median_share)],
        "related_claim": "",
    })
    return out, flagged


def risk_level(score: float) -> str:
    return "High" if score >= 80 else "Medium" if score >= 50 else "Low"


def run_detection(claims: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = prepare(claims)
    dups = find_duplicates(df)
    imp, provider_days = find_impossible_days(df, load_reference())
    upc, upcoders = find_upcoding(df)
    hits = pd.concat([dups, imp, upc], ignore_index=True)

    agg = (hits.sort_values("score", ascending=False)
               .groupby("claim_id")
               .agg(risk_score=("score", "max"), rules=("rule", lambda r: ", ".join(sorted(set(r)))),
                    reasons=("reason", lambda r: " | ".join(r)), related_claim=("related_claim", "first"))
               .reset_index())
    scored = df.merge(agg, on="claim_id", how="left")
    scored["risk_score"] = scored.risk_score.fillna(0)
    scored["flagged"] = scored.risk_score > 0
    scored["risk_level"] = np.where(scored.flagged, scored.risk_score.map(risk_level), "")
    scored[["rules", "reasons", "related_claim"]] = scored[["rules", "reasons", "related_claim"]].fillna("")

    # Case grouping (each claim belongs to one case; priority: impossible day > upcoding > duplicate)
    scored["case_id"] = ""
    imp_mask = scored.rules.str.contains("impossible_day")
    scored.loc[imp_mask, "case_id"] = ("DAY-" + scored.loc[imp_mask, "provider_id"] + "-"
                                       + scored.loc[imp_mask, "service_date"].dt.strftime("%Y%m%d"))
    upc_mask = scored.rules.str.contains("upcoding") & ~imp_mask
    scored.loc[upc_mask, "case_id"] = "UPC-" + scored.loc[upc_mask, "provider_id"]
    dup_mask = scored.flagged & ~imp_mask & ~upc_mask
    scored.loc[dup_mask, "case_id"] = "DUP-" + scored.loc[dup_mask, "claim_id"]
    return {"scored": scored, "provider_days": provider_days, "upcoders": upcoders}


def build_cases(scored: pd.DataFrame) -> pd.DataFrame:
    f = scored[scored.flagged]
    cases = (f.groupby("case_id")
               .agg(rules=("rules", "first"), provider_id=("provider_id", "first"), provider_name=("provider_name", "first"),
                    specialty=("specialty", "first"), service_date=("service_date", "first"),
                    claims=("claim_id", "count"), billed_amount=("billed_amount", "sum"),
                    risk_score=("risk_score", "max"), reasons=("reasons", "first"),
                    claim_ids=("claim_id", lambda c: list(c)))
               .reset_index()
               .sort_values(["risk_score", "billed_amount"], ascending=False))
    prefix = cases.case_id.str[:4]
    cases["case_type"] = np.select([prefix == "DAY-", prefix == "UPC-",
                                    cases.rules.str.contains("near_duplicate") & ~cases.rules.str.contains(r"\bduplicate", regex=True)],
                                   ["impossible_day", "upcoding", "near_duplicate"], "duplicate")
    cases = cases.drop(columns="rules")
    cases["risk_level"] = cases.risk_score.map(risk_level)
    cases["route"] = cases.risk_level.map({"High": "Refer to SIU", "Medium": "Manual review", "Low": "Auto-approve"})
    return cases


def evaluate(scored: pd.DataFrame) -> pd.DataFrame:
    """Precision/recall against planted ground truth (synthetic data only)."""
    rows = []
    for label, rule in [("Duplicate (exact)", "duplicate"), ("Near-duplicate", "near_duplicate"),
                        ("Impossible day", "impossible_day"), ("Upcoding", "upcoding")]:
        pred = scored.rules.str.split(", ").apply(lambda r: rule in r)
        truth = scored.fraud_type == rule
        tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
        rows.append({"check": label, "planted": int(truth.sum()), "flagged": int(pred.sum()),
                     "true_pos": tp, "false_pos": fp, "missed": fn,
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
    pred, truth = scored.flagged, scored.planted_fraud.astype(bool)
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    rows.append({"check": "Overall", "planted": int(truth.sum()), "flagged": int(pred.sum()),
                 "true_pos": tp, "false_pos": fp, "missed": fn,
                 "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
    return pd.DataFrame(rows)
