"""Feedback loop: investigator decisions -> labels -> rule tuning.

Decisions are appended to output/feedback.csv (latest decision per case wins), so they survive app restarts.
They are used to:
  1. Hide closed cases and carry status into the queue.
  2. Boost the risk of providers who already have confirmed fraud (repeat offenders).
  3. Measure each rule's real-world precision and suggest threshold changes.
  4. Export claim-level labels (output/labels.csv) for training an ML model later.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "output" / "feedback.csv"
LABELS = ROOT / "output" / "labels.csv"
COLS = ["timestamp", "case_id", "case_type", "provider_id", "decision", "reviewer", "note"]

CONFIRMED, NOT_FRAUD, REFERRED = "Confirmed fraud", "Not fraud", "Referred to SIU"
DECISIONS = [CONFIRMED, NOT_FRAUD, REFERRED]
REPEAT_OFFENDER_BOOST = 10   # points added to open cases of providers with confirmed fraud
MIN_REVIEWS = 5              # reviews needed before suggesting a rule change


def load() -> pd.DataFrame:
    """All decisions, latest per case."""
    if not FILE.exists():
        return pd.DataFrame(columns=COLS)
    fb = pd.read_csv(FILE, dtype=str).fillna("")
    return fb.sort_values("timestamp").drop_duplicates("case_id", keep="last")


def record(case: pd.Series, decision: str, reviewer: str = "investigator", note: str = "") -> None:
    FILE.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "case_id": case.case_id,
                         "case_type": case.case_type, "provider_id": case.provider_id,
                         "decision": decision, "reviewer": reviewer, "note": note}])
    row.to_csv(FILE, mode="a", header=not FILE.exists(), index=False)


def apply(cases: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    """Add status + repeat-offender boost to cases."""
    out = cases.copy()
    status = dict(zip(fb.case_id, fb.decision))
    out["status"] = out.case_id.map(lambda c: status.get(c, "Open"))
    offenders = set(fb.loc[fb.decision == CONFIRMED, "provider_id"])
    still_open = ~out.status.isin([CONFIRMED, NOT_FRAUD])  # Open or Referred
    boost = still_open & out.provider_id.isin(offenders)
    out["repeat_offender"] = boost
    out.loc[boost, "risk_score"] = (out.loc[boost, "risk_score"] + REPEAT_OFFENDER_BOOST).clip(upper=100)
    out.loc[boost, "reasons"] = out.loc[boost, "reasons"] + " | Provider has confirmed fraud on another case"
    out["risk_level"] = out.risk_score.map(lambda s: "High" if s >= 80 else "Medium" if s >= 50 else "Low")
    out["route"] = out.risk_level.map({"High": "Refer to SIU", "Medium": "Manual review", "Low": "Auto-approve"})
    return out.sort_values(["risk_score", "billed_amount"], ascending=False)


def rule_stats(fb: pd.DataFrame) -> pd.DataFrame:
    """Real-world precision per rule from closed decisions, with a tuning suggestion."""
    closed = fb[fb.decision.isin([CONFIRMED, NOT_FRAUD])]
    rows = []
    for rule, g in closed.groupby("case_type"):
        n, conf = len(g), int((g.decision == CONFIRMED).sum())
        p = conf / n
        if n < MIN_REVIEWS:
            tip = f"Need {MIN_REVIEWS - n} more reviews"
        elif p < 0.5:
            tip = "Too many false alarms: tighten threshold"
        elif p > 0.9:
            tip = "Very reliable: consider loosening threshold to catch more"
        else:
            tip = "Keep current threshold"
        rows.append({"rule": rule, "reviewed": n, "confirmed": conf, "not_fraud": n - conf,
                     "precision": p, "suggestion": tip})
    return pd.DataFrame(rows, columns=["rule", "reviewed", "confirmed", "not_fraud", "precision", "suggestion"])


def export_labels(cases: pd.DataFrame, scored: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    """Claim-level labels (1 = confirmed fraud, 0 = not fraud) for future model training."""
    closed = fb[fb.decision.isin([CONFIRMED, NOT_FRAUD])][["case_id", "decision", "timestamp"]]
    lab = cases[["case_id", "claim_ids"]].merge(closed, on="case_id").explode("claim_ids")
    lab = lab.rename(columns={"claim_ids": "claim_id"})
    lab["label"] = (lab.decision == CONFIRMED).astype(int)
    lab = lab.merge(scored.drop(columns=["planted_fraud", "fraud_type"], errors="ignore"), on=["claim_id", "case_id"])
    LABELS.parent.mkdir(exist_ok=True)
    lab.to_csv(LABELS, index=False)
    return lab
