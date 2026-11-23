"""Seed demo review outcomes for the synthetic dataset (SYN-2026Q3) from the planted ground truth.

    python src/simulate_feedback.py [--n 60] [--reset]

Each simulated review records a structured outcome with a reason and evidence, plus line-level findings
(planted lines -> substantiated, other lines in the case -> not substantiated). Reviewer is 'demo seed (simulated)'.
"""
import argparse
import sqlite3

import pandas as pd

import detect
import feedback as fbk

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--reset", action="store_true", help="delete all synthetic-dataset review data first")
a = ap.parse_args()
DS = fbk.SAMPLE_DATASET

if a.reset and fbk.DB.exists():
    with sqlite3.connect(fbk.DB) as c:
        for t in ("events", "case_state", "line_findings", "audit"):
            c.execute(f"DELETE FROM {t} WHERE dataset_id=?", (DS,))

scored = detect.run_detection(pd.read_csv(detect.ROOT / "data" / "claims.csv", dtype={"cpt_code": str}))["scored"]
cases = detect.build_cases(scored)
fbk.ensure_cases(DS, cases)
sample = cases.sample(min(a.n, len(cases)), random_state=7)
truth = scored.set_index("claim_id").planted_fraud.astype(bool)
WHO, ROLE = "demo seed (simulated)", "Investigator"

for i, (_, c) in enumerate(sample.iterrows()):
    planted = [x for x in c.claim_ids if truth.get(x, False)]
    if planted:
        reason = {"duplicate": fbk.REASONS[fbk.SUBST][1], "near_duplicate": fbk.REASONS[fbk.SUBST][1],
                  "impossible_day": fbk.REASONS[fbk.SUBST][0], "upcoding": fbk.REASONS[fbk.SUBST][2]}[c.case_type]
        outcome = fbk.SIU if i % 9 == 0 else fbk.SUBST
        fbk.record(DS, c, outcome, WHO, ROLE, reason=reason if outcome == fbk.SUBST else fbk.REASONS[fbk.SIU][0],
                   evidence=", ".join(planted[:3]), note="Simulated review from synthetic ground truth",
                   amount=float(scored[scored.claim_id.isin(planted)].billed_amount.sum()),
                   owner="SIU · M. Lee" if outcome == fbk.SIU else None,
                   next_step="Records request" if outcome == fbk.SIU else "")
    else:
        outcome = fbk.INSUFF if i % 7 == 0 else fbk.NO_ISSUE
        fbk.record(DS, c, outcome, WHO, ROLE, reason=fbk.REASONS[outcome][0], evidence=c.claim_ids[0],
                   note="Simulated review from synthetic ground truth")
    if outcome != fbk.INSUFF:
        fbk.save_line_findings(DS, c.case_id, {x: ("sub" if x in planted else "not") for x in c.claim_ids[:40]}, WHO)

fb = fbk.current(DS)
print(f"Recorded {len(sample)} simulated reviews -> {fbk.DB}")
print(fbk.rule_stats(fb).to_string(index=False))
