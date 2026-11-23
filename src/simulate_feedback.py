"""Demo only: simulate investigator reviews using the synthetic ground truth.

Reviews a random sample of flagged cases and records Confirmed fraud / Not fraud, so the
Feedback loop page has data. Real deployments get these decisions from investigators in the app.

    python src/simulate_feedback.py [--n 60] [--reset]
"""
import argparse

import pandas as pd

import detect
import feedback

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--reset", action="store_true", help="delete existing feedback first")
a = ap.parse_args()

if a.reset and feedback.FILE.exists():
    feedback.FILE.unlink()

scored = detect.run_detection(pd.read_csv(detect.ROOT / "data" / "claims.csv", dtype={"cpt_code": str}))["scored"]
cases = detect.build_cases(scored)
sample = cases.sample(min(a.n, len(cases)), random_state=7)
truth = scored.set_index("claim_id").planted_fraud.astype(bool)

for _, c in sample.iterrows():
    # A case is real fraud if any of its claims was planted (upcoding is judged at provider level)
    is_fraud = truth.loc[c.claim_ids].any()
    feedback.record(c, feedback.CONFIRMED if is_fraud else feedback.NOT_FRAUD, reviewer="simulated",
                    note="Simulated review from synthetic ground truth")

fb = feedback.load()
print(f"Recorded {len(sample)} simulated reviews -> {feedback.FILE}")
print(feedback.rule_stats(fb).to_string(index=False))
