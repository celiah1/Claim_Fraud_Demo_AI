"""Run the full pipeline from the command line.

    python run_pipeline.py                 # rules + Claude briefs for High-risk cases
    python run_pipeline.py --no-ai         # rules only, zero token cost
    python run_pipeline.py --levels High Medium
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import detect  # noqa: E402
import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=str(ROOT / "data" / "claims.csv"))
ap.add_argument("--no-ai", action="store_true")
ap.add_argument("--levels", nargs="+", default=["High", "Medium"])
a = ap.parse_args()

claims = pd.read_csv(a.input, dtype={"cpt_code": str})
res = detect.run_detection(claims)
scored, pdays = res["scored"], res["provider_days"]
cases = detect.build_cases(scored)

out = ROOT / "output"
out.mkdir(exist_ok=True)
scored[scored.flagged].drop(columns=["planted_fraud", "fraud_type"], errors="ignore") \
    .to_csv(out / "flagged_claims.csv", index=False)
cases.drop(columns=["claim_ids"]).to_csv(out / "cases.csv", index=False)

print(f"Scanned {len(scored):,} claims | flagged {scored.flagged.sum():,} | cases {len(cases):,}")
print(cases.risk_level.value_counts().to_string())

if "planted_fraud" in scored:
    ev = detect.evaluate(scored)
    ev.to_csv(out / "evaluation.csv", index=False)
    print("\nEvaluation vs planted fraud:\n" + ev.to_string(index=False, float_format="{:.1%}".format))

briefs = summarize.generate_briefs(cases, scored, pdays, levels=a.levels, use_claude=not a.no_ai,
                                   upcoders=res["upcoders"])
src = pd.Series([b["source"] for b in briefs.values()]).value_counts()
print(f"\nBriefs: {src.to_dict()}  -> output/briefs.json")
