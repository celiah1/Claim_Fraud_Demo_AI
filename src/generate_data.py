"""Generate a synthetic health insurance claims dataset with planted fraud.

Planted patterns (ground truth columns `planted_fraud` / `fraud_type`):
  - duplicate        : exact re-submission of an existing claim
  - near_duplicate   : re-submission shifted by 1 day and/or amount changed slightly
  - impossible_day   : provider bills far more service time in a day than is possible
  - upcoding         : provider routinely bills office visits at the highest level (99215)

Usage:  python src/generate_data.py  [--claims 8000] [--seed 42]
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# CPT code -> (description, typical minutes, base fee $)
CPT = {
    "99213": ("Office visit, established, low", 20, 110),
    "99214": ("Office visit, established, moderate", 30, 165),
    "99215": ("Office visit, established, high", 40, 230),
    "99204": ("Office visit, new, moderate", 45, 250),
    "97110": ("Therapeutic exercise, 15 min", 15, 45),
    "97140": ("Manual therapy, 15 min", 15, 48),
    "90834": ("Psychotherapy, 45 min", 45, 140),
    "90837": ("Psychotherapy, 60 min", 60, 185),
    "20610": ("Joint injection, major", 15, 120),
    "73721": ("MRI lower extremity", 45, 650),
    "71046": ("Chest X-ray, 2 views", 10, 60),
    "80053": ("Comprehensive metabolic panel", 5, 35),
    "85025": ("Complete blood count", 5, 25),
}

SPECIALTY_CPT = {
    "Family Medicine": ["99213", "99214", "99215", "99204", "85025"],
    "Orthopedics": ["99214", "99215", "99204", "20610", "73721"],
    "Physical Therapy": ["97110", "97140", "97110", "97140"],
    "Psychiatry": ["90834", "90837", "99214"],
    "Radiology": ["73721", "71046", "71046"],
    "Laboratory": ["80053", "85025", "80053"],
}

ICD = {
    "Family Medicine": ["J06.9", "I10", "E11.9", "Z00.00"],
    "Orthopedics": ["M17.11", "M25.561", "S83.511A"],
    "Physical Therapy": ["M54.50", "M25.561", "M62.81"],
    "Psychiatry": ["F32.1", "F41.1", "F43.10"],
    "Radiology": ["M25.561", "R05.9", "S83.511A"],
    "Laboratory": ["E11.9", "I10", "Z00.00"],
}

FIRST = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "J.", "K.", "L.", "M.", "N.", "P.", "R.", "S."]
LAST = ["Patel", "Moreno", "Osei", "Nguyen", "Kim", "Garcia", "Smith", "Chen", "Okafor", "Rossi",
        "Silva", "Haddad", "Novak", "Larsen", "Ito", "Mensah", "Walsh", "Brooks", "Ahmed", "Lopez"]
ORG = ["Westside Imaging", "CityLab Diagnostics", "Northgate Radiology", "Prime Lab Services",
       "Harbor Imaging", "Summit Diagnostics"]


def make_providers(rng: np.random.Generator, n: int = 60) -> pd.DataFrame:
    rows = []
    specs = list(SPECIALTY_CPT)
    weights = np.array([0.30, 0.15, 0.15, 0.15, 0.13, 0.12])
    org_i = 0
    for i in range(n):
        spec = rng.choice(specs, p=weights)
        if spec in ("Radiology", "Laboratory") and org_i < len(ORG):
            name = ORG[org_i]
            org_i += 1
        else:
            name = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"
        rows.append({"provider_id": f"PRV-{1000 + i}", "provider_name": name, "specialty": spec})
    return pd.DataFrame(rows)


def generate(n_claims: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    providers = make_providers(rng)
    members = [f"MBR-{100000 + i}" for i in range(2500)]
    start = date(2026, 7, 1)
    workdays = [start + timedelta(d) for d in range(92) if (start + timedelta(d)).weekday() < 5]

    # --- Legitimate claims: each provider works a set of days with a normal load ---
    rows = []
    per_provider = n_claims // len(providers)
    for _, p in providers.iterrows():
        codes = SPECIALTY_CPT[p.specialty]
        remaining = per_provider
        days = rng.permutation(workdays)
        di = 0
        while remaining > 0 and di < len(days):
            k = int(min(remaining, rng.integers(3, 12)))
            for _ in range(k):
                cpt = str(rng.choice(codes))
                fee = CPT[cpt][2] * rng.uniform(0.9, 1.15)
                rows.append({
                    "member_id": str(rng.choice(members)),
                    "provider_id": p.provider_id,
                    "service_date": days[di],
                    "cpt_code": cpt,
                    "icd10_code": str(rng.choice(ICD[p.specialty])),
                    "units": 1,
                    "billed_amount": round(fee, 2),
                    "planted_fraud": False,
                    "fraud_type": "",
                })
            remaining -= k
            di += 1
    df = pd.DataFrame(rows)

    # --- Plant impossible days: 5 providers, 2-3 days each with a huge load ---
    fraud_rows = []
    targets = providers[providers.specialty.isin(["Orthopedics", "Physical Therapy", "Psychiatry", "Family Medicine"])]
    for _, p in targets.sample(5, random_state=seed).iterrows():
        codes = SPECIALTY_CPT[p.specialty]
        for d in rng.choice(workdays, size=int(rng.integers(2, 4)), replace=False):
            # mark that provider-day's existing claims as part of the scheme too
            mask = (df.provider_id == p.provider_id) & (df.service_date == d)
            df.loc[mask, ["planted_fraud", "fraud_type"]] = [True, "impossible_day"]
            for _ in range(int(rng.integers(28, 40))):
                cpt = str(rng.choice(codes))
                fraud_rows.append({
                    "member_id": str(rng.choice(members)), "provider_id": p.provider_id,
                    "service_date": d, "cpt_code": cpt,
                    "icd10_code": str(rng.choice(ICD[p.specialty])),
                    # timed 15-min codes are billed in multiple units in this scheme
                    "units": (u := int(rng.integers(2, 5)) if CPT[cpt][1] == 15 else 1),
                    "billed_amount": round(CPT[cpt][2] * u * rng.uniform(0.95, 1.2), 2),
                    "planted_fraud": True, "fraud_type": "impossible_day",
                })
    df = pd.concat([df, pd.DataFrame(fraud_rows)], ignore_index=True)
    df["service_date"] = pd.to_datetime(df.service_date)
    df["submitted_date"] = df.service_date + pd.to_timedelta(rng.integers(1, 21, len(df)), unit="D")

    # --- Plant upcoding: 4 providers bill most office visits at the highest level ---
    imp_providers = set(df.loc[df.fraud_type == "impossible_day", "provider_id"])
    em_providers = providers[providers.specialty.isin(["Family Medicine", "Orthopedics"])
                             & ~providers.provider_id.isin(imp_providers)]
    for _, p in em_providers.sample(4, random_state=seed + 3).iterrows():
        m = (df.provider_id == p.provider_id) & df.cpt_code.isin(["99213", "99214"]) & ~df.planted_fraud
        m &= rng.random(len(df)) < 0.75
        df.loc[m, "cpt_code"] = "99215"
        df.loc[m, "billed_amount"] = (CPT["99215"][2] * rng.uniform(0.9, 1.15, m.sum())).round(2)
        df.loc[m, ["planted_fraud", "fraud_type"]] = [True, "upcoding"]

    # --- Plant duplicates (copies of legitimate claims, re-submitted later) ---
    legit = df[~df.planted_fraud]
    exact = legit.sample(110, random_state=seed + 1).copy()
    exact["submitted_date"] += pd.to_timedelta(rng.integers(5, 40, len(exact)), unit="D")
    exact["planted_fraud"], exact["fraud_type"] = True, "duplicate"
    near = legit.drop(exact.index).sample(70, random_state=seed + 2).copy()
    shift = rng.integers(0, 2, len(near))  # 0 or 1 day
    near["service_date"] = near.service_date + pd.to_timedelta(shift, unit="D")
    near["submitted_date"] += pd.to_timedelta(rng.integers(5, 40, len(near)), unit="D")
    factor = np.where(shift == 0, rng.uniform(1.03, 1.09, len(near)), rng.uniform(0.97, 1.06, len(near)))
    near["billed_amount"] = (near.billed_amount * factor).round(2)
    near["planted_fraud"], near["fraud_type"] = True, "near_duplicate"
    df = pd.concat([df, exact, near], ignore_index=True)

    # --- Finalize: shuffle, assign IDs, submission dates, joins ---
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.merge(providers, on="provider_id", how="left")
    df["cpt_description"] = df.cpt_code.map(lambda c: CPT[c][0])
    df.insert(0, "claim_id", [f"CLM-{100000 + i}" for i in range(len(df))])
    cols = ["claim_id", "member_id", "provider_id", "provider_name", "specialty", "service_date",
            "submitted_date", "cpt_code", "cpt_description", "icd10_code", "units", "billed_amount",
            "planted_fraud", "fraud_type"]
    return df[cols], providers


def write_reference():
    ref = pd.DataFrame([{"cpt_code": k, "description": v[0], "minutes": v[1], "base_fee": v[2]}
                        for k, v in CPT.items()])
    ref.to_csv(DATA_DIR / "cpt_reference.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    claims, _ = generate(a.claims, a.seed)
    claims.to_csv(DATA_DIR / "claims.csv", index=False)
    write_reference()
    print(f"Wrote {len(claims):,} claims -> data/claims.csv")
    print(claims.fraud_type.replace("", "legitimate").value_counts().to_string())
