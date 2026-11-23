# ClaimGuard: Health Claim Fraud Detection (MVP)

An AI-powered workflow that flags suspicious health insurance claims and writes investigator case briefs.
See `architecture.svg` for the design and `app_mockup.html` for the original UI mockup.

## What it detects
| Check | Rule | Risk |
|---|---|---|
| Duplicate billing | Same member, provider, CPT, date, amount, submitted again | High (92) |
| Near-duplicate | Same member/provider/CPT, date within ±1 day, amount within 10% | Medium (68) |
| Impossible days | Provider's billed time (CPT minutes × units) above 12 h/day; above 16 h counts as High | 60–97 |
| Upcoding | Provider's share of office visits billed at 99215 is >25 points and >2 std devs above specialty peers (leave-one-out) | 55–88 |

Detection is pure Python, so it has **no token cost**. Claude only writes briefs for flagged cases (High risk by default).

## Setup (Windows)
```bash
pip install -r requirements.txt
```
To have Claude write the briefs, set an API key (otherwise rule-based briefs are used):
```bash
setx ANTHROPIC_API_KEY "your-key"
```

## Run
```bash
python src/generate_data.py      # synthetic claims -> data/claims.csv
python run_pipeline.py           # detect + briefs -> output/
python -m streamlit run app.py   # web app at http://localhost:8501
```
`run_pipeline.py --no-ai` skips Claude; `--levels High Medium` writes briefs for more cases.

## Files
| Path | Purpose |
|---|---|
| `src/generate_data.py` | Synthetic claims with planted fraud (ground truth columns) |
| `src/detect.py` | Rules, risk scoring, case grouping, evaluation |
| `src/summarize.py` | Claude case briefs (Opus 5.5, low effort, JSON output, prompt caching, cached in `output/briefs.json`) |
| `run_pipeline.py` | Command-line pipeline |
| `app.py` | Streamlit app: Overview, Investigator queue, Providers, Model evaluation |
| `output/` | `flagged_claims.csv`, `cases.csv`, `evaluation.csv`, `briefs.json` |

## Using real data
Provide a CSV with these columns: `claim_id, member_id, provider_id, provider_name, specialty, service_date,
submitted_date, cpt_code, cpt_description, icd10_code, units, billed_amount`. Upload it in the app sidebar.
Add any new CPT codes to `data/cpt_reference.csv` (minutes per unit). **De-identify PHI before sending
anything to an external API.**

## Results on synthetic data
| Check | Precision | Recall |
|---|---|---|
| Duplicate / near-duplicate / impossible day | 95–100% | 100% |
| Upcoding (claim level) | 53% | 100% |
| Upcoding (provider level) | 4 of 4 | 4 of 4 |

Upcoding is a provider-level signal. The rule flags all of a flagged provider's 99215 claims, including the genuine ones, so claim-level precision is lower by design; the audit decides which visits were upcoded. Real fraud is messier, so expect lower numbers.

## Feedback loop
Investigators close cases in the queue as **Confirmed fraud**, **Not fraud** or **Referred to SIU**, with an optional note.
Decisions are saved to `output/feedback.csv`, so they survive restarts, and they are used to:
- hide closed cases from the queue (a toggle shows them again),
- add +10 risk to open cases of providers who already have confirmed fraud (repeat offenders),
- measure each rule's real confirmed rate and suggest tightening or loosening it (after 5 or more reviews),
- export claim-level training labels to `output/labels.csv`.

The **Feedback loop** page shows all of this. For a demo, `python src/simulate_feedback.py --reset` records
60 reviews based on the synthetic ground truth.

## ML risk model
`python src/ml_model.py` trains a gradient-boosted tree model (scikit-learn) on 12 explainable claim features,
such as provider billed hours per day, near-matching claims, top-level visit share and amount against the code median.
- **Labels:** investigator decisions from `output/labels.csv` once there are 50 or more of each class. Until then it uses the synthetic ground truth (demo only).
- **Validation:** time-based. It trains on service dates before 1 Sep 2026 and tests on later claims.
- **Synthetic test results:** ROC AUC 0.995, precision 90% and recall 85% at score 80, top-100 precision 100%.
- **In the app:** every claim gets an `ml_score` (0–100), shown in the queue. The Model evaluation page shows the metrics, what drives the score, and claims the ML flags that the rules missed (candidates for new rules).
- **Caveat:** in synthetic data, re-submitted duplicates always arrive late, so `submit_lag_days` matters more here than it may in real data.

## Quick start
Double-click `Start_ClaimGuard.bat`. It opens the app at http://localhost:8501.

## Roadmap
1. Upcoding audit: Claude reads clinical notes against E/M documentation guidelines
2. Retrain the ML model on real investigator labels
3. Batch API for nightly runs (50% cheaper)
4. Connect to the claims database, schedule runs, add sign-in for users
