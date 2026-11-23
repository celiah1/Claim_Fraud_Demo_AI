# ClaimGuard: Walkthrough Script and ML Architecture

## Part 1. Walkthrough script (about 10 minutes)

Open http://localhost:8501 (or the URL in `.claude/launch.json`). Start on **Overview**.

**1. Set the scene (30 sec)**
> "ClaimGuard screens health insurance claims for fraud. It uses cheap, explainable rules first, a machine learning score second, and an LLM only for the cases a human needs to read. The data here is synthetic, with fraud planted on purpose so we can measure the detectors."

**2. Data and sidebar (1 min)**
- Point to **Claims file**. Empty means the synthetic dataset (about 8,500 claims, 60 providers, Jul to Sep 2026). A CSV upload uses the same checks.
- Point to **Service date range** and **Checks**. Turning a check off removes its flags everywhere.

**3. Overview (2 min)**
- KPIs: flagged claims, dollars at risk, high-risk cases, AI briefs.
- "Every number here comes from rules, so each one can be explained line by line."

**4. Providers (2 min)**
- Impossible days: a provider billing more than 12 hours (Medium) or 16 hours (High) of time in one day.
- Upcoding: a provider billing a high share of office visits as 99215, compared with specialty peers. Flagged providers show in red.

**5. Investigator queue (3 min), the core of the demo**
- Open a High case. Read the reasons, the rule score and the ML score.
- Read the case brief. Say: "Today this brief is rule-based because the LLM account has no credit. Once credit is added, Claude writes it, and it must list innocent explanations."
- Click **Referred to SIU** (or **Not fraud**) and add a note.
- Show the **Show closed cases** toggle.
- If another case from the same provider is open, point to the **Repeat offender** warning. Explain the +10 boost.

**6. Feedback loop (1 min)**
- Each decision is logged. After 5 reviews per check, the page suggests tightening or loosening that check.
- The labels download is the training data for the ML model.

**7. Model evaluation (1.5 min)**
- Rule results first, then the ML model.
- Be honest about the numbers (see the caveat in Part 2).

**8. Close (30 sec)**
> "Next steps: add LLM credit so Claude reads the clinical notes for upcoding and writes the briefs. Retrain the ML model on real investigator labels. Then run the batch job nightly."

---

## Part 2. How the machine learning works

### Pipeline

```
data/claims.csv
   |
   v
detect.py ----------> rule flags, reasons, risk_score (0-100)
   |
   v
ml_model.build_features ---> 12 numeric features per claim
   |
   v
HistGradientBoostingClassifier  (trained by ml_model.train)
   |
   v
predict_proba x 100 --> ml_score (0-100) per claim
   |
   v
app.py queue: rule score + ML score side by side
```

### Model

| Item | Value |
|---|---|
| Library | scikit-learn |
| Algorithm | `HistGradientBoostingClassifier` (gradient-boosted decision trees) |
| Settings | `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=15`, `class_weight="balanced"`, `random_state=0` |
| Output | probability of fraud, scaled to 0-100 |
| Saved to | `models/risk_model.joblib`, metrics in `models/metrics.json` |
| Flag threshold | ml_score >= 80 |

Why this model: it handles non-linear feature interactions on tabular data, needs no feature scaling, and its feature importance can be explained. `class_weight="balanced"` offsets the fact that fraud is rare.

### Features (12)

| Feature | Meaning |
|---|---|
| `amount_vs_code_median` | Billed amount divided by the median for the same CPT code |
| `units` | Units billed |
| `submit_lag_days` | Days between service and submission |
| `provider_day_hours` | Provider's total billed hours that day |
| `provider_day_claims` | Provider's claim count that day |
| `provider_day_hours_vs_peers` | Provider-day hours divided by specialty median |
| `provider_top_level_share` | Share of office visits billed 99215 |
| `provider_top_level_z` | That share, in standard deviations vs specialty peers |
| `is_99215` | Claim is a top-level office visit |
| `near_match_claims` | Other claims with same member, provider, code within 1 day |
| `exact_match_claims` | Other claims identical in member, provider, code, date, amount |
| `member_claims_90d` | Member's claim count in the period |

Feedback (investigator decisions) is **not** a feature. The repeat-offender boost changes only the rule score.

### Training labels

Chosen in this order (`get_labels`):
1. **Investigator decisions** from `output/labels.csv`, used only when there are at least 50 confirmed and at least 50 not-fraud claims.
2. **Synthetic ground truth** (`planted_fraud`), used as a demo fallback. The app says so on the metrics page.

Right now the model is trained on source 2.

### Validation

- **Time-based split** at 2026-09-01. Train on claims before that date (5,537), test on later claims (2,961). This avoids random-split leakage.
- Metrics: ROC AUC, average precision, precision and recall at score 80, precision in the top 100.
- **Permutation importance** on the test set (5 repeats, scored by average precision) ranks the features.

### Current results (synthetic labels)

| Metric | Value |
|---|---|
| ROC AUC | 0.995 |
| Average precision | 0.969 |
| Precision at 80 | 0.900 |
| Recall at 80 | 0.854 |
| Precision, top 100 | 1.000 |

Top features: `provider_day_hours` (0.47), `submit_lag_days` (0.10).

### Caveats to say out loud

1. **The results are inflated.** The labels come from the same generator that planted the patterns, and the features measure those same patterns (hours per day, 99215 share, duplicates). The model is largely re-learning the rules. Treat AUC 0.995 as a sanity check, not as proof of real-world accuracy.
2. **The time split does little here.** Planted fraud is spread across all dates, so the test period looks like the training period.
3. **Four ML-only flags**, none of them planted fraud. These are the claims worth a look.
4. **Real accuracy is unknown** until the model is retrained on real investigator labels (50+ per class) and tested on a holdout of real claims.

### Retraining

```bash
"C:\Users\LilyLoi\AppData\Local\Programs\Python\Python312\python.exe" src\ml_model.py
```

It reads the current labels and overwrites `models/` with a new model and metrics.

### Where the LLM fits

The LLM is not part of the ML model. It writes case briefs for flagged cases only. Planned: it reads clinical notes for upcoding. It has no effect on the ML score.
