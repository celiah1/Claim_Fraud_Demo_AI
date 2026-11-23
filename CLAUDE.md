# Project context
Health insurance claim fraud detection MVP ("ClaimGuard"). Python + pandas + Streamlit + Claude API.

- Python: `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` (not on PATH; `python` alone fails).
- Pipeline: `src/generate_data.py` -> `src/detect.py` (rules, zero token cost) -> `src/summarize.py` (Claude briefs for flagged cases only) -> `app.py`.
- App preview config: `.claude/launch.json` (name `claimguard`, port 8501).
- Palette: navy #0f2d52, blue #2b5f94 / #3b7cc4, teal #0f766e for the LLM step only, with red/amber/green for risk levels only.
- Model: claude-opus-5-5, effort low, json_schema output, server-side fallbacks, system prompt cached.
- ANTHROPIC_API_KEY is set (user env) but the account had no credit on 2026-11-17, so run `run_pipeline.py --no-ai`; briefs are rule-based (`source: "rules"`).
- Checks: duplicate, near-duplicate, impossible day, upcoding (peer comparison, added in phase 2).
- Feedback loop: `src/feedback.py` (decisions in output/feedback.csv, latest per case wins). `src/simulate_feedback.py --reset` seeds demo reviews.
- ML model: `src/ml_model.py` (run from src/ or with src on sys.path), saved to models/. Uses feedback labels at 50+ per class, else planted ground truth. Time split at 2026-09-01.
- Launcher: `Start_ClaimGuard.bat` -> http://claimguard.localhost:8503 (8501 is used by the user's separate "VS code Demo" Northstar app).
- Next: Claude reads clinical notes for upcoding (needs API credit), Batch API nightly run (needs credit).
- Verify the app headlessly with streamlit.testing AppTest if the browser pane is hidden.
- Keep detection rules explainable; every flag needs a reason string.
