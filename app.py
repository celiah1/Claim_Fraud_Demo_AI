"""ClaimGuard: health claim screening app.  Run:  streamlit run app.py

Layout follows the locked design docs/design/layout_redesign_v5_FROZEN.html:
How to Use (start) -> ① Bring in claims (Claim Intake, Data Upload) -> ② Find unusual claims (Flag Overview,
Provider Risk) -> ③ Review & improve (Investigator Queue, Review Outcomes); Leadership (Executive Dashboard);
Analyst & admin (Model Evaluation, Data & Access, Under the Hood); "Ask your data" on every page.

Screening only: review outcomes are recorded here; payment, denial, recovery and formal fraud findings stay in the
authorized claims / SIU processes.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go_
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import assistant  # noqa: E402
import detect  # noqa: E402
import feedback  # noqa: E402
import intake  # noqa: E402
import ml_model  # noqa: E402
import summarize  # noqa: E402

NAVY, BLUE, BLUE2, TEAL, GREY = "#0f2d52", "#2b5f94", "#3b7cc4", "#0f766e", "#94a3b8"
RISK = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}
FLAG = assistant.FLAG          # case_type -> flag reason (screening language)
SHORT = {"duplicate": "Repeated claim", "near_duplicate": "Near-repeated claim",
         "impossible_day": "Unusual billed hours", "upcoding": "Visit level above peers"}
FLAG_COLOR = {FLAG["duplicate"]: NAVY, FLAG["near_duplicate"]: BLUE2, FLAG["impossible_day"]: BLUE,
              FLAG["upcoding"]: "#93b4d9"}
GUIDE = "How to Use"
ORDER = ["Claim Intake", "Data Upload", "Flag Overview", "Provider Risk", "Investigator Queue", "Review Outcomes",
         "Model Evaluation"]                      # the 7-step reference tour
STAGES = [("① Bring in claims", ORDER[0:2]), ("② Find unusual claims", ORDER[2:4]), ("③ Review & improve", ORDER[4:6])]
EXTRA_NAV = [("Leadership", ["Executive Dashboard"]), ("Analyst & admin", ["Model Evaluation", "Data & Access", "Under the Hood"])]
NAV_ICON = {"Executive Dashboard": "📊", "Model Evaluation": "7", "Data & Access": "🔒", "Under the Hood": "⚙"}
SHORTNAME = dict(zip(ORDER, ["Intake", "Upload", "Overview", "Providers", "Queue", "Outcomes", "Model"]))
FILTER_PAGES = {"Flag Overview", "Provider Risk", "Investigator Queue", "Review Outcomes", "Executive Dashboard"}
REQUIRED = intake.REQUIRED
SAMPLE_NAME = "Sample dataset (synthetic, 8,498 claims)"
USERS = {"A. Chan": "Investigator", "B. Ho": "Investigator", "J. Wong": "Supervisor", "R. Patel": "Analyst"}
RULESET = "v1.4"
RULE_INFO = {
    "impossible_day": dict(
        rule="Estimated billed hours per provider-day",
        thr=f"over {detect.IMPOSSIBLE_HOURS_MED} h → routine review · over {detect.IMPOSSIBLE_HOURS_HIGH} h → priority review",
        formula=f"base = 60 + 3 × (estimated hours − {detect.IMPOSSIBLE_HOURS_MED}), at least 85 above "
                f"{detect.IMPOSSIBLE_HOURS_HIGH} h, capped at 97",
        src="src/detect.py · find_impossible_days()",
        sql="SELECT c.provider_id, c.service_date,\n       SUM(c.units * COALESCE(r.minutes, 15)) / 60.0 AS est_hours\n"
            "FROM claims c LEFT JOIN cpt_reference r USING (cpt_code)\nGROUP BY 1, 2 HAVING est_hours > 12;"),
    "duplicate": dict(
        rule="Same member, provider, code, service date and amount",
        thr="exact match → priority review", formula="base = 92",
        src="src/detect.py · find_duplicates()",
        sql="SELECT b.claim_id, a.claim_id AS original\nFROM claims a JOIN claims b\n  ON a.member_id = b.member_id "
            "AND a.provider_id = b.provider_id\n AND a.cpt_code = b.cpt_code AND a.service_date = b.service_date\n"
            " AND a.billed_amount = b.billed_amount AND b.submitted_date > a.submitted_date;"),
    "near_duplicate": dict(
        rule=f"Same member, provider and code within ±{detect.NEAR_DUP_DAYS} day and ±{detect.NEAR_DUP_AMOUNT_PCT:.0%} amount",
        thr="near match → routine review", formula="base = 68",
        src="src/detect.py · find_duplicates()",
        sql="SELECT b.claim_id, a.claim_id AS original\nFROM claims a JOIN claims b\n  ON a.member_id = b.member_id "
            "AND a.provider_id = b.provider_id AND a.cpt_code = b.cpt_code\n AND ABS(b.service_date - a.service_date) <= 1\n"
            " AND ABS(b.billed_amount - a.billed_amount) <= 0.10 * a.billed_amount;"),
    "upcoding": dict(
        rule="Share of office visits billed 99215 vs specialty peers",
        thr=f"> leave-one-out peer median + {detect.UPCODE_EXCESS:.0%} points and z > {detect.UPCODE_Z:g}, "
            f"≥ {detect.UPCODE_MIN_VISITS} visits",
        formula="base = 55 + 5 × z, capped at 88", src="src/detect.py · find_upcoding()",
        sql="SELECT provider_id, AVG(cpt_code = '99215') AS top_share\nFROM claims WHERE cpt_code IN ('99213','99214','99215')\n"
            "GROUP BY provider_id HAVING COUNT(*) >= 20;\n-- flag if top_share > peer median + 0.25 AND z > 2"),
}

st.set_page_config(page_title="ClaimGuard", page_icon="🛡", layout="wide")
SH = "0 1px 2px rgba(15,45,82,.04),0 4px 14px rgba(15,45,82,.05)"
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stApp"],[data-testid="stApp"] :is(p,div,span,label,button,input,textarea,h1,h2,h3,h4,li,td,th):not([data-testid="stIconMaterial"]):not([class*="material"]){{font-family:Inter,"Segoe UI",sans-serif}}
[data-testid="stApp"]{{background:#f6f8fb}}
[data-testid="stHeader"]{{background:transparent}}
[data-testid="stSidebar"]{{background:#fff;border-right:1px solid #e6eaf0}}
[data-testid="stSidebar"] :is(h1,h2,h3,p,label,span,small){{color:#334155}}
[data-testid="stSidebar"] .stButton button{{width:100%;border-radius:9px;padding:8px 12px;min-height:0;justify-content:flex-start}}
[data-testid="stSidebar"] .stButton button > div{{justify-content:flex-start}}
[data-testid="stSidebar"] .stButton button p{{font-size:14.5px !important}}
[data-testid="stSidebar"] .stButton button[kind="secondary"]{{background:transparent;border:0;box-shadow:none}}
[data-testid="stSidebar"] .stButton button[kind="secondary"] p{{color:#475569 !important;font-weight:500}}
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover{{background:#f1f5fa}}
[data-testid="stSidebar"] .stButton button[kind="primary"]{{background:#eef4fc;border:0;box-shadow:inset 4px 0 0 {BLUE2}}}
[data-testid="stSidebar"] .stButton button[kind="primary"] p{{color:{NAVY} !important;font-weight:700}}
.st-key-nav-help button{{background:#eef4fc !important;border:1px solid #cfdff2 !important;justify-content:center !important}}
.st-key-nav-help button p{{color:{NAVY} !important;font-weight:600 !important}}
.brand{{display:flex;align-items:center;gap:12px;margin:2px 0 14px}}
.brand .bname{{font-size:26px;font-weight:800;color:{NAVY};letter-spacing:-.3px;line-height:1.1}}
.brand .btag{{font-size:12.5px;color:#64748b;margin-top:2px}}
.stagelbl{{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#94a3b8;margin:14px 4px 4px}}
.ds{{border:1px solid #e6eaf0;background:#f8fafc;border-radius:12px;padding:10px 12px;font-size:12.5px;color:#64748b;margin-top:10px}}
.ds b{{color:{NAVY}}}
.topwrap{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#fff;border:1px solid #e6eaf0;border-radius:12px;padding:8px 12px}}
.tk{{font-size:12px;color:#94a3b8;font-weight:600;padding:3px 8px;border-radius:20px;white-space:nowrap}}
.tk.done{{color:#16a34a}}.tk.on{{color:#fff;background:{NAVY}}}.tk-sep{{color:#cbd5e1}}
.synth{{display:inline-flex;gap:6px;background:#fff7e6;border:1px solid #f6d28b;color:#8a5a00;font-size:12px;font-weight:700;padding:4px 10px;border-radius:20px;white-space:nowrap;margin-left:auto}}
.synth.xs{{font-size:10px;padding:1px 6px;margin-left:4px;vertical-align:2px}}
.crumb{{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{BLUE2};margin-top:14px}}
[data-testid="stMain"] h2{{font-size:2.3rem !important;font-weight:800;margin:0;padding:0;color:{NAVY};letter-spacing:-.02em}}
.purpose{{font-size:16px;color:#64748b;margin:4px 0 16px;max-width:900px}}
.todo{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:8px}}
.todo div{{background:#fff;border:1px solid #e6eaf0;border-radius:12px;padding:12px 14px;font-size:13px;color:#64748b;box-shadow:{SH}}}
.todo b{{display:block;color:{NAVY};font-size:14px;margin-bottom:2px}}
.todo .k{{font-size:11px;font-weight:700;color:{BLUE2};text-transform:uppercase;letter-spacing:.06em}}
.check{{background:#fff;border:1px solid #e6eaf0;border-radius:14px;padding:14px 18px;font-size:15px;color:#1e293b;box-shadow:{SH}}}
.ok{{color:#16a34a;font-weight:700}}.bad{{color:#dc2626;font-weight:700}}
.sec{{display:flex;align-items:center;gap:12px;margin:28px 0 4px}}
.sec .n{{background:#eef4fc;color:{BLUE};font-weight:700;width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}}
.sec .t{{font-size:21px;font-weight:700;color:{NAVY}}}
.take{{font-size:15px;color:#64748b;margin:0 0 12px 42px}}
.sec:not(:has(.n)) + .take{{margin-left:0}}.take b{{color:#1e293b}}
.kpi,.route,.card{{background:#fff;border:1px solid #e6eaf0;border-radius:14px;box-shadow:{SH}}}
.kpi{{padding:16px 18px;height:100%}}.kpi.t .l{{color:{TEAL}}}
.kpi .l{{font-size:14px;font-weight:600;color:#64748b}}.kpi .v{{font-size:30px;font-weight:800;color:{NAVY};margin-top:2px}}.kpi .d{{font-size:13px;color:#64748b}}
.route{{padding:16px;height:100%}}.route .h{{font-size:15px;font-weight:700;margin-bottom:6px}}.route .v{{font-size:28px;font-weight:800;color:{NAVY}}}.route .d{{font-size:14px;color:#64748b}}
.card{{padding:16px 18px;margin-bottom:12px}}.card h3{{margin:0 0 4px;color:{NAVY};font-size:18px}}.card .sub{{margin:0 0 10px;color:#64748b;font-size:14px}}
.steps{{background:#fff;border:1px solid #e6eaf0;border-radius:14px;padding:14px 18px;margin-bottom:14px;box-shadow:{SH}}}
.steps ol{{margin:6px 0 0;padding-left:20px;font-size:15px;color:#1e293b;line-height:1.7}}
.next{{background:#eef4fc;border:1px solid #cfdff2;border-radius:14px;padding:14px 18px;font-size:15px;color:#1e293b}}
.nextwrap{{margin-top:26px}}
.callout{{border-left:4px solid {BLUE2};background:#eef4fc;border-radius:8px;padding:10px 14px;font-size:15px;color:#1e293b;margin:0 0 14px}}
.warn{{background:#fff7e6;border:1px solid #f6d28b;color:#6b4700;border-radius:12px;padding:11px 14px;font-size:14px;margin-bottom:14px}}
.legend3{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.legend3 div{{border-radius:12px;padding:12px 14px;font-size:13.5px}}.legend3 b{{display:block;font-size:14.5px;margin-bottom:2px}}
.hero{{background:linear-gradient(125deg,#0f2d52,#2b5f94);color:#fff;border-radius:16px;padding:22px 26px;height:100%}}
.hero .c{{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#9cc3ee}}
.hero h3{{margin:4px 0 8px;font-size:24px;color:#fff}}.hero p{{margin:0;color:#d5e5f7;font-size:14.5px;line-height:1.5}}
.qs{{background:#fff;border:1px solid #e6eaf0;border-radius:16px;padding:16px 18px;box-shadow:{SH};height:100%}}.qs h4{{margin:0 0 6px;color:{NAVY};font-size:16px}}
.q{{display:flex;gap:10px;padding:8px 0;border-top:1px solid #f0f3f7;font-size:13px;color:#64748b}}.q:first-of-type{{border:0}}
.q i{{font-style:normal;flex-shrink:0;width:24px;height:24px;border-radius:7px;background:#eef4fc;color:{BLUE};font-weight:800;display:flex;align-items:center;justify-content:center;font-size:12px}}.q b{{display:block;color:{NAVY};font-size:13.5px}}
.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.fstage{{background:#fff;border:1px solid #e6eaf0;border-radius:14px;box-shadow:{SH};padding:14px 16px;border-top:4px solid {BLUE2}}}
.fstage h4{{margin:0 0 2px;color:{NAVY};font-size:16px}}.fstage p{{margin:0 0 8px;font-size:13px;color:#64748b}}
.fstage div{{font-size:14px;padding:6px 0;border-top:1px solid #f0f3f7}}.fstage em{{font-style:normal;color:#64748b;font-size:12.5px;display:block}}
.roles{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.role{{border:1px solid #e6eaf0;border-radius:12px;padding:14px;background:#fafcff;font-size:13px;color:#64748b}}.role b{{display:block;color:{NAVY};font-size:14px;margin-bottom:4px}}
.role ol{{margin:6px 0 0;padding-left:18px;color:#1e293b;line-height:1.6}}
.gloss div{{padding:7px 0;border-bottom:1px solid #f0f3f7;font-size:14px}}.gloss b{{color:{NAVY}}}
.chead{{display:flex;gap:14px;align-items:center}}
.score{{width:56px;height:56px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:21px;flex-shrink:0}}
.s-High{{background:#fde8e8;color:#dc2626}}.s-Medium{{background:#fdf1e1;color:#a35d07}}.s-Low{{background:#e3f4ea;color:#16a34a}}
.chead h3{{margin:0;font-size:20px;color:{NAVY}}}.chead p{{margin:2px 0 0;color:#64748b;font-size:13.5px}}
.facts{{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;padding-top:12px;border-top:1px solid #f0f3f7;font-size:13px;color:#64748b}}.facts b{{color:#1e293b}}
.st{{font-size:11px;font-weight:700;padding:3px 8px;border-radius:12px;white-space:nowrap}}
.st-Open{{background:#eef2f7;color:#475569}}.st-conf{{background:#fde8e8;color:#dc2626}}.st-clear{{background:#e3f4ea;color:#16a34a}}.st-ref{{background:#fdf1e1;color:#a35d07}}
.bh{{display:flex;gap:10px;align-items:center;margin:0 0 10px}}
.bh i{{font-style:normal;width:26px;height:26px;border-radius:8px;background:{NAVY};color:#fff;font-weight:800;font-size:13px;display:flex;align-items:center;justify-content:center}}
.bh b{{color:{NAVY};font-size:16px}}.bh span{{color:#64748b;font-size:12.5px;margin-left:auto}}
.reason{{font-size:15px;line-height:1.55;margin:0 0 6px}}.reason b{{color:{NAVY}}}
.scoreexp{{display:flex;gap:12px;align-items:flex-start;background:#f8fafc;border:1px solid #e6eaf0;border-radius:10px;padding:10px 12px;font-size:13px;color:#64748b;margin:4px 0 10px}}
.scale{{display:flex;height:8px;border-radius:4px;width:150px;flex-shrink:0;margin-top:5px;position:relative}}.scale i{{flex:1}}
.scale em{{position:absolute;top:-4px;width:3px;height:16px;background:{NAVY};border-radius:2px}}
.brief{{background:#f0fdfa;border:1px solid #99f6e4;border-radius:12px;padding:14px 16px;color:#1e293b;font-size:14px}}
.brief h4{{color:{TEAL};font-size:11.5px;text-transform:uppercase;letter-spacing:.5px;margin:10px 0 3px}}.brief ul{{margin:0;padding-left:18px}}
.ai{{background:{TEAL};color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:10px}}
.rb{{background:#64748b;color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:10px}}
.recorded{{background:#f6f8fb;border:1px solid #e6eaf0;border-radius:10px;padding:10px 12px;font-size:14px;margin:6px 0}}
.hint{{font-size:12.5px;color:#64748b}}
.st-key-qlist .stButton button{{justify-content:flex-start;text-align:left;border-radius:10px;padding:8px 12px;min-height:0}}
.st-key-qlist .stButton button p{{font-size:13px;text-align:left}}
.st-key-qlist .stButton button > div,.st-key-qlist .stButton button [data-testid="stMarkdownContainer"]{{justify-content:flex-start;text-align:left;width:100%}}
.st-key-qlist .stButton button[kind="primary"]{{background:#eef4fc;color:{NAVY};border:1px solid #cfdff2;box-shadow:inset 4px 0 0 {BLUE2}}}
.st-key-qlist .stButton button[kind="primary"] p{{color:{NAVY};font-weight:700}}
.st-key-askbtn button{{background:{TEAL} !important;color:#fff !important;border:0 !important;font-weight:700}}
.st-key-askbtn button p{{color:#fff !important;font-weight:700}}
.st-key-askcase button{{background:#f0fdfa !important;border:1px solid #b6efe4 !important}}.st-key-askcase button p{{color:{TEAL} !important;font-weight:600}}
[data-testid="stPlotlyChart"]{{background:#fff;border:1px solid #e6eaf0;padding:6px;border-radius:14px;box-shadow:{SH}}}
[data-testid="stMain"] .stButton button[kind="primary"],[data-testid="stMain"] .stDownloadButton button{{border-radius:9px;font-weight:600}}
.chatctx{{font-size:12px;color:#64748b;background:#f8fafb;border:1px solid #e6eaf0;border-radius:8px;padding:6px 10px;margin-bottom:8px}}.chatctx b{{color:{TEAL}}}
.guard{{background:#fff7e6;border:1px solid #f6d28b;border-radius:8px;padding:7px 9px;font-size:12.5px;color:#6b4700;margin-top:6px}}
.st-key-qlist .stButton button p{{line-height:1.45}}
.ctx{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0 12px}}.ctx div{{border:1px solid #e6eaf0;border-radius:10px;padding:9px 11px;font-size:13px;color:#334155;background:#fff}}
.ctx b{{display:block;color:{NAVY};font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px}}
.sbs{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}}.sbs>div{{border-radius:10px;padding:10px 12px;font-size:13.5px}}
.sbs ul{{margin:4px 0 0;padding-left:18px;line-height:1.55}}.sbs h5{{margin:0;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}}
.sup{{background:#fef2f2;border:1px solid #fecaca}}.sup h5{{color:#dc2626}}.alt{{background:#f0fdf4;border:1px solid #bbf7d0}}.alt h5{{color:#16a34a}}
.prio{{border:1px solid #e6eaf0;border-radius:10px;padding:10px 12px;margin-bottom:10px;background:#f8fafc;font-size:13.5px}}
.prio table{{width:100%;font-size:13px;margin-top:6px;border-collapse:collapse}}.prio td{{padding:4px 8px;border-bottom:1px solid #eef1f5}}.prio td:first-child{{color:#64748b;width:190px}}
.pill{{font-size:11px;font-weight:700;padding:2px 8px;border-radius:12px}}.pill.md{{background:#fdf1e1;color:#a35d07}}.pill.ok{{background:#e3f4ea;color:#16a34a}}.pill.gr{{background:#eef2f7;color:#475569}}
.dup{{width:100%;border-collapse:collapse;font-size:13px}}.dup th{{text-align:left;color:#64748b;font-weight:600;width:170px;padding:5px 8px}}.dup td{{padding:5px 8px;border-bottom:1px solid #f0f3f7}}.dup td.diff{{background:#fff4e5;font-weight:700}}
.pilot{{background:linear-gradient(120deg,#0f2d52,#2b5f94);color:#fff;border-radius:16px;padding:18px 22px;margin-bottom:14px}}
.pilot h3{{margin:0 0 4px;font-size:19px;color:#fff}}.pilot p{{margin:0;color:#d5e5f7;font-size:14px}}
.crit{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}}.crit div{{background:rgba(255,255,255,.1);border-radius:10px;padding:10px 12px;font-size:12.5px;color:#d5e5f7}}
.crit b{{display:block;font-size:18px;color:#fff}}.crit i{{font-style:normal;display:block;margin-top:4px;color:#fff;font-weight:600}}
.mrow{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f0f3f7;font-size:14px}}.mrow b{{color:{NAVY}}}
.roletbl td,.roletbl th{{text-align:center;padding:6px}}.roletbl td:first-child,.roletbl th:first-child{{text-align:left}}.yes{{color:#16a34a;font-weight:800}}.no{{color:#cbd5e1}}
.arch{{width:100%;display:block}}
.who{{font-size:12px;color:#64748b;margin-top:2px}}
@media(max-width:900px){{.ctx,.sbs,.crit{{grid-template-columns:1fr}}}}
@media(max-width:900px){{.todo,.flow,.roles,.legend3{{grid-template-columns:1fr}}}}
</style>""", unsafe_allow_html=True)


# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load(file_bytes: bytes | None):
    src = pd.read_csv(pd.io.common.BytesIO(file_bytes), dtype={"cpt_code": str}) if file_bytes \
        else pd.read_csv(ROOT / "data" / "claims.csv", dtype={"cpt_code": str})
    res = detect.run_detection(src)
    scored = res["scored"]
    ml = ml_model.score(scored)
    scored["ml_score"] = ml if ml is not None else np.nan
    cases = detect.build_cases(scored)
    cases["ml_score"] = cases.claim_ids.map(lambda ids: scored.loc[scored.claim_id.isin(ids), "ml_score"].max())
    # Billed hours for EVERY provider-day (detect only returns the days over the threshold)
    mins = scored.cpt_code.map(detect.load_reference()).fillna(15) * scored.units
    all_days = (scored.assign(minutes=mins)
                .groupby(["provider_id", "provider_name", "specialty", "service_date"])
                .agg(billed_hours=("minutes", lambda m: m.sum() / 60), claims=("claim_id", "count"))
                .reset_index())
    all_days["peer_median_hours"] = all_days.groupby("specialty").billed_hours.transform("median")
    return scored, all_days, res["upcoders"], cases


def go(p, case_id=None):
    st.session_state.page = p
    if case_id:
        st.session_state.case_id = case_id


def page_header(title, purpose, step=None):
    stage = next((s[0][0] for s in STAGES if title in s[1]), None)
    crumb = "Start here" if step is None else (f"Stage {stage} · Step {step} of {len(ORDER)}" if stage else
                                               f"Analyst view · step {step} of the reference tour")
    st.markdown(f'<div class="crumb">{crumb}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown(f'<div class="purpose">{purpose}</div>', unsafe_allow_html=True)


def todo(*items):
    st.markdown('<div class="todo">' + "".join(f'<div><span class="k">{k}</span><b>{b}</b>{t}</div>' for k, b, t in items)
                + "</div>", unsafe_allow_html=True)


def next_bar(text, label, target, key, case_id=None):
    st.markdown('<div class="nextwrap"></div>', unsafe_allow_html=True)
    l, r = st.columns([3.2, 1])
    l.markdown(f'<div class="next"><b>Next:</b> {text}</div>', unsafe_allow_html=True)
    r.button(label, key=key, type="primary", width="stretch", on_click=go, args=(target, case_id))


def section(n, title, takeaway=""):
    num = f'<div class="n">{n}</div>' if n != "" else ""
    st.markdown(f'<div class="sec">{num}<div class="t">{title}</div></div>'
                + (f'<div class="take">{takeaway}</div>' if takeaway else ""), unsafe_allow_html=True)


def kpi(col, label, value, detail, teal=False):
    col.markdown(f'<div class="kpi{" t" if teal else ""}"><div class="l">{label}</div>'
                 f'<div class="v">{value}</div><div class="d">{detail}</div></div>', unsafe_allow_html=True)


def style(fig, height=360, title=None):
    fig.update_layout(height=height, plot_bgcolor="#fff", paper_bgcolor="#fff", margin=dict(t=50 if title else 20, b=10, l=10, r=10),
                      title=dict(text=title, font=dict(size=17, color="#0f172a")) if title else None,
                      font=dict(size=14, color="#0f172a"), legend=dict(font=dict(size=13), title_text=""))
    fig.update_xaxes(tickfont=dict(size=13, color="#000"), title_font=dict(size=14, color="#000"), gridcolor="#eef2f7")
    fig.update_yaxes(tickfont=dict(size=13, color="#000"), title_font=dict(size=14, color="#000"), gridcolor="#eef2f7")
    return fig


def money(x):
    if x >= 1_000_000:
        return f"${x / 1_000_000:,.2f}M"
    return f"${x / 1000:,.0f}K" if x >= 1000 else f"${x:,.0f}"


def status_tag(s):
    cls = {feedback.SUBST: "conf", feedback.NO_ISSUE: "clear", feedback.SIU: "ref", feedback.PENDING: "ref",
           feedback.INSUFF: "Open"}.get(s, "Open")
    return f'<span class="st st-{cls}">{s}</span>'


def log_export(what):
    feedback.log(DS, ss.user, "export", what)


# ---------------- State ----------------
ss = st.session_state
ss.setdefault("page", None)
ss.setdefault("data_bytes", None)
ss.setdefault("data_name", SAMPLE_NAME)
ss.setdefault("visited", set())
ss.setdefault("case_id", None)
ss.setdefault("chat", [])
ss.setdefault("user", "A. Chan")
for k, v in {"f_dup": True, "f_imp": True, "f_upc": True, "q_routes": ["Priority review", "Routine review"],
             "q_reasons": list(FLAG.values()), "q_view": "all"}.items():
    ss.setdefault(k, v)
ROLE = USERS[ss.user]
is_sample = ss.data_bytes is None or ss.data_name.startswith("Sample")
DS = feedback.SAMPLE_DATASET if is_sample else "UPL-" + hashlib.md5(ss.data_bytes).hexdigest()[:8]

scored, pdays, upcoders, cases = load(ss.data_bytes)
dmin, dmax = scored.service_date.min().date(), scored.service_date.max().date()
feedback.ensure_cases(DS, cases, seed_owner="A. Chan" if is_sample else None)
ss.setdefault("f_rng", (dmin, dmax))
if ss.f_rng[0] < dmin or ss.f_rng[-1] > dmax:
    ss.f_rng = (dmin, dmax)

# Returning investigators open in their queue; first-time visitors start with the guide
if ss.page is None:
    ev0 = feedback.events(DS)
    mine_before = (ev0.reviewer == ss.user).any() if len(ev0) else False
    ss.page = "Investigator Queue" if mine_before else GUIDE
    if mine_before:
        ss.q_view = "mine"
page = ss.page
if page in ORDER:
    ss.visited.add(page)

# ---------------- Filters (kept in session state, edited behind the ⚙ Filters button) ----------------
rng = ss.f_rng
if len(rng) == 2:
    s, e = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    scored = scored[scored.service_date.between(s, e)]
    cases = cases[cases.service_date.between(s, e)]
allowed = ({"duplicate", "near_duplicate"} if ss.f_dup else set()) | ({"impossible_day"} if ss.f_imp else set()) \
    | ({"upcoding"} if ss.f_upc else set())
cases = cases[cases.case_type.isin(allowed)]
fb = feedback.current(DS)
fb = fb[fb.case_id.isin(cases.case_id)]
all_cases = cases
cases = feedback.apply(cases, DS)
cases["flag_reason"] = cases.case_type.map(FLAG)
flag_ids = set(c for ids in cases.claim_ids for c in ids)
flagged = scored[scored.claim_id.isin(flag_ids)].drop(columns="fraud_type", errors="ignore") \
    .merge(cases[["case_id", "flag_reason", "case_type"]], on="case_id", how="left")
briefs = summarize.load_cache()
open_cases = cases[~cases.status.isin(feedback.CLOSED)]
reviewed = fb[fb.decision.isin(feedback.FINAL)]
DATA = {"scored": scored, "cases": cases, "flagged": flagged, "all_days": pdays, "fb": fb,
        "intake_df": ss.get("intake_df")}

# ---------------- Sidebar ----------------
LOGO = f"""<div class="brand"><svg width="40" height="45" viewBox="0 0 46 52" xmlns="http://www.w3.org/2000/svg">
<path d="M23 2 L42 9 V24 C42 37 33 46 23 50 C13 46 4 37 4 24 V9 Z" fill="{BLUE2}"/>
<path d="M14 26 L20.5 32.5 L33 19" fill="none" stroke="#fff" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg><div><div class="bname">ClaimGuard</div><div class="btag">Health claim screening</div></div></div>"""

with st.sidebar:
    st.markdown(LOGO, unsafe_allow_html=True)
    st.button("❓ How to Use ClaimGuard", key="nav-help", on_click=go, args=(GUIDE,), width="stretch",
              type="primary" if page == GUIDE else "secondary")
    mine_open = int(((open_cases.owner == ss.user) & open_cases.status.isin([feedback.OPEN, feedback.PENDING])).sum())
    for label, pages in STAGES + EXTRA_NAV:
        st.markdown(f'<div class="stagelbl">{label}</div>', unsafe_allow_html=True)
        for p in pages:
            if p in ORDER and p != "Model Evaluation":
                n = ORDER.index(p) + 1
                mark = "✓" if (p in ss.visited and p != page) else str(n)
            else:
                mark = NAV_ICON[p]
            extra = f"  · {mine_open} mine" if p == "Investigator Queue" else ""
            st.button(f"{mark}  {p}{extra}", key=f"nav-{p}", on_click=go, args=(p,), width="stretch",
                      type="primary" if page == p else "secondary")
    st.markdown('<div class="stagelbl">Signed in (demo)</div>', unsafe_allow_html=True)
    st.selectbox("Signed in as", list(USERS), key="user", label_visibility="collapsed",
                 format_func=lambda u: f"{u} · {USERS[u]}", help="Demo sign-in. The pilot uses single sign-on (SSO).")
    st.markdown(f'<div class="ds"><b>{ss.data_name}</b><br>{dmin:%d %b} – {dmax:%d %b %Y}<br>Org <b>{feedback.ORG}</b> · '
                f'dataset <b>{DS}</b>{" (synthetic, isolated from pilot data)" if is_sample else ""}</div>',
                unsafe_allow_html=True)
    st.button("Change dataset", key="to-upload", on_click=go, args=("Data Upload",), width="stretch")
    st.markdown('<div class="ds"><b>Flag ≠ fraud.</b> A flag means "worth a review". Reviewers record outcomes; '
                'formal fraud findings stay with SIU.</div>', unsafe_allow_html=True)


# ---------------- Ask your data (dialog) ----------------
@st.dialog("✦ Ask your data", width="large")
def chat_dialog(first_q=None):
    ctx_case = ss.case_id if page == "Investigator Queue" else None
    st.markdown(f'<div class="chatctx">Looking at: <b>{page}</b>' + (f' · case <b>{ctx_case}</b>' if ctx_case else "")
                + f' · data: {ss.data_name} · answers come from the loaded claims only</div>', unsafe_allow_html=True)

    def ask(q):
        a = assistant.answer(q, DATA, page=page, case_id=ctx_case, provider_id=ss.get("prov_pid"))
        ss.chat.append(("user", q, None))
        ss.chat.append(("assistant", a, None))

    if first_q and not ss.get("_asked_first"):
        ss._asked_first = True
        ask(first_q)
    box = st.container(height=420)
    with box:
        if not ss.chat:
            st.chat_message("assistant", avatar=":material/auto_awesome:").markdown(
                "Hi! Ask me about the claims in ClaimGuard. I answer from the loaded data only and show how I got "
                "each answer.")
        for i, (role, msg, _) in enumerate(ss.chat):
            if role == "user":
                st.chat_message("user").markdown(msg)
                continue
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.markdown(msg.text)
                if msg.table is not None and len(msg.table):
                    st.dataframe(msg.table, hide_index=True, width="stretch")
                if msg.guard:
                    st.markdown(f'<div class="guard">{msg.guard}</div>', unsafe_allow_html=True)
                if msg.how or msg.query:
                    with st.expander("How I got this"):
                        if msg.how:
                            st.caption(msg.how)
                        if msg.query:
                            st.code(msg.query, language="sql")
                for j, (lab, pg, cid) in enumerate(msg.actions):
                    if st.button(lab, key=f"chat-act-{i}-{j}"):
                        go(pg, cid)
                        st.rerun()
    st.caption("Suggested questions")
    sugg = assistant.SUGGESTIONS.get(page, assistant.SUGGESTIONS[GUIDE])
    cols = st.columns(len(sugg))
    for c, q in zip(cols, sugg):
        if c.button(q, key=f"sugg-{q}", width="stretch"):
            ask(q)
            st.rerun(scope="fragment")
    with st.form("chatform", clear_on_submit=True, border=False):
        l, r = st.columns([5, 1])
        q = l.text_input("Question", placeholder="Ask a question about the claims…", label_visibility="collapsed")
        if r.form_submit_button("Ask", width="stretch", type="primary") and q.strip():
            ask(q)
            st.rerun(scope="fragment")
    st.caption("Synthetic demo · answers are rule-based today (Claude when enabled) · AI can be wrong: check "
               "'How I got this' · read-only, it can't change data or decisions")


# ---------------- Top bar: progress tracker, demo badge, filters, assistant ----------------
cur = ORDER.index(page) if page in ORDER else -1
track = '<span class="tk-sep">›</span>'.join(
    f'<span class="tk {"on" if i == cur else "done" if p in ss.visited else ""}">'
    f'{"✓" if (p in ss.visited and i != cur) else i + 1} {SHORTNAME[p]}</span>' for i, p in enumerate(ORDER))
badge = '<span class="synth" title="All claims, providers and decisions in the sample are made up.">⚠ Synthetic demo data · outcomes can be reversed (kept in history)</span>' if is_sample else ""
t1, t2, t3 = st.columns([6, 1.05, 1.25], vertical_alignment="center")
t1.markdown(f'<div class="topwrap">{track}{badge}</div>', unsafe_allow_html=True)
if page in FILTER_PAGES:
    with t2.popover("⚙ Filters", width="stretch"):
        st.markdown("**Applies to all analysis pages**")
        v = st.date_input("Service date range", ss.f_rng, min_value=dmin, max_value=dmax, key="w_rng")
        if isinstance(v, tuple) and len(v) == 2 and v != ss.f_rng:
            ss.f_rng = v
            st.rerun()
        for k, lab in [("f_dup", "Repeated claims"), ("f_imp", "Unusual billed hours"), ("f_upc", "Visit level above peers")]:
            nv = st.checkbox(lab, ss[k], key=f"w_{k}")
            if nv != ss[k]:
                ss[k] = nv
                st.rerun()
        if page == "Investigator Queue":
            st.markdown("**Queue only**")
            rt = st.multiselect("Routing", ["Priority review", "Routine review", "No additional review recommended"],
                                ss.q_routes, key="w_rt")
            rs = st.multiselect("Flag reason", list(FLAG.values()), ss.q_reasons, key="w_rs")
            if (rt, rs) != (ss.q_routes, ss.q_reasons):
                ss.q_routes, ss.q_reasons = rt, rs
                st.rerun()
if t3.button("✦ Ask your data", key="askbtn", width="stretch"):
    ss._asked_first = False
    chat_dialog()

# ================= How to Use =================
if page == GUIDE:
    page_header("How to Use ClaimGuard",
                "ClaimGuard screens health claims with explainable rules and helps investigators decide which ones "
                "need a closer look. It works in three stages. Follow them left to right, or jump straight to the part "
                "your role needs.")
    l, r = st.columns([1.3, 1])
    with l:
        st.markdown('<div class="hero"><div class="c">For claims investigators &amp; payment-integrity teams</div>'
                    '<h3>See it in 1 minute: review a real-looking case</h3><p>Opens the top flagged case from the '
                    'sample data. You\'ll see why it was flagged, what to check, and where to record a decision. No '
                    'upload needed.</p></div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1.15, 1.25, 1])
        top_case = open_cases.case_id.iloc[0] if len(open_cases) else None
        b1.button("▶ Review a sample case", type="primary", width="stretch", on_click=go,
                  args=("Investigator Queue", top_case), key="g-sample")
        b2.button("Prepare documents", width="stretch", on_click=go, args=("Claim Intake",), key="g-prep")
        b3.button("7-step reference tour", width="stretch", on_click=go, args=("Claim Intake",), key="g-tour")
        st.caption(f"Returning investigators open straight in **My cases** ({mine_open} open for {ss.user}). "
                   "Leadership: see the **Executive Dashboard** in the sidebar.")
    with r:
        st.markdown('<div class="qs"><h4>Every case answers three questions</h4>'
                    '<div class="q"><i>1</i><div><b>Why was this flagged?</b>The rule, the comparison and the exact claim rows.</div></div>'
                    '<div class="q"><i>2</i><div><b>What should I check next?</b>A checklist for that flag reason.</div></div>'
                    '<div class="q"><i>3</i><div><b>Where do I record the outcome?</b>A structured outcome with a reason and evidence. Every change is kept in the case history.</div></div>'
                    '</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="border-left:4px solid #0f766e;margin-top:16px"><h3>Pilot scope: duplicate billing '
                '(shadow pilot)</h3><p class="sub" style="margin:0">Investigators review ClaimGuard\'s duplicate-billing flags '
                '<b>alongside</b> the existing process. Payments, denials and recoveries stay in the claims system. Success '
                'criteria are agreed in advance: see the Executive Dashboard.</p></div>', unsafe_allow_html=True)
    section("", "The Workflow", "Each stage feeds the next. The tracker at the top of every page shows where you are.")
    st.markdown("""<div class="flow">
      <div class="fstage"><h4>① Bring in claims</h4><p>Get claims into a clean, standard table.</p>
        <div>Claim Intake<em>messy PDFs, emails, spreadsheets → rows</em></div><div>Data Upload<em>check and load a claims CSV</em></div></div>
      <div class="fstage" style="border-top-color:#2b5f94"><h4>② Find unusual claims</h4><p>Rules score every claim and show where the flags are.</p>
        <div>Flag Overview<em>how many, how much billed, why</em></div><div>Provider Risk<em>who is behind it</em></div></div>
      <div class="fstage" style="border-top-color:#0f2d52"><h4>③ Review &amp; improve</h4><p>People decide; their decisions make the system smarter.</p>
        <div>Investigator Queue<em>work cases one by one</em></div><div>Review Outcomes<em>what reviewers found, with denominators</em></div><div>Executive Dashboard<em>yield, backlog, verified value</em></div></div>
    </div>""", unsafe_allow_html=True)

    section("", "Pick Your Path", "You don't need every page. Start from the one that matches your job.")
    st.markdown("""<div class="roles">
      <div class="role"><b>🧾 Claims processor</b>I receive claim documents.<ol><li>Claim Intake: upload, fix red rows</li><li>Add rows to the dataset</li><li>Check Flag Overview for new flags</li></ol></div>
      <div class="role"><b>🔍 Investigator</b>I review flagged cases.<ol><li>Investigator Queue: click the top case</li><li>Read why it was flagged, tick the checks</li><li>Record an outcome with a reason and evidence</li></ol></div>
      <div class="role"><b>📊 Manager</b>I need the big picture.<ol><li>Executive Dashboard: yield, backlog, value</li><li>Flag Overview: where flags come from</li><li>Provider Risk: top providers</li></ol></div>
    </div>""", unsafe_allow_html=True)

    l, r = st.columns(2)
    with l:
        section("", "Words You'll See")
        st.markdown("""<div class="card gloss">
          <div><b>Review priority (0–100)</b>: how strongly the rules suggest a review: the rule's base score plus any adjustment, both shown in the case. 80+ priority review, 50–79 routine review, below 50 no additional review recommended. <b>Not</b> a probability of fraud.</div>
          <div><b>Billed · Flagged · Substantiated</b>: what was charged · what a rule found unusual · what a reviewer substantiated with evidence.</div>
          <div><b>Outcomes</b>: No issue found · Billing or coding issue substantiated · Insufficient evidence · Refer to SIU. Formal fraud findings stay with SIU.</div>
          <div><b>Case</b>: one or more related flagged claims, reviewed together.</div>
          <div><b>Repeated claim</b>: the same service billed twice.</div>
          <div><b>Unusual billed hours</b>: estimated hours from billed codes and units exceed what fits in a day.</div>
          <div><b>Visit level above peers</b>: billing the most expensive visit far more often than peers.</div>
          <div><b>Ready / Check / Fix</b>: intake row status: usable, read the warning, or missing a value.</div></div>""",
                    unsafe_allow_html=True)
    with r:
        section("", "Common Questions")
        for qq, aa in [
            ("Where do I start?", "Click **Review a sample case** above for the 1-minute version, or **7-step reference tour** to walk through every page."),
            ("Does a flag mean the provider committed fraud?", "No. A flag means \"worth a review\". Investigators record outcomes such as \"Billing or coding issue substantiated\"; formal fraud findings stay with the authorized SIU process."),
            ("Why can't ClaimGuard read my scanned image?", "Intake reads text PDFs, emails and spreadsheets. Scanned images need the Claude AI reader, coming once API credit is added."),
            ("Will my decisions change the scores?", "Not automatically. Outcomes are logged with full history. Any change to rules, the priority adjustment or the model needs independent validation and approval first."),
            ("Can I just ask a question?", "Yes. Click **✦ Ask your data** at the top right of any page. It answers from the loaded claims, shows how it got the answer, and suggests questions for the page you're on."),
            ("Is this real data?", "No. The sample is synthetic, including the accuracy figures, and is stored as its own dataset. Outcomes can be reversed; the history keeps every change.")]:
            with st.expander(qq, expanded=qq == "Where do I start?"):
                st.markdown(aa)

# ================= Claim Intake =================
if page == "Claim Intake":
    page_header("Claim Intake", "Turn messy claim documents (receipts, superbills, emails, odd spreadsheets) into clean "
                "claim rows the screening rules can check.", 1)
    todo(("You will", "Upload documents", "PDF receipts, emails or spreadsheets"),
         ("Then", "Fix rows marked red", "a required value is missing"),
         ("Finish by", "Adding rows to the dataset", "so the screening rules see them"))

    section(1, "Upload Messy Claims", "Text PDFs, .txt / .eml notes, or .csv / .xlsx files with any column names. "
            "Scanned images are not supported yet (that needs OCR or Claude).")
    l, r = st.columns([1.5, 1])
    with l:
        ups = st.file_uploader("Claim documents", type=["pdf", "txt", "eml", "md", "csv", "xlsx"],
                               accept_multiple_files=True, key="intake_up")
    with r:
        st.markdown('<div class="check"><b>Extraction engine</b><br>● Rules: active, $0<br>'
                    '<span style="color:#94a3b8">○ Claude: off (no API credit yet)</span></div>', unsafe_allow_html=True)
        st.write("")
        use_samples = st.button("📂 Load messy samples", width="stretch")
    if use_samples:
        ss.intake_files = [(p.name, p.read_bytes()) for p in sorted((ROOT / "data" / "messy").iterdir())
                           if p.suffix.lower() in (".pdf", ".txt", ".eml", ".md", ".csv", ".xlsx")]
        ss.pop("intake_df", None)
    elif ups:
        files = [(u.name, u.getvalue()) for u in ups]
        if files != ss.get("intake_files"):
            ss.intake_files = files
            ss.pop("intake_df", None)

    files = ss.get("intake_files")
    if not files:
        st.markdown('<div class="next">⬆ Upload documents or click <b>Load messy samples</b> to start. You can also '
                    'skip this step: the sample dataset is already loaded.</div>', unsafe_allow_html=True)
        next_bar("the sample dataset is already loaded; check it before screening.", "Go to Data Upload →",
                 "Data Upload", "nx-intake0")
    else:
        if "intake_df" not in ss:
            with st.spinner("Extracting claim lines..."):
                ss.intake_df, ss.intake_src = intake.extract(files)
                # Preserve the raw input unchanged, per batch
                ss.intake_batch = "B-" + pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
                raw_dir = ROOT / "output" / "raw" / ss.intake_batch
                raw_dir.mkdir(parents=True, exist_ok=True)
                for n_, b_ in files:
                    (raw_dir / Path(n_).name).write_bytes(b_)
                feedback.log(DS, ss.user, "intake batch", f"{ss.intake_batch}: {len(files)} files, {len(ss.intake_df)} lines")
        idf = ss.intake_df
        st.caption("Documents: " + " · ".join(n for n, _ in files))

        n_fix, n_chk, n_ok = (idf.status == intake.FIX).sum(), (idf.status == intake.CHECK).sum(), (idf.status == intake.READY).sum()
        section(2, "What Was Extracted", f"<b>{len(idf)}</b> claim lines from <b>{len(files)}</b> documents. "
                f"The <b>How filled</b> column in the table below lists every value the rules inferred, and why.")
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Claim Lines", f"{len(idf)}", f"from {len(files)} documents")
        kpi(c2, "🟢 Ready", f"{n_ok}", "nothing to check")
        kpi(c3, "🟠 Check", f"{n_chk}", "usable, but read the warning")
        kpi(c4, "🔴 Fix", f"{n_fix}", "a required value is missing")

        section(3, "Review & Fix", "Start with 🔴 rows. Edit any cell to correct it, then press <b>Re-check</b>. "
                "Rows marked <span class='bad'>Fix</span> are left out of the export until they are complete.")
        show = ["status", "claim_id", "member_id", "provider_id", "provider_name", "specialty", "service_date",
                "submitted_date", "cpt_code", "cpt_description", "icd10_code", "units", "billed_amount", "issues",
                "inferred", "source"]
        badge_ = {intake.READY: "🟢 Ready", intake.CHECK: "🟠 Check", intake.FIX: "🔴 Fix"}
        edited = st.data_editor(
            idf[show].assign(status=idf.status.map(badge_)), hide_index=True, width="stretch", height=420,
            key="intake_editor", disabled=["status", "issues", "inferred", "source"],
            column_config={
                "status": st.column_config.TextColumn("Status", width="small"),
                "service_date": st.column_config.DateColumn("service_date"),
                "submitted_date": st.column_config.DateColumn("submitted_date"),
                "units": st.column_config.NumberColumn("units", min_value=0, step=1),
                "billed_amount": st.column_config.NumberColumn("billed_amount", format="$%.2f", min_value=0),
                "issues": st.column_config.TextColumn("Issues", width="large"),
                "inferred": st.column_config.TextColumn("How filled", width="large"),
            })
        b1, _ = st.columns([1, 4])
        if b1.button("↻ Re-check edits", width="stretch"):
            merged = idf.copy()
            for c in show:
                if c not in ("status", "issues", "inferred", "source"):
                    merged[c] = edited[c].values
            ss.intake_df = intake.validate(merged)
            st.rerun()

        with st.expander("📄 Source text of each document (what the rules read)"):
            doc = st.selectbox("Document", list(ss.intake_src))
            st.code(ss.intake_src[doc], language=None)

        idf2 = ss.intake_df
        section(4, f"Reconciliation · Batch {ss.get('intake_batch', '-')}",
                "From input to accepted rows, with totals. Raw files are kept unchanged in output/raw/.")
        amt = pd.to_numeric(idf2.billed_amount, errors="coerce").fillna(0)
        rec = (idf2.assign(amt=amt, ok=idf2.status != intake.FIX)
               .groupby("source").agg(lines_in=("claim_id", "count"), accepted=("ok", "sum"),
                                      amount_in=("amt", "sum"), amount_accepted=("amt", lambda a: a[idf2.loc[a.index, "status"] != intake.FIX].sum()))
               .reset_index())
        rec["rejected"] = rec.lines_in - rec.accepted
        tot = pd.DataFrame([{"source": "Total", **rec.drop(columns="source").sum().to_dict()}])
        rec = pd.concat([rec, tot], ignore_index=True)[["source", "lines_in", "accepted", "rejected", "amount_in", "amount_accepted"]]
        l, r = st.columns([1.3, 1])
        with l:
            st.dataframe(rec, hide_index=True, width="stretch", column_config={
                "amount_in": st.column_config.NumberColumn("$ in", format="$%.2f"),
                "amount_accepted": st.column_config.NumberColumn("$ accepted", format="$%.2f")})
            st.caption("Rejected rows are missing a required value. Stated document totals that don't match their lines are "
                       "listed as warnings in the Issues column.")
        with r:
            infl = [(cid, f, why) for cid, notes in idf2[["claim_id", "_notes"]].values for f, why in (notes or {}).items()
                    if f in ("member_id", "provider_id", "service_date", "billed_amount", "submitted_date")]
            st.markdown("**Inferred values · original shown**")
            st.dataframe(pd.DataFrame(infl, columns=["claim", "field", "original / how filled"]).head(40),
                         hide_index=True, width="stretch", height=220)
            resub = idf2[idf2.issues.str.contains("possible duplicate", na=False)]
            if len(resub):
                st.markdown(f'<div class="hint"><b>{len(resub)} line(s) look like resubmissions</b> of other lines in this batch. '
                            'They are kept and linked, then flagged for review as possible duplicates, rather than silently '
                            'added as new claims. A claim marked as a replacement would replace the original instead.</div>',
                            unsafe_allow_html=True)

        section(5, "Export", "Only rows that are Ready or Check are exported.")
        out = intake.to_claims(ss.intake_df)
        e1, e2, e3 = st.columns([1.2, 1.6, 2])
        e1.download_button(f"⬇ Download clean CSV ({len(out)} rows)", out.to_csv(index=False).encode(),
                           "intake_claims.csv", "text/csv", width="stretch", disabled=out.empty,
                           on_click=log_export, args=("intake clean CSV",))
        if e2.button(f"▶ Add {len(out)} rows to the dataset & continue", type="primary", width="stretch",
                     disabled=out.empty):
            base = pd.read_csv(pd.io.common.BytesIO(ss.data_bytes), dtype={"cpt_code": str}) \
                if ss.data_bytes else pd.read_csv(ROOT / "data" / "claims.csv", dtype={"cpt_code": str})
            base = base[~base.claim_id.astype(str).str.startswith("INT-")]   # replace any earlier intake rows
            new = out.assign(planted_fraud=False, fraud_type="") if "planted_fraud" in base else out
            combined = pd.concat([base, new], ignore_index=True)
            ss.data_bytes = combined.to_csv(index=False).encode()
            ss.data_name = f"Sample + {len(out)} intake claims ({len(combined):,} claims)"
            ss.page = "Data Upload"
            st.rerun()
        e3.markdown('<div style="font-size:14px;color:#64748b;padding-top:4px">Adding to the dataset lets the '
                    'repeated-claim, billed-hours and visit-level checks compare these claims with each provider\'s '
                    'history.</div>', unsafe_allow_html=True)

# ================= Data Upload =================
if page == "Data Upload":
    page_header("Data Upload", "Check that the claims table has everything the screening rules need, then run them.", 2)
    todo(("You will", "Choose a dataset", "sample data, or your own CSV"),
         ("Then", "Read the green checks", f"all {len(REQUIRED)} columns must be present"),
         ("Finish by", "Running the analysis", "takes a few seconds"))
    section(1, "Choose Your Data", "Upload your own claims CSV, or keep using the built-in sample dataset.")
    l, r = st.columns([1.4, 1])
    with l:
        up = st.file_uploader("Upload a claims CSV", type="csv", key="uploader")
    with r:
        st.markdown(f'<div class="check">Currently using:<br><b>{ss.data_name}</b></div>', unsafe_allow_html=True)
        st.write("")
        if ss.data_bytes is not None and st.button("↺ Switch back to sample data"):
            ss.data_bytes = None
            ss.data_name = SAMPLE_NAME
            st.rerun()
        st.download_button("⬇ Download a template (sample data)", (ROOT / "data" / "claims.csv").read_bytes(),
                           "claims_template.csv", "text/csv")

    if up is not None:
        raw = up.getvalue()
        try:
            df = pd.read_csv(pd.io.common.BytesIO(raw), dtype={"cpt_code": str})
        except Exception as e:  # unreadable file
            st.error(f"Could not read this file as CSV: {e}")
            df = None
        src_name = up.name
    else:
        df = pd.read_csv(pd.io.common.BytesIO(ss.data_bytes), dtype={"cpt_code": str}) \
            if ss.data_bytes else pd.read_csv(ROOT / "data" / "claims.csv", dtype={"cpt_code": str})
        src_name = ss.data_name

    if df is not None:
        missing = [c for c in REQUIRED if c not in df.columns]
        dates = pd.to_datetime(df.get("service_date"), errors="coerce") if "service_date" in df else pd.Series(dtype="datetime64[ns]")
        bad_dates = int(dates.isna().sum()) if len(dates) else 0
        dup_ids = int(df.claim_id.duplicated().sum()) if "claim_id" in df else 0
        n_int = int(df.claim_id.astype(str).str.startswith("INT-").sum()) if "claim_id" in df else 0
        section(2, "Check The Data",
                f"<b>{src_name}</b>: {len(df):,} rows, {len(df.columns)} columns. "
                + ("<span class='ok'>All required columns are present.</span>" if not missing
                   else f"<span class='bad'>{len(missing)} required column(s) missing.</span>"))
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Claims", f"{len(df):,}", f"incl. {n_int} from intake" if n_int else
            f"{df.provider_id.nunique() if 'provider_id' in df else '-'} providers")
        kpi(c2, "Members", f"{df.member_id.nunique():,}" if "member_id" in df else "-", "unique patients")
        kpi(c3, "Period Covered", f"{(dates.max() - dates.min()).days + 1} days" if dates.notna().any() else "-",
            f"{dates.min():%d %b} – {dates.max():%d %b %Y} · {bad_dates} unreadable" if dates.notna().any() else "")
        kpi(c4, "Columns", f"{len(REQUIRED) - len(missing)}/{len(REQUIRED)} {'✓' if not missing else '✗'}",
            f"{dup_ids} duplicate claim IDs")
        st.write("")
        checks = "".join(f"<span class='{'ok' if c in df.columns else 'bad'}'>{'✓' if c in df.columns else '✗'}</span> "
                         f"{c} &nbsp;&nbsp; " for c in REQUIRED)
        st.markdown(f'<div class="check"><b>Required columns</b><br>{checks}</div>', unsafe_allow_html=True)

        section(3, "Preview", "The first 100 rows, exactly as they will be analysed.")
        st.dataframe(df.head(100), hide_index=True, width="stretch", height=320)

        if up is not None:
            section(4, "Run The Analysis")
            if missing:
                st.error("Fix the missing columns and upload again. Download the template above to see the expected format.")
            elif st.button(f"▶ Use {up.name} and run the analysis", type="primary"):
                ss.data_bytes = raw
                ss.data_name = f"{up.name} ({len(df):,} claims)"
                ss.page = "Flag Overview"
                st.rerun()
        else:
            next_bar("this dataset is loaded and screened. See what the rules flagged.", "Go to Flag Overview →",
                     "Flag Overview", "nx-upload")

# ================= Flag Overview =================
if page == "Flag Overview":
    page_header("Flag Overview", "How many claims the rules flagged, how much was billed on them, and which flag "
                "reasons drive it. Amounts are <b>billed</b>, not substantiated losses.", 3)
    billed_flag = flagged.billed_amount.sum()
    by_reason = flagged.groupby("flag_reason").billed_amount.sum().sort_values(ascending=False)
    n_conf = int((fb.decision == feedback.SUBST).sum())
    st.markdown(f'<div class="callout">📌 <b>Key finding:</b> {len(cases)} flagged cases · {money(billed_flag)} billed. '
                + (f'"{by_reason.index[0]}" accounts for the most billed dollars. ' if len(by_reason) else "")
                + f'{n_conf} of {len(reviewed)} reviewed cases substantiated so far.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Billed On Flagged Claims", money(billed_flag), f"of {money(scored.billed_amount.sum())} billed in total")
    kpi(c2, "Flagged Cases", f"{len(cases)}", f"{int((cases.risk_level == 'High').sum())} in priority review (80+) · "
        f"{len(flagged):,} claims")
    prov = flagged.groupby("provider_name").billed_amount.sum().sort_values(ascending=False)
    kpi(c3, "Providers With Flags", f"{flagged.provider_id.nunique()}",
        f"top 5 hold {prov.head(5).sum() / max(billed_flag, 1):.0%} of flagged $")
    kpi(c4, "Substantiated After Review", f"{n_conf}", f"of {len(reviewed)} reviewed ({n_conf / max(len(reviewed), 1):.0%}) · "
        f"{int((cases.status == feedback.OPEN).sum())} not yet reviewed", teal=True)

    section(1, "Three Different Numbers", "Don't mix them up.")
    st.markdown('<div class="legend3"><div style="background:#f1f5fa"><b>Billed</b>What providers charged. Every claim '
                'has this.</div><div style="background:#fdf1e1"><b>Flagged</b>A rule found something unusual. Worth a '
                'review, not proof.</div><div style="background:#fde8e8"><b>Substantiated</b>A reviewer recorded "Billing or coding '
                'issue substantiated" with evidence.</div></div>', unsafe_allow_html=True)

    section(2, "Where The Flags Come From",
            f"<b>{by_reason.index[0]}</b> accounts for the largest share of billed dollars on flagged claims "
            f"(<b>{by_reason.iloc[0] / max(billed_flag, 1):.0%}</b>)." if len(by_reason) else "")
    l, r = st.columns([1, 1.4])
    with l:
        d = by_reason.reset_index()
        fig = px.pie(d, names="flag_reason", values="billed_amount", hole=0.55, color="flag_reason",
                     color_discrete_map=FLAG_COLOR)
        fig.update_traces(textinfo="percent", textfont_size=15, hovertemplate="%{label}: $%{value:,.0f}")
        st.plotly_chart(style(fig, 360, "Billed On Flagged Claims, By Flag Reason"), width="stretch")
    with r:
        wk = flagged.assign(week=flagged.service_date.dt.to_period("W").dt.start_time)
        wk = wk.groupby(["week", "flag_reason"]).billed_amount.sum().reset_index()
        fig = px.bar(wk, x="week", y="billed_amount", color="flag_reason", color_discrete_map=FLAG_COLOR,
                     labels={"week": "Week Of Service", "billed_amount": "Billed On Flagged Claims"})
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(style(fig, 360, "Billed On Flagged Claims, By Week"), width="stretch")
    stats = feedback.rule_stats(fb).set_index("rule") if len(fb) else pd.DataFrame()
    tbl = cases.groupby("case_type").agg(cases=("case_id", "count"), billed=("billed_amount", "sum")).reset_index()
    tbl["substantiated_reviewed"] = tbl.case_type.map(
        lambda t: f"{int(stats.loc[t, 'substantiated'])} of {int(stats.loc[t, 'reviewed'])}"
        if t in stats.index else "0 reviewed")
    tbl.insert(0, "flag_reason", tbl.case_type.map(FLAG))
    st.dataframe(tbl.drop(columns="case_type").sort_values("billed", ascending=False), hide_index=True, width="stretch",
                 column_config={"flag_reason": "Flag Reason", "cases": "Flagged Cases",
                                "billed": st.column_config.NumberColumn("Billed On Flagged Claims", format="$%.0f"),
                                "substantiated_reviewed": "Substantiated / Reviewed"})

    section(3, "Who Is Behind The Flags",
            f"Flags are concentrated: the top 5 providers account for <b>{prov.head(5).sum() / max(billed_flag, 1):.0%}</b> "
            "of billed dollars on flagged claims.")
    top = (flagged[flagged.provider_name.isin(prov.head(10).index)]
           .groupby(["provider_name", "flag_reason"]).billed_amount.sum().reset_index())
    fig = px.bar(top, x="billed_amount", y="provider_name", color="flag_reason", orientation="h",
                 color_discrete_map=FLAG_COLOR, category_orders={"provider_name": list(prov.head(10).index)},
                 labels={"billed_amount": "Billed On Flagged Claims", "provider_name": ""})
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(style(fig, 420, "Top 10 Providers By Billed On Flagged Claims"), width="stretch")

    section(4, "How Open Cases Are Routed", "Routing sets review effort only. Payment, denial, recovery and provider "
            "actions stay in the authorized claims process.")
    c1, c2, c3 = st.columns(3)
    for col, lvl, route_, desc in [(c1, "High", "Priority review", "Review priority 80+"),
                                   (c2, "Medium", "Routine review", "Review priority 50–79"),
                                   (c3, "Low", "No additional review recommended", "Below 50 · claims continue normally")]:
        g = open_cases[open_cases.route == route_]
        col.markdown(f'<div class="route" style="border-top:4px solid {RISK[lvl]}"><div class="h" style="color:{RISK[lvl]}">'
                     f'{route_}</div><div class="v">{len(g)} cases</div>'
                     f'<div class="d">{money(g.billed_amount.sum())} billed · {desc}</div></div>', unsafe_allow_html=True)
    next_bar("find out which providers are behind the flags.", "Go to Provider Risk →", "Provider Risk", "nx-ov")

# ================= Provider Risk =================
if page == "Provider Risk":
    page_header("Provider Risk", "Which providers are behind the flagged billing. Pick a provider to see their profile "
                "against peers. Comparisons use the <b>rendering</b> provider within specialty; providers with fewer than "
                f"{detect.UPCODE_MIN_VISITS} office visits are left out of visit-level comparisons as low-volume.", 4)
    todo(("You will", "Scan the ranking", "most billed on flagged claims first"),
         ("Then", "Pick a provider", "tick a row; the profile below updates"),
         ("Finish by", "Opening their top case", "in the Investigator Queue"))
    p = (flagged.groupby(["provider_id", "provider_name", "specialty"])
         .agg(flagged_claims=("claim_id", "count"), billed_flagged=("billed_amount", "sum"),
              flag_reasons=("flag_reason", lambda t: ", ".join(sorted(set(t.dropna())))), max_score=("risk_score", "max"))
         .reset_index().sort_values("billed_flagged", ascending=False))
    em = scored[scored.cpt_code.isin(detect.EM_CODES)]
    mix = (em.groupby(["provider_id", "provider_name", "specialty"])
             .agg(visits=("claim_id", "count"), top=("cpt_code", lambda c: (c == "99215").mean()))
             .reset_index().query("visits >= @detect.UPCODE_MIN_VISITS"))
    mix["flag"] = mix.provider_id.isin(upcoders.provider_id).map({True: "Flagged", False: "Within Peer Range"})
    over = pdays[pdays.billed_hours > detect.IMPOSSIBLE_HOURS_MED]
    total_flag = max(p.billed_flagged.sum(), 1)

    section(1, "The Overall Picture",
            f"<b>{len(p)} of {scored.provider_id.nunique()} providers</b> have at least one flag, but the money is "
            f"concentrated: the top 5 hold <b>{p.billed_flagged.head(5).sum() / total_flag:.0%}</b> of flagged billed dollars.")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Providers With Flags", f"{len(p)}", f"of {scored.provider_id.nunique()} providers billing")
    kpi(c2, "With Unusual Billed Hours", f"{over.provider_id.nunique()}", f"{len(over)} days over 12 estimated hours")
    kpi(c3, "Visit Level Above Peers", f"{len(upcoders)}", "far above peers on 99215")
    kpi(c4, "Top 5 Share", f"{p.billed_flagged.head(5).sum() / total_flag:.0%}", money(p.billed_flagged.head(5).sum()), teal=True)
    st.write("")
    spec = flagged.groupby(["specialty", "flag_reason"]).billed_amount.sum().reset_index()
    fig = px.bar(spec, x="billed_amount", y="specialty", color="flag_reason", orientation="h",
                 color_discrete_map=FLAG_COLOR, labels={"billed_amount": "Billed On Flagged Claims", "specialty": ""})
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style(fig, 300, "Billed On Flagged Claims By Specialty And Flag Reason"), width="stretch")

    section(2, "Provider Ranking", "Tick the box at the start of a row to see that provider's profile in step 3. "
            "The top provider is shown until you pick one.")
    pr = p.reset_index(drop=True)
    sel = st.dataframe(pr, hide_index=True, width="stretch", height=300, on_select="rerun",
                       selection_mode="single-row", key="prov_table", column_config={
                           "provider_id": "ID", "provider_name": "Provider", "specialty": "Specialty",
                           "flagged_claims": "Flagged Claims",
                           "billed_flagged": st.column_config.NumberColumn("Billed On Flagged", format="$%.0f"),
                           "flag_reasons": "Flag Reasons",
                           "max_score": st.column_config.ProgressColumn("Highest Review Priority", min_value=0, max_value=100, format="%d")})
    pid = None
    if len(pr):
        row = pr.iloc[sel.selection.rows[0]] if sel.selection.rows else pr.iloc[0]
        pid, pname, pspec = row.provider_id, row.provider_name, row.specialty
        ss.prov_pid = pid
        mine_days = pdays[pdays.provider_id == pid].sort_values("service_date")
        n_imp = int((mine_days.billed_hours > detect.IMPOSSIBLE_HOURS_MED).sum())
        mine_em = em[em.provider_id == pid]
        peers_em = em[(em.specialty == pspec) & (em.provider_id != pid)]
        my_top = (mine_em.cpt_code == "99215").mean() if len(mine_em) else np.nan
        peer_top = (peers_em.cpt_code == "99215").mean() if len(peers_em) else np.nan
        my_cases = cases[cases.provider_id == pid]
        section(3, f"Provider Profile: {pname}",
                f"{pspec} · <b>{money(row.billed_flagged)}</b> billed on flagged claims across <b>{len(my_cases)} cases</b> "
                f"({row.flag_reasons}).")
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "Billed On Flagged", money(row.billed_flagged), f"{row.billed_flagged / total_flag:.1%} of all flagged $")
        kpi(c2, "Flagged Claims", f"{row.flagged_claims}", f"of {int((scored.provider_id == pid).sum())} claims billed")
        kpi(c3, "Days Over 12 Est. Hours", f"{n_imp}", f"longest day {mine_days.billed_hours.max():.1f} h" if len(mine_days) else "-")
        kpi(c4, "Top-Level Visits", f"{my_top:.0%}" if my_top == my_top else "-",
            f"peer median {peer_top:.0%}" if peer_top == peer_top else "no office visits")
        kpi(c5, "Open Cases", f"{int((~my_cases.status.isin(feedback.CLOSED)).sum())}",
            f"{int((my_cases.status == feedback.SUBST).sum())} substantiated after review", teal=True)
        st.write("")
        l, r = st.columns(2)
        with l:
            d = mine_days.assign(level=np.select(
                [mine_days.billed_hours > detect.IMPOSSIBLE_HOURS_HIGH, mine_days.billed_hours > detect.IMPOSSIBLE_HOURS_MED],
                ["Over 16 h (high)", "12–16 h (review)"], "Normal day"))
            fig = px.bar(d, x="service_date", y="billed_hours", color="level",
                         color_discrete_map={"Over 16 h (high)": RISK["High"], "12–16 h (review)": RISK["Medium"], "Normal day": BLUE2},
                         labels={"service_date": "Service Date", "billed_hours": "Estimated Hours"})
            fig.add_hline(y=detect.IMPOSSIBLE_HOURS_MED, line_dash="dash", line_color=RISK["Medium"])
            st.plotly_chart(style(fig, 340, "Estimated Hours On Each Working Day"), width="stretch")
        with r:
            if len(mine_em):
                lv = pd.concat([mine_em.cpt_code.value_counts(normalize=True).rename(pname),
                                peers_em.cpt_code.value_counts(normalize=True).rename(f"{pspec} peers")], axis=1) \
                    .reindex(detect.EM_CODES).fillna(0).reset_index(names="code").melt(id_vars="code")
                lv["code"] = lv.code.map({"99213": "99213 Low", "99214": "99214 Moderate", "99215": "99215 High"})
                fig = px.bar(lv, x="code", y="value", color="variable", barmode="group", text_auto=".0%",
                             color_discrete_sequence=[RISK["High"] if pid in set(upcoders.provider_id) else NAVY, GREY],
                             labels={"code": "Office Visit Level", "value": "Share Of Visits", "variable": ""})
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(style(fig, 340, "Office Visit Levels Vs Peers"), width="stretch")
            else:
                st.info(f"{pname} bills no office visits, so the visit-level check does not apply.")
        st.markdown("**Their Cases**")
        st.dataframe(my_cases[["case_id", "flag_reason", "claims", "billed_amount", "risk_score", "route", "status"]],
                     hide_index=True, width="stretch", height=220, column_config={
                         "case_id": "Case", "flag_reason": "Flag Reason", "claims": "Claims",
                         "billed_amount": st.column_config.NumberColumn("Billed", format="$%.0f"),
                         "risk_score": st.column_config.NumberColumn("Review Priority", format="%d"),
                         "route": "Routing", "status": "Status"})

    section(4, "Estimated Billed Hours Across All Providers",
            f"Each dot is one provider on one day. Almost all of the {len(pdays):,} provider-days sit below 10 hours; "
            f"<b>{len(over)}</b> go above 12." + (f" <b>{pname}</b>'s days are ringed in red." if pid else ""))
    fig = px.scatter(pdays, x="service_date", y="billed_hours", color="specialty", hover_name="provider_name",
                     color_discrete_sequence=[NAVY, BLUE, BLUE2, TEAL, "#64748b", GREY],
                     labels={"service_date": "Service Date", "billed_hours": "Estimated Hours", "specialty": ""})
    if pid:
        m = pdays[pdays.provider_id == pid]
        fig.add_scatter(x=m.service_date, y=m.billed_hours, mode="markers", name=pname, hovertext=m.provider_name,
                        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=RISK["High"], width=2.5)))
    fig.add_hline(y=detect.IMPOSSIBLE_HOURS_MED, line_dash="dash", line_color=RISK["Medium"],
                  annotation_text="12 h: review", annotation_font_size=13)
    fig.add_hline(y=detect.IMPOSSIBLE_HOURS_HIGH, line_dash="dash", line_color=RISK["High"],
                  annotation_text="16 h: high risk", annotation_font_size=13)
    st.plotly_chart(style(fig, 400), width="stretch")

    section(5, "Visit Levels Across All Providers",
            f"Each dot is a provider. <b>{len(upcoders)} providers</b> (red) bill the top-level visit far more often "
            "than peers." + (f" <b>{pname}</b> is ringed." if pid and pid in set(mix.provider_id) else ""))
    fig = px.strip(mix, x="specialty", y="top", color="flag", hover_name="provider_name",
                   color_discrete_map={"Flagged": RISK["High"], "Within Peer Range": BLUE},
                   labels={"specialty": "", "top": "Share Of Visits Billed 99215"})
    fig.update_traces(marker_size=12)
    if pid and pid in set(mix.provider_id):
        m = mix[mix.provider_id == pid]
        fig.add_scatter(x=m.specialty, y=m.top, mode="markers", name=pname,
                        marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="#000", width=2.5)))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 380), width="stretch")
    top_case = None
    if pid:
        oc = open_cases[open_cases.provider_id == pid]
        top_case = oc.case_id.iloc[0] if len(oc) else None
    next_bar(f"review {pname}'s top open case." if top_case else "work through the open cases.",
             "Open their top case →" if top_case else "Go to the Queue →", "Investigator Queue", "nx-prov", top_case)


# ================= Investigator Queue =================
def case_story(case):
    """Plain-language reason, comparison, context, claim rows and checklist for one case."""
    ctx = summarize.case_context(case, scored, pdays, upcoders)
    ref = detect.load_reference()
    rows = scored[scored.claim_id.isin(case.claim_ids)].copy()
    cols = ["claim_id", "member_id", "service_date", "cpt_code", "cpt_description", "units", "billed_amount"]
    info = {"dup": None}
    if case.case_type == "impossible_day":
        h, peer = ctx["billed_hours_that_day"], ctx["specialty_peer_median_hours_per_day"]
        others = ctx["other_flagged_days_same_provider"]
        reason = (f"<b>{h:.1f} estimated hours based on billed codes and units</b> for {case.provider_name} (rendering) on "
                  f"{case.service_date:%d %b %Y}. {case.specialty} providers in this data have a median of {peer:.1f} "
                  f"estimated hours a day. Over {detect.IMPOSSIBLE_HOURS_HIGH} estimated hours in one day is placed in "
                  f"priority review; over {detect.IMPOSSIBLE_HOURS_MED} in routine review.")
        cmp = [("This provider, that day", h, RISK["High"] if h > detect.IMPOSSIBLE_HOURS_HIGH else RISK["Medium"]),
               ("Specialty median day", peer, BLUE2), ("Priority-review line", detect.IMPOSSIBLE_HOURS_HIGH, GREY)]
        unit = "h"
        rows["ref_minutes"] = rows.cpt_code.map(ref)
        rows["est_minutes"] = rows.ref_minutes.fillna(15) * rows.units
        rows["share_of_day"] = (rows.est_minutes / max(rows.est_minutes.sum(), 1) * 100).round(1)
        cols = cols[:-1] + ["est_minutes", "share_of_day", "billed_amount"]
        sp = pdays[pdays.specialty == case.specialty]
        info["cohort"] = (f"{sp.provider_id.nunique()} {case.specialty.lower()} rendering providers · {dmin:%d %b} – "
                          f"{dmax:%d %b %Y} · {len(sp):,} provider-days")
        info["assume"] = ("Time per line = units × the code's typical minutes in the CPT reference; codes without a "
                          "reference time count 15 min per unit.")
        miss = int(rows.ref_minutes.isna().sum())
        info["coverage"] = (f"{len(rows) - miss} of {len(rows)} lines have a reference time"
                            + (f"; {miss} counted at the 15-min default." if miss else "."))
        checks = [(f"Request treatment and attendance records for {case.service_date:%d %b}", "Do the notes support every billed unit for each member?"),
                  ("Confirm who rendered each service", "Compare against the practice's staff rota for that day"),
                  ("Compare with this provider's other long days", (", ".join(others[:3]) or "No other days over 12 h")),
                  ("Check members with several claims that day", "Were they billed again elsewhere?")]
        extra_expl = ["Group practice: other clinicians' time billed under one rendering ID",
                      "Therapy assistants' time billed under the supervising clinician"]
    elif case.case_type == "upcoding":
        reason = (f"<b>{ctx['share_billed_at_99215']} of this provider's {ctx['office_visits_billed']} office visits were billed at "
                  f"the highest level (99215).</b> The {case.specialty} peer median is {ctx['specialty_peer_median_share_99215']} "
                  f"({ctx['std_devs_above_peers']} standard deviations above {ctx['peer_providers_compared']} peers).")
        cmp = [("This provider", float(ctx["share_billed_at_99215"].rstrip("%")), RISK["High"]),
               ("Specialty peer median", float(ctx["specialty_peer_median_share_99215"].rstrip("%")), BLUE2)]
        unit = "%"
        info["cohort"] = (f"{ctx['peer_providers_compared']} {case.specialty.lower()} providers with ≥ {detect.UPCODE_MIN_VISITS} office "
                          f"visits · {dmin:%d %b} – {dmax:%d %b %Y} · provider excluded from own peer group")
        info["assume"] = "Established-patient office visits 99213–99215 only."
        info["coverage"] = (f"{ctx['office_visits_billed']} visits ≥ the {detect.UPCODE_MIN_VISITS}-visit minimum, so this is "
                            "not a low-volume comparison.")
        checks = [("Sample 10 high-level visits", "Do the notes show the complexity a 99215 requires?"),
                  ("Check the patient mix", "Diagnoses and age profile compared with peers"),
                  ("Compare month by month", "Has the share always been high, or did it jump?")]
        extra_expl = ["Different setting (e.g. urgent care) compared with office-based peers"]
    else:
        f, o = ctx["flagged_claim"], ctx["original_claim"] or {}
        reason = (f"<b>Claim {f['claim_id']} repeats {o.get('claim_id', 'an earlier claim')}:</b> same member, provider and "
                  f"procedure ({f['cpt_code']}). {case.reasons}")
        cmp = [(f"Original {o.get('claim_id', '')}", float(o.get("billed_amount", 0) or 0), BLUE2),
               (f"Later {f['claim_id']}", float(f["billed_amount"]), RISK["High"])]
        unit = "$"
        rows = pd.concat([scored[scored.claim_id == o.get("claim_id")], rows]).drop_duplicates("claim_id")
        info["cohort"] = "Matched against every claim for this member in the loaded dataset"
        info["assume"] = ("Exact: member, provider, code, service date and amount match. Near: same member, provider and code "
                          f"within ±{detect.NEAR_DUP_DAYS} day and ±{detect.NEAR_DUP_AMOUNT_PCT:.0%} amount.")
        info["coverage"] = "All match fields present on both claims."
        nd = "not in this dataset"
        info["dup"] = [("Claim ID", o.get("claim_id", "-"), f["claim_id"]),
                       ("Submitted", o.get("submitted_date", "")[:10], f["submitted_date"][:10]),
                       ("Service date", o.get("service_date", "")[:10], f["service_date"][:10]),
                       ("Code", o.get("cpt_code", ""), f["cpt_code"]),
                       ("Billed", f"${o.get('billed_amount', '')}", f"${f['billed_amount']}"),
                       ("Allowed · paid", nd, nd), ("Replacement / frequency code", nd, nd), ("Adjustment / reversal", nd, nd)]
        checks = [("Check whether the original claim was paid", "If it was, paying this one would pay twice."),
                  ("Ask the provider whether this is a corrected claim", "A valid replacement voids or replaces the original."),
                  ("Compare dates and amounts", f"Original {o.get('service_date', '')[:10]} ${o.get('billed_amount', '')} · "
                                                f"later {f['service_date'][:10]} ${f['billed_amount']}")]
        extra_expl = ["Corrected claim sent without the replacement code", "Two genuine same-day services"]
    info["billing"] = (f"Rendering: {case.provider_name} ({case.provider_id})<br>Billing provider: not in this dataset "
                       "(added for the pilot)")
    return ctx, reason, cmp, unit, rows[cols], checks, info, extra_expl


def in_view(df, view):
    if view == "mine":
        return df[(df.owner == ss.user) & df.status.isin([feedback.OPEN, feedback.PENDING])]
    if view == "unassigned":
        return df[df.owner.isna() & df.status.isin([feedback.OPEN, feedback.PENDING])]
    if view == "referred":
        return df[df.status == feedback.SIU]
    return df


if page == "Investigator Queue":
    page_header("Investigator Queue", "Your cases first. Click a case; it opens beside the list with its evidence, "
                "checklist, outcome form and full history.", 5)
    todo(("1 · Why", "Verify the evidence", "rule, comparison, claim rows, explanations"),
         ("2 · Check", "Work the checklist", "saved with the case"),
         ("3 · Record", "Record a structured outcome", "reason + evidence · full history kept"))
    base = cases[cases.route.isin(ss.q_routes) | cases.status.isin(feedback.FINAL + [feedback.PENDING])]
    base = base[base.flag_reason.isin(ss.q_reasons)]
    VIEWS = {"mine": "My cases", "unassigned": "Unassigned", "referred": "Referred", "all": "All"}
    counts = {k: len(in_view(base, k)) for k in VIEWS}
    left, right = st.columns([1, 1.85], gap="medium")
    with left:
        st.segmented_control("View", list(VIEWS), key="q_view", format_func=lambda k: f"{VIEWS[k]} {counts[k]}",
                             label_visibility="collapsed")
        qs = st.text_input("Find", key="q_search", placeholder="🔎 Find by case, claim or provider ID…",
                           label_visibility="collapsed").strip().lower()
        q = in_view(base, ss.q_view or "all")
        if qs:
            hit = q.apply(lambda r: qs in r.case_id.lower() or qs in r.provider_id.lower() or qs in r.provider_name.lower()
                          or any(qs in c.lower() for c in r.claim_ids), axis=1)
            q = q[hit] if len(q) else q
        st.markdown(f"**{len(q)} case{'s' if len(q) != 1 else ''} · {money(q.billed_amount.sum())} billed**")
        with st.container(height=760, key="qlist"):
            if q.empty:
                st.info("No cases in this view.")
            for r_ in q.head(80).itertuples():
                icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[r_.risk_level]
                done = "" if r_.status == feedback.OPEN else f" · ✓ {r_.status}"
                due = f"{r_.due:%d %b}" if pd.notna(r_.due) else "-"
                lab = (f"{icon} **{r_.risk_score:.0f}** · {SHORT[r_.case_type]}{done}  \n{r_.provider_name} · "
                       f"${r_.billed_amount:,.0f}  \n👤 {r_.owner or 'Unassigned'} · {r_.age_days}d old · due {due}")
                st.button(lab, key=f"case-{r_.case_id}", width="stretch",
                          type="primary" if r_.case_id == ss.case_id else "secondary",
                          on_click=go, args=("Investigator Queue", r_.case_id))
            if len(q) > 80:
                st.caption(f"Showing the top 80 of {len(q)}. Narrow the list with ⚙ Filters or search.")
    if ss.case_id not in set(cases.case_id):
        ss.case_id = q.case_id.iloc[0] if len(q) else None
    with right:
        if ss.case_id is None:
            st.info("Pick a case from the list.")
        else:
            case = cases[cases.case_id == ss.case_id].iloc[0]
            if ss.get("_viewed") != (DS, case.case_id, ss.user):
                ss._viewed = (DS, case.case_id, ss.user)
                feedback.log(DS, ss.user, "view case", case.case_id)
            ctx, reason, cmp, unit, rows, checks, info, extra_expl = case_story(case)
            RI = RULE_INFO[case.case_type]
            hist = feedback.history(DS, case.case_id)
            last = hist.iloc[0] if len(hist) else None
            st.markdown(
                f'<div class="card"><div class="chead"><div class="score s-{case.risk_level}" title="Review priority">{case.risk_score:.0f}</div>'
                f'<div><div class="crumb" style="margin:0">Case {case.case_id} · {status_tag(case.status)} · {case.route}</div>'
                f'<h3>{FLAG[case.case_type]}</h3><p>{case.provider_name} ({case.provider_id}) · {case.specialty}</p></div></div>'
                f'<div class="facts"><span>Service date <b>{case.service_date:%d %b %Y}</b></span><span>Claims <b>{int(case.claims)}</b></span>'
                f'<span>Billed <b>${case.billed_amount:,.2f}</b></span><span>Owner <b>{case.owner or "Unassigned"}</b></span>'
                f'<span>Age <b>{case.age_days} days</b></span><span>Due <b>{case.due:%d %b %Y}</b></span>'
                f'<span>Last action <b>{case.last_action}</b></span></div></div>', unsafe_allow_html=True)
            a1, a2, _ = st.columns([1.1, 1.3, 2])
            if not case.owner and ROLE in ("Investigator", "Supervisor") and case.status in (feedback.OPEN, feedback.PENDING):
                if a1.button("👤 Assign to me", key="assign"):
                    feedback.assign(DS, case, ss.user, ss.user, ROLE)
                    st.rerun()
            if a2.button("✦ Ask about this case", key="askcase"):
                ss.chat.clear()
                ss._asked_first = False
                chat_dialog("Why was this case flagged?")

            # ---- 1. Why was this flagged? ----
            with st.container(border=True):
                st.markdown('<div class="bh"><i>1</i><b>Why was this flagged?</b><span>Rule-based, explainable</span></div>'
                            f'<p class="reason">{reason}</p>', unsafe_allow_html=True)
                fig = go_.Figure(go_.Bar(x=[v for _, v, _ in cmp], y=[l_ for l_, _, _ in cmp], orientation="h",
                                         marker_color=[c for _, _, c in cmp],
                                         text=[(f"${v:,.2f}" if unit == "$" else f"{v:.0f}%" if unit == "%" else f"{v:.1f} h")
                                               for _, v, _ in cmp], textposition="outside"))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(range=[0, max(v for _, v, _ in cmp) * 1.25 or 1])
                fig.update_layout(showlegend=False)
                st.plotly_chart(style(fig, 70 + 42 * len(cmp)), width="stretch", config={"displayModeBar": False})
                if case.case_type == "impossible_day":
                    others = pdays[pdays.provider_id != case.provider_id]
                    mine = pdays[pdays.provider_id == case.provider_id]
                    hi = mine[mine.service_date == case.service_date]
                    f2 = go_.Figure()
                    f2.add_scatter(x=others.service_date, y=others.billed_hours, mode="markers", name="All other provider-days",
                                   marker=dict(size=4, color="#cbd5e1"), hoverinfo="skip")
                    f2.add_scatter(x=mine.service_date, y=mine.billed_hours, mode="markers", name="This provider",
                                   marker=dict(size=7, color=NAVY))
                    f2.add_scatter(x=hi.service_date, y=hi.billed_hours, mode="markers", name="Flagged day",
                                   marker=dict(size=13, color=RISK["High"]))
                    f2.add_hline(y=detect.IMPOSSIBLE_HOURS_HIGH, line_dash="dash", line_color=RISK["High"])
                    f2.update_layout(legend=dict(orientation="h", y=1.15))
                    f2.update_yaxes(title_text="Est. hours")
                    st.plotly_chart(style(f2, 250, "This Provider's Days In Context"), width="stretch",
                                    config={"displayModeBar": False})
                adj_txt = (f"<b>+{feedback.ADJ_VALUE}</b> Prior substantiated billing issue on another case "
                           f"<span class='pill md'>{'applied' if feedback.ADJ_ENABLED else 'not validated · not applied'}</span>"
                           if case.adjustment_eligible else "None")
                st.markdown(f'<div class="prio"><div><b>Review priority {case.risk_score:.0f}</b> · {case.route}</div><table>'
                            f'<tr><td>Rule</td><td>{RI["rule"]}</td></tr><tr><td>Threshold</td><td>{RI["thr"]}</td></tr>'
                            f'<tr><td>Base score from the rule</td><td><b>{case.base_score:.0f}</b></td></tr>'
                            f'<tr><td>Adjustment</td><td>{adj_txt}</td></tr></table>'
                            '<div class="hint">Review priority says how strongly the rules suggest a review. It is not a '
                            'probability of fraud.</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ctx"><div><b>Peer cohort</b>{info["cohort"]}</div><div><b>Time / match assumptions</b>'
                            f'{info["assume"]}</div><div><b>Code coverage</b>{info["coverage"]}</div><div><b>Billing vs rendering</b>'
                            f'{info["billing"]}</div></div>', unsafe_allow_html=True)
                b = briefs.get(case.case_id) or summarize.rule_based_brief(ctx) | {"source": "rules"}
                sup = "".join(f"<li>{x}</li>" for x in b["evidence"][:5])
                alt = "".join(f"<li>{x}</li>" for x in list(dict.fromkeys(b["innocent_explanations"] + extra_expl)))
                st.markdown(f'<div class="sbs"><div class="sup"><h5>Supports a review</h5><ul>{sup}</ul></div>'
                            f'<div class="alt"><h5>Possible legitimate explanations</h5><ul>{alt}</ul></div></div>',
                            unsafe_allow_html=True)
                if info["dup"]:
                    trs = "".join(f'<tr><th>{k}</th><td>{a}</td><td class="{"diff" if a != b_ else ""}">{b_}</td></tr>'
                                  for k, a, b_ in info["dup"])
                    st.markdown(f'<div class="card" style="padding:10px 12px"><b style="color:{NAVY}">Original vs later submission</b>'
                                f'<table class="dup"><tr><th></th><th>Original</th><th>Later submission</th></tr>{trs}</table>'
                                '<div class="hint">Highlighted cells differ. The pilot adds paid amounts and replacement codes '
                                'so legitimate corrections are easy to recognise.</div></div>', unsafe_allow_html=True)
                lf = feedback.line_findings(DS, case.case_id).set_index("claim_id").finding
                with st.expander(f"▸ Show the {len(rows)} claim rows used, with each line's contribution and your line findings"):
                    ed = rows.copy()
                    ed["line_finding"] = ed.claim_id.map(lf).map(feedback.FINDINGS).fillna("Not reviewed")
                    can_edit = ROLE in ("Investigator", "Supervisor")
                    edited = st.data_editor(
                        ed, hide_index=True, width="stretch", height=min(40 + 35 * len(ed), 360), key=f"lf-{case.case_id}",
                        disabled=[c for c in ed.columns if c != "line_finding" or not can_edit],
                        column_config={"billed_amount": st.column_config.NumberColumn("billed", format="$%.2f"),
                                       "service_date": st.column_config.DateColumn("service_date"),
                                       "est_minutes": st.column_config.NumberColumn("est. minutes"),
                                       "share_of_day": st.column_config.NumberColumn("share of day %"),
                                       "line_finding": st.column_config.SelectboxColumn("Line finding", options=list(feedback.FINDINGS.values()))})
                    vals = edited.line_finding.value_counts()
                    st.caption(f"Line findings: {vals.get('Substantiated', 0)} substantiated · {vals.get('Not substantiated', 0)} "
                               f"not substantiated · {vals.get('Uncertain', 0)} uncertain · {vals.get('Not reviewed', 0)} not reviewed. "
                               "Only substantiated / not-substantiated lines can become training labels.")
                    c1, c2 = st.columns([1, 1])
                    inv = {v: k for k, v in feedback.FINDINGS.items()}
                    if can_edit and c1.button("💾 Save line findings", key="savelf"):
                        feedback.save_line_findings(DS, case.case_id, dict(zip(edited.claim_id, edited.line_finding.map(inv))), ss.user)
                        st.toast("Line findings saved")
                    c2.download_button("⬇ Download these rows", rows.to_csv(index=False), f"{case.case_id}_claims.csv",
                                       "text/csv", key=f"dl-{case.case_id}", on_click=log_export, args=(f"{case.case_id} rows",))
                tag = '<span class="ai">✦ CLAUDE</span>' if b.get("source") == "claude" else \
                    '<span class="rb">RULE-BASED</span> <span style="font-size:11px;color:#64748b">· ✦ AI version when enabled, each statement linked to the rows above</span>'
                ev = "".join(f"<li>{x}</li>" for x in b["evidence"])
                st.markdown(f'<div class="brief"><div style="display:flex;justify-content:space-between;gap:8px"><b>CASE BRIEF</b>'
                            f'<span>{tag}</span></div><p style="margin:6px 0 0"><b>{b["headline"]}</b></p>'
                            f'<h4>Finding</h4><p style="margin:0">{b["finding"]}</p><h4>Evidence</h4><ul>{ev}</ul>'
                            f'<h4>Recommended next step</h4><p style="margin:0">{b["recommended_action"]}</p></div>',
                            unsafe_allow_html=True)
                with st.expander("⚙ Technical details: rule logic, score formula, data lineage"):
                    st.markdown(f"**Rule:** {RI['rule']}  \n**Score formula:** {RI['formula']}  \n**Source:** `{RI['src']}`  \n"
                                f"**Ruleset:** {RULESET} · **dataset:** {DS} · **org:** {feedback.ORG}")
                    st.code(RI["sql"], language="sql")
                    st.caption("Equivalent SQL for review. The rule runs in pandas; every flag keeps its reason string and input claim IDs.")

            # ---- 2. Checklist ----
            stt = feedback.state(DS).set_index("case_id")
            saved = stt.loc[case.case_id]
            import json as _json
            ticks = _json.loads(saved.checklist or "[]") + [False] * len(checks)
            with st.container(border=True):
                st.markdown(f'<div class="bh"><i>2</i><b>What should I check next?</b><span>{sum(ticks[:len(checks)])} of '
                            f'{len(checks)} done · saved with the case</span></div>', unsafe_allow_html=True)
                new_ticks = [st.checkbox(f"**{t}**  \n{sub}", value=bool(ticks[i]), key=f"chk-{case.case_id}-{i}")
                             for i, (t, sub) in enumerate(checks)]
                if new_ticks != [bool(x) for x in ticks[:len(checks)]]:
                    feedback.save_checklist(DS, case.case_id, new_ticks)
                    st.rerun()
                notes = st.text_area("Evidence notes (saved with the case)", value=saved.evidence_notes or "",
                                     key=f"evn-{case.case_id}", height=70)
                if notes != (saved.evidence_notes or "") and st.button("💾 Save notes", key="savenotes"):
                    feedback.save_notes(DS, case.case_id, notes)
                    st.toast("Notes saved")
                    st.rerun()

            # ---- 3. Outcome ----
            with st.container(border=True):
                st.markdown(f'<div class="bh"><i>3</i><b>Record the outcome</b><span>Signed in as {ss.user} ({ROLE}) · '
                            'every change is kept</span></div>', unsafe_allow_html=True)
                if case.status == feedback.OPEN:
                    if ROLE not in ("Investigator", "Supervisor"):
                        st.info(f"{ROLE}s can view cases but not record outcomes.")
                    else:
                        caps = {feedback.SUBST: "Evidence shows the billing is wrong", feedback.NO_ISSUE: "Records support the billing",
                                feedback.INSUFF: "Can't conclude either way", feedback.SIU: "Possible intent; SIU owns formal findings"}
                        out = st.radio("Outcome", feedback.FINAL, index=None, key=f"out-{case.case_id}",
                                       format_func=lambda o: "Refer to SIU" if o == feedback.SIU else o, captions=[caps[o] for o in feedback.FINAL])
                        if out:
                            f1, f2 = st.columns(2)
                            reason_ = f1.selectbox("Reason (required)", feedback.REASONS[out], index=None, key=f"rs-{case.case_id}-{out}")
                            if out != feedback.INSUFF:
                                evl = f2.multiselect("Evidence: claim lines (required: lines or a document reference)",
                                                     list(rows.claim_id), key=f"evl-{case.case_id}")
                            else:
                                evl = []
                            evt = st.text_input("Evidence: document reference (records request #, page…)", key=f"evt-{case.case_id}")
                            owner_, nxt, amt = None, "", None
                            if out == feedback.SIU:
                                g1, g2 = st.columns(2)
                                owner_ = g1.selectbox("SIU owner (required)", feedback.SIU_OWNERS, index=None, key=f"siuo-{case.case_id}")
                                nxt = g2.text_input("Next step (required)", key=f"siun-{case.case_id}", placeholder="e.g. Records request, due 8 Dec")
                            if out == feedback.SUBST:
                                amt = st.number_input("Substantiated amount ($, counted once per claim line in reports)",
                                                      min_value=0.0, value=float(case.billed_amount), key=f"amt-{case.case_id}")
                            note = st.text_area("Note for the record", key=f"note-{case.case_id}", height=70)
                            if st.button("Save outcome", type="primary", key="saveout"):
                                ev_ = ", ".join(evl + ([evt] if evt.strip() else []))
                                err = ("Choose a reason." if not reason_ else
                                       "Add at least one evidence reference (a claim line or a document reference)."
                                       if out != feedback.INSUFF and not ev_ else
                                       "A referral needs an SIU owner and a next step." if out == feedback.SIU and (not owner_ or not nxt.strip())
                                       else "")
                                if err:
                                    st.error(err)
                                else:
                                    status_, appr = out, ""
                                    if out == feedback.SUBST and (amt or 0) > feedback.APPROVAL_LIMIT and ROLE != "Supervisor":
                                        status_, appr = feedback.PENDING, "Awaiting supervisor approval"
                                    feedback.record(DS, case, status_, ss.user, ROLE, reason=reason_, evidence=ev_, note=note,
                                                    amount=amt, owner=owner_, next_step=nxt, approval=appr)
                                    st.toast(f"{case.case_id}: {status_}")
                                    st.rerun()
                elif case.status == feedback.PENDING:
                    st.warning(f"Substantiated amount is over ${feedback.APPROVAL_LIMIT:,} and needs supervisor approval before the "
                               "case closes.")
                    if ROLE == "Supervisor":
                        if st.button("✓ Approve substantiation", type="primary", key="approve"):
                            p = hist[hist.new_status == feedback.PENDING].iloc[0]
                            feedback.record(DS, case, feedback.SUBST, ss.user, ROLE, reason="Approved substantiation",
                                            evidence=p.evidence, amount=p.amount, approval=f"Approved by {ss.user}")
                            st.rerun()
                    else:
                        st.caption("Switch the demo sign-in to J. Wong (Supervisor) in the sidebar to approve.")
                if case.status != feedback.OPEN:
                    st.markdown(f'<div class="recorded">✅ Current status: <b>{case.status}</b>'
                                + (f' · owner <b>{case.owner}</b>' if case.status == feedback.SIU else "")
                                + (f'<div class="who">Last change {last.ts[:16].replace("T", " ")} by {last.reviewer}: {last.reason}</div>'
                                   if last is not None else "") + '</div>', unsafe_allow_html=True)
                    if ROLE in ("Investigator", "Supervisor"):
                        r1, r2, r3 = st.columns([1.6, 1, 1])
                        rr = r1.selectbox("Reason for reversing", feedback.REVERSE_REASONS, index=None, key=f"rev-{case.case_id}",
                                          label_visibility="collapsed", placeholder="Reason for reversing…")
                        if r2.button("↶ Reverse outcome", key="reverse", width="stretch"):
                            if not rr:
                                st.error("Choose a reason for reversing first.")
                            else:
                                feedback.record(DS, case, feedback.OPEN, ss.user, ROLE, reason=f"Outcome reversed: {rr}", kind="reversal")
                                st.rerun()
                        nxt_ = in_view(base, "mine")
                        nxt_ = nxt_[nxt_.case_id != case.case_id]
                        nxt_ = nxt_ if len(nxt_) else open_cases[(open_cases.status == feedback.OPEN) & (open_cases.case_id != case.case_id)]
                        if len(nxt_):
                            r3.button("Next case →", key="nextcase", type="primary", width="stretch", on_click=go,
                                      args=("Investigator Queue", nxt_.case_id.iloc[0]))
                        st.caption("Reversing adds a new entry to the history; nothing is deleted.")
                st.markdown('<p class="hint">Payment, denial, recovery and provider actions stay in the authorized claims '
                            'process. ClaimGuard records the review outcome only. Formal fraud findings belong to SIU.</p>',
                            unsafe_allow_html=True)

            # ---- 4. History ----
            with st.container(border=True):
                st.markdown(f'<div class="bh"><i>4</i><b>Case history</b><span>Full audit trail · dataset {DS} · org '
                            f'{feedback.ORG}</span></div>', unsafe_allow_html=True)
                if hist.empty:
                    st.caption("Flagged by rule; no actions yet.")
                else:
                    h = hist.assign(when=hist.ts.str[:16].str.replace("T", " "),
                                    change=hist.prior_status + " → " + hist.new_status)
                    st.dataframe(h[["when", "reviewer", "role", "kind", "change", "reason", "evidence", "approval", "note"]],
                                 hide_index=True, width="stretch", height=min(40 + 35 * len(h), 280))
    st.download_button("⬇ Export flagged claims", flagged.drop(columns=["planted_fraud"], errors="ignore").to_csv(index=False),
                       "flagged_claims.csv", "text/csv", on_click=log_export, args=("flagged claims CSV",))
    next_bar("after a few outcomes, see what reviewers found.", "Go to Review Outcomes →", "Review Outcomes", "nx-q")

# ================= Review Outcomes =================
if page == "Review Outcomes":
    page_header("Review Outcomes", "What reviewers found, with denominators. Outcomes are kept as records; they do "
                "<b>not</b> change rules or the model until independently validated and approved.", 6)
    if is_sample:
        st.markdown(f'<div class="warn">⚠ <b>Synthetic sample outcomes.</b> Most reviews were generated for the demo '
                    f'(reviewer "demo seed (simulated)") and are stored in dataset {DS}, isolated from any pilot data. '
                    'Your outcomes in the Queue are added here.</div>', unsafe_allow_html=True)
    n_rev = len(reviewed)
    cnt = reviewed.decision.value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Cases Reviewed", f"{n_rev}", f"of {len(cases)} flagged ({n_rev / max(len(cases), 1):.0%})")
    kpi(c2, "Issue Substantiated", f"{cnt.get(feedback.SUBST, 0)}", "billing or coding")
    kpi(c3, "No Issue Found", f"{cnt.get(feedback.NO_ISSUE, 0)}", "records support billing")
    kpi(c4, "Insufficient Evidence", f"{cnt.get(feedback.INSUFF, 0)}", "inconclusive")
    kpi(c5, "Referred To SIU", f"{cnt.get(feedback.SIU, 0)}", f"{int((cases.status == feedback.PENDING).sum())} pending approval",
        teal=True)

    stats = feedback.rule_stats(fb)
    l, r = st.columns([1.2, 1])
    with l:
        section(1, "Substantiation Rate Among Reviewed Cases", "Always shown with its denominator.")
        rate = cnt.get(feedback.SUBST, 0) / max(n_rev, 1)
        st.markdown(f'<div class="card"><div style="display:flex;align-items:baseline;gap:10px"><div style="font-size:34px;'
                    f'font-weight:800;color:{NAVY}">{rate:.0%}</div><div style="color:#64748b">{cnt.get(feedback.SUBST, 0)} of '
                    f'{n_rev} reviewed cases</div></div><div class="hint" style="margin-top:6px"><b>Read with care:</b> reviewed '
                    'cases are a <b>selected subset</b> (priority review first), not a random sample of the queue, so this rate '
                    'does not describe all flags. The pilot also reviews a random sample of unflagged claims.</div></div>',
                    unsafe_allow_html=True)
        if not stats.empty:
            stats["flag_reason"] = stats.rule.map(FLAG)
            st.dataframe(stats[["flag_reason", "reviewed", "substantiated", "no_issue", "insufficient", "siu", "rate"]],
                         hide_index=True, width="stretch", column_config={
                             "flag_reason": "Flag Reason", "reviewed": "Reviewed", "substantiated": "Substantiated",
                             "no_issue": "No Issue", "insufficient": "Insufficient", "siu": "SIU",
                             "rate": st.column_config.ProgressColumn("Substantiation Rate", min_value=0, max_value=1, format="percent")})
    with r:
        section(2, "From Outcomes To Labels", "Case outcomes are not copied onto every claim line.")
        lf = feedback.line_findings(DS)
        rev_lines = set(c for ids in cases[cases.case_id.isin(reviewed.case_id)].claim_ids for c in ids)
        all_lines = set(c for ids in cases.claim_ids for c in ids)
        lfc = lf.finding.value_counts()
        tbl = pd.DataFrame([
            ("Substantiated", lfc.get("sub", 0), "Yes · positive"), ("Not substantiated", lfc.get("not", 0), "Yes · negative"),
            ("Uncertain", lfc.get("unc", 0), "No · unlabeled"),
            ("Not reviewed (in reviewed cases)", max(len(rev_lines) - len(lf[lf.finding != "nr"]), 0), "No · unlabeled"),
            ("Lines in unreviewed cases", len(all_lines - rev_lines), "No · unlabeled")],
            columns=["Line-level finding", "Lines", "Used as training label?"])
        st.dataframe(tbl, hide_index=True, width="stretch")
        labels = feedback.export_labels(DS, scored)
        st.caption(f"{len(labels):,} labelled lines saved to output/labels.csv. The ML model needs "
                   f"{ml_model.MIN_LABELS_PER_CLASS}+ per class; until then it uses the synthetic ground truth.")
        st.download_button("⬇ Download line-level labels", labels.to_csv(index=False), "labels.csv", "text/csv",
                           on_click=log_export, args=("labels CSV",))

    section(3, "Proposed Rule Changes · Approval Required",
            "Suggestions from review outcomes. None take effect until validated on later data and approved.")
    props = []
    for s_ in stats.itertuples() if not stats.empty else []:
        if s_.reviewed >= 5 and s_.rate < 0.5:
            props.append((f"{FLAG[s_.rule]}: tighten threshold", f"{s_.substantiated} of {s_.reviewed} substantiated ({s_.rate:.0%})",
                          "Awaiting independent validation"))
    props.append(("Priority adjustment +10 'prior substantiated billing issue'", "Effect not yet measured",
                  "Shown but not applied until validated"))
    nlab = labels.label.value_counts()
    props.append(("Retrain the ML model on line-level labels",
                  f"{nlab.get(1, 0)} positive / {nlab.get(0, 0)} negative lines (minimum {ml_model.MIN_LABELS_PER_CLASS} each)",
                  "Not enough labels" if min(nlab.get(1, 0), nlab.get(0, 0)) < ml_model.MIN_LABELS_PER_CLASS
                  else "Awaiting independent validation"))
    st.dataframe(pd.DataFrame(props, columns=["Proposal", "Evidence", "Status"]), hide_index=True, width="stretch")

    section(4, "Outcome Log", "Every outcome, reversal, approval and assignment, newest first.")
    ev = feedback.events(DS).iloc[::-1]
    ev = ev[ev.case_id.isin(cases.case_id)]
    st.dataframe(ev.assign(when=ev.ts.str[:16].str.replace("T", " "), change=ev.prior_status + " → " + ev.new_status)
                 [["when", "case_id", "reviewer", "role", "kind", "change", "reason", "evidence", "approval"]],
                 hide_index=True, width="stretch", height=300)
    next_bar("see the leadership view of yield, backlog and verified value.", "Go to Executive Dashboard →",
             "Executive Dashboard", "nx-fb")

# ================= Model Evaluation =================
if page == "Model Evaluation":
    page_header("Model Evaluation", "For analysts: how well the scores separate claims that turned out to be problems from normal "
                "claims, and how many false alarms they raise.", 7)
    has_truth = "planted_fraud" in scored and not scored.planted_fraud.isna().all()
    syn = '<span class="synth xs">synthetic</span>' if has_truth else ""
    if has_truth:
        st.markdown('<div class="warn">⚠ <b>Synthetic result.</b> These figures come from made-up data with planted '
                    'problems. They show how accuracy will be reported, <b>not</b> how accurate ClaimGuard is on real '
                    'claims. Real accuracy is measured in a pilot.</div>', unsafe_allow_html=True)
        ev = detect.evaluate(scored)
        ov = ev[ev.check == "Overall"].iloc[0]
        section(1, "Rule Accuracy", f"The rules catch <b>{ov.recall:.0%}</b> of planted problems, and <b>{ov.precision:.0%}</b> "
                "of what they flag was planted.")
        d = ev[ev.check != "Overall"].melt(id_vars="check", value_vars=["precision", "recall"])
        d["variable"] = d.variable.map({"precision": "Precision (Flags That Were Problems)", "recall": "Recall (Problems Caught)"})
        d["check"] = d.check.map(lambda c: {"Duplicate (exact)": SHORT["duplicate"], "Near-duplicate": SHORT["near_duplicate"],
                                            "Impossible day": SHORT["impossible_day"], "Upcoding": SHORT["upcoding"]}.get(c, c))
        fig = px.bar(d, x="check", y="value", color="variable", barmode="group", color_discrete_sequence=[NAVY, BLUE2],
                     labels={"check": "", "value": "", "variable": ""}, text_auto=".0%")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
        st.plotly_chart(style(fig, 340, "Rule Accuracy (Synthetic)"), width="stretch")

        section(2, "Choosing The Score Cut-Off", "A higher cut-off sends fewer, more certain claims to investigators.")
        truth = scored.planted_fraud.astype(bool)
        cut = []
        for c_ in (50, 80, 90):
            s_ = scored.risk_score >= c_
            cut.append({"Cut-off": f"{c_}+" + (" (current high)" if c_ == 80 else ""), "Claims sent to review": int(s_.sum()),
                        "Share that were problems": truth[s_].mean() if s_.any() else 0,
                        "Problems caught": (s_ & truth).sum() / max(truth.sum(), 1)})
        st.dataframe(pd.DataFrame(cut), hide_index=True, width="stretch", column_config={
            "Share that were problems": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Problems caught": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
        st.caption("Synthetic. Ask the assistant \"What happens if I raise the cut-off to 85?\" for other values.")
    else:
        section(1, "Rule Accuracy", "Needs ground-truth labels, which real data doesn't have. Use Review Outcomes instead.")

    m = ml_model.load_metrics()
    section(3, "ML Risk Model", "A second opinion: a machine-learning model scores every claim from 0 to 100.")
    if m is None:
        st.info("No model trained yet. Run `python src/ml_model.py`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, f"Ranking Quality (AUC) {syn}", f"{m['roc_auc']:.3f}", "1.0 = perfect, 0.5 = coin toss")
        kpi(c2, f"Precision At Score 80 {syn}", f"{m['precision_at_80']:.0%}", "flags that were problems")
        kpi(c3, f"Recall At Score 80 {syn}", f"{m['recall_at_80']:.0%}", "problems that get caught")
        n_sub = int((fb.decision == feedback.SUBST).sum())
        kpi(c4, "Substantiated Among Reviewed", f"{n_sub / max(len(reviewed), 1):.0%}",
            f"{n_sub} of {len(reviewed)} reviewed cases (selected subset)", teal=True)
        st.caption(f"Labels: {m['label_source']}. Trained on {m['train_claims']:,} claims before "
                   f"{ml_model.SPLIT_DATE}, tested on {m['test_claims']:,} later claims it had never seen.")
        imp = pd.DataFrame(m["importance"]).query("importance > 0.001").sort_values("importance")
        section(4, "What Drives The ML Score", f"<b>{imp.description.iloc[-1]}</b> is the strongest signal." if len(imp) else "")
        l, r = st.columns([1.3, 1])
        with l:
            fig = px.bar(imp, x="importance", y="description", orientation="h", color_discrete_sequence=[BLUE],
                         labels={"importance": "Importance", "description": ""})
            st.plotly_chart(style(fig, 380), width="stretch")
        with r:
            only = scored[(scored.ml_score >= ml_model.ML_FLAG) & ~scored.flagged]
            st.markdown(f"**Claims The ML Flags But The Rules Missed: {len(only)}**")
            st.caption("Worth a manual look. Repeated patterns here are candidates for new rules.")
            st.dataframe(only[["claim_id", "provider_name", "cpt_code", "billed_amount", "ml_score"]],
                         hide_index=True, width="stretch", height=280, column_config={
                             "claim_id": "Claim", "provider_name": "Provider", "cpt_code": "CPT",
                             "billed_amount": st.column_config.NumberColumn("Billed", format="$%.0f"),
                             "ml_score": "ML Score"})
    section("", "Before Citing Real-World Performance")
    st.markdown('<div class="card" style="border-left:4px solid #d97706"><ul style="margin:0;padding-left:18px;font-size:14.5px;'
                'line-height:1.7"><li><b>Point-in-time features:</b> some model features use group statistics (e.g. peer '
                'medians) computed across the whole loaded data (<code>src/ml_model.py · build_features()</code>). For a valid '
                'test they must be computed only from claims available <i>at scoring time</i>; a date split alone does not '
                'guarantee that.</li><li><b>Later data:</b> validate on a period after the training data, with labels from '
                'line-level reviewed findings only.</li><li><b>Approval:</b> independent validation and sign-off before any '
                'model is used to set review priority.</li></ul></div>', unsafe_allow_html=True)
    st.markdown('<div class="nextwrap"></div>', unsafe_allow_html=True)
    l, r = st.columns([3.2, 1])
    l.markdown('<div class="next"><b>You\'ve seen the whole workflow.</b> Go back to the guide any time.</div>',
               unsafe_allow_html=True)
    r.button("❓ How to Use →", key="nx-model", type="primary", width="stretch", on_click=go, args=(GUIDE,))

# ================= Executive Dashboard =================
if page == "Executive Dashboard":
    st.markdown('<div class="crumb">Leadership</div>', unsafe_allow_html=True)
    st.markdown("## Executive Dashboard")
    st.markdown(f'<div class="purpose">Three questions: are we finding actionable cases, can the team keep up, and what '
                f'financial value did we verify? Reporting cohort: cases flagged for services {ss.f_rng[0]:%d %b} – '
                f'{ss.f_rng[-1]:%d %b %Y}.</div>', unsafe_allow_html=True)
    if is_sample:
        st.markdown('<div class="warn">⚠ <b>Synthetic demo figures.</b> In the pilot, financial measures are reconciled to '
                    'the claims system before they are reported.</div>', unsafe_allow_html=True)
    cnt = reviewed.decision.value_counts()
    dup = feedback.rule_stats(fb)
    dup = dup[dup.rule == "duplicate"]
    dy = f"{dup.rate.iloc[0]:.0%}" if len(dup) else "–"
    rev_all = reviewed
    complete = ((rev_all.reason.fillna("") != "") & ((rev_all.evidence.fillna("") != "") | (rev_all.decision == feedback.INSUFF))).mean() \
        if len(rev_all) else 0
    st.markdown(f"""<div class="pilot"><h3>Shadow pilot · duplicate billing</h3><p>Investigators review ClaimGuard's
      duplicate-billing queue alongside the existing process; payments stay in the claims system. A random sample of 200
      unflagged claims is also reviewed to estimate what the rules miss. Agreed success criteria and where we are now:</p>
      <div class="crit"><div><b>≥ 40%</b>review yield (substantiated ÷ reviewed)<i>now {dy}</i></div>
      <div><b>≤ 30 min</b>median handling time<i>not tracked yet (pilot adds timers)</i></div>
      <div><b>100%</b>outcomes with reason + evidence<i>now {complete:.0%}</i></div>
      <div><b>0</b>legitimate claims delayed<i>0 · shadow mode, no payment holds</i></div></div></div>""",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        section("", "1 · Are We Finding Actionable Cases?")
        fun = pd.DataFrame({"stage": ["Flagged", "Assigned", "Reviewed", "Substantiated", "No issue found", "Insufficient", "Referred to SIU"],
                            "cases": [len(cases), int(cases.owner.notna().sum()), len(rev_all), cnt.get(feedback.SUBST, 0),
                                      cnt.get(feedback.NO_ISSUE, 0), cnt.get(feedback.INSUFF, 0), cnt.get(feedback.SIU, 0)]})
        fig = px.bar(fun, x="cases", y="stage", orientation="h", text="cases",
                     color="stage", color_discrete_sequence=[BLUE2, BLUE2, NAVY, RISK["High"], RISK["Low"], GREY, RISK["Medium"]])
        fig.update_yaxes(autorange="reversed", title_text="")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 300), width="stretch", config={"displayModeBar": False})
        st_ = feedback.rule_stats(fb)
        if len(st_):
            st.dataframe(st_.assign(flag_reason=st_.rule.map(SHORT))[["flag_reason", "reviewed", "substantiated", "rate"]],
                         hide_index=True, width="stretch", column_config={
                             "rate": st.column_config.ProgressColumn("Yield", min_value=0, max_value=1, format="percent")})
    with c2:
        section("", "2 · Can The Team Keep Up?")
        op = cases[cases.status.isin([feedback.OPEN, feedback.PENDING])]
        bk = pd.cut(op.age_days, [-1, 6, 30, 10_000], labels=["Under 7 days", "7–30 days", "Over 30 days"]).value_counts().reindex(
            ["Under 7 days", "7–30 days", "Over 30 days"]).fillna(0).reset_index()
        bk.columns = ["age", "cases"]
        fig = px.bar(bk, x="age", y="cases", text="cases", color="age",
                     color_discrete_sequence=[BLUE2, RISK["Medium"], RISK["High"]], labels={"age": "", "cases": "Open cases"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 230, "Open Backlog By Age"), width="stretch", config={"displayModeBar": False})
        ev = feedback.events(DS)
        fo = ev[ev.new_status.isin(feedback.FINAL) & ev.case_id.isin(cases.case_id)].groupby("case_id").ts.min()
        cr = cases.set_index("case_id").created_at
        ttr = (pd.to_datetime(fo) - cr.reindex(fo.index)).dt.total_seconds() / 86400
        st.markdown(f'<div class="mrow"><span>Median time to outcome</span><b>{ttr.median():.1f} days</b></div>'
                    if len(ttr.dropna()) else '<div class="mrow"><span>Median time to outcome</span><b>–</b></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="mrow"><span>Open cases past due date</span><b>{int((op.due < pd.Timestamp.now()).sum())}</b></div>'
                    f'<div class="mrow"><span>Unassigned open cases</span><b>{int(op.owner.isna().sum())}</b></div>'
                    '<div class="mrow"><span>Median handling time</span><b>not tracked yet</b></div>'
                    '<div class="mrow"><span>Legitimate claims delayed</span><b>0 (shadow mode)</b></div>', unsafe_allow_html=True)
        wl = op.assign(owner=op.owner.fillna("Unassigned")).groupby("owner").size().rename("open").reset_index()
        done7 = ev[(pd.to_datetime(ev.ts) > pd.Timestamp.now() - pd.Timedelta(days=7)) & ev.new_status.isin(feedback.FINAL)] \
            .groupby("reviewer").size().rename("outcomes_7d").reset_index().rename(columns={"reviewer": "owner"})
        st.dataframe(wl.merge(done7, on="owner", how="outer").fillna(0), hide_index=True, width="stretch")
    with c3:
        section("", "3 · What Value Did We Verify?")
        sub_amt = float(fb.loc[fb.decision == feedback.SUBST, "amount"].fillna(0).sum())
        st.markdown(f'<div class="mrow"><span>Billed on flagged claims</span><b>{money(flagged.billed_amount.sum())}</b></div>'
                    '<div class="mrow"><span>Allowed</span><b>not in dataset</b></div>'
                    '<div class="mrow"><span>Paid</span><b>not in dataset</b></div>'
                    f'<div class="mrow"><span>Substantiated (reviewer-entered)</span><b style="color:#dc2626">{money(sub_amt)}</b></div>'
                    '<div class="mrow"><span>Prevented payment</span><b>$0 · shadow pilot</b></div>'
                    '<div class="mrow"><span>Recovered (reconciled)</span><b>claims system not connected</b></div>',
                    unsafe_allow_html=True)
        overlap = sum(len(ids) for ids in cases.claim_ids) - len(flag_ids)
        st.markdown(f'<div class="hint" style="margin-top:8px">Billed is counted once per claim line: {overlap} line(s) '
                    'appear in more than one case. Allowed, paid, prevented and recovered amounts come from the claims '
                    'system in the pilot, not from ClaimGuard.</div>', unsafe_allow_html=True)

# ================= Data & Access =================
if page == "Data & Access":
    st.markdown('<div class="crumb">Analyst &amp; admin</div>', unsafe_allow_html=True)
    st.markdown("## Data & Access")
    st.markdown('<div class="purpose">The data boundary: what data comes in, who can do what, what leaves, how long it is '
                'kept, and which AI features may see it.</div>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        section("", "Data Scope")
        st.markdown(f"""<div class="card"><div class="mrow"><span>Organisation</span><b>{feedback.ORG}</b></div>
          <div class="mrow"><span>Current dataset</span><b>{DS}{" (synthetic)" if is_sample else ""}</b></div>
          <div class="mrow"><span>Pilot data</span><b>separate dataset ID; never mixed with synthetic</b></div>
          <div class="mrow"><span>Storage</span><b>SQLite (transactional) · per-org, per-dataset case IDs</b></div>
          <div class="mrow"><span>Retention (pilot)</span><b>pilot + 90 days, then deleted with certificate</b></div>
          <p class="hint" style="margin-top:8px"><b>Included:</b> pseudonymised member/provider IDs, dates, codes, units, billed
          (+ allowed/paid in pilot). <b>Excluded:</b> names, addresses, phone numbers, free-text clinical notes until approved.</p></div>""",
                    unsafe_allow_html=True)
        section("", "AI Data Flows", "Each external flow is reviewed before it is enabled.")
        has_claude = any(b.get("source") == "claude" for b in briefs.values())
        st.dataframe(pd.DataFrame([
            ("Case brief (rule-based)", "On", "None"),
            ("Case brief (Claude)", "On" if has_claude else "Off · needs data-flow review and API credit",
             "Case facts only, pseudonymised; statements must link to claim rows"),
            ("Ask your data (rule-based)", "On", "None (runs locally)"),
            ("Document reading for scans (Claude)", "Off · needs review", "Document images")],
            columns=["Feature", "Status", "Data sent outside"]), hide_index=True, width="stretch")
    with r:
        section("", "Who Can Do What", "Demo roles; the pilot takes roles from single sign-on.")
        Y, N = "✓", "–"
        st.dataframe(pd.DataFrame([
            ("View cases", Y, Y, Y), ("Record outcome", Y, Y, N), (f"Approve substantiation > ${feedback.APPROVAL_LIMIT:,}", N, Y, N),
            ("Export case rows (logged)", Y, Y, N), ("Export aggregates", N, Y, Y), ("Propose rule / model change", N, N, Y)],
            columns=["Action", "Investigator", "Supervisor", "Analyst"]), hide_index=True, width="stretch")
        st.caption("Demo users: " + " · ".join(f"{u} ({r_})" for u, r_ in USERS.items()) + ". Change user in the sidebar.")
        section("", "Audit Log", "Views, exports, outcomes and assignments, newest first.")
        st.dataframe(feedback.audit(DS, 100), hide_index=True, width="stretch", height=320)

# ================= Under the Hood =================
if page == "Under the Hood":
    st.markdown('<div class="crumb">Analyst &amp; admin</div>', unsafe_allow_html=True)
    st.markdown("## Under the Hood")
    st.markdown('<div class="purpose">How ClaimGuard works technically: the pipeline, the data contract, the rules and their '
                'scores, the model, the assistant and the pilot deployment plan.</div>', unsafe_allow_html=True)
    section("", "Pipeline", "Every stage is deterministic and logged; AI steps are optional and off by default.")
    st.markdown("""<div class="card"><svg class="arch" viewBox="0 0 1000 250" role="img" style="font-family:Inter,Segoe UI,sans-serif">
    <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10z" fill="#64748b"/></marker></defs>
    <g font-size="12" text-anchor="middle">
     <rect x="10" y="30" width="140" height="74" rx="10" fill="#eef4fc" stroke="#3b7cc4"/><text x="80" y="58" font-weight="700" fill="#0f2d52">1 · Ingest</text><text x="80" y="76" fill="#334155">PDF · email · CSV · XLSX</text><text x="80" y="92" fill="#334155">837 / bordereaux (next)</text>
     <rect x="180" y="30" width="140" height="74" rx="10" fill="#eef4fc" stroke="#3b7cc4"/><text x="250" y="58" font-weight="700" fill="#0f2d52">2 · Normalise</text><text x="250" y="76" fill="#334155">12-column schema</text><text x="250" y="92" fill="#334155">reasons for every inference</text>
     <rect x="350" y="30" width="140" height="74" rx="10" fill="#eef4fc" stroke="#3b7cc4"/><text x="420" y="58" font-weight="700" fill="#0f2d52">3 · Screen</text><text x="420" y="76" fill="#334155">4 explainable rules</text><text x="420" y="92" fill="#334155">base score + reason</text>
     <rect x="520" y="30" width="140" height="74" rx="10" fill="#eef4fc" stroke="#3b7cc4"/><text x="590" y="58" font-weight="700" fill="#0f2d52">4 · Build cases</text><text x="590" y="76" fill="#334155">group related lines</text><text x="590" y="92" fill="#334155">review priority + route</text>
     <rect x="690" y="30" width="140" height="74" rx="10" fill="#0f2d52"/><text x="760" y="58" font-weight="700" fill="#fff">5 · Review</text><text x="760" y="76" fill="#cfe2f7">queue · checklist</text><text x="760" y="92" fill="#cfe2f7">structured outcome</text>
     <rect x="860" y="30" width="130" height="74" rx="10" fill="#eef4fc" stroke="#3b7cc4"/><text x="925" y="58" font-weight="700" fill="#0f2d52">6 · Report</text><text x="925" y="76" fill="#334155">dashboard · exports</text><text x="925" y="92" fill="#334155">audit log</text>
     <rect x="520" y="160" width="140" height="70" rx="10" fill="#f0fdfa" stroke="#0f766e"/><text x="590" y="186" font-weight="700" fill="#0f766e">✦ AI (optional)</text><text x="590" y="203" fill="#334155">case brief · Ask your data</text><text x="590" y="219" fill="#334155">rule-based fallback</text>
     <rect x="690" y="160" width="140" height="70" rx="10" fill="#fff7e6" stroke="#d97706"/><text x="760" y="186" font-weight="700" fill="#8a5a00">Line-level labels</text><text x="760" y="203" fill="#334155">substantiated / not only</text><text x="760" y="219" fill="#334155">unlabeled otherwise</text>
     <rect x="860" y="160" width="130" height="70" rx="10" fill="#fff7e6" stroke="#d97706"/><text x="925" y="186" font-weight="700" fill="#8a5a00">Validate + approve</text><text x="925" y="203" fill="#334155">before any rule or</text><text x="925" y="219" fill="#334155">model change</text>
    </g>
    <g stroke="#64748b" stroke-width="1.6" fill="none" marker-end="url(#ar)"><path d="M150,67 H178"/><path d="M320,67 H348"/><path d="M490,67 H518"/><path d="M660,67 H688"/><path d="M830,67 H858"/><path d="M590,104 V158"/><path d="M760,104 V158"/><path d="M830,195 H858"/><path d="M925,160 C925,135 420,150 420,106" stroke-dasharray="5 4"/></g>
    </svg></div>""", unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        section("", f"Data Contract ({len(REQUIRED)} Required Columns)")
        st.dataframe(pd.DataFrame([
            ("claim_id", "string", "unique per line; INT- generated if missing"), ("member_id", "string", "pseudonymised in pilot"),
            ("provider_id", "string", "rendering provider; billing ID added in pilot"), ("provider_name · specialty", "string", "from provider master"),
            ("service_date · submitted_date", "date", "ISO 8601"), ("cpt_code · cpt_description", "string", "checked against CPT reference"),
            ("icd10_code", "string", "format-checked"), ("units", "int > 0", ""), ("billed_amount", "decimal > 0", "allowed / paid added in pilot")],
            columns=["Column", "Type", "Notes"]), hide_index=True, width="stretch")
    with r:
        section("", f"Rules And Scores · Ruleset {RULESET}")
        st.dataframe(pd.DataFrame([(FLAG[k], v["thr"], v["formula"], v["src"]) for k, v in RULE_INFO.items()],
                                  columns=["Rule", "Condition", "Base score", "Source"]), hide_index=True, width="stretch")
        st.caption("Review priority = base score + adjustment (currently not applied). Routing: ≥ 80 priority review · "
                   "50–79 routine review · < 50 no additional review recommended.")
    l, r = st.columns(2)
    with l:
        section("", "ML Second Opinion", "Shown beside the rules; never overrides them.")
        m = ml_model.load_metrics() or {}
        st.dataframe(pd.DataFrame([
            ("Algorithm", "HistGradientBoostingClassifier (scikit-learn)"), ("Features", f"{len(ml_model.FEATURES)} claim, provider-day and peer features"),
            ("Split", f"train before {ml_model.SPLIT_DATE}, test after"), ("Labels", m.get("label_source", "-")),
            ("Known gap", "peer statistics are computed over the whole frame; must be rebuilt point-in-time before real validation")],
            columns=["Item", "Detail"]), hide_index=True, width="stretch")
    with r:
        section("", "✦ Ask Your Data", "Question → intent → read-only query → answer with its working.")
        st.code("question: \"Has Dr. Osei billed long days before?\"\nintent:   provider_long_days(provider = PRV-…)\n"
                "query:    SELECT service_date, SUM(units*minutes)/60 AS hours\n          FROM claims JOIN cpt_reference USING (cpt_code)\n"
                "          WHERE provider_id = :pid GROUP BY 1 HAVING hours > 12\nguards:   read-only · member IDs only · no guilt judgements",
                language="text")
        st.caption("Rule-based today (src/assistant.py). A Claude version would translate questions into the same guarded queries.")
    section("", "Stack And Pilot Deployment")
    st.markdown("""<div class="flow" style="grid-template-columns:repeat(4,1fr)">
      <div class="fstage"><h4>Today (demo)</h4><p>Python 3.12 · pandas · Streamlit · Plotly · scikit-learn · SQLite review store</p></div>
      <div class="fstage"><h4>Pilot target</h4><p>FastAPI service · PostgreSQL · SSO roles · audit tables · per-org case IDs</p></div>
      <div class="fstage"><h4>Hosting</h4><p>Cloud region approved for the data (e.g. Hong Kong) · encrypted at rest and in transit</p></div>
      <div class="fstage"><h4>AI (optional)</h4><p>Claude via an approved route · JSON-schema output · prompt caching · Batch API · rule-based fallback</p></div>
    </div>""", unsafe_allow_html=True)
