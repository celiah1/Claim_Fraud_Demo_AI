"""ClaimGuard: health claim fraud detection app.  Run:  streamlit run app.py

Pages follow a story:  Fraud Overview (what is happening) -> Provider Risk (who is driving it)
-> Investigator Queue (act on it) -> Feedback Loop (learn from decisions) -> Model Evaluation (can we trust it).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import detect  # noqa: E402
import feedback  # noqa: E402
import intake  # noqa: E402
import ml_model  # noqa: E402
import summarize  # noqa: E402

NAVY, BLUE, BLUE2, TEAL, GREY = "#0f2d52", "#2b5f94", "#3b7cc4", "#0f766e", "#94a3b8"
RISK = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}
TYPE_LABEL = {"duplicate": "Duplicate Billing", "near_duplicate": "Near-Duplicate",
              "impossible_day": "Impossible Day", "upcoding": "Upcoding"}
TYPE_COLOR = {"Duplicate Billing": NAVY, "Near-Duplicate": BLUE2, "Impossible Day": BLUE, "Upcoding": "#93b4d9"}
PAGES = {"Claim Intake": "🧾", "Data Upload": "📥", "Fraud Overview": "📊", "Provider Risk": "🩺", "Investigator Queue": "🔍",
         "Feedback Loop": "🔁", "Model Evaluation": "✅"}
REQUIRED = ["claim_id", "member_id", "provider_id", "provider_name", "specialty", "service_date", "submitted_date",
            "cpt_code", "cpt_description", "icd10_code", "units", "billed_amount"]

st.set_page_config(page_title="ClaimGuard", page_icon="🛡", layout="wide")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stApp"],[data-testid="stApp"] :is(p,div,span,label,button,input,h1,h2,h3,h4,li,td,th):not([data-testid="stIconMaterial"]):not([class*="material"]){{font-family:Inter,"Segoe UI",sans-serif}}
[data-testid="stApp"]{{background:#f6f8fb}}
[data-testid="stHeader"]{{background:transparent}}
[data-testid="stSidebar"]{{background:#fff;border-right:1px solid #e6eaf0}}
[data-testid="stSidebar"] :is(h1,h2,h3,p,label,span,small) {{color:#334155}}
[data-testid="stSidebar"] .stButton button{{width:100%;border-radius:9px;padding:9px 14px;min-height:0;justify-content:flex-start}}
[data-testid="stSidebar"] .stButton button > div{{justify-content:flex-start}}
[data-testid="stSidebar"] .stButton button p{{font-size:15px !important}}
[data-testid="stSidebar"] .stButton button[kind="secondary"]{{background:transparent;border:0;box-shadow:none}}
[data-testid="stSidebar"] .stButton button[kind="secondary"] p{{color:#475569 !important;font-weight:500}}
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover{{background:#f1f5fa}}
[data-testid="stSidebar"] .stButton button[kind="primary"]{{background:#eef4fc;border:0;box-shadow:inset 4px 0 0 {BLUE2}}}
[data-testid="stSidebar"] .stButton button[kind="primary"] p{{color:{NAVY} !important;font-weight:700}}
[data-testid="stSidebar"] .stButton button[kind="tertiary"]{{background:{TEAL};justify-content:center}}
[data-testid="stSidebar"] .stButton button[kind="tertiary"] p{{color:#fff !important;font-weight:600}}
.brand{{display:flex;align-items:center;gap:14px;margin:4px 0 18px}}
.brand .bname{{font-size:28px;font-weight:800;color:{NAVY};letter-spacing:-.3px;line-height:1.1}}
.brand .btag{{font-size:13px;color:#64748b;margin-top:3px}}
[data-testid="stSidebar"] [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]{{background:#f8fafc;border:1px solid #e6eaf0 !important;border-radius:14px;padding:12px 12px 14px}}
.navlabel{{font-size:11px;font-weight:700;letter-spacing:1.2px;color:#94a3b8;margin:18px 0 6px}}
.src{{background:#eef4fc;border-radius:9px;padding:10px 12px;font-size:13px;color:#334155;margin-top:6px}}
.src b{{color:{NAVY}}}
.check{{background:#fff;border:1px solid #e6eaf0;border-radius:14px;padding:14px 18px;font-size:15px;color:#1e293b;box-shadow:0 1px 2px rgba(15,45,82,.04),0 4px 14px rgba(15,45,82,.05)}}
.ok{{color:#16a34a;font-weight:700}}.bad{{color:#dc2626;font-weight:700}}
[data-testid="stMain"] h2{{font-size:2.5rem !important;font-weight:800;margin-bottom:0;color:{NAVY};letter-spacing:-.02em}}
.purpose{{font-size:16px;color:#64748b;margin:2px 0 24px;padding:0;background:none;border:0}}
.purpose b{{color:{BLUE};font-weight:600}}
.sec{{display:flex;align-items:center;gap:12px;margin:32px 0 4px}}
.sec .n{{background:#eef4fc;color:{BLUE};font-weight:700;width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}}
.sec .t{{font-size:21px;font-weight:700;color:{NAVY}}}
.take{{font-size:15px;color:#64748b;margin:0 0 12px 42px}}
.take b{{color:#1e293b}}
.kpi,.route,.steps{{background:#fff;border:1px solid #e6eaf0;border-radius:14px;box-shadow:0 1px 2px rgba(15,45,82,.04),0 4px 14px rgba(15,45,82,.05)}}
.kpi{{padding:16px 18px;height:100%}}
.kpi.t .l{{color:{TEAL}}}
.kpi .l{{font-size:14px;font-weight:600;color:#64748b}}.kpi .v{{font-size:32px;font-weight:800;color:{NAVY};margin-top:2px}}.kpi .d{{font-size:13px;color:#64748b}}
.route{{padding:16px;height:100%}}
.route .h{{font-size:15px;font-weight:700;margin-bottom:6px}}.route .v{{font-size:30px;font-weight:800;color:{NAVY}}}
.route .d{{font-size:14px;color:#64748b}}
.steps{{padding:14px 18px;margin-bottom:14px}}
.steps ol{{margin:6px 0 0 0;padding-left:20px;font-size:15px;color:#1e293b;line-height:1.7}}
.next{{background:#eef4fc;border:1px solid #cfdff2;border-radius:14px;padding:14px 18px;font-size:16px;color:#1e293b;margin-top:18px}}
[data-testid="stMain"] [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"][class*="border"],[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{{border-radius:14px}}
[data-testid="stPlotlyChart"]{{background:#fff;border:1px solid #e6eaf0;padding:6px;box-shadow:0 1px 2px rgba(15,45,82,.04),0 4px 14px rgba(15,45,82,.05)}}
[data-testid="stMain"] .stButton button[kind="primary"],[data-testid="stMain"] .stDownloadButton button{{border-radius:9px;font-weight:600}}
.brief{{background:#f0fdfa;border:1px solid #99f6e4;border-radius:14px;padding:16px 18px;color:#1e293b}}
.brief h4{{color:{TEAL};font-size:12px;text-transform:uppercase;letter-spacing:.5px;margin:12px 0 4px}}
.ai{{background:{TEAL};color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:10px}}
.rb{{background:#64748b;color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:10px}}
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


def page_header(title, purpose):
    st.markdown(f"## {title}")
    st.markdown(f'<div class="purpose"><b>Purpose:</b> {purpose}</div>', unsafe_allow_html=True)


def section(n, title, takeaway=""):
    st.markdown(f'<div class="sec"><div class="n">{n}</div><div class="t">{title}</div></div>'
                + (f'<div class="take">{takeaway}</div>' if takeaway else ""), unsafe_allow_html=True)


def kpi(col, label, value, detail, teal=False):
    col.markdown(f'<div class="kpi{" t" if teal else ""}"><div class="l">{label}</div>'
                 f'<div class="v">{value}</div><div class="d">{detail}</div></div>', unsafe_allow_html=True)


def style(fig, height=360, title=None):
    fig.update_layout(height=height, plot_bgcolor="#fff", paper_bgcolor="#fff", margin=dict(t=50 if title else 20, b=10, l=10, r=10),
                      title=dict(text=title, font=dict(size=18, color="#0f172a")) if title else None,
                      font=dict(size=14, color="#0f172a"), legend=dict(font=dict(size=13), title_text=""))
    fig.update_xaxes(tickfont=dict(size=13, color="#000"), title_font=dict(size=14, color="#000"), gridcolor="#eef2f7")
    fig.update_yaxes(tickfont=dict(size=13, color="#000"), title_font=dict(size=14, color="#000"), gridcolor="#eef2f7")
    return fig


def money(x):
    if x >= 1_000_000:
        return f"${x / 1_000_000:,.2f}M"
    return f"${x / 1000:,.0f}K" if x >= 1000 else f"${x:,.0f}"


# ---------------- Sidebar ----------------
st.session_state.setdefault("page", "Claim Intake")
st.session_state.setdefault("data_bytes", None)
st.session_state.setdefault("data_name", "Sample dataset (synthetic, 8,498 claims)")


def go(p):
    st.session_state.page = p


LOGO = f"""<div class="brand"><svg width="46" height="52" viewBox="0 0 46 52" xmlns="http://www.w3.org/2000/svg">
<path d="M23 2 L42 9 V24 C42 37 33 46 23 50 C13 46 4 37 4 24 V9 Z" fill="{BLUE2}"/>
<path d="M14 26 L20.5 32.5 L33 19" fill="none" stroke="#fff" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg><div><div class="bname">ClaimGuard</div><div class="btag">Health claim fraud detection</div></div></div>"""

with st.sidebar:
    st.markdown(LOGO, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="navlabel" style="margin-top:0">PAGES</div>', unsafe_allow_html=True)
        for p in PAGES:
            label = f"{p}  ✦ AI" if p == "Investigator Queue" else f"{p}  · NEW" if p == "Claim Intake" else p
            st.button(label, key=f"nav-{p}", on_click=go, args=(p,), width="stretch",
                      type="primary" if st.session_state.page == p else "secondary")
    page = st.session_state.page

    scored, pdays, upcoders, cases = load(st.session_state.data_bytes)
    dmin, dmax = scored.service_date.min().date(), scored.service_date.max().date()
    with st.container(border=True):
        st.markdown('<div class="navlabel" style="margin-top:0">DATASET</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:14px;color:#0f2d52;font-weight:600">{st.session_state.data_name}</div>',
                    unsafe_allow_html=True)
        st.button("Change dataset", key="to-upload", on_click=go, args=("Data Upload",), width="stretch",
                  type="secondary")
    run_ai, levels = False, ["High"]
    if page == "Investigator Queue":
        with st.container(border=True):
            st.markdown('<div class="navlabel" style="margin-top:0">AI CASE BRIEFS</div>', unsafe_allow_html=True)
            levels = st.multiselect("Write briefs for", ["High", "Medium"], ["High"])
            run_ai = st.button("✦ Generate AI briefs", type="primary", width="stretch")
    with st.container(border=True):
        st.markdown('<div class="navlabel" style="margin-top:0">FILTERS</div>', unsafe_allow_html=True)
        rng = st.date_input("Service date range", (dmin, dmax), min_value=dmin, max_value=dmax)
        use_dup = st.checkbox("Duplicate billing", True)
        use_imp = st.checkbox("Impossible days", True)
        use_upc = st.checkbox("Upcoding (peer comparison)", True)

# Filters
if len(rng) == 2:
    s, e = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    scored = scored[scored.service_date.between(s, e)]
    cases = cases[cases.service_date.between(s, e)]
allowed = ({"duplicate", "near_duplicate"} if use_dup else set()) | ({"impossible_day"} if use_imp else set()) \
    | ({"upcoding"} if use_upc else set())
cases = cases[cases.case_type.isin(allowed)]
fb = feedback.load()
all_cases = cases
cases = feedback.apply(cases, fb)
cases["fraud_type"] = cases.case_type.map(TYPE_LABEL)
flag_ids = set(c for ids in cases.claim_ids for c in ids)
flagged = scored[scored.claim_id.isin(flag_ids)].drop(columns="fraud_type", errors="ignore").merge(cases[["case_id", "fraud_type"]], on="case_id", how="left")

briefs = summarize.load_cache()
open_cases = cases[~cases.status.isin([feedback.CONFIRMED, feedback.NOT_FRAUD])]

# ================= 00. Claim Intake =================
if page == "Claim Intake":
    page_header("Claim Intake",
                "Turn messy claim documents (receipts, superbills, emails, odd spreadsheets) into clean claim rows, "
                "review them, and send them to the fraud checks.")
    st.markdown('<div class="steps"><b>How it works</b><ol><li><b>Upload</b> PDFs, text/email files or spreadsheets '
                '(or load the messy samples).</li><li><b>Extract</b>: rules pull out member, provider, date, procedure '
                'code, units and amount, and explain every value they fill in.</li><li><b>Review</b>: fix any row '
                'marked <span class="bad">Fix</span> and read the <b>Check</b> warnings.</li><li><b>Export</b> a clean CSV, '
                'or add the rows to the dataset and run detection.</li></ol></div>', unsafe_allow_html=True)

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
        st.session_state.intake_files = [(p.name, p.read_bytes()) for p in sorted((ROOT / "data" / "messy").iterdir())
                                         if p.suffix.lower() in (".pdf", ".txt", ".eml", ".md", ".csv", ".xlsx")]
        st.session_state.pop("intake_df", None)
    elif ups:
        files = [(u.name, u.getvalue()) for u in ups]
        if files != st.session_state.get("intake_files"):
            st.session_state.intake_files = files
            st.session_state.pop("intake_df", None)

    files = st.session_state.get("intake_files")
    if not files:
        st.markdown('<div class="next">⬆ Upload documents or click <b>Load messy samples</b> to start.</div>',
                    unsafe_allow_html=True)
    else:
        if "intake_df" not in st.session_state:
            with st.spinner("Extracting claim lines..."):
                st.session_state.intake_df, st.session_state.intake_src = intake.extract(files)
        idf = st.session_state.intake_df
        st.caption("Documents: " + " · ".join(n for n, _ in files))

        n_fix, n_chk, n_ok = (idf.status == intake.FIX).sum(), (idf.status == intake.CHECK).sum(), (idf.status == intake.READY).sum()
        section(2, "What Was Extracted", f"<b>{len(idf)}</b> claim lines from <b>{len(files)}</b> documents. "
                f"The <b>How filled</b> column in the table below lists every value the rules inferred, and why.")
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Claim Lines", f"{len(idf)}", f"from {len(files)} documents")
        kpi(c2, "Ready", f"{n_ok}", "nothing to check")
        kpi(c3, "Check", f"{n_chk}", "usable, but read the warning")
        kpi(c4, "Fix", f"{n_fix}", "a required value is missing")

        section(3, "Review & Fix", "Edit any cell to correct it, then press <b>Re-check</b>. Rows marked "
                "<span class='bad'>Fix</span> are left out of the export until they are complete.")
        show = ["status", "claim_id", "member_id", "provider_id", "provider_name", "specialty", "service_date",
                "submitted_date", "cpt_code", "cpt_description", "icd10_code", "units", "billed_amount", "issues",
                "inferred", "source"]
        badge = {intake.READY: "🟢 Ready", intake.CHECK: "🟠 Check", intake.FIX: "🔴 Fix"}
        edited = st.data_editor(
            idf[show].assign(status=idf.status.map(badge)), hide_index=True, width="stretch", height=420, key="intake_editor",
            disabled=["status", "issues", "inferred", "source"],
            column_config={
                "status": st.column_config.TextColumn("Status", width="small"),
                "service_date": st.column_config.DateColumn("service_date"),
                "submitted_date": st.column_config.DateColumn("submitted_date"),
                "units": st.column_config.NumberColumn("units", min_value=0, step=1),
                "billed_amount": st.column_config.NumberColumn("billed_amount", format="$%.2f", min_value=0),
                "issues": st.column_config.TextColumn("Issues", width="large"),
                "inferred": st.column_config.TextColumn("How filled", width="large"),
            })
        b1, b2 = st.columns([1, 4])
        if b1.button("↻ Re-check edits", width="stretch"):
            merged = idf.copy()
            for c in show:
                if c not in ("status", "issues", "inferred", "source"):
                    merged[c] = edited[c].values
            st.session_state.intake_df = intake.validate(merged)
            st.rerun()

        with st.expander("📄 Source text of each document (what the rules read)"):
            doc = st.selectbox("Document", list(st.session_state.intake_src))
            st.code(st.session_state.intake_src[doc], language=None)

        section(4, "Export", "Only rows that are Ready or Check are exported.")
        out = intake.to_claims(st.session_state.intake_df)
        e1, e2, e3 = st.columns([1.2, 1.6, 2])
        e1.download_button(f"⬇ Download clean CSV ({len(out)} rows)", out.to_csv(index=False).encode(),
                           "intake_claims.csv", "text/csv", width="stretch", disabled=out.empty)
        if e2.button(f"▶ Add {len(out)} rows to the dataset & analyse", type="primary", width="stretch",
                     disabled=out.empty):
            base = pd.read_csv(pd.io.common.BytesIO(st.session_state.data_bytes), dtype={"cpt_code": str}) \
                if st.session_state.data_bytes else pd.read_csv(ROOT / "data" / "claims.csv", dtype={"cpt_code": str})
            base = base[~base.claim_id.astype(str).str.startswith("INT-")]   # replace any earlier intake rows
            new = out.assign(planted_fraud=False, fraud_type="") if "planted_fraud" in base else out
            combined = pd.concat([base, new], ignore_index=True)
            st.session_state.data_bytes = combined.to_csv(index=False).encode()
            st.session_state.data_name = f"Sample + {len(out)} intake claims ({len(combined):,} claims)"
            st.session_state.page = "Data Upload"
            st.rerun()
        e3.markdown('<div style="font-size:14px;color:#64748b;padding-top:4px">Adding to the dataset lets the '
                    'duplicate, impossible-day and upcoding checks compare these claims with each provider\'s '
                    'history.</div>', unsafe_allow_html=True)

# ================= 0. Data Upload =================
if page == "Data Upload":
    page_header("Data Upload",
                "Choose the claims data to analyse, check it has everything the detection needs, and preview it "
                "before running the analysis.")
    section(1, "Choose Your Data", "Upload your own claims CSV, or keep using the built-in sample dataset.")
    l, r = st.columns([1.4, 1])
    with l:
        up = st.file_uploader("Upload a claims CSV", type="csv", key="uploader")
    with r:
        st.markdown(f'<div class="check">Currently using:<br><b>{st.session_state.data_name}</b></div>',
                    unsafe_allow_html=True)
        st.write("")
        if st.session_state.data_bytes is not None and st.button("↺ Switch back to sample data"):
            st.session_state.data_bytes = None
            st.session_state.data_name = "Sample dataset (synthetic, 8,498 claims)"
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
        df = pd.read_csv(pd.io.common.BytesIO(st.session_state.data_bytes), dtype={"cpt_code": str}) \
            if st.session_state.data_bytes else pd.read_csv(ROOT / "data" / "claims.csv", dtype={"cpt_code": str})
        src_name = st.session_state.data_name

    if df is not None:
        missing = [c for c in REQUIRED if c not in df.columns]
        dates = pd.to_datetime(df.get("service_date"), errors="coerce") if "service_date" in df else pd.Series(dtype="datetime64[ns]")
        bad_dates = int(dates.isna().sum()) if len(dates) else 0
        dup_ids = int(df.claim_id.duplicated().sum()) if "claim_id" in df else 0
        section(2, "Check The Data",
                f"<b>{src_name}</b>: {len(df):,} rows, {len(df.columns)} columns. "
                + ("<span class='ok'>All required columns are present.</span>" if not missing
                   else f"<span class='bad'>{len(missing)} required column(s) missing.</span>"))
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Claims", f"{len(df):,}", f"{df.provider_id.nunique() if 'provider_id' in df else '-'} providers")
        kpi(c2, "Members", f"{df.member_id.nunique():,}" if "member_id" in df else "-", "unique patients")
        kpi(c3, "Period Covered", f"{(dates.max() - dates.min()).days + 1} days" if dates.notna().any() else "-",
            f"{dates.min():%d %b} – {dates.max():%d %b %Y} · {bad_dates} unreadable" if dates.notna().any() else "")
        kpi(c4, "Total Billed", money(df.billed_amount.sum()) if "billed_amount" in df else "-",
            f"{dup_ids} duplicate claim IDs")
        st.write("")
        checks = "".join(f"<span class='{'ok' if c in df.columns else 'bad'}'>{'✓' if c in df.columns else '✗'}</span> "
                         f"{c} &nbsp;&nbsp; " for c in REQUIRED)
        st.markdown(f'<div class="check"><b>Required columns</b><br>{checks}</div>', unsafe_allow_html=True)

        section(3, "Preview", "The first 100 rows, exactly as they will be analysed.")
        st.dataframe(df.head(100), hide_index=True, width="stretch", height=320)

        section(4, "Run The Analysis")
        if up is not None:
            if missing:
                st.error("Fix the missing columns and upload again. Download the template above to see the expected format.")
            elif st.button(f"▶ Use {up.name} and run the analysis", type="primary"):
                st.session_state.data_bytes = raw
                st.session_state.data_name = f"{up.name} ({len(df):,} claims)"
                st.session_state.page = "Fraud Overview"
                st.rerun()
        else:
            st.markdown('<div class="next">➡ This dataset is already loaded. Open <b>Fraud Overview</b> in the sidebar to '
                        'see the results.</div>', unsafe_allow_html=True)
            st.button("📊 Go to Fraud Overview", type="primary", on_click=go, args=("Fraud Overview",))

# ================= 1. Fraud Overview =================
if page == "Fraud Overview":
    page_header("Fraud Overview",
                "See how much suspicious billing was found this period, where it comes from, and what needs action.")

    at_risk = flagged.billed_amount.sum()
    section(1, "The Big Picture",
            f"Out of <b>{len(scored):,} claims</b> scanned ({dmin:%d %b} to {dmax:%d %b %Y}), "
            f"<b>{len(flagged):,}</b> look suspicious, worth <b>{money(at_risk)}</b>.")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Claims Flagged", f"{len(flagged):,}", f"{len(flagged) / max(len(scored), 1):.1%} of all claims")
    kpi(c2, "Dollars At Risk", money(at_risk), f"across {len(cases)} cases")
    hi = open_cases[open_cases.risk_level == "High"]
    kpi(c3, "High Risk Cases", f"{len(hi)}", f"{hi.provider_id.nunique()} providers involved")
    n_ai = sum(1 for b in briefs.values() if b["source"] == "claude")
    kpi(c4, "Case Briefs Ready", f"{len(briefs)}", f"{n_ai} written by Claude", teal=True)

    by_type = flagged.groupby("fraud_type").billed_amount.sum().sort_values(ascending=False)
    top_type = by_type.index[0] if len(by_type) else "-"
    section(2, "Where The Risk Comes From",
            f"<b>{top_type}</b> accounts for the largest share of dollars at risk "
            f"(<b>{by_type.iloc[0] / max(at_risk, 1):.0%}</b>)." if len(by_type) else "")
    l, r = st.columns([1, 1.4])
    with l:
        d = by_type.reset_index()
        fig = px.pie(d, names="fraud_type", values="billed_amount", hole=0.55, color="fraud_type",
                     color_discrete_map=TYPE_COLOR)
        fig.update_traces(textinfo="percent", textfont_size=15, hovertemplate="%{label}: $%{value:,.0f}")
        st.plotly_chart(style(fig, 360, "Dollars At Risk By Fraud Type"), width="stretch")
    with r:
        wk = flagged.assign(week=flagged.service_date.dt.to_period("W").dt.start_time)
        wk = wk.groupby(["week", "fraud_type"]).billed_amount.sum().reset_index()
        fig = px.bar(wk, x="week", y="billed_amount", color="fraud_type", color_discrete_map=TYPE_COLOR,
                     labels={"week": "Week Of Service", "billed_amount": "Dollars At Risk"})
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(style(fig, 360, "Dollars At Risk By Week"), width="stretch")

    prov = flagged.groupby("provider_name").billed_amount.sum().sort_values(ascending=False)
    top5 = prov.head(5).sum() / max(at_risk, 1)
    section(3, "Who Is Driving It",
            f"Risk is concentrated: the top 5 providers account for <b>{top5:.0%}</b> of dollars at risk. "
            "See the Provider Risk page for detail.")
    top = (flagged[flagged.provider_name.isin(prov.head(10).index)]
           .groupby(["provider_name", "fraud_type"]).billed_amount.sum().reset_index())
    fig = px.bar(top, x="billed_amount", y="provider_name", color="fraud_type", orientation="h",
                 color_discrete_map=TYPE_COLOR, category_orders={"provider_name": list(prov.head(10).index)},
                 labels={"billed_amount": "Dollars At Risk", "provider_name": ""})
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(style(fig, 420, "Top 10 Providers By Dollars At Risk"), width="stretch")

    section(4, "What Happens Next", "Every open case is routed by risk level.")
    c1, c2, c3 = st.columns(3)
    for col, lvl, action, desc in [(c1, "High", "Refer To SIU", "Strong evidence; investigate now"),
                                   (c2, "Medium", "Manual Review", "Check with the provider or records"),
                                   (c3, "Low", "Auto-Approve", "Pay as normal")]:
        g = open_cases[open_cases.risk_level == lvl]
        col.markdown(f'<div class="route" style="border-top:4px solid {RISK[lvl]}"><div class="h" style="color:{RISK[lvl]}">'
                     f'{lvl} Risk → {action}</div><div class="v">{len(g)} cases</div>'
                     f'<div class="d">{money(g.billed_amount.sum())} billed · {desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="next">➡ <b>Next step:</b> open the <b>Investigator Queue</b> page in the sidebar to '
                'review cases one by one.</div>', unsafe_allow_html=True)

# ================= 2. Provider Risk =================
if page == "Provider Risk":
    page_header("Provider Risk",
                "Find which providers are behind the suspicious billing. Pick a provider to see their evidence "
                "for each pattern compared with their peers.")
    p = (flagged.groupby(["provider_id", "provider_name", "specialty"])
         .agg(flagged_claims=("claim_id", "count"), dollars_at_risk=("billed_amount", "sum"),
              fraud_types=("fraud_type", lambda t: ", ".join(sorted(set(t)))), max_score=("risk_score", "max"))
         .reset_index().sort_values("dollars_at_risk", ascending=False))
    em = scored[scored.cpt_code.isin(detect.EM_CODES)]
    mix = (em.groupby(["provider_id", "provider_name", "specialty"])
             .agg(visits=("claim_id", "count"), top=("cpt_code", lambda c: (c == "99215").mean()))
             .reset_index().query("visits >= @detect.UPCODE_MIN_VISITS"))
    mix["flag"] = mix.provider_id.isin(upcoders.provider_id).map({True: "Flagged", False: "Within Peer Range"})
    over = pdays[pdays.billed_hours > detect.IMPOSSIBLE_HOURS_MED]
    total_risk = max(p.dollars_at_risk.sum(), 1)

    # ---- 1. Overall picture ----
    section(1, "The Overall Picture",
            f"<b>{len(p)} providers</b> have at least one flagged claim, but the money is concentrated: the top 5 "
            f"hold <b>{p.dollars_at_risk.head(5).sum() / total_risk:.0%}</b> of dollars at risk.")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Providers Flagged", f"{len(p)}", f"of {scored.provider_id.nunique()} providers billing")
    kpi(c2, "With Impossible Days", f"{over.provider_id.nunique()}", f"{len(over)} days over 12 hours")
    kpi(c3, "Upcoding Outliers", f"{len(upcoders)}", "far above peers on 99215")
    kpi(c4, "Top 5 Share Of Risk", f"{p.dollars_at_risk.head(5).sum() / total_risk:.0%}",
        money(p.dollars_at_risk.head(5).sum()), teal=True)
    st.write("")
    spec = flagged.groupby(["specialty", "fraud_type"]).billed_amount.sum().reset_index()
    fig = px.bar(spec, x="billed_amount", y="specialty", color="fraud_type", orientation="h",
                 color_discrete_map=TYPE_COLOR, labels={"billed_amount": "Dollars At Risk", "specialty": ""})
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style(fig, 300, "Dollars At Risk By Specialty And Fraud Type"), width="stretch")

    # ---- 2. Ranking (selectable) ----
    section(2, "Provider Ranking",
            "Tick the box at the start of a row to see that provider's full profile in step 3. "
            "The top provider is shown until you pick one.")
    pr = p.reset_index(drop=True)
    sel = st.dataframe(pr, hide_index=True, width="stretch", height=300,
                       on_select="rerun", selection_mode="single-row", key="prov_table", column_config={
        "provider_id": "ID", "provider_name": "Provider", "specialty": "Specialty", "flagged_claims": "Flagged Claims",
        "dollars_at_risk": st.column_config.NumberColumn("Dollars At Risk", format="$%.0f"),
        "fraud_types": "Fraud Types", "max_score": st.column_config.ProgressColumn("Highest Risk Score", min_value=0, max_value=100, format="%d")})

    pid = None
    if len(pr):
        row = pr.iloc[sel.selection.rows[0]] if sel.selection.rows else pr.iloc[0]
        pid, pname, pspec = row.provider_id, row.provider_name, row.specialty

        # ---- 3. Profile of the selected provider ----
        mine_days = pdays[pdays.provider_id == pid].sort_values("service_date")
        n_imp = int((mine_days.billed_hours > detect.IMPOSSIBLE_HOURS_MED).sum())
        mine_em = em[em.provider_id == pid]
        peers_em = em[(em.specialty == pspec) & (em.provider_id != pid)]
        my_top = (mine_em.cpt_code == "99215").mean() if len(mine_em) else np.nan
        peer_top = (peers_em.cpt_code == "99215").mean() if len(peers_em) else np.nan
        my_cases = cases[cases.provider_id == pid]
        section(3, f"Provider Profile: {pname}",
                f"{pspec} · <b>{money(row.dollars_at_risk)}</b> at risk across <b>{len(my_cases)} cases</b> "
                f"({row.fraud_types}).")
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "Dollars At Risk", money(row.dollars_at_risk), f"{row.dollars_at_risk / total_risk:.1%} of all risk")
        kpi(c2, "Flagged Claims", f"{row.flagged_claims}",
            f"of {int((scored.provider_id == pid).sum())} claims billed")
        kpi(c3, "Impossible Days", f"{n_imp}", f"busiest day {mine_days.billed_hours.max():.1f} hrs" if len(mine_days) else "-")
        kpi(c4, "Top-Level Visits", f"{my_top:.0%}" if my_top == my_top else "-",
            f"peer median {peer_top:.0%}" if peer_top == peer_top else "no office visits")
        kpi(c5, "Open Cases", f"{int((~my_cases.status.isin([feedback.CONFIRMED, feedback.NOT_FRAUD])).sum())}",
            f"{int((my_cases.status == feedback.CONFIRMED).sum())} confirmed fraud", teal=True)
        st.write("")

        l, r = st.columns(2)
        with l:
            d = mine_days.assign(level=np.select(
                [mine_days.billed_hours > detect.IMPOSSIBLE_HOURS_HIGH, mine_days.billed_hours > detect.IMPOSSIBLE_HOURS_MED],
                ["Over 16 h (high risk)", "12–16 h (review)"], "Normal day"))
            fig = px.bar(d, x="service_date", y="billed_hours", color="level",
                         color_discrete_map={"Over 16 h (high risk)": RISK["High"], "12–16 h (review)": RISK["Medium"],
                                             "Normal day": BLUE2},
                         labels={"service_date": "Service Date", "billed_hours": "Billed Hours"})
            fig.add_hline(y=detect.IMPOSSIBLE_HOURS_MED, line_dash="dash", line_color=RISK["Medium"])
            st.plotly_chart(style(fig, 340, "Billed Hours On Each Working Day"), width="stretch")
        with r:
            if len(mine_em):
                lv = pd.concat([
                    mine_em.cpt_code.value_counts(normalize=True).rename(pname),
                    peers_em.cpt_code.value_counts(normalize=True).rename(f"{pspec} peers")], axis=1) \
                    .reindex(detect.EM_CODES).fillna(0).reset_index(names="code").melt(id_vars="code")
                lv["code"] = lv.code.map({"99213": "99213 Low", "99214": "99214 Moderate", "99215": "99215 High"})
                fig = px.bar(lv, x="code", y="value", color="variable", barmode="group", text_auto=".0%",
                             color_discrete_sequence=[RISK["High"] if pid in set(upcoders.provider_id) else NAVY, GREY],
                             labels={"code": "Office Visit Level", "value": "Share Of Visits", "variable": ""})
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(style(fig, 340, "Office Visit Levels Vs Peers"), width="stretch")
            else:
                st.info(f"{pname} bills no office visits, so upcoding does not apply.")

        l, r = st.columns([1, 1.3])
        with l:
            ft = flagged[flagged.provider_id == pid].groupby("fraud_type").billed_amount.sum().reset_index()
            fig = px.pie(ft, names="fraud_type", values="billed_amount", hole=0.55, color="fraud_type",
                         color_discrete_map=TYPE_COLOR)
            fig.update_traces(textinfo="percent", textfont_size=14, hovertemplate="%{label}: $%{value:,.0f}")
            st.plotly_chart(style(fig, 300, "Their Risk By Fraud Type"), width="stretch")
        with r:
            st.markdown("**Their Cases**")
            st.dataframe(my_cases[["case_id", "fraud_type", "claims", "billed_amount", "risk_level", "status"]],
                         hide_index=True, width="stretch", height=250, column_config={
                             "case_id": "Case", "fraud_type": "Fraud Type", "claims": "Claims",
                             "billed_amount": st.column_config.NumberColumn("Billed", format="$%.0f"),
                             "risk_level": "Risk", "status": "Status"})

    # ---- 4. Impossible days, all providers ----
    section(4, "Impossible Days Across All Providers",
            f"Each dot is one provider on one day. Almost all of the {len(pdays):,} provider-days sit below 10 hours; "
            f"<b>{len(over)}</b> go above 12, which one person cannot physically work."
            + (f" <b>{pname}</b>'s days are ringed in red." if pid else ""))
    fig = px.scatter(pdays, x="service_date", y="billed_hours", color="specialty", hover_name="provider_name",
                     color_discrete_sequence=[NAVY, BLUE, BLUE2, TEAL, "#64748b", GREY],
                     labels={"service_date": "Service Date", "billed_hours": "Billed Hours", "specialty": ""})
    if pid:
        m = pdays[pdays.provider_id == pid]
        fig.add_scatter(x=m.service_date, y=m.billed_hours, mode="markers", name=pname, hovertext=m.provider_name,
                        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=RISK["High"], width=2.5)))
    fig.add_hline(y=detect.IMPOSSIBLE_HOURS_MED, line_dash="dash", line_color=RISK["Medium"],
                  annotation_text="12 h: review", annotation_font_size=13)
    fig.add_hline(y=detect.IMPOSSIBLE_HOURS_HIGH, line_dash="dash", line_color=RISK["High"],
                  annotation_text="16 h: high risk", annotation_font_size=13)
    st.plotly_chart(style(fig, 400), width="stretch")

    # ---- 5. Upcoding, all providers ----
    section(5, "Upcoding Across All Providers",
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
    st.markdown('<div class="next">➡ <b>Next step:</b> open the <b>Investigator Queue</b> to review these '
                'providers\' cases and record a decision.</div>', unsafe_allow_html=True)

# ================= 3. Investigator Queue =================
if page == "Investigator Queue":
    st.markdown('<div style="display:flex;align-items:center;gap:10px">'
                '<svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true">'
                '<path d="M12 1.5l2.1 7.4 7.4 2.1-7.4 2.1L12 20.5l-2.1-7.4-7.4-2.1 7.4-2.1z" fill="#0f766e"/>'
                '<path d="M19 15.5l.9 2.6 2.6.9-2.6.9-.9 2.6-.9-2.6-2.6-.9 2.6-.9z" fill="#3b7cc4"/>'
                '</svg><h2 style="margin:0">Investigator Queue</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="purpose"><b>Purpose:</b> Work through flagged cases one at a time: read the '
                'evidence, then record a decision.</div>', unsafe_allow_html=True)
    st.markdown("""<div class="steps"><b>How to use this page</b><ol>
        <li><b>Filter</b> the list (optional). Cases are sorted highest risk first.</li>
        <li><b>Pick a case:</b> tick the box at the start of a row in <b>step 1</b>, or choose it from the dropdown in <b>step 2</b>.</li>
        <li><b>Read the case</b> in step 2: what was found, the evidence, and innocent explanations to rule out.</li>
        <li><b>Record your decision</b> in step 3. <i>Confirmed fraud</i> or <i>Not fraud</i> closes the case and removes it from the list; <i>Referred to SIU</i> keeps it open.</li>
        <li>Every decision is saved and feeds the <b>Feedback Loop</b> page.</li>
        </ol></div>""", unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1, 1, 1])
    lv = f1.multiselect("Risk level", ["High", "Medium", "Low"], ["High", "Medium"])
    ft = f2.multiselect("Fraud type", list(TYPE_COLOR), list(TYPE_COLOR))
    show_closed = f3.toggle("Show closed cases", False)
    if run_ai:
        bar = st.progress(0.0, "Writing case briefs...")
        summarize.generate_briefs(cases, scored, pdays[pdays.billed_hours > detect.IMPOSSIBLE_HOURS_MED],
                                  levels=tuple(levels), progress=bar.progress, upcoders=upcoders)
        bar.empty()
        briefs = summarize.load_cache()
    q = cases if show_closed else open_cases.assign(fraud_type=open_cases.case_type.map(TYPE_LABEL))
    q = q[q.risk_level.isin(lv) & q.fraud_type.isin(ft)]
    section(1, "Pick A Case From The List",
            f"<b>{len(q)} cases</b> match your filters, worth <b>{money(q.billed_amount.sum())}</b>. "
            "Tick the box at the start of a row to open it in step 2.")

    show = q[["case_id", "provider_name", "fraud_type", "claims", "billed_amount", "risk_score", "ml_score",
              "risk_level", "status"]].reset_index(drop=True)
    if True:
        sel = st.dataframe(show, hide_index=True, width="stretch", height=330, on_select="rerun",
                           key="queue_table", selection_mode="single-row", column_config={
                               "case_id": "Case", "provider_name": "Provider", "fraud_type": "Fraud Type",
                               "claims": "Claims", "billed_amount": st.column_config.NumberColumn("Billed", format="$%.0f"),
                               "risk_score": st.column_config.ProgressColumn("Rule Score", min_value=0, max_value=100, format="%d"),
                               "ml_score": st.column_config.NumberColumn("ML Score", format="%d"),
                               "risk_level": "Risk", "status": "Status"})
    if show.empty:
        st.info("No cases match these filters.")
    else:
        # The table tick and the dropdown both drive the selected case
        rows = sel.selection.rows
        picked = show.case_id.iloc[rows[0]] if rows else show.case_id.iloc[0]
        labels = {r.case_id: f"{r.case_id}  ·  {r.provider_name}  ·  {r.fraud_type}  ·  ${r.billed_amount:,.0f}  ·  {r.risk_level} risk"
                  for r in show.itertuples()}
        section(2, "Read The Case",
                "This is the case ticked in step 1 (the first case if none is ticked). You can also switch case here.")
        cid = st.selectbox("Selected case", list(labels), index=list(labels).index(picked),
                           format_func=labels.get, key=f"pick-{picked}")
        case = show[show.case_id == cid].iloc[0]
        color = RISK.get(case.risk_level, BLUE)
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, lab, val in [(c1, "Case", case.case_id), (c2, "Provider", case.provider_name),
                              (c3, "Fraud Type", case.fraud_type), (c4, "Billed", f"${case.billed_amount:,.0f}"),
                              (c5, "Risk", f"{case.risk_level} · {case.risk_score:.0f}")]:
            col.markdown(f'<div class="kpi" style="border-top-color:{color};padding:10px 14px"><div class="d">{lab}</div>'
                         f'<div style="font-size:18px;font-weight:700;color:#0f172a">{val}</div></div>',
                         unsafe_allow_html=True)
        st.write("")
        l, r = st.columns([1.5, 1])
        with l:
            b = briefs.get(case.case_id)
            if not b:
                st.info("No brief yet for this case. Use **Generate AI briefs** in the sidebar.")
            else:
                badge = '<span class="ai">✦ CLAUDE</span>' if b["source"] == "claude" else '<span class="rb">RULE-BASED</span>'
                ev = "".join(f"<li>{x}</li>" for x in b["evidence"])
                alt = "".join(f"<li>{x}</li>" for x in b["innocent_explanations"])
                st.markdown(f"""<div class="brief"><div style="display:flex;justify-content:space-between">
                    <b>Case Brief</b>{badge}</div>
                    <p style="margin:8px 0 0;font-size:16px"><b>{b['headline']}</b></p>
                    <h4>Finding</h4><p>{b['finding']}</p><h4>Evidence</h4><ul>{ev}</ul>
                    <h4>Rule Out</h4><ul>{alt}</ul>
                    <h4>Recommended Action</h4><p>{b['recommended_action']}</p>
                    <p style="font-size:13px;color:#475569">Estimated exposure ${b['estimated_exposure_usd']:,.0f}</p></div>""",
                            unsafe_allow_html=True)
        with r:
            st.markdown(f'<div class="sec" style="margin-top:0"><div class="n">3</div><div class="t">Record Your Decision</div></div>'
                        f'<div class="take" style="margin-left:0">Current status: <b>{case.status}</b></div>',
                        unsafe_allow_html=True)
            if q.loc[q.case_id == case.case_id, "repeat_offender"].any():
                st.warning("Repeat offender: this provider has confirmed fraud on another case (+10 risk).")
            note = st.text_area("Note (optional)", key=f"note-{case.case_id}", height=90)
            for label, help_ in [(feedback.CONFIRMED, "Closes the case. Counts as fraud."),
                                 (feedback.NOT_FRAUD, "Closes the case. Counts as a false alarm."),
                                 (feedback.REFERRED, "Keeps the case open for the investigations unit.")]:
                if st.button(label, key=label, type="primary" if label == feedback.CONFIRMED else "secondary",
                             width="stretch", help=help_):
                    feedback.record(all_cases[all_cases.case_id == case.case_id].iloc[0], label, note=note)
                    st.toast(f"{case.case_id}: {label}")
                    st.rerun()
            st.caption("Closed cases leave the list. Use 'Show closed cases' above to see them again.")
    st.download_button("⬇ Export flagged claims", flagged.drop(columns=["planted_fraud"], errors="ignore")
                       .to_csv(index=False), "flagged_claims.csv", "text/csv")

# ================= 4. Feedback Loop =================
if page == "Feedback Loop":
    page_header("Feedback Loop",
                "Show how investigator decisions make the system smarter: they measure how accurate each rule really "
                "is and become training data for the ML model.")
    closed = fb[fb.decision.isin([feedback.CONFIRMED, feedback.NOT_FRAUD])]
    section(1, "Review Progress",
            f"Investigators have reviewed <b>{len(fb)}</b> of <b>{len(all_cases)}</b> flagged cases "
            f"({len(fb) / max(len(all_cases), 1):.0%}).")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Cases Reviewed", f"{len(fb)}", f"of {len(all_cases)} flagged")
    kpi(c2, "Confirmed Fraud", f"{(fb.decision == feedback.CONFIRMED).sum()}", "become fraud labels")
    kpi(c3, "Not Fraud", f"{(fb.decision == feedback.NOT_FRAUD).sum()}", "become false-alarm labels")
    kpi(c4, "Repeat Offender Boosts", f"{int(cases.repeat_offender.sum())}", "open cases given +10 risk", teal=True)

    stats = feedback.rule_stats(fb)
    section(2, "How Accurate Each Rule Really Is",
            "The share of reviewed cases that turned out to be real fraud, with a suggested tuning action.")
    if stats.empty:
        st.info("No closed decisions yet. Mark cases as Confirmed fraud or Not fraud in the Investigator Queue.")
    else:
        stats["rule"] = stats.rule.map(TYPE_LABEL)
        l, r = st.columns([1, 1.2])
        with l:
            fig = px.bar(stats, x="rule", y=["confirmed", "not_fraud"], barmode="stack",
                         color_discrete_map={"confirmed": NAVY, "not_fraud": GREY},
                         labels={"rule": "", "value": "Cases Reviewed", "variable": ""})
            fig.for_each_trace(lambda t: t.update(name={"confirmed": "Confirmed Fraud", "not_fraud": "Not Fraud"}[t.name]))
            st.plotly_chart(style(fig, 340, "Review Outcomes By Rule"), width="stretch")
        with r:
            st.dataframe(stats, hide_index=True, width="stretch", column_config={
                "rule": "Rule", "reviewed": "Reviewed", "confirmed": "Confirmed", "not_fraud": "Not Fraud",
                "precision": st.column_config.ProgressColumn("Confirmed Rate", min_value=0, max_value=1, format="percent"),
                "suggestion": "Suggested Action"})
            st.caption(f"Suggestions need at least {feedback.MIN_REVIEWS} reviews per rule. Thresholds live in src/detect.py.")

    section(3, "Decisions And Training Labels", "Every decision is logged; closed cases become labelled training data.")
    l, r = st.columns([1.4, 1])
    with l:
        st.dataframe(fb.sort_values("timestamp", ascending=False), hide_index=True, width="stretch", height=300)
    with r:
        if len(closed):
            labels = feedback.export_labels(all_cases, scored, fb)
            kpi(st, "Labelled Claims", f"{len(labels):,}",
                f"{labels.label.sum():,} fraud · {(labels.label == 0).sum():,} not fraud", teal=True)
            st.write("")
            st.caption("Saved to output/labels.csv. With 50+ of each class, the ML model retrains on these "
                       "real decisions.")
            st.download_button("⬇ Download labels", labels.to_csv(index=False), "labels.csv", "text/csv")
        else:
            st.info("Labels appear here once cases are closed.")

# ================= 5. Model Evaluation =================
if page == "Model Evaluation":
    page_header("Model Evaluation",
                "Check how far the detection can be trusted: how many real fraud cases the rules and the ML model "
                "catch, and how many false alarms they raise.")
    has_truth = "planted_fraud" in scored and not scored.planted_fraud.isna().all()
    if has_truth:
        ev = detect.evaluate(scored)
        ov = ev[ev.check == "Overall"].iloc[0]
        section(1, "Rule Accuracy",
                f"The rules catch <b>{ov.recall:.0%}</b> of known fraud, and <b>{ov.precision:.0%}</b> of what they flag "
                "is real fraud. Measured against fraud planted in the synthetic data.")
        d = ev[ev.check != "Overall"].melt(id_vars="check", value_vars=["precision", "recall"])
        d["variable"] = d.variable.map({"precision": "Precision (Flags That Are Fraud)", "recall": "Recall (Fraud That Gets Caught)"})
        fig = px.bar(d, x="check", y="value", color="variable", barmode="group",
                     color_discrete_sequence=[NAVY, BLUE2], labels={"check": "", "value": "", "variable": ""},
                     text_auto=".0%")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("Upcoding precision is lower by design: it flags all of a suspicious provider's top-level visits, "
                   "and the audit decides which ones were upcoded. Real fraud is messier, so expect lower scores.")
    else:
        section(1, "Rule Accuracy", "Needs ground-truth labels, which real data doesn't have. Use the Feedback Loop page instead.")

    m = ml_model.load_metrics()
    section(2, "ML Risk Model", "A second opinion: a machine-learning model scores every claim from 0 to 100.")
    if m is None:
        st.info("No model trained yet. Run `python src/ml_model.py`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Ranking Quality (AUC)", f"{m['roc_auc']:.3f}", "1.0 = perfect")
        kpi(c2, "Precision At Score 80", f"{m['precision_at_80']:.0%}", "flags that are fraud")
        kpi(c3, "Recall At Score 80", f"{m['recall_at_80']:.0%}", "fraud that gets caught")
        kpi(c4, "Top 100 Precision", f"{m['precision_top100']:.0%}", "of the highest-scored claims", teal=True)
        st.caption(f"Labels: {m['label_source']}. Trained on {m['train_claims']:,} claims before "
                   f"{ml_model.SPLIT_DATE}, tested on {m['test_claims']:,} later claims it had never seen.")
        imp = pd.DataFrame(m["importance"]).query("importance > 0.001").sort_values("importance")
        section(3, "What Drives The ML Score",
                f"<b>{imp.description.iloc[-1]}</b> is the strongest signal." if len(imp) else "")
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
