"""Claim Intake: turn messy claim documents into rows of the standard claims schema.

Rule-based extraction ($0, no API). Every value the rules fill in or doubt gets a reason
string, so a reviewer can see why. `extract(..., method="claude")` is reserved for an
LLM extractor with the same output schema once API credit is available.

Supported inputs: text PDFs (.pdf), plain text / email (.txt .eml .md), spreadsheets (.csv .xlsx).
"""
from __future__ import annotations

import difflib
import io
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["claim_id", "member_id", "provider_id", "provider_name", "specialty", "service_date", "submitted_date",
            "cpt_code", "cpt_description", "icd10_code", "units", "billed_amount"]
MUST_HAVE = ["member_id", "provider_id", "service_date", "cpt_code", "units", "billed_amount"]
READY, CHECK, FIX = "Ready", "Check", "Fix"

DATE_RX = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]{3}-\d{1,2}-\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b")
MONEY_RX = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2})")
CPT_RX = re.compile(r"(?<![\w.$-])(\d{5})(?![\w.])")
MEMBER_RX = re.compile(r"\bMBR[-\s]?(\d{6})\b", re.I)
PROVIDER_RX = re.compile(r"\bPRV[-\s]?(\d{4})\b", re.I)
ICD_RX = re.compile(r"\b([A-TV-Z]\d{2}\.\d{1,4})\b", re.I)
CLAIM_REF_RX = re.compile(r"\bCLM-?[\dO]{6}\b", re.I)


# ---------------- reading files ----------------
def read_file(name: str, data: bytes):
    """Return ('text', str) or ('table', DataFrame)."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        # layout mode keeps items printed on the same row (code, units, price) on one line
        return "text", "\n".join(page.extract_text(extraction_mode="layout") or "" for page in reader.pages)
    if ext in (".csv", ".xlsx", ".xls"):
        df = pd.read_csv(io.BytesIO(data), dtype=str) if ext == ".csv" else pd.read_excel(io.BytesIO(data), dtype=str)
        return "table", df
    return "text", data.decode("utf-8", errors="replace")


# ---------------- helpers ----------------
def _date(tok: str):
    try:
        return pd.to_datetime(tok, format="mixed", dayfirst=False).date()
    except (ValueError, TypeError):
        return None


def _money(tok: str) -> float:
    return float(tok.replace(",", "").replace("$", "").strip())


def reference() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "cpt_reference.csv", dtype={"cpt_code": str}).set_index("cpt_code")


def provider_history() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "claims.csv", usecols=["provider_id", "provider_name", "specialty"])
    return df.drop_duplicates("provider_id").set_index("provider_id")


# ---------------- text documents ----------------
def parse_text(text: str, source: str) -> tuple[list[dict], list[str]]:
    """Extract claim lines from one free-text document. Returns (rows, document-level warnings)."""
    lines = [ln.strip() for ln in text.splitlines()]
    doc_warn: list[str] = []
    notes: dict[str, str] = {}

    m = MEMBER_RX.search(text)
    member = f"MBR-{m.group(1)}" if m else None
    if m and m.group(0).upper() != member:
        notes["member_id"] = f"normalised from '{m.group(0)}'"
    p = PROVIDER_RX.search(text)
    prov = f"PRV-{p.group(1)}" if p else None
    if p and p.group(0).upper() != prov:
        notes["provider_id"] = f"normalised from '{p.group(0)}'"
    icd = ICD_RX.search(text)
    icd10 = icd.group(1).upper() if icd else None

    # Dates: a labelled service date wins; submitted = a line mentioning submit/sent; skip fax headers
    service = submitted = None
    for ln in lines:
        low = ln.lower()
        for tok in DATE_RX.findall(ln):
            d = _date(tok)
            if d is None:
                continue
            if "submit" in low or low.startswith("sent"):
                submitted = submitted or d
            elif re.search(r"\b(dos|date of service|service date|date\s*:)", low):
                service = d
    if service is None:   # fall back to the first date not on a fax / submitted / sent line
        for ln in lines:
            low = ln.lower()
            if "fax" in low or "submit" in low or low.startswith("sent"):
                continue
            ds = [_date(t) for t in DATE_RX.findall(ln)]
            ds = [d for d in ds if d]
            if ds:
                service = ds[0]
                notes["service_date"] = f"no 'date of service' label; used first date on the document ('{ln[:40]}')"
                break

    ref = CLAIM_REF_RX.search(text)
    if ref and "O" in ref.group(0).upper()[4:]:
        doc_warn.append(f"Claim reference '{ref.group(0)}' looks mistyped (letter O instead of zero)")

    # Line items: lines with a 5-digit procedure code. If the form has tick boxes, keep ticked lines only.
    ticked_form = any(re.match(r"^[Xx✓]\s+\d{5}\b", ln) for ln in lines)
    per_unit_doc = bool(re.search(r"rate\s*ea|per unit|unit price", text, re.I))
    items: dict[str, dict] = {}
    for ln in lines:
        clean = re.sub(r"\([^)]*\)", " ", ln)               # drop '(15 min)' style asides
        codes = [c for c in CPT_RX.findall(clean) if not re.search(rf"(MBR|PRV|CLM)[-\s]?\w*{c}", ln, re.I)]
        if not codes:
            continue
        code = codes[0]
        if ticked_form and not re.match(r"^[Xx✓]\s", ln):
            continue
        rest = clean.split(code, 1)[1]
        amounts = [_money(a) for a in MONEY_RX.findall(rest)]
        no_money = MONEY_RX.sub(" ", rest)
        u = re.search(r"(?:\bx\s?|units?\s*:?\s*)(\d{1,2})\b", no_money, re.I) or \
            (re.findall(r"(?<![\w.])(\d{1,2})(?![\w.])", no_money) or [None])[-1]
        units = int(u.group(1)) if isinstance(u, re.Match) else (int(u) if u else None)
        desc = re.sub(r"\b(units?|ea|x|qty|total|amt|cpt)\b", " ", re.sub(r"[^A-Za-z ,/&-]", " ", rest), flags=re.I)
        if not desc.strip(" ,-/&"):                                 # e.g. '96127 (brief emotional assessment)'
            paren = re.search(rf"{code}\s*\(([^)]*)\)", ln)
            desc = paren.group(1) if paren else ""
        it = items.setdefault(code, {"cpt_code": code, "units": None, "billed_amount": None, "notes": {},
                                     "desc": re.sub(r"\s+", " ", desc).strip(" ,-/&")[:40]})
        if units and it["units"] is None:
            it["units"] = units
        if amounts and it["billed_amount"] is None:
            amt = amounts[0]   # the first amount on a line is the line total ...
            if units and units > 1 and (per_unit_doc or re.search(r"\bea\b", rest, re.I)):
                it["notes"]["billed_amount"] = f"{units} units x ${amt:,.2f} per unit"   # ... unless priced 'each'
                amt = round(amt * units, 2)
            it["billed_amount"] = amt

    # Stated total vs extracted lines
    tot = None
    for ln in lines:
        mt = re.search(r"total[^$\d\n]{0,25}\$?\s?(\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2})", ln, re.I)
        if mt:
            tot = _money(mt.group(1))
            break
    line_sum = sum(i["billed_amount"] or 0 for i in items.values())
    if tot is not None and items and abs(tot - line_sum) > 0.5:
        doc_warn.append(f"Stated total ${tot:,.2f} does not match the extracted lines (${line_sum:,.2f})")

    rows = []
    for it in items.values():
        row = {"member_id": member, "provider_id": prov, "service_date": service, "submitted_date": submitted,
               "cpt_code": it["cpt_code"], "icd10_code": icd10, "units": it["units"],
               "billed_amount": it["billed_amount"], "_doc_desc": it["desc"], "source": source}
        row["_notes"] = {**notes, **it["notes"]}
        row["_doc_warn"] = list(doc_warn)
        rows.append(row)
    if not rows:
        doc_warn.append("No procedure codes found. This document may need manual entry or Claude extraction.")
    return rows, doc_warn


# ---------------- spreadsheets ----------------
SYNONYMS = {
    "claim_id": ["claim", "claim no", "claim number", "claim #", "ref"],
    "member_id": ["member", "member #", "mbr", "patient id", "member no", "subscriber"],
    "provider_id": ["provider", "prov id", "provider #", "npi", "prv"],
    "provider_name": ["dr name", "doctor", "provider name", "physician", "rendering provider"],
    "specialty": ["speciality", "spec", "dept", "department"],
    "service_date": ["dos", "date of service", "svc date", "visit date", "date"],
    "submitted_date": ["submitted", "date submitted", "received", "claim date"],
    "cpt_code": ["cpt", "procedure", "proc code", "code", "hcpcs"],
    "cpt_description": ["description", "procedure description", "service"],
    "icd10_code": ["dx", "diagnosis", "icd", "icd10", "icd-10"],
    "units": ["qty", "quantity", "unit", "units"],
    "billed_amount": ["amt", "amount", "amt $", "charge", "charges", "billed", "total", "fee"],
}


def map_columns(cols) -> dict[str, str]:
    """Map messy column names to the schema. Returns {original: schema_field}."""
    out, used = {}, set()
    norm = {c: re.sub(r"[^a-z0-9 #$-]", "", str(c).lower().replace("_", " ")).strip() for c in cols}
    for field, syns in SYNONYMS.items():
        cands = [field.replace("_", " ")] + syns
        best, score = None, 0.0
        for c, n in norm.items():
            if c in out:
                continue
            s = max(difflib.SequenceMatcher(None, n, x).ratio() for x in cands)
            if n in cands:
                s = 1.0
            if s > score:
                best, score = c, s
        if best is not None and score >= 0.75 and field not in used:
            out[best] = field
            used.add(field)
    return out


def parse_table(df: pd.DataFrame, source: str) -> tuple[list[dict], list[str]]:
    mapping = map_columns(df.columns)
    warn = [f"Columns matched: " + ", ".join(f"'{k}' → {v}" for k, v in mapping.items() if k != v)] \
        if any(k != v for k, v in mapping.items()) else []
    rows = []
    for _, r in df.rename(columns=mapping).iterrows():
        row = {f: r.get(f) for f in REQUIRED if f in r and pd.notna(r.get(f))}
        notes = {}
        if "member_id" in row:
            m = MEMBER_RX.search(str(row["member_id"]))
            row["member_id"] = f"MBR-{m.group(1)}" if m else row["member_id"]
        if "provider_id" in row:
            m = PROVIDER_RX.search(str(row["provider_id"]))
            row["provider_id"] = f"PRV-{m.group(1)}" if m else row["provider_id"]
        for f in ("service_date", "submitted_date"):
            if f in row:
                row[f] = _date(str(row[f]))
        if "billed_amount" in row:
            ms = MONEY_RX.findall(str(row["billed_amount"]))
            row["billed_amount"] = _money(ms[0]) if ms else pd.to_numeric(row["billed_amount"], errors="coerce")
        if "units" in row:
            row["units"] = pd.to_numeric(row["units"], errors="coerce")
        row.update(source=source, _notes=notes, _doc_warn=list(warn), _doc_desc="")
        rows.append(row)
    return rows, warn


# ---------------- fill gaps + validate ----------------
def complete(rows: list[dict]) -> pd.DataFrame:
    """Fill what can be looked up, generate IDs, and attach status + reasons."""
    ref, hist = reference(), provider_history()
    out = []
    for i, r in enumerate(rows, 1):
        notes = dict(r.get("_notes", {}))
        if not r.get("claim_id"):
            r["claim_id"] = f"INT-{i:04d}"
            notes["claim_id"] = "generated (no claim number on the document)"
        pid = r.get("provider_id")
        if pid in hist.index:
            h = hist.loc[pid]
            if not r.get("provider_name"):
                r["provider_name"] = h.provider_name
                notes["provider_name"] = f"from {pid} claim history"
            if not r.get("specialty"):
                r["specialty"] = h.specialty
                notes["specialty"] = f"from {pid} claim history"
        code = r.get("cpt_code")
        if code and not r.get("cpt_description"):
            if code in ref.index:
                r["cpt_description"] = ref.loc[code, "description"]
                notes["cpt_description"] = "from CPT reference"
            elif r.get("_doc_desc"):
                r["cpt_description"] = r["_doc_desc"].title()
                notes["cpt_description"] = "taken from the document text"
        if not r.get("submitted_date"):
            r["submitted_date"] = date.today()
            notes["submitted_date"] = "not on the document; set to today (intake date)"
        row = {f: r.get(f) for f in REQUIRED}
        row["source"] = r.get("source", "")
        row["_notes"] = notes
        row["_doc_warn"] = r.get("_doc_warn", [])
        out.append(row)
    df = pd.DataFrame(out, columns=REQUIRED + ["source", "_notes", "_doc_warn"])
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute status + issues. Safe to call again after a reviewer edits the table."""
    ref, hist = reference(), provider_history()
    df = df.copy()
    statuses, issues, inferred = [], [], []
    for _, r in df.iterrows():
        fix, chk = [], []
        for f in MUST_HAVE:
            v = r.get(f)
            if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan", "None", "NaT"):
                fix.append(f"{f} missing")
        if r.get("units") is not None and pd.notna(r.get("units")) and float(r["units"]) <= 0:
            fix.append("units must be above 0")
        if r.get("billed_amount") is not None and pd.notna(r.get("billed_amount")) and float(r["billed_amount"]) <= 0:
            fix.append("billed_amount must be above 0")
        code = str(r.get("cpt_code") or "")
        if code and code not in ref.index:
            chk.append(f"CPT {code} is not in the CPT reference (time per unit unknown)")
        pid = r.get("provider_id")
        if pid and pid not in hist.index:
            chk.append(f"{pid} has no claim history (new provider)")
        if not r.get("icd10_code") or pd.isna(r.get("icd10_code")):
            chk.append("no diagnosis code")
        chk += [w for w in (r.get("_doc_warn") or []) if not w.startswith("Columns matched")]
        statuses.append(FIX if fix else CHECK if chk else READY)
        issues.append("; ".join(fix + chk))
        inferred.append("; ".join(f"{k}: {v}" for k, v in (r.get("_notes") or {}).items()))
    # Same member + provider + date + code more than once in this batch (e.g. a resubmitted claim)
    key = df[["member_id", "provider_id", "service_date", "cpt_code"]].astype(str).agg("|".join, axis=1)
    for i, k in enumerate(key):
        twins = [df.claim_id.iloc[j] for j in range(len(df)) if j != i and key.iloc[j] == k]
        if twins:
            issues[i] = "; ".join(filter(None, [issues[i], f"possible duplicate of {', '.join(twins)} in this batch"]))
            if statuses[i] == READY:
                statuses[i] = CHECK
    df["status"], df["issues"], df["inferred"] = statuses, issues, inferred
    return df


def provider_mismatch(text: str, rows: pd.DataFrame) -> list[str]:
    """Flag when the names on the document don't match the provider on file."""
    hist = provider_history()
    out = []
    for pid in rows.provider_id.dropna().unique():
        if pid in hist.index:
            on_file = hist.loc[pid, "provider_name"]
            surname = re.sub(r"^Dr\.?\s*\w\.\s*", "", on_file).split()[-1]
            if surname.lower() not in text.lower():
                out.append(f"{pid} is on file as {on_file} ({hist.loc[pid, 'specialty']}), "
                           f"but that name does not appear on this document")
    return out


def extract(files: list[tuple[str, bytes]], method: str = "rules"):
    """Main entry. files = [(name, bytes)]. Returns (rows DataFrame, {name: text or table preview})."""
    if method != "rules":
        raise NotImplementedError("Claude extraction needs API credit; use method='rules'.")
    all_rows, sources = [], {}
    for name, data in files:
        kind, content = read_file(name, data)
        if kind == "text":
            rows, _ = parse_text(content, name)
            if rows:
                extra = provider_mismatch(content, pd.DataFrame(rows))
                for r in rows:
                    r["_doc_warn"] += extra
            sources[name] = content
        else:
            rows, _ = parse_table(content, name)
            sources[name] = content.head(50).to_csv(index=False)
        all_rows += rows
    if not all_rows:
        return pd.DataFrame(columns=REQUIRED + ["source", "status", "issues", "inferred"]), sources
    return complete(all_rows), sources


def to_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Rows ready for detection, in exactly the REQUIRED schema."""
    ok = df[df.status != FIX][REQUIRED].copy()
    ok["units"] = pd.to_numeric(ok.units).astype(int)
    ok["billed_amount"] = pd.to_numeric(ok.billed_amount).round(2)
    for f in ("service_date", "submitted_date"):
        ok[f] = pd.to_datetime(ok[f]).dt.strftime("%Y-%m-%d")
    return ok


if __name__ == "__main__":
    folder = ROOT / "data" / "messy"
    rows, _ = extract([(p.name, p.read_bytes()) for p in sorted(folder.iterdir()) if p.is_file()])
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70):
        print(rows.drop(columns=["_notes", "_doc_warn"]))
