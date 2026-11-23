"""Review store: case outcomes, full history, case workflow, line-level findings and audit log.

Storage is SQLite (output/claimguard.db): transactional, and every row carries an org_id and dataset_id so cases
from different datasets (e.g. synthetic demo vs pilot) never mix. Nothing is overwritten: each outcome, reversal,
assignment or approval is a new event, and the current status of a case is its latest status event.

Outcomes are structured (screening language, not fraud findings):
  No issue found · Billing or coding issue substantiated · Insufficient evidence · Referred to SIU
A substantiation above APPROVAL_LIMIT waits for supervisor approval. Formal fraud findings stay with SIU.

Training labels come only from line-level findings (substantiated / not substantiated); uncertain and unreviewed
lines stay unlabeled. Nothing here changes rules or the model automatically.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "output" / "claimguard.db"
LEGACY_CSV = ROOT / "output" / "feedback.csv"
LABELS = ROOT / "output" / "labels.csv"
ORG = "DEMO-RE"
SAMPLE_DATASET = "SYN-2026Q3"

OPEN = "Open"
SUBST = "Billing or coding issue substantiated"
NO_ISSUE = "No issue found"
INSUFF = "Insufficient evidence"
SIU = "Referred to SIU"
PENDING = "Pending supervisor approval"
FINAL = [SUBST, NO_ISSUE, INSUFF, SIU]          # a review outcome has been recorded
CLOSED = [SUBST, NO_ISSUE, INSUFF]              # case leaves the active queue
APPROVAL_LIMIT = 5000                           # substantiated amount needing supervisor approval
DUE_DAYS = 10
ADJ_ENABLED = False                             # "prior substantiated issue" +10: shown, not applied until validated
ADJ_VALUE = 10

REASONS = {
    SUBST: ["Units billed exceed documented time", "Same service billed twice (not a valid replacement)",
            "Visit level not supported by notes", "Other (describe in note)"],
    NO_ISSUE: ["Records support the billing", "Legitimate corrected / replacement claim",
               "Group billing explains the volume", "Other (describe in note)"],
    INSUFF: ["Records not received by due date", "Records incomplete or illegible", "Other (describe in note)"],
    SIU: ["Pattern suggests possible intent: needs SIU", "Prior substantiated issues on other cases",
          "Other (describe in note)"],
}
REVERSE_REASONS = ["New records received", "Recorded on the wrong case", "Supervisor review disagreed"]
FINDINGS = {"nr": "Not reviewed", "sub": "Substantiated", "not": "Not substantiated", "unc": "Uncertain"}
SIU_OWNERS = ["SIU · M. Lee", "SIU · R. Patel"]
LEGACY_MAP = {"Confirmed fraud": SUBST, "Not fraud": NO_ISSUE, "Referred to SIU": SIU, "Open": OPEN}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, org_id TEXT, dataset_id TEXT, case_id TEXT, case_type TEXT,
  provider_id TEXT, ts TEXT, reviewer TEXT, role TEXT, kind TEXT, prior_status TEXT, new_status TEXT,
  reason TEXT, evidence TEXT, note TEXT, amount REAL, owner TEXT, next_step TEXT, approval TEXT);
CREATE INDEX IF NOT EXISTS ev_case ON events(org_id, dataset_id, case_id);
CREATE TABLE IF NOT EXISTS case_state (
  org_id TEXT, dataset_id TEXT, case_id TEXT, created_at TEXT, owner TEXT, due TEXT, checklist TEXT,
  evidence_notes TEXT, PRIMARY KEY (org_id, dataset_id, case_id));
CREATE TABLE IF NOT EXISTS line_findings (
  org_id TEXT, dataset_id TEXT, case_id TEXT, claim_id TEXT, finding TEXT, reviewer TEXT, ts TEXT,
  PRIMARY KEY (org_id, dataset_id, case_id, claim_id));
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, org_id TEXT, dataset_id TEXT, ts TEXT, user TEXT, action TEXT, detail TEXT);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def conn():
    DB.parent.mkdir(exist_ok=True)
    fresh = not DB.exists()
    c = sqlite3.connect(DB, timeout=10)
    try:
        c.executescript(SCHEMA)
        if fresh:
            _migrate_legacy(c)
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _migrate_legacy(c):
    """Import the old CSV decision log into the synthetic dataset (kept as history, marked as demo seed)."""
    if not LEGACY_CSV.exists():
        return
    old = pd.read_csv(LEGACY_CSV, dtype=str).fillna("")
    prior: dict[str, str] = {}
    for r in old.sort_values("timestamp", kind="stable").itertuples():
        new = LEGACY_MAP.get(r.decision, r.decision)
        who = "demo seed (simulated)" if r.reviewer in ("simulated", "investigator") else r.reviewer
        c.execute("INSERT INTO events (org_id,dataset_id,case_id,case_type,provider_id,ts,reviewer,role,kind,"
                  "prior_status,new_status,reason,evidence,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (ORG, SAMPLE_DATASET, r.case_id, r.case_type, r.provider_id, r.timestamp, who, "Investigator",
                   "outcome", prior.get(r.case_id, OPEN), new, "Imported from earlier demo log", "",
                   r.note or "Simulated review from synthetic ground truth"))
        prior[r.case_id] = new


# ---------------- writes ----------------
def record(ds: str, case: pd.Series, new_status: str, user: str, role: str, reason: str = "", evidence: str = "",
           note: str = "", amount: float | None = None, owner: str | None = None, next_step: str = "",
           approval: str = "", kind: str = "outcome") -> None:
    with conn() as c:
        prior = _status(c, ds, case.case_id)
        c.execute("INSERT INTO events (org_id,dataset_id,case_id,case_type,provider_id,ts,reviewer,role,kind,prior_status,"
                  "new_status,reason,evidence,note,amount,owner,next_step,approval) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (ORG, ds, case.case_id, case.case_type, case.provider_id, now(), user, role, kind, prior, new_status,
                   reason, evidence, note, amount, owner, next_step, approval))
        if owner:
            c.execute("UPDATE case_state SET owner=? WHERE org_id=? AND dataset_id=? AND case_id=?",
                      (owner, ORG, ds, case.case_id))
        _audit(c, ds, user, f"{kind}: {case.case_id}", f"{prior} → {new_status}")


def assign(ds: str, case: pd.Series, owner: str, user: str, role: str) -> None:
    with conn() as c:
        st_ = _status(c, ds, case.case_id)
        c.execute("UPDATE case_state SET owner=? WHERE org_id=? AND dataset_id=? AND case_id=?", (owner, ORG, ds, case.case_id))
        c.execute("INSERT INTO events (org_id,dataset_id,case_id,case_type,provider_id,ts,reviewer,role,kind,prior_status,"
                  "new_status,reason,owner) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (ORG, ds, case.case_id, case.case_type, case.provider_id, now(), user, role, "assignment", st_, st_,
                   f"Assigned to {owner}", owner))
        _audit(c, ds, user, f"assign: {case.case_id}", owner)


def save_checklist(ds, case_id, ticks: list[bool]) -> None:
    with conn() as c:
        c.execute("UPDATE case_state SET checklist=? WHERE org_id=? AND dataset_id=? AND case_id=?",
                  (json.dumps(ticks), ORG, ds, case_id))


def save_notes(ds, case_id, text: str) -> None:
    with conn() as c:
        c.execute("UPDATE case_state SET evidence_notes=? WHERE org_id=? AND dataset_id=? AND case_id=?",
                  (text, ORG, ds, case_id))


def save_line_findings(ds, case_id, findings: dict[str, str], user: str) -> None:
    with conn() as c:
        for claim, f in findings.items():
            c.execute("INSERT OR REPLACE INTO line_findings VALUES (?,?,?,?,?,?,?)", (ORG, ds, case_id, claim, f, user, now()))
        _audit(c, ds, user, f"line findings: {case_id}", f"{sum(v != 'nr' for v in findings.values())} lines reviewed")


def log(ds, user, action, detail="") -> None:
    with conn() as c:
        _audit(c, ds, user, action, detail)


def _audit(c, ds, user, action, detail):
    c.execute("INSERT INTO audit (org_id,dataset_id,ts,user,action,detail) VALUES (?,?,?,?,?,?)",
              (ORG, ds, now(), user, action, detail))


def ensure_cases(ds: str, cases: pd.DataFrame, seed_owner: str | None = None) -> None:
    """Register cases the first time they are seen (created_at = first event or now; due = +DUE_DAYS)."""
    with conn() as c:
        have = {r[0] for r in c.execute("SELECT case_id FROM case_state WHERE org_id=? AND dataset_id=?", (ORG, ds))}
        new = cases[~cases.case_id.isin(have)]
        if new.empty:
            return
        first = dict(c.execute("SELECT case_id, MIN(ts) FROM events WHERE org_id=? AND dataset_id=? GROUP BY case_id", (ORG, ds)))
        rows = []
        for cid in new.case_id:
            created = min(first.get(cid, now()), now())
            due = (datetime.fromisoformat(created) + timedelta(days=DUE_DAYS)).date().isoformat()
            rows.append((ORG, ds, cid, created, None, due, "[]", ""))
        c.executemany("INSERT INTO case_state VALUES (?,?,?,?,?,?,?,?)", rows)
        if seed_owner and not have:   # demo only: supervisor assigns two open cases on first load
            status = dict(c.execute("SELECT case_id, new_status FROM events WHERE id IN (SELECT MAX(id) FROM events "
                                    "WHERE org_id=? AND dataset_id=? AND kind!='assignment' GROUP BY case_id)", (ORG, ds)))
            open_ = cases[cases.case_id.map(lambda x: status.get(x, OPEN) == OPEN)].sort_values("risk_score", ascending=False)
            for _, r in open_.head(2).iterrows():
                c.execute("UPDATE case_state SET owner=? WHERE org_id=? AND dataset_id=? AND case_id=?", (seed_owner, ORG, ds, r.case_id))
                c.execute("INSERT INTO events (org_id,dataset_id,case_id,case_type,provider_id,ts,reviewer,role,kind,prior_status,"
                          "new_status,reason,owner) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (ORG, ds, r.case_id, r.case_type, r.provider_id, now(), "J. Wong (demo seed)", "Supervisor",
                           "assignment", OPEN, OPEN, f"Assigned to {seed_owner}", seed_owner))


# ---------------- reads ----------------
def _status(c, ds, case_id) -> str:
    r = c.execute("SELECT new_status FROM events WHERE org_id=? AND dataset_id=? AND case_id=? AND kind!='assignment' "
                  "ORDER BY id DESC LIMIT 1", (ORG, ds, case_id)).fetchone()
    return r[0] if r else OPEN


def events(ds: str) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM events WHERE org_id=? AND dataset_id=? ORDER BY id", c, params=(ORG, ds))


def history(ds: str, case_id: str) -> pd.DataFrame:
    e = events(ds)
    return e[e.case_id == case_id].iloc[::-1]


def current(ds: str) -> pd.DataFrame:
    """Latest status per case (compatible shape: case_id, case_type, provider_id, decision, timestamp, reviewer)."""
    e = events(ds)
    e = e[e.kind != "assignment"]
    cols = ["case_id", "case_type", "provider_id", "decision", "timestamp", "reviewer", "reason", "evidence", "amount"]
    if e.empty:
        return pd.DataFrame(columns=cols)
    last = e.sort_values("id").drop_duplicates("case_id", keep="last")
    last = last.rename(columns={"new_status": "decision", "ts": "timestamp"})
    return last[last.decision != OPEN][cols]


def state(ds: str) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT case_id, created_at, owner, due, checklist, evidence_notes FROM case_state "
                           "WHERE org_id=? AND dataset_id=?", c, params=(ORG, ds))


def line_findings(ds: str, case_id: str | None = None) -> pd.DataFrame:
    with conn() as c:
        q = "SELECT case_id, claim_id, finding, reviewer, ts FROM line_findings WHERE org_id=? AND dataset_id=?"
        p = [ORG, ds]
        if case_id:
            q += " AND case_id=?"
            p.append(case_id)
        return pd.read_sql(q, c, params=p)


def audit(ds: str, n: int = 50) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT ts, user, action, detail FROM audit WHERE org_id=? AND dataset_id=? ORDER BY id DESC LIMIT ?",
                           c, params=(ORG, ds, n))


def route(p: float) -> str:
    return "Priority review" if p >= 80 else "Routine review" if p >= 50 else "No additional review recommended"


def apply(cases: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Add status, workflow fields, base score / adjustment / review priority and routing to cases."""
    out = cases.copy()
    cur = current(ds)
    status = dict(zip(cur.case_id, cur.decision))
    out["status"] = out.case_id.map(lambda c: status.get(c, OPEN))
    stt = state(ds).set_index("case_id")
    out["owner"] = out.case_id.map(stt.owner).where(lambda s: s.notna(), None)
    out["created_at"] = pd.to_datetime(out.case_id.map(stt.created_at))
    out["due"] = pd.to_datetime(out.case_id.map(stt.due))
    out["age_days"] = (pd.Timestamp.now().normalize() - out.created_at.dt.normalize()).dt.days.fillna(0).astype(int)
    e = events(ds)
    last = e.sort_values("id").drop_duplicates("case_id", keep="last").set_index("case_id") if len(e) else pd.DataFrame()
    out["last_action"] = out.case_id.map(lambda c: f"{last.loc[c, 'reason'] or last.loc[c, 'new_status']} · "
                                         f"{last.loc[c, 'ts'][:10]}" if c in last.index else "Flagged by rule")
    out["base_score"] = out.risk_score
    prior_subst = set(cur.loc[cur.decision == SUBST, "provider_id"])
    still = ~out.status.isin(CLOSED)
    out["adjustment_eligible"] = still & out.provider_id.isin(prior_subst) & ~out.case_id.isin(cur.loc[cur.decision == SUBST, "case_id"])
    out["adjustment"] = out.adjustment_eligible.map(lambda b: ADJ_VALUE if (b and ADJ_ENABLED) else 0)
    out["risk_score"] = (out.base_score + out.adjustment).clip(upper=100)
    out["risk_level"] = out.risk_score.map(lambda s: "High" if s >= 80 else "Medium" if s >= 50 else "Low")
    out["route"] = out.risk_score.map(route)
    return out.sort_values(["risk_score", "billed_amount"], ascending=False)


def rule_stats(fb: pd.DataFrame) -> pd.DataFrame:
    """Substantiation rate among reviewed cases, per rule (reviewed = any recorded outcome)."""
    rev = fb[fb.decision.isin(FINAL)]
    rows = []
    for rule, g in rev.groupby("case_type"):
        n, sub = len(g), int((g.decision == SUBST).sum())
        rows.append({"rule": rule, "reviewed": n, "substantiated": sub, "no_issue": int((g.decision == NO_ISSUE).sum()),
                     "insufficient": int((g.decision == INSUFF).sum()), "siu": int((g.decision == SIU).sum()),
                     "rate": sub / n})
    return pd.DataFrame(rows, columns=["rule", "reviewed", "substantiated", "no_issue", "insufficient", "siu", "rate"])


def export_labels(ds: str, scored: pd.DataFrame) -> pd.DataFrame:
    """Claim-line labels from line-level findings only (1 substantiated, 0 not substantiated)."""
    lf = line_findings(ds)
    lab = lf[lf.finding.isin(["sub", "not"])].drop_duplicates("claim_id", keep="last")
    lab = lab.assign(label=(lab.finding == "sub").astype(int))[["case_id", "claim_id", "label", "reviewer", "ts"]]
    LABELS.parent.mkdir(exist_ok=True)
    lab.to_csv(LABELS, index=False)
    return lab
