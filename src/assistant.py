"""'Ask your data' assistant.

Today it is rule-based: each question is matched to an intent, and the intent runs a pandas query on the
loaded claims. Every answer carries 'how' (plain-words method) and 'query' (equivalent SQL) so users can check it.
`answer(..., method="claude")` is reserved for a Claude version (question -> query, with the same guardrails)
once API credit and a permitted route are available.

Guardrails: read-only, member IDs only (never names), no judgement of guilt, answers only from the loaded data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

import detect
import feedback

FLAG = {"duplicate": "Repeated claim (exact duplicate)", "near_duplicate": "Near-repeated claim",
        "impossible_day": "Unusual billed hours in one day", "upcoding": "Visit level above specialty peers"}

SUGGESTIONS = {
    "How to Use": ["What can I ask you?", "Which providers have the most flagged billing?",
                   "How much was billed on flagged claims, by flag reason?"],
    "Claim Intake": ["How many intake rows need fixing?", "What period does the data cover?", "What can I ask you?"],
    "Data Upload": ["What period does the data cover?", "Which providers have the most flagged billing?",
                    "What can I ask you?"],
    "Flag Overview": ["How much was billed on flagged claims, by flag reason?",
                      "Which providers have the most flagged billing?", "Which flag reason has the lowest substantiation rate?"],
    "Provider Risk": ["Which providers have the most flagged billing?", "Has this provider billed long days before?",
                      "Compare this provider's visit levels with peers"],
    "Investigator Queue": ["Why was this case flagged?", "Has this provider billed long days before?",
                           "Which members are on this case?", "Did this provider commit fraud?"],
    "Review Outcomes": ["Which flag reason has the lowest substantiation rate?", "How many cases are still open?",
                        "What can I ask you?"],
    "Executive Dashboard": ["How much was billed on flagged claims, by flag reason?",
                            "Which flag reason has the lowest substantiation rate?", "How many cases are still open?"],
    "Data & Access": ["What can I ask you?", "What period does the data cover?"],
    "Under the Hood": ["What can I ask you?", "Explain precision and recall simply"],
    "Model Evaluation": ["What happens if I raise the cut-off to 90?", "Explain precision and recall simply",
                         "Which flag reason has the lowest substantiation rate?"],
}


@dataclass
class Answer:
    text: str
    table: pd.DataFrame | None = None
    how: str = ""
    query: str = ""
    actions: list[tuple[str, str, str | None]] = field(default_factory=list)  # (label, page, case_id)
    guard: str = ""


def _money(x: float) -> str:
    return f"${x / 1000:,.0f}K" if x >= 1000 else f"${x:,.0f}"


def _provider_in(q: str, scored: pd.DataFrame, fallback: str | None) -> str | None:
    """Provider mentioned in the question (ID or surname), else the one in context."""
    m = re.search(r"prv[-\s]?(\d{4})", q)
    if m:
        return f"PRV-{m.group(1)}"
    names = scored.drop_duplicates("provider_id")[["provider_id", "provider_name"]]
    for pid, name in names.itertuples(index=False):
        last = re.sub(r"[^a-z ]", " ", str(name).lower()).split()
        if last and len(last[-1]) > 3 and re.search(rf"\b{re.escape(last[-1])}\b", q):
            return pid
    return fallback


def answer(question: str, d: dict, page: str = "", case_id: str | None = None,
           provider_id: str | None = None, method: str = "rules") -> Answer:
    """d = {'scored', 'cases', 'flagged', 'all_days', 'fb', 'intake_df' (optional)}."""
    if method != "rules":
        raise NotImplementedError("Claude assistant needs API credit; use method='rules'.")
    q = question.lower().strip()
    scored, cases, flagged, days, fb = d["scored"], d["cases"], d["flagged"], d["all_days"], d["fb"]
    case = cases[cases.case_id == case_id].iloc[0] if case_id and (cases.case_id == case_id).any() else None
    ctx_pid = provider_id or (case.provider_id if case is not None else None)
    pid = _provider_in(q, scored, ctx_pid)
    pname = scored.loc[scored.provider_id == pid, "provider_name"].iloc[0] if pid and (scored.provider_id == pid).any() else None

    def has(*words):
        return any(w in q for w in words)

    # --- guardrail: no judgement of guilt ---
    if has("commit fraud", "committed fraud", "is he a", "is she a", "criminal", "guilty", "fraudster", "a fraud?"):
        who = pname or "this provider"
        return Answer(f"I can't say whether {who} committed fraud. The data can only show that billing is "
                      "**unusual** compared with peers, and there can be legitimate reasons.",
                      guard="Reviewers record outcomes with evidence; formal fraud findings belong to SIU. I'm read-only.",
                      actions=[("Go to the case checklist", "Investigator Queue", case_id)] if case_id else [])

    if has("what can i ask", "help", "what can you"):
        return Answer("I answer questions about the claims loaded in ClaimGuard: totals, providers, flag reasons, "
                      "cases and review decisions. For example:\n\n- *Which providers have the most flagged billing?*\n"
                      "- *Has Dr. Osei billed long days before?* (name or PRV- ID)\n- *What if the cut-off is 90?*",
                      guard="I can't change data or record decisions, and I won't judge whether someone committed "
                            "fraud. That's the investigator's call.")

    if has("substantiation rate", "false alarm", "most often wrong", "cleared"):
        st_ = feedback.rule_stats(fb)
        if st_.empty:
            return Answer("No review outcomes yet, so I can't measure substantiation rates.")
        st_ = st_.sort_values("rate")
        worst = st_.iloc[0]
        t = st_.assign(flag_reason=st_.rule.map(FLAG), rate=(st_.rate * 100).round().astype(int).astype(str) + "%")
        return Answer(f"**{FLAG.get(worst.rule, worst.rule)}** has the lowest substantiation rate: {worst.substantiated} "
                      f"of {worst.reviewed} reviewed cases ({worst.rate:.0%}). Reviewed cases are a selected subset, not a "
                      "random sample of all flags.",
                      t[["flag_reason", "reviewed", "substantiated", "rate"]],
                      "Share of reviewed cases with outcome 'Billing or coding issue substantiated', by the rule that "
                      "flagged them.",
                      "SELECT case_type, COUNT(*),\n       SUM(status = 'Billing or coding issue substantiated')\n"
                      "FROM case_outcomes WHERE status IN (<recorded outcomes>) GROUP BY case_type",
                      [("Open Review Outcomes", "Review Outcomes", None)])

    if has("flag reason", "by reason", "by pattern", "billed on flagged", "by type"):
        g = flagged.groupby("case_type").billed_amount.sum().sort_values(ascending=False)
        t = pd.DataFrame({"flag_reason": g.index.map(FLAG), "billed": g.map(_money).values})
        n_conf = int((fb.decision == feedback.SUBST).sum())
        n_rev = int(fb.decision.isin(feedback.FINAL).sum())
        return Answer(f"**{_money(g.sum())}** was billed on {len(cases)} flagged cases. Remember: this is "
                      f"**billed**, not substantiated. {n_conf} of {n_rev} reviewed cases are substantiated so far.", t,
                      "Summed the billed amount on flagged claims, grouped by the rule that flagged them.",
                      "SELECT flag_reason, SUM(billed_amount) FROM flagged_claims\nGROUP BY flag_reason ORDER BY 2 DESC",
                      [("Open Flag Overview", "Flag Overview", None)])

    if has("most flagged", "top provider", "which providers", "providers have"):
        g = (flagged.groupby(["provider_id", "provider_name"]).agg(cases=("case_id", "nunique"), billed=("billed_amount", "sum"))
             .reset_index().sort_values("billed", ascending=False).head(5))
        top = g.iloc[0]
        t = g.assign(billed=g.billed.map(_money))
        return Answer(f"The top 5 providers hold **{_money(g.billed.sum())}** "
                      f"of the {_money(flagged.billed_amount.sum())} billed on flagged claims. "
                      f"**{top.provider_name}** ({top.provider_id}) is first.", t,
                      "Grouped flagged claims by provider and summed the billed amount.",
                      "SELECT provider_id, COUNT(DISTINCT case_id), SUM(billed_amount)\nFROM flagged_claims "
                      "GROUP BY provider_id ORDER BY 3 DESC LIMIT 5",
                      [("Open Provider Risk", "Provider Risk", None)])

    if has("long day", "billed long", "hours", "before", "done this"):
        if not pid:
            return Answer("Which provider? Mention a name or ID (e.g. PRV-1031), or open a case first.")
        m = days[(days.provider_id == pid) & (days.billed_hours > detect.IMPOSSIBLE_HOURS_MED)].sort_values("billed_hours", ascending=False)
        peer = days.loc[days.provider_id == pid, "peer_median_hours"].iloc[0] if (days.provider_id == pid).any() else float("nan")
        if m.empty:
            return Answer(f"No. **{pname}** has no days over {detect.IMPOSSIBLE_HOURS_MED} billed hours. "
                          f"Their specialty's median day is {peer:.1f} h.")
        t = pd.DataFrame({"date": m.service_date.dt.strftime("%d %b %Y"), "billed_hours": m.billed_hours.round(1),
                          "claims": m.claims})
        return Answer(f"Yes. **{pname}** has **{len(m)} day(s) over {detect.IMPOSSIBLE_HOURS_MED} billed hours**. "
                      f"The longest is {m.billed_hours.iloc[0]:.1f} h; the specialty median day is {peer:.1f} h.", t,
                      "Billed minutes per day = units × minutes per procedure code (CPT reference), summed per "
                      "provider-day; kept days over 12 hours.",
                      f"SELECT service_date, SUM(units*minutes)/60 AS hours, COUNT(*)\nFROM claims JOIN cpt_reference "
                      f"USING (cpt_code)\nWHERE provider_id='{pid}' GROUP BY 1 HAVING hours > 12",
                      [("Open Provider Risk", "Provider Risk", None)])

    if has("visit level", "99215", "upcod", "top-level", "top level"):
        if not pid:
            return Answer("Which provider? Mention a name or ID, or open a case first.")
        em = scored[scored.cpt_code.isin(detect.EM_CODES)]
        me = em[em.provider_id == pid]
        if me.empty:
            return Answer(f"**{pname}** bills no office visits, so visit levels don't apply.")
        spec = me.specialty.iloc[0]
        peers = em[(em.specialty == spec) & (em.provider_id != pid)]
        share = lambda x: (x.cpt_code == "99215").mean()
        peer_med = peers.groupby("provider_id").apply(share, include_groups=False).median()
        t = pd.DataFrame({"": [pname, f"{spec} peer median"],
                          "share_billed_99215": [f"{share(me):.0%}", f"{peer_med:.0%}"]})
        return Answer(f"**{pname}** billed **{share(me):.0%}** of {len(me)} office visits at the top level (99215). "
                      f"The {spec} peer median is **{peer_med:.0%}**.", t,
                      "Share of established-patient visits coded 99215, compared with the median of other providers "
                      "in the same specialty (the provider is excluded from their own peer group).",
                      f"SELECT AVG(cpt_code='99215') FROM claims WHERE provider_id='{pid}'\n  AND cpt_code IN "
                      "('99213','99214','99215')")

    if has("why was", "why is", "why did"):
        if case is None:
            return Answer("Open a case in the Investigator Queue first, then ask me why it was flagged.",
                          actions=[("Go to the Queue", "Investigator Queue", None)])
        return Answer(f"**{case.case_id}** was flagged by the rule **{FLAG.get(case.case_type, case.case_type)}**.\n\n"
                      f"Rule output: {case.reasons}\n\nIts review priority is **{case.risk_score:.0f}** (base "
                      f"{case.base_score:.0f}, adjustment {case.adjustment:.0f}): how strongly the rules suggest a "
                      "review, not a probability of fraud.",
                      how=f"Rule output for case {case.case_id}; the {int(case.claims)} claim rows are listed under "
                          "'Show the claim rows used' in the case.",
                      actions=[("Show the case", "Investigator Queue", case.case_id)])

    if has("member", "patient"):
        if case is None:
            return Answer("Open a case first and I'll list the members on it (IDs only).")
        c = scored[scored.claim_id.isin(case.claim_ids)]
        mins = c.cpt_code.map(detect.load_reference()).fillna(15) * c.units
        t = (c.assign(hours=mins / 60).groupby("member_id").agg(claims=("claim_id", "count"), billed_hours=("hours", "sum"))
             .reset_index().sort_values("claims", ascending=False))
        t["billed_hours"] = t.billed_hours.round(2)
        return Answer(f"Case **{case.case_id}** covers **{len(c)} claims for {c.member_id.nunique()} members**. "
                      "I show member IDs only, not names.", t.head(10),
                      "Claims in this case, grouped by member.",
                      f"SELECT member_id, COUNT(*), SUM(units*minutes)/60 FROM claims\nWHERE claim_id IN (<case "
                      f"{case.case_id}>) GROUP BY member_id")

    m = re.search(r"(?:cut-?off|threshold|score)\D{0,15}(\d{2})", q)
    if m or has("cut-off", "cutoff", "threshold"):
        cut = int(m.group(1)) if m else 90
        if "planted_fraud" not in scored or scored.planted_fraud.isna().all():
            return Answer("I need labelled outcomes to answer that, and this dataset has none yet. Use Review Outcomes "
                          "once investigators have recorded line-level findings.")
        truth = scored.planted_fraud.astype(bool)
        rows = []
        for c_ in sorted({50, 80, cut}):
            sel = scored.risk_score >= c_
            rows.append({"cut_off": f"{c_}+", "claims_sent_to_review": int(sel.sum()),
                         "share_that_were_problems": f"{truth[sel].mean():.0%}" if sel.any() else "-",
                         "problems_caught": f"{(sel & truth).sum() / max(truth.sum(), 1):.0%}"})
        t = pd.DataFrame(rows)
        r = t[t.cut_off == f"{cut}+"].iloc[0]
        return Answer(f"At a cut-off of **{cut}**, **{r.claims_sent_to_review}** claims go to review; "
                      f"{r.share_that_were_problems} of them were problems, and you'd catch {r.problems_caught} of all "
                      "problems. A higher cut-off means fewer, surer cases, but more missed.", t,
                      "Counted claims at each rule-score cut-off and compared with the planted labels.",
                      "SELECT COUNT(*), AVG(planted_fraud) FROM scored WHERE risk_score >= :cut",
                      [("Open Model Evaluation", "Model Evaluation", None)],
                      guard="Synthetic figures from planted problems, not real-world accuracy.")

    if has("precision", "recall"):
        return Answer("**Precision**: of the cases ClaimGuard sends you, how many are real problems.\n\n"
                      "**Recall**: of all real problems, how many ClaimGuard catches.\n\n"
                      "Raising the cut-off usually raises precision and lowers recall.",
                      guard="In this demo both are measured on synthetic data.")

    if has("still open", "how many open", "open cases", "not reviewed"):
        open_ = cases[~cases.status.isin(feedback.CLOSED)]
        return Answer(f"**{len(open_)} of {len(cases)}** flagged cases are still open "
                      f"({int((open_.risk_level == 'High').sum())} in priority review). {len(fb)} outcomes are recorded.",
                      how="Cases without a closing outcome (substantiated, no issue found, insufficient evidence). "
                          "SIU referrals stay open with SIU.",
                      actions=[("Go to the Queue", "Investigator Queue", None)])

    if has("intake", "fixing", "need fix"):
        idf = d.get("intake_df")
        if idf is None or idf.empty:
            return Answer("No documents have been processed in Claim Intake yet in this session.",
                          actions=[("Open Claim Intake", "Claim Intake", None)])
        fx = idf[idf.status == "Fix"]
        return Answer(f"**{len(fx)} of {len(idf)}** intake rows need fixing; {(idf.status == 'Check').sum()} more have "
                      "warnings.", fx[["claim_id", "source", "issues"]].head(10),
                      "Rows whose status is 'Fix' (a required value is missing).",
                      actions=[("Open Claim Intake", "Claim Intake", None)])

    if has("period", "date range", "cover", "how many claims"):
        return Answer(f"The loaded data covers **{scored.service_date.min():%d %b %Y} – {scored.service_date.max():%d %b %Y}**: "
                      f"{len(scored):,} claims from {scored.provider_id.nunique()} providers and "
                      f"{scored.member_id.nunique():,} members.")

    if has("duplicate", "repeated"):
        dup = cases[cases.case_type.isin(["duplicate", "near_duplicate"])]
        return Answer(f"**{len(dup)}** repeated-claim cases, {_money(dup.billed_amount.sum())} billed. Exact duplicates: "
                      f"{(dup.case_type == 'duplicate').sum()}; near-repeats (±1 day, ±10%): "
                      f"{(dup.case_type == 'near_duplicate').sum()}.",
                      how="Cases flagged by the duplicate and near-duplicate rules.")

    return Answer("I couldn't match that to the claims data. I can answer about providers, flag reasons, cases, "
                  "decisions and score cut-offs. Try one of the suggested questions.")
