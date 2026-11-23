"""AI investigator case briefs, written by Claude.

Only flagged cases are sent to Claude (by default High risk only), which keeps token cost low.
Without API credentials, a rule-based brief is produced instead so the app still works.
Briefs are cached in output/briefs.json so each case is only paid for once.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "output" / "briefs.json"
MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a health insurance Special Investigations Unit (SIU) analyst.
You receive a fraud case that automated rules have flagged, with the supporting claim data.
Write a concise, factual brief for a human investigator.

Rules:
- Use only the facts in the case data. Never invent numbers, names or events.
- Explain why the pattern is suspicious in plain language.
- Note innocent explanations an investigator should rule out (e.g. billing system error, group sessions, multiple clinicians billing under one NPI).
- The recommended action must be one investigators can take: request records, contact the provider, recover payment, refer to SIU, or close.
- This is a lead for review, not a finding of fraud. Keep the tone neutral."""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One sentence summary of the issue"},
        "finding": {"type": "string", "description": "2-3 sentences explaining what was found"},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "3-5 specific facts from the data"},
        "innocent_explanations": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {"type": "string"},
        "estimated_exposure_usd": {"type": "number"},
    },
    "required": ["headline", "finding", "evidence", "innocent_explanations", "recommended_action",
                 "estimated_exposure_usd"],
    "additionalProperties": False,
}


def case_context(case: pd.Series, scored: pd.DataFrame, provider_days: pd.DataFrame,
                 upcoders: pd.DataFrame | None = None) -> dict:
    """Assemble the compact facts Claude needs for one case."""
    claims = scored[scored.claim_id.isin(case.claim_ids)]
    ctx = {
        "case_id": case.case_id, "case_type": case.case_type, "risk_score": int(case.risk_score),
        "provider": f"{case.provider_name} ({case.provider_id}), {case.specialty}",
        "claims_in_case": int(case.claims), "total_billed_usd": round(float(case.billed_amount), 2),
        "rule_reasons": case.reasons,
    }
    if case.case_type == "impossible_day":
        day = provider_days[(provider_days.provider_id == case.provider_id)
                            & (provider_days.service_date == case.service_date)].iloc[0]
        other = provider_days[(provider_days.provider_id == case.provider_id)
                              & (provider_days.service_date != case.service_date)]
        ctx.update({
            "service_date": f"{case.service_date:%Y-%m-%d}",
            "billed_hours_that_day": round(float(day.billed_hours), 1),
            "specialty_peer_median_hours_per_day": round(float(day.peer_median_hours), 1),
            "distinct_patients": int(claims.member_id.nunique()),
            "procedure_mix": claims.groupby(["cpt_code", "cpt_description"]).units.sum()
                                   .reset_index().rename(columns={"units": "units_billed"}).to_dict("records"),
            "other_flagged_days_same_provider": [f"{d:%Y-%m-%d} ({h:.1f} hrs)" for d, h in
                                                 zip(other.service_date, other.billed_hours)],
        })
    elif case.case_type == "upcoding":
        u = upcoders[upcoders.provider_id == case.provider_id].iloc[0]
        em = scored[(scored.provider_id == case.provider_id) & scored.cpt_code.isin(["99213", "99214", "99215"])]
        ctx.update({
            "office_visits_billed": int(u.em_visits),
            "level_mix": em.cpt_code.value_counts().to_dict(),
            "share_billed_at_99215": f"{u.top_share:.0%}",
            "specialty_peer_median_share_99215": f"{u.peer_median_share:.0%}",
            "peer_providers_compared": int(u.peers),
            "std_devs_above_peers": round(float(u.z), 1),
            "billed_on_99215_claims_usd": round(float(case.billed_amount), 2),
            "top_diagnoses_on_99215_claims": claims.icd10_code.value_counts().head(3).to_dict(),
        })
    else:
        c = claims.iloc[0]
        orig = scored[scored.claim_id == c.related_claim]
        fields = ["claim_id", "member_id", "service_date", "submitted_date", "cpt_code", "cpt_description",
                  "billed_amount"]
        ctx.update({
            "flagged_claim": {k: str(c[k]) for k in fields},
            "original_claim": {k: str(orig.iloc[0][k]) for k in fields} if len(orig) else None,
        })
    return ctx


def rule_based_brief(ctx: dict) -> dict:
    """Fallback brief when Claude is not available."""
    if ctx["case_type"] == "impossible_day":
        return {
            "headline": f"{ctx['provider']} billed {ctx['billed_hours_that_day']} hours on {ctx['service_date']}.",
            "finding": (f"Billed time is {ctx['billed_hours_that_day']} hours across {ctx['distinct_patients']} patients, "
                        f"against a specialty median of {ctx['specialty_peer_median_hours_per_day']} hours."),
            "evidence": [f"{p['units_billed']} x {p['cpt_code']} {p['cpt_description']}" for p in ctx["procedure_mix"]][:5],
            "innocent_explanations": ["Multiple clinicians billing under one provider ID", "Data entry error"],
            "recommended_action": "Request appointment logs and clinical notes for the date; refer to SIU if unsupported.",
            "estimated_exposure_usd": ctx["total_billed_usd"],
        }
    if ctx["case_type"] == "upcoding":
        # Exposure estimate: difference between 99215 and 99214 fees on the excess share of visits
        excess = float(ctx["share_billed_at_99215"].rstrip("%")) - float(ctx["specialty_peer_median_share_99215"].rstrip("%"))
        exposure = round(ctx["office_visits_billed"] * excess / 100 * (230 - 165), 2)
        return {
            "headline": f"{ctx['provider']} bills {ctx['share_billed_at_99215']} of office visits at the highest level.",
            "finding": (f"{ctx['share_billed_at_99215']} of {ctx['office_visits_billed']} established-patient visits were billed as 99215, "
                        f"against a peer median of {ctx['specialty_peer_median_share_99215']} "
                        f"({ctx['std_devs_above_peers']} standard deviations above {ctx['peer_providers_compared']} peers)."),
            "evidence": [f"{k}: {v} visits" for k, v in ctx["level_mix"].items()]
                        + [f"Common diagnoses: {', '.join(ctx['top_diagnoses_on_99215_claims'])}"],
            "innocent_explanations": ["Genuinely sicker patient panel", "Provider sees referral-only complex cases"],
            "recommended_action": "Request a sample of clinical notes for 99215 visits and audit them against E/M documentation guidelines.",
            "estimated_exposure_usd": exposure,
        }
    f, o = ctx["flagged_claim"], ctx["original_claim"] or {}
    return {
        "headline": f"Claim {f['claim_id']} appears to duplicate {o.get('claim_id', 'an earlier claim')}.",
        "finding": f"Same member, provider and procedure ({f['cpt_code']}) billed again: {ctx['rule_reasons']}.",
        "evidence": [f"Flagged: {f['service_date'][:10]}, ${f['billed_amount']}, submitted {f['submitted_date'][:10]}",
                     f"Original: {o.get('service_date', '')[:10]}, ${o.get('billed_amount', '')}, submitted {o.get('submitted_date', '')[:10]}"],
        "innocent_explanations": ["Corrected claim re-submitted without a void", "Two genuine same-day services"],
        "recommended_action": "Hold payment and confirm with the provider whether this is a corrected claim.",
        "estimated_exposure_usd": float(f["billed_amount"]),
    }


def claude_brief(client, ctx: dict) -> dict:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                       # if a request is declined, retry on a fallback model
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "Case data:\n" + json.dumps(ctx, indent=1, default=str)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this case")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def get_client():
    """Return an Anthropic client if credentials resolve, else None."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        client.models.retrieve(MODEL)  # cheap credential check
        return client
    except Exception:
        return None


def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def generate_briefs(cases: pd.DataFrame, scored: pd.DataFrame, provider_days: pd.DataFrame,
                    levels=("High",), use_claude: bool = True, progress=None,
                    upcoders: pd.DataFrame | None = None) -> dict:
    cache = load_cache()
    client = get_client() if use_claude else None
    todo = cases[cases.risk_level.isin(levels)]
    for i, (_, case) in enumerate(todo.iterrows()):
        cached = cache.get(case.case_id)
        if cached and (cached["source"] == "claude" or client is None):
            continue
        ctx = case_context(case, scored, provider_days, upcoders)
        brief, source = None, "rules"
        if client is not None:
            try:
                brief, source = claude_brief(client, ctx), "claude"
            except Exception as e:  # keep the pipeline running; fall back to rules
                print(f"Claude failed on {case.case_id}: {e}")
        cache[case.case_id] = {"source": source, **(brief or rule_based_brief(ctx))}
        if progress:
            progress((i + 1) / len(todo))
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache
