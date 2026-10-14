"""Explainable urgency score (0-100) and SLA tier.

Blends three signals so a support lead can see *why* a ticket was escalated:
  1. the model's predicted priority distribution (learned from historical labels),
  2. how negative the customer sounds,
  3. business-risk keywords (outage, breach, double charge, ...).
"""
from __future__ import annotations

import re

RISK_KEYWORDS: dict[str, tuple[int, str]] = {
    r"\b(outage|(is|are|went|been) down|offline|not responding|unavailable)\b": (12, "service outage language"),
    r"\b(breach|hack(ed)?|compromised|unauthori[sz]ed|ransomware|phishing|vulnerab\w*)\b": (15, "security risk"),
    r"\b(data loss|lost (all|my|our) data|corrupt\w*)\b": (12, "data loss"),
    r"\b(urgent(ly)?|asap|immediately|critical|emergency)\b": (8, "customer flagged urgency"),
    r"\b(charged twice|double charg\w*|overcharg\w*|refund|fraud\w*)\b": (8, "money at stake"),
    r"\b(cancel\w*|switch(ing)? to|competitor|lawyer|legal action)\b": (8, "churn / legal risk"),
    r"\b(all (users|employees|customers)|entire (team|company|organi[sz]ation)|production)\b": (8, "wide blast radius"),
}

SLA_TIERS = [  # (min score, tier, first-response target)
    (80, "P1", "1 hour"),
    (60, "P2", "4 hours"),
    (40, "P3", "1 business day"),
    (0, "P4", "3 business days"),
]


def urgency_score(priority_probs: dict[str, float], sentiment_negative: float, text: str) -> dict:
    drivers: list[str] = []

    model_part = 70 * (priority_probs.get("high", 0) + 0.45 * priority_probs.get("medium", 0))
    if priority_probs.get("high", 0) >= 0.5:
        drivers.append(f"model predicts high priority ({priority_probs['high']:.0%})")

    sentiment_part = 15 * sentiment_negative
    if sentiment_negative >= 0.6:
        drivers.append("strongly negative customer tone")

    keyword_part = 0
    lowered = text.lower()
    for pattern, (points, reason) in RISK_KEYWORDS.items():
        if re.search(pattern, lowered):
            keyword_part += points
            drivers.append(reason)
    keyword_part = min(keyword_part, 25)

    score = int(round(min(100, model_part + sentiment_part + keyword_part)))
    tier, sla = next((t, s) for floor, t, s in SLA_TIERS if score >= floor)
    return {
        "score": score,
        "tier": tier,
        "first_response_sla": sla,
        "drivers": drivers or ["no strong urgency signals"],
        "components": {
            "model": round(model_part, 1),
            "sentiment": round(sentiment_part, 1),
            "keywords": keyword_part,
        },
    }
