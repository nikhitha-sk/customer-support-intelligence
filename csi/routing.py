"""Routing decision shared by the agent, the batch API and the evaluation.

1. Very urgent tickets always go to a human lead.
2. If the model is confident about one queue, route there automatically.
3. Error analysis showed most mistakes are between Technical, IT and Product Support. So if the
   model is unsure *which* of those three, but sure it is one of them, route to the shared
   technical group and let that group pick the exact team.
4. Otherwise a human decides, with the model's suggestion pre-filled.
"""
from __future__ import annotations

from csi.config import settings

TECH_GROUP = ("Technical Support", "IT Support", "Product Support")
GROUP_LABEL = "Technical (group)"


def route(queue_probs: dict[str, float], urgency: int, threshold: float | None = None,
          escalation: int | None = None) -> dict:
    threshold = settings.route_confidence_threshold if threshold is None else threshold
    escalation = settings.escalation_urgency if escalation is None else escalation
    queue = max(queue_probs, key=queue_probs.get)
    conf = queue_probs[queue]
    group_conf = sum(queue_probs.get(q, 0.0) for q in TECH_GROUP)

    if urgency >= escalation:
        decision, target, reasons = "human_review", queue, [f"urgency {urgency} >= {escalation}"]
        if conf < threshold:
            reasons.append(f"queue confidence {conf:.0%} < {threshold:.0%}")
    elif conf >= threshold:
        decision, target, reasons = "auto_route", queue, []
    elif queue in TECH_GROUP and group_conf >= threshold:
        decision, target = "auto_route_group", GROUP_LABEL
        reasons = [f"unsure between technical teams ({conf:.0%} {queue}), "
                   f"but {group_conf:.0%} sure it is technical"]
    else:
        decision, target, reasons = "human_review", queue, [f"queue confidence {conf:.0%} < {threshold:.0%}"]
    return {"decision": decision, "target": target, "suggested_queue": queue, "confidence": round(conf, 4),
            "group_confidence": round(group_conf, 4), "reasons": reasons}
