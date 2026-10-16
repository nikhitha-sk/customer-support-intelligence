"""Agentic triage workflow built with LangGraph.

    preprocess -> classify -> assess_urgency --(urgent / low confidence)--> escalate --+
                                            \\------------------------------------------+--> retrieve
    retrieve -> draft_reply -> guardrail --(ungrounded & retries left)--> draft_reply (strict)
                                         \\--> finalize -> END

Every node appends to `trace`, so the UI can show the agent's reasoning step by step.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from csi.config import settings
from csi.genai import draft_reply, grounding_check
from csi.models.predictor import get_classifier
from csi.nlp.sentiment import get_sentiment
from csi.nlp.urgency import urgency_score
from csi.rag.embeddings import embed
from csi.rag.retriever import get_retriever
from csi.routing import route
from csi.text import build_ticket_text, redact_pii

MAX_DRAFT_ATTEMPTS = 2


class TriageState(TypedDict, total=False):
    subject: str
    body: str
    text: str
    classification: dict
    sentiment: dict
    urgency: dict
    routing: dict
    kb_hits: list
    similar_tickets: list
    draft: dict
    attempts: int
    guardrail: dict
    action_plan: dict
    trace: list


def _log(state: TriageState, step: str, start: float, detail: str) -> list:
    return [*state.get("trace", []), {"step": step, "ms": round((time.time() - start) * 1000), "detail": detail}]


def preprocess(state: TriageState) -> TriageState:
    t = time.time()
    text = redact_pii(build_ticket_text(state.get("subject"), state["body"]))
    return {"text": text, "attempts": 0, "trace": _log(state, "preprocess", t, "normalised text, masked PII")}


def classify(state: TriageState) -> TriageState:
    t = time.time()
    clf = get_classifier()
    pred = clf.predict([state["text"]])[0]
    q = pred["queue"]
    return {"classification": {**pred, "backend": clf.backend},
            "trace": _log(state, "classify", t,
                          f"queue={q['label']} ({q['confidence']:.0%}), priority={pred['priority']['label']}, "
                          f"type={pred['type']['label']}")}


def assess_urgency(state: TriageState) -> TriageState:
    t = time.time()
    sent = get_sentiment().score([state["text"]])[0]
    urg = urgency_score(state["classification"]["priority"]["probabilities"], sent["negative"], state["text"])
    routing = route(state["classification"]["queue"]["probabilities"], urg["score"])
    return {"sentiment": sent, "urgency": urg, "routing": routing,
            "trace": _log(state, "assess_urgency", t,
                          f"sentiment={sent['label']}, urgency={urg['score']} ({urg['tier']}), "
                          f"routing={routing['decision']}")}


def escalate(state: TriageState) -> TriageState:
    t = time.time()
    routing = {**state["routing"], "escalated_to": "duty team lead",
               "notify": "page on-call" if state["urgency"]["tier"] == "P1" else "add to review queue"}
    return {"routing": routing,
            "trace": _log(state, "escalate", t, "; ".join(state["routing"]["reasons"]))}


def retrieve(state: TriageState) -> TriageState:
    t = time.time()
    r = get_retriever()
    qv = embed([state["text"]])
    queue = state["classification"]["queue"]["label"]
    kb = r.search_kb(state["text"], k=settings.top_k, query_vec=qv)
    similar = r.similar_tickets(state["text"], k=2, queue=queue, query_vec=qv)
    return {"kb_hits": kb, "similar_tickets": similar,
            "trace": _log(state, "retrieve", t,
                          f"{len(kb)} KB chunks ({', '.join(h['chunk_id'] for h in kb)}), "
                          f"{len(similar)} similar resolved tickets")}


def draft(state: TriageState) -> TriageState:
    t = time.time()
    attempts = state.get("attempts", 0) + 1
    d = draft_reply(state["text"], state["classification"]["queue"]["label"], state["kb_hits"],
                    state["similar_tickets"], strict=attempts > 1)
    return {"draft": d, "attempts": attempts,
            "trace": _log(state, "draft_reply", t, f"attempt {attempts} via {d['provider']}")}


def guardrail(state: TriageState) -> TriageState:
    t = time.time()
    g = grounding_check(state["draft"]["reply"], state["kb_hits"], state["similar_tickets"])
    return {"guardrail": g,
            "trace": _log(state, "guardrail", t,
                          f"groundedness={g['groundedness']:.0%}, citations={len(g['citations'])}, "
                          f"passed={g['passed']}")}


def finalize(state: TriageState) -> TriageState:
    t = time.time()
    routing, u, g = state["routing"], state["urgency"], state["guardrail"]
    needs_review = routing["decision"] == "human_review" or not g["passed"]
    plan = {
        "assign_to_queue": routing["target"],
        "suggested_queue": routing["suggested_queue"],
        "priority": u["tier"],
        "first_response_sla": u["first_response_sla"],
        "reply_mode": "agent must review draft" if needs_review else "draft ready to send after quick check",
        "escalated": "escalated_to" in routing,
    }
    return {"action_plan": plan, "trace": _log(state, "finalize", t, plan["reply_mode"])}


def _after_urgency(state: TriageState) -> str:
    return "escalate" if state["routing"]["decision"] == "human_review" else "retrieve"


def _after_guardrail(state: TriageState) -> str:
    g = state["guardrail"]
    if not g["passed"] and state["attempts"] < MAX_DRAFT_ATTEMPTS and state["draft"]["provider"] != "offline-extractive":
        return "draft_reply"
    return "finalize"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(TriageState)
    for name, fn in [("preprocess", preprocess), ("classify", classify), ("assess_urgency", assess_urgency),
                     ("escalate", escalate), ("retrieve", retrieve), ("draft_reply", draft),
                     ("guardrail", guardrail), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "classify")
    g.add_edge("classify", "assess_urgency")
    g.add_conditional_edges("assess_urgency", _after_urgency, {"escalate": "escalate", "retrieve": "retrieve"})
    g.add_edge("escalate", "retrieve")
    g.add_edge("retrieve", "draft_reply")
    g.add_edge("draft_reply", "guardrail")
    g.add_conditional_edges("guardrail", _after_guardrail, {"draft_reply": "draft_reply", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run_triage(subject: str | None, body: str) -> dict:
    start = time.time()
    state = build_graph().invoke({"subject": subject or "", "body": body, "trace": []})
    state["total_ms"] = round((time.time() - start) * 1000)
    return dict(state)
