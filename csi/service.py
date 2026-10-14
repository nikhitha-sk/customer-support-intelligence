"""Business-logic facade shared by the FastAPI service and the Streamlit dashboard."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

from csi.agent import run_triage
from csi.config import settings
from csi.genai import summarize_thread
from csi.models.predictor import get_classifier
from csi.nlp.sentiment import get_sentiment
from csi.nlp.topics import get_topic_model
from csi.nlp.urgency import urgency_score
from csi.rag.embeddings import embed
from csi.rag.retriever import get_retriever
from csi.text import build_ticket_text, redact_pii

LOG_SCHEMA = """CREATE TABLE IF NOT EXISTS triage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT, source TEXT, subject TEXT, text TEXT,
    queue TEXT, queue_confidence REAL, priority TEXT, type TEXT,
    sentiment TEXT, sentiment_score REAL, urgency INTEGER, tier TEXT,
    topic INTEGER, topic_label TEXT, routing TEXT, groundedness REAL, latency_ms INTEGER
)"""


def _db() -> sqlite3.Connection:
    settings.paths.artifacts.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.paths.warehouse)
    con.execute(LOG_SCHEMA)
    return con


def log_rows(rows: list[dict]) -> None:
    if not rows:
        return
    con = _db()
    cols = ["received_at", "source", "subject", "text", "queue", "queue_confidence", "priority", "type",
            "sentiment", "sentiment_score", "urgency", "tier", "topic", "topic_label", "routing",
            "groundedness", "latency_ms"]
    con.executemany(f"INSERT INTO triage_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [[r.get(c) for c in cols] for r in rows])
    con.commit()
    con.close()


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = _db()
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def classify_batch(tickets: list[dict], source: str = "batch", log: bool = True) -> list[dict]:
    """Fast path (no LLM): classification + sentiment + urgency + theme for many tickets."""
    start = time.time()
    texts = [redact_pii(build_ticket_text(t.get("subject"), t["body"])) for t in tickets]
    preds, sents = [], []
    for i in range(0, len(texts), 32):
        preds += get_classifier().predict(texts[i:i + 32])
        sents += get_sentiment().score(texts[i:i + 32])
    topics = get_topic_model().assign(vectors=embed(texts))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    per_ticket_ms = round((time.time() - start) * 1000 / max(1, len(texts)))

    results = []
    for t, text, p, s, tp in zip(tickets, texts, preds, sents, topics):
        u = urgency_score(p["priority"]["probabilities"], s["negative"], text)
        results.append({
            "subject": t.get("subject", ""), "text": text,
            "queue": p["queue"]["label"], "queue_confidence": p["queue"]["confidence"],
            "priority": p["priority"]["label"], "type": p["type"]["label"],
            "sentiment": s["label"], "sentiment_score": s["score"],
            "urgency": u["score"], "tier": u["tier"], "sla": u["first_response_sla"],
            "urgency_drivers": "; ".join(u["drivers"]),
            "topic": tp["topic"], "topic_label": tp["label"],
            "routing": "human_review" if (u["score"] >= settings.escalation_urgency or
                                          p["queue"]["confidence"] < settings.route_confidence_threshold)
            else "auto_route",
            "received_at": now, "source": source, "latency_ms": per_ticket_ms,
        })
    if log:
        log_rows(results)
    return results


def triage(subject: str | None, body: str, source: str = "api") -> dict:
    result = run_triage(subject, body)
    c, u, s = result["classification"], result["urgency"], result["sentiment"]
    tp = get_topic_model().assign([result["text"]])[0]
    result["topic"] = tp
    log_rows([{
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
        "subject": subject or "", "text": result["text"], "queue": c["queue"]["label"],
        "queue_confidence": c["queue"]["confidence"], "priority": c["priority"]["label"],
        "type": c["type"]["label"], "sentiment": s["label"], "sentiment_score": s["score"],
        "urgency": u["score"], "tier": u["tier"], "topic": tp["topic"], "topic_label": tp["label"],
        "routing": result["routing"]["decision"], "groundedness": result["guardrail"]["groundedness"],
        "latency_ms": result["total_ms"],
    }])
    return result


def search(query_text: str, k: int = 4) -> dict:
    r = get_retriever()
    qv = embed([query_text])
    return {"kb": r.search_kb(query_text, k=k, query_vec=qv),
            "similar_tickets": r.similar_tickets(query_text, k=k, query_vec=qv)}


def summarize(text: str) -> dict:
    return summarize_thread(text)


def load_report(name: str) -> dict:
    path = settings.paths.reports / name
    return json.loads(path.read_text()) if path.exists() else {}
