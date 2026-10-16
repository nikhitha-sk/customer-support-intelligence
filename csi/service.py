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
from csi.routing import route
from csi.text import build_ticket_text, redact_pii

LOG_SCHEMA = """CREATE TABLE IF NOT EXISTS triage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT, source TEXT, subject TEXT, text TEXT,
    queue TEXT, queue_confidence REAL, priority TEXT, type TEXT,
    sentiment TEXT, sentiment_score REAL, urgency INTEGER, tier TEXT,
    topic INTEGER, topic_label TEXT, routing TEXT, groundedness REAL, latency_ms INTEGER
)"""

# Agent feedback on drafted replies. Over time this becomes a labelled evaluation set:
# "helpful" ratings for the reply generator and corrected queues for the router.
FEEDBACK_SCHEMA = """CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT, triage_id INTEGER, helpful INTEGER, predicted_queue TEXT,
    corrected_queue TEXT, provider TEXT, groundedness REAL, comment TEXT
)"""


def _db() -> sqlite3.Connection:
    settings.paths.artifacts.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.paths.warehouse)
    con.execute(LOG_SCHEMA)
    con.execute(FEEDBACK_SCHEMA)
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_rows(rows: list[dict]) -> list[int]:
    """Store triaged tickets and return their triage_log ids (used to attach feedback later)."""
    if not rows:
        return []
    con = _db()
    cols = ["received_at", "source", "subject", "text", "queue", "queue_confidence", "priority", "type",
            "sentiment", "sentiment_score", "urgency", "tier", "topic", "topic_label", "routing",
            "groundedness", "latency_ms"]
    sql = f"INSERT INTO triage_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    ids = [con.execute(sql, [r.get(c) for c in cols]).lastrowid for r in rows]
    con.commit()
    con.close()
    return ids


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
    now = _now()
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
            "routing": (r := route(p["queue"]["probabilities"], u["score"]))["decision"],
            "route_to": r["target"],
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
    result["triage_id"] = log_rows([{
        "received_at": _now(), "source": source,
        "subject": subject or "", "text": result["text"], "queue": c["queue"]["label"],
        "queue_confidence": c["queue"]["confidence"], "priority": c["priority"]["label"],
        "type": c["type"]["label"], "sentiment": s["label"], "sentiment_score": s["score"],
        "urgency": u["score"], "tier": u["tier"], "topic": tp["topic"], "topic_label": tp["label"],
        "routing": result["routing"]["decision"], "groundedness": result["guardrail"]["groundedness"],
        "latency_ms": result["total_ms"],
    }])[0]
    return result


def record_feedback(triage_id: int, helpful: bool, corrected_queue: str | None = None,
                    comment: str = "", provider: str | None = None) -> int:
    """Save an agent's rating of a drafted reply. Raises KeyError if the triage id does not exist."""
    con = _db()
    try:
        row = con.execute("SELECT queue, groundedness FROM triage_log WHERE id = ?", (triage_id,)).fetchone()
        if row is None:
            raise KeyError(triage_id)
        fid = con.execute(
            "INSERT INTO feedback (created_at, triage_id, helpful, predicted_queue, corrected_queue, provider, "
            "groundedness, comment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), triage_id, int(helpful), row[0], corrected_queue or None, provider, row[1],
             redact_pii(comment or "")[:1000]),
        ).lastrowid
        con.commit()
        return fid
    finally:
        con.close()


def feedback_summary() -> dict:
    """Headline numbers for the feedback loop: how often drafts help and how often agents fix the queue."""
    df = query("SELECT * FROM feedback")
    if df.empty:
        return {"ratings": 0}
    corrected = df["corrected_queue"].notna() & (df["corrected_queue"] != df["predicted_queue"])
    return {
        "ratings": int(len(df)),
        "helpful_rate": round(float(df["helpful"].mean()), 3),
        "queue_corrections": int(corrected.sum()),
        "router_agreement": round(float(1 - corrected.mean()), 3),
        "avg_groundedness_helpful": _mean(df.loc[df.helpful == 1, "groundedness"]),
        "avg_groundedness_not_helpful": _mean(df.loc[df.helpful == 0, "groundedness"]),
    }


def _mean(s: pd.Series) -> float | None:
    s = s.dropna()
    return round(float(s.mean()), 3) if len(s) else None


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
