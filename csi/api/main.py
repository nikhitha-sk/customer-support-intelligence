"""REST API for the Customer Support Intelligence Platform.

Run:  uvicorn csi.api.main:app --port 8000      (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from csi import service
from csi.llm import get_llm
from csi.models.predictor import get_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

app = FastAPI(
    title="Customer Support Intelligence API",
    description="Ticket routing, urgency scoring, theme detection and RAG-grounded reply drafting.",
    version="1.0.0",
)


class Ticket(BaseModel):
    subject: str | None = Field(default="", max_length=500)
    body: str = Field(min_length=10, max_length=10000)


class BatchRequest(BaseModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=500)


class TextRequest(BaseModel):
    text: str = Field(min_length=10, max_length=20000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    llm = get_llm()
    return {"status": "ok", "classifier": get_classifier().backend,
            "llm_provider": llm.provider, "llm_model": llm.model}


@app.post("/v1/classify")
def classify(ticket: Ticket) -> dict:
    """Fast path: queue, priority, type, sentiment, urgency and theme (no LLM call)."""
    return service.classify_batch([ticket.model_dump()], source="api")[0]


@app.post("/v1/classify/batch")
def classify_batch(req: BatchRequest) -> dict:
    results = service.classify_batch([t.model_dump() for t in req.tickets], source="api-batch")
    results.sort(key=lambda r: -r["urgency"])
    return {"count": len(results), "results": results}


@app.post("/v1/triage")
def triage(ticket: Ticket) -> dict:
    """Full agentic workflow: classify -> assess urgency -> (escalate) -> retrieve -> draft -> guardrail."""
    try:
        return service.triage(ticket.subject, ticket.body, source="api")
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Model or index artifact missing: {exc}. Run the training pipeline first.")


@app.post("/v1/summarize")
def summarize(req: TextRequest) -> dict:
    return service.summarize(req.text)


@app.post("/v1/search")
def search(req: SearchRequest) -> dict:
    return service.search(req.query, req.k)


@app.get("/v1/insights/queues")
def queue_insights() -> list[dict]:
    df = service.query("""
        SELECT queue, COUNT(*) AS tickets,
               ROUND(AVG(priority = 'high') * 100, 1) AS pct_high_priority,
               ROUND(AVG(type = 'Incident') * 100, 1) AS pct_incidents
        FROM tickets GROUP BY queue ORDER BY tickets DESC""")
    return df.to_dict(orient="records")


@app.get("/v1/model/metrics")
def model_metrics() -> dict:
    return {"baseline": service.load_report("metrics_baseline.json"),
            "distilbert": service.load_report("metrics_distilbert.json"),
            "retrieval": service.load_report("metrics_retrieval.json")}
