"""REST API for the Customer Support Intelligence Platform.

Run:  uvicorn csi.api.main:app --port 8000      (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from csi import service
from csi.api.security import rate_limit, require_api_key
from csi.config import settings
from csi.llm import get_llm
from csi.models.predictor import get_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

app = FastAPI(
    title="Customer Support Intelligence API",
    description="Ticket routing, urgency scoring, theme detection and RAG-grounded reply drafting.",
    version="1.1.0",
)
# Every /v1 route needs the API key (when CSI_API_KEY is set) and is rate limited per client.
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key), Depends(rate_limit)])


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


class Feedback(BaseModel):
    triage_id: int = Field(description="The `triage_id` returned by /v1/triage")
    helpful: bool
    corrected_queue: str | None = Field(default=None, description="The right queue, if the prediction was wrong")
    comment: str = Field(default="", max_length=1000)
    provider: str | None = Field(default=None, description="LLM provider that wrote the draft")


@app.get("/health")
def health() -> dict:
    llm = get_llm()
    return {"status": "ok", "classifier": get_classifier().backend,
            "llm_provider": llm.provider, "llm_model": llm.model}


@v1.post("/classify")
def classify(ticket: Ticket) -> dict:
    """Fast path: queue, priority, type, sentiment, urgency and theme (no LLM call)."""
    return service.classify_batch([ticket.model_dump()], source="api")[0]


@v1.post("/classify/batch")
def classify_batch(req: BatchRequest) -> dict:
    results = service.classify_batch([t.model_dump() for t in req.tickets], source="api-batch")
    results.sort(key=lambda r: -r["urgency"])
    return {"count": len(results), "results": results}


@v1.post("/triage")
def triage(ticket: Ticket) -> dict:
    """Full agentic workflow: classify -> assess urgency -> (escalate) -> retrieve -> draft -> guardrail."""
    try:
        return service.triage(ticket.subject, ticket.body, source="api")
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Model or index artifact missing: {exc}. Run the training pipeline first.")


@v1.post("/summarize")
def summarize(req: TextRequest) -> dict:
    return service.summarize(req.text)


@v1.post("/search")
def search(req: SearchRequest) -> dict:
    return service.search(req.query, req.k)


@v1.post("/feedback", status_code=201)
def feedback(fb: Feedback) -> dict:
    """Agent rating of a drafted reply (and the right queue if routing was wrong)."""
    if fb.corrected_queue and fb.corrected_queue not in settings.queues:
        raise HTTPException(422, f"Unknown queue '{fb.corrected_queue}'")
    try:
        fid = service.record_feedback(fb.triage_id, fb.helpful, fb.corrected_queue, fb.comment, fb.provider)
    except KeyError:
        raise HTTPException(404, f"No triaged ticket with id {fb.triage_id}")
    return {"feedback_id": fid}


@v1.get("/feedback/summary")
def feedback_summary() -> dict:
    return service.feedback_summary()


@v1.get("/insights/queues")
def queue_insights() -> list[dict]:
    df = service.query("""
        SELECT queue, COUNT(*) AS tickets,
               ROUND(AVG(priority = 'high') * 100, 1) AS pct_high_priority,
               ROUND(AVG(type = 'Incident') * 100, 1) AS pct_incidents
        FROM tickets GROUP BY queue ORDER BY tickets DESC""")
    return df.to_dict(orient="records")


@v1.get("/model/metrics")
def model_metrics() -> dict:
    return {"baseline": service.load_report("metrics_baseline.json"),
            "distilbert": service.load_report("metrics_distilbert.json"),
            "retrieval": service.load_report("metrics_retrieval.json")}


app.include_router(v1)
