"""Benchmark drafted replies per LLM provider on the same test tickets.

For every ticket the sources are retrieved once, then each provider drafts a reply and the guardrail
scores it. Like the agent, a failed draft gets one strict redraft (LLM providers only).
Metrics: groundedness, guardrail pass rate (first try and after redraft), share of replies with
valid citations, PII leaks, and latency.

Run:  python -m csi.eval_replies --n 20                        # every provider that is reachable
      python -m csi.eval_replies --n 20 --providers offline     # just the offline fallback
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from csi.config import settings
from csi.genai import draft_reply, grounding_check
from csi.llm import LLMClient
from csi.rag.embeddings import embed
from csi.rag.retriever import get_retriever

log = logging.getLogger("csi.eval_replies")


def evaluate_provider(llm: LLMClient, cases: list[dict]) -> dict:
    rows = []
    for c in cases:
        start = time.time()
        d = draft_reply(c["text"], c["queue"], c["kb"], c["similar"], llm=llm)
        g = grounding_check(d["reply"], c["kb"], c["similar"])
        first_pass = g["passed"]
        if not first_pass and d["provider"] != "offline-extractive":
            d = draft_reply(c["text"], c["queue"], c["kb"], c["similar"], strict=True, llm=llm)
            g = grounding_check(d["reply"], c["kb"], c["similar"])
        rows.append({
            "ticket_id": c["ticket_id"], "provider": d["provider"], "groundedness": g["groundedness"],
            "passed_first_try": first_pass, "passed": g["passed"],
            "has_valid_citation": bool(g["citations"]) and not g["invalid_citations"],
            "pii_leak": g["pii_leak"], "words": len(d["reply"].split()),
            "latency_s": round(time.time() - start, 2), "reply": d["reply"],
        })
        log.info("%s %s groundedness=%.2f passed=%s %.1fs", llm.provider, c["ticket_id"],
                 g["groundedness"], g["passed"], rows[-1]["latency_s"])
    df = pd.DataFrame(rows)
    fell_back = (df["provider"] == "offline-extractive").mean() if llm.provider != "offline" else 0.0
    return {
        "provider": llm.provider, "model": llm.model, "tickets": len(df),
        "mean_groundedness": round(float(df["groundedness"].mean()), 3),
        "guardrail_pass_rate_first_try": round(float(df["passed_first_try"].mean()), 3),
        "guardrail_pass_rate_after_redraft": round(float(df["passed"].mean()), 3),
        "valid_citation_rate": round(float(df["has_valid_citation"].mean()), 3),
        "pii_leaks": int(df["pii_leak"].sum()),
        "mean_words": round(float(df["words"].mean()), 1),
        "latency_p50_s": round(float(np.percentile(df["latency_s"], 50)), 2),
        "latency_p95_s": round(float(np.percentile(df["latency_s"], 95)), 2),
        "fell_back_to_offline": round(float(fell_back), 3),
        "example": {"ticket_id": rows[0]["ticket_id"], "reply": rows[0]["reply"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--providers", nargs="+", default=["offline", "ollama", "groq"])
    args = ap.parse_args()
    p = settings.paths

    test = pd.read_parquet(p.processed / "tickets.parquet").query("split == 'test'")
    test = test.sample(n=min(args.n, len(test)), random_state=settings.seed)
    r = get_retriever()
    vecs = embed(test["text"].tolist())
    cases = [{"ticket_id": row.ticket_id, "text": row.text, "queue": row.queue,
              "kb": r.search_kb(row.text, k=settings.top_k, query_vec=vecs[i:i + 1]),
              "similar": r.similar_tickets(row.text, k=2, queue=row.queue, query_vec=vecs[i:i + 1])}
             for i, row in enumerate(test.itertuples())]

    results = []
    for name in args.providers:
        llm = LLMClient(name)
        if llm.provider != name:
            log.warning("provider %s is not reachable, skipping", name)
            continue
        results.append(evaluate_provider(llm, cases))

    report = {"tickets": len(cases), "sampled_from": "test split", "results": results}
    (p.reports / "metrics_replies.json").write_text(json.dumps(report, indent=2))
    for res in results:
        log.info("%s", {k: v for k, v in res.items() if k != "example"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    main()
