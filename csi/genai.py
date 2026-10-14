"""GenAI tasks: grounded reply drafting (RAG), thread summarisation, and the grounding guardrail."""
from __future__ import annotations

import re

import numpy as np

from csi.llm import LLMUnavailable, get_llm
from csi.rag.embeddings import embed
from csi.text import redact_pii, split_sentences

REPLY_SYSTEM = """You are an assistant that drafts replies for customer-support agents.
Rules:
- Use ONLY facts found in the SOURCES. Never invent policies, prices, timelines or links.
- After every sentence that uses a fact, cite its source id in square brackets, e.g. [KB-002#1] or [T000123].
- If the sources do not answer the question, say what extra information you need from the customer.
- Be empathetic and concise: at most 150 words. Address the customer as <name>.
- Never include passwords, card numbers or other personal data. Do not promise outcomes you cannot guarantee.
Return only the reply text."""

STRICT_SUFFIX = """
IMPORTANT: Your previous draft contained statements that were not supported by the sources.
Rewrite it using only sentences that are directly supported by the sources, each with a citation."""

SUMMARY_SYSTEM = """You summarise customer-support threads for a busy support lead.
Respond in exactly this format:
Issue: <one sentence>
Customer impact: <one sentence>
Actions taken so far: <one sentence or 'None'>
Next best action: <one sentence>
Sentiment: <calm / frustrated / angry>"""


def format_sources(kb_hits: list[dict], similar: list[dict]) -> str:
    parts = [f"[{h['chunk_id']}] {h['text']}" for h in kb_hits]
    parts += [f"[{t['ticket_id']}] Past ticket ({t['queue']}): {t['subject']}\nResolution sent: {t['answer'][:500]}"
              for t in similar]
    return "\n\n".join(parts)


def _fallback_reply(ticket: str, kb_hits: list[dict], similar: list[dict]) -> str:
    """Extractive reply used when no LLM is reachable: picks the source sentences most
    similar to the ticket and stitches them into a cited template."""
    candidates: list[tuple[str, str]] = []
    for h in kb_hits[:3]:
        body = h["text"].split(": ", 1)[-1]
        candidates += [(s, h["chunk_id"]) for s in split_sentences(body)]
    if not candidates and similar:
        candidates += [(s, similar[0]["ticket_id"]) for s in split_sentences(similar[0]["answer"])[:3]]
    if not candidates:
        return ("Dear <name>,\n\nThank you for contacting us. To help you quickly, could you share more details "
                "about the issue, including any error messages and when it started?\n\nBest regards,\nSupport Team")
    sims = embed([c[0] for c in candidates]) @ embed([ticket])[0]
    best = sorted(np.argsort(-sims)[:3])
    body = " ".join(f"{candidates[i][0]} [{candidates[i][1]}]" for i in best)
    return f"Dear <name>,\n\nThank you for reaching out, and sorry for the trouble. {body}\n\nBest regards,\nSupport Team"


def draft_reply(ticket: str, queue: str, kb_hits: list[dict], similar: list[dict], strict: bool = False) -> dict:
    llm = get_llm()
    sources = format_sources(kb_hits, similar)
    user = f"TICKET (routed to {queue}):\n{redact_pii(ticket)}\n\nSOURCES:\n{sources}"
    try:
        resp = llm.chat(REPLY_SYSTEM + (STRICT_SUFFIX if strict else ""), user, temperature=0.1 if strict else 0.3)
        return {"reply": redact_pii(resp.text), "provider": resp.provider, "model": resp.model,
                "latency_s": resp.latency_s}
    except LLMUnavailable:
        return {"reply": _fallback_reply(ticket, kb_hits, similar), "provider": "offline-extractive",
                "model": "none", "latency_s": 0.0}
    except Exception as exc:  # network errors, rate limits...
        return {"reply": _fallback_reply(ticket, kb_hits, similar), "provider": "offline-extractive",
                "model": "none", "latency_s": 0.0, "error": str(exc)[:200]}


_BOILERPLATE = re.compile(r"^(dear|hi|hello|best|kind|regards|thank|thanks|sincerely|support team|<name>)", re.I)


def grounding_check(reply: str, kb_hits: list[dict], similar: list[dict], threshold: float = 0.5) -> dict:
    """Share of substantive reply sentences that are semantically supported by a retrieved source."""
    sentences = [s for s in split_sentences(re.sub(r"\[[^\]]+\]", "", reply))
                 if len(s.split()) >= 5 and not _BOILERPLATE.match(s)]
    source_texts = [h["text"] for h in kb_hits] + [t["answer"] for t in similar if t.get("answer")]
    valid_ids = {h["chunk_id"] for h in kb_hits} | {t["ticket_id"] for t in similar}
    cited = set(re.findall(r"\[([A-Z]+-?\d+(?:#\d+)?)\]", reply))
    source_sents = [s for txt in source_texts for s in split_sentences(txt)]
    if not sentences or not source_sents:
        support = []
    else:
        support = (embed(sentences) @ embed(source_sents).T).max(1).tolist()
    grounded = sum(s >= threshold for s in support)
    score = grounded / len(support) if support else 1.0
    return {
        "groundedness": round(score, 3),
        "unsupported_sentences": [s for s, v in zip(sentences, support) if v < threshold],
        "citations": sorted(cited),
        "invalid_citations": sorted(cited - valid_ids),
        "pii_leak": redact_pii(reply) != reply,
        "passed": score >= 0.7 and not (cited - valid_ids),
    }


def summarize_thread(text: str) -> dict:
    llm = get_llm()
    try:
        resp = llm.chat(SUMMARY_SYSTEM, redact_pii(text), temperature=0.1, max_tokens=250)
        return {"summary": resp.text, "provider": resp.provider, "model": resp.model, "latency_s": resp.latency_s}
    except Exception:
        # Extractive fallback: the sentences closest to the thread's overall meaning (embedding centrality).
        sents = split_sentences(redact_pii(text))
        if len(sents) <= 3:
            summary = " ".join(sents)
        else:
            vecs = embed(sents)
            centrality = vecs @ vecs.mean(0)
            summary = " ".join(sents[i] for i in sorted(np.argsort(-centrality)[:3]))
        return {"summary": summary, "provider": "offline-extractive", "model": "none", "latency_s": 0.0}
