"""Fast unit tests (no model downloads needed)."""
from pathlib import Path

import pandas as pd

from csi.data.pipeline import clean_text, _dedup_key
from csi.models.metrics import coverage_precision_curve
from csi.nlp.urgency import urgency_score
from csi.rag.chunking import chunk_article, window_sentences
from csi.rag.retriever import rrf, tokenize
from csi.text import redact_pii

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"


def test_redact_pii_masks_contact_and_payment_data():
    text = "Mail me at jane.doe@example.com or +91 98765 43210, card 4111 1111 1111 1111, ip 10.0.0.1"
    out = redact_pii(text)
    assert "jane.doe" not in out and "4111" not in out and "98765" not in out and "10.0.0.1" not in out
    assert {"<email>", "<phone>", "<card>", "<ip>"} <= set(out.replace(",", " ").split())


def test_clean_text_fixes_mojibake_and_escaped_newlines():
    assert clean_text("ich mÃ¶chte\\nhelfen") == "ich möchte\nhelfen"
    assert clean_text(None) == ""


def test_dedup_key_ignores_case_punctuation_and_spacing():
    assert _dedup_key("Hello,  World!") == _dedup_key("hello world")


def test_urgency_escalates_security_breach():
    probs = {"high": 0.8, "medium": 0.15, "low": 0.05}
    u = urgency_score(probs, 0.9, "URGENT: our servers were hacked, unauthorized access to production")
    assert u["tier"] == "P1" and u["score"] >= 80
    assert "security risk" in u["drivers"]


def test_urgency_low_for_calm_question():
    u = urgency_score({"high": 0.05, "medium": 0.2, "low": 0.75}, 0.05, "Which plans do you offer?")
    assert u["tier"] == "P4"


def test_rrf_rewards_documents_ranked_well_by_both():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert max(fused, key=fused.get) in (1, 2) and fused[4] < fused[2]


def test_tokenize_drops_stopwords():
    assert tokenize("Please help, the VPN is down!") == ["help", "vpn", "down"]


def test_window_sentences_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = window_sentences(text, size=200, overlap=50)
    assert len(chunks) > 3 and all(len(c) <= 260 for c in chunks)


def test_kb_articles_chunk_by_section():
    chunks = chunk_article(KB_DIR / "KB-002-billing-and-payments.md", 600, 100)
    assert chunks and all(c.doc_id == "KB-002" for c in chunks)
    assert any(c.section == "Duplicate or incorrect charges" for c in chunks)


def test_coverage_precision_curve_monotonic_coverage():
    y = ["a", "b", "a", "b"]
    p = ["a", "a", "a", "b"]
    conf = [0.9, 0.4, 0.8, 0.6]
    rows = coverage_precision_curve(y, p, conf, thresholds=[0.3, 0.5, 0.85])
    assert [r["coverage"] for r in rows] == [1.0, 0.75, 0.25]
    assert rows[1]["precision"] == 1.0


def test_customer_facing_drops_agent_instructions():
    from csi.genai import customer_facing
    assert customer_facing("Agents must share the refund reference number with the customer.") is None
    assert customer_facing("Before replying, check the internal status dashboard.") is None
    assert customer_facing("Escalate any request for an exception above 500 USD to a team lead.") is None


def test_customer_facing_rewrites_to_second_person():
    from csi.genai import customer_facing
    assert customer_facing("Ask the customer to restart the application.") == "Please restart the application."
    assert customer_facing("The customer is entitled to a free replacement.") == "You are entitled to a free replacement."
    assert customer_facing("Customers can reset their password.") == "You can reset your password."
    assert customer_facing("If the replacement costs more, the customer pays the difference.") == \
        "If the replacement costs more, you pay the difference."
    assert customer_facing("Ask the customer to keep the device close and to disable VPNs.") == \
        "Please keep the device close and disable VPNs."
    assert customer_facing("Enterprise customers can request an exception.") == "Enterprise customers can request an exception."
    assert customer_facing("Refunds are issued to the original payment method.") == \
        "Refunds are issued to the original payment method."


def test_feedback_loop_roundtrip(tmp_path, monkeypatch):
    import dataclasses

    import pytest

    from csi import service
    from csi.config import Paths, settings

    paths = Paths(artifacts=tmp_path, warehouse=tmp_path / "test.db")
    monkeypatch.setattr(service, "settings", dataclasses.replace(settings, paths=paths))

    ids = service.log_rows([{"queue": "IT Support", "groundedness": 0.9},
                            {"queue": "Billing and Payments", "groundedness": 0.4}])
    assert ids == [1, 2]
    service.record_feedback(ids[0], helpful=True, corrected_queue="IT Support")
    service.record_feedback(ids[1], helpful=False, corrected_queue="Customer Service", comment="mail me a@b.com")
    s = service.feedback_summary()
    assert s["ratings"] == 2 and s["helpful_rate"] == 0.5
    assert s["queue_corrections"] == 1 and s["router_agreement"] == 0.5
    assert s["avg_groundedness_helpful"] == 0.9
    assert "a@b.com" not in service.query("SELECT comment FROM feedback")["comment"].iloc[1]
    with pytest.raises(KeyError):
        service.record_feedback(999, helpful=True)


def test_customer_facing_drops_instructions_after_a_condition():
    from csi.genai import customer_facing
    assert customer_facing("If an incident is already open, link the customer's ticket to the parent incident.") is None
    assert customer_facing("For questions about compatibility, share the product data sheet.") is None


def test_section_ranking_keeps_each_section_once():
    from csi.rag.evaluate_kb import section_ranking
    hits = [{"doc_id": "KB-1", "section": "A"}, {"doc_id": "KB-1", "section": "A"}, {"doc_id": "KB-2", "section": "B"}]
    assert section_ranking(hits) == [("KB-1", "A"), ("KB-2", "B")]


def test_route_sends_unsure_technical_tickets_to_the_group():
    from csi.routing import GROUP_LABEL, route
    probs = {"Technical Support": 0.40, "IT Support": 0.30, "Product Support": 0.15, "Billing and Payments": 0.15}
    r = route(probs, urgency=30, threshold=0.6, escalation=80)
    assert r["decision"] == "auto_route_group" and r["target"] == GROUP_LABEL
    assert r["suggested_queue"] == "Technical Support"


def test_route_confident_urgent_and_unsure_cases():
    from csi.routing import route
    sure = {"Billing and Payments": 0.9, "Customer Service": 0.1}
    assert route(sure, urgency=30, threshold=0.6, escalation=80)["decision"] == "auto_route"
    assert route(sure, urgency=95, threshold=0.6, escalation=80)["decision"] == "human_review"
    unsure = {"Billing and Payments": 0.5, "Customer Service": 0.5}
    assert route(unsure, urgency=30, threshold=0.6, escalation=80)["decision"] == "human_review"
