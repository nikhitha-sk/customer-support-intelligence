"""Streamlit dashboard: insights for support leads + live 'paste a ticket' triage demo.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from csi import service
from csi.config import settings

st.set_page_config(page_title="Support Intelligence", page_icon="🎧", layout="wide")
st.title("Customer Support Intelligence Platform")
st.caption("NLP + GenAI + RAG: route tickets, score urgency, discover complaint themes, draft grounded replies.")

SAMPLES = {
    "Double charge (billing)": ("Charged twice for my subscription",
                                "Hi, I was charged twice for my annual plan this month and need a refund urgently. "
                                "This is the second time and I am considering cancelling."),
    "Security incident": ("Possible data breach",
                          "We noticed unauthorized access to several employee accounts last night and suspect our "
                          "cloud storage has been compromised. Please advise immediately on containment."),
    "Smart-home setup": ("Device won't connect",
                         "My new smart plug will not pair with the app. I tried resetting it several times. "
                         "Does it work with Google Assistant?"),
    "Outage": ("Dashboard down for all users",
               "Our analytics dashboard has been offline for two hours and the entire team cannot work. "
               "Is there an ongoing outage? When will it be fixed?"),
}


@st.cache_data(ttl=600)
def sql(q: str) -> pd.DataFrame:
    return service.query(q)


tab_overview, tab_themes, tab_triage, tab_batch, tab_summary, tab_models = st.tabs(
    ["📊 Overview", "🧭 Complaint themes", "⚡ Live triage", "📥 Batch triage", "📝 Summarize", "📈 Model performance"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    kpi = sql("""SELECT COUNT(*) n, AVG(priority='high') high, AVG(type='Incident') inc,
                 COUNT(DISTINCT queue) queues FROM tickets""").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets analysed", f"{int(kpi.n):,}")
    c2.metric("High priority", f"{kpi.high:.0%}")
    c3.metric("Incidents", f"{kpi.inc:.0%}")
    c4.metric("Support queues", int(kpi.queues))

    q_sql = """SELECT queue, priority, COUNT(*) AS tickets FROM tickets GROUP BY queue, priority"""
    by_q = sql(q_sql)
    order = by_q.groupby("queue")["tickets"].sum().sort_values().index.tolist()
    fig = px.bar(by_q, y="queue", x="tickets", color="priority", orientation="h",
                 category_orders={"queue": order[::-1], "priority": ["high", "medium", "low"]},
                 color_discrete_map={"high": "#d1495b", "medium": "#edae49", "low": "#66a182"},
                 title="Ticket volume by queue and priority")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("SQL behind this chart"):
        st.code(q_sql, language="sql")

    col1, col2 = st.columns(2)
    risk = sql("""SELECT queue, ROUND(AVG(priority='high')*100,1) AS pct_high, COUNT(*) AS tickets
                  FROM tickets GROUP BY queue ORDER BY pct_high DESC""")
    col1.plotly_chart(px.bar(risk, x="pct_high", y="queue", orientation="h",
                             title="% high-priority tickets by queue"), use_container_width=True)
    types = sql("SELECT type, COUNT(*) AS tickets FROM tickets GROUP BY type")
    col2.plotly_chart(px.pie(types, names="type", values="tickets", hole=0.5, title="Ticket types"),
                      use_container_width=True)

    fb = service.feedback_summary()
    if fb["ratings"]:
        st.subheader("Agent feedback on drafted replies")
        f1, f2, f3 = st.columns(3)
        f1.metric("Ratings", fb["ratings"])
        f2.metric("Drafts rated helpful", f"{fb['helpful_rate']:.0%}")
        f3.metric("Router agreement", f"{fb['router_agreement']:.0%}",
                  f"{fb['queue_corrections']} queue corrections", delta_color="off")

    live = sql("SELECT * FROM triage_log ORDER BY id DESC LIMIT 200")
    if not live.empty:
        st.subheader("Recently triaged tickets (live log)")
        st.dataframe(live[["received_at", "source", "subject", "queue", "tier", "urgency", "sentiment",
                           "topic_label", "routing"]], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Themes
with tab_themes:
    topics = service.load_report("topics.json")
    if topics:
        tdf = pd.DataFrame(topics["topics"])
        st.markdown(f"**{topics['k']} themes** discovered with embedding clustering + class-based TF-IDF "
                    f"(silhouette {topics['silhouette']}). Top-right = fix first: frequent *and* urgent.")
        fig = px.scatter(tdf, x="size", y="high_priority_rate", size="size", color="top_queue",
                         hover_name="label", hover_data={"keywords": True},
                         labels={"size": "Tickets", "high_priority_rate": "High-priority rate"},
                         title="Theme prioritisation matrix")
        fig.add_hline(y=tdf.high_priority_rate.median(), line_dash="dot")
        fig.add_vline(x=tdf["size"].median(), line_dash="dot")
        st.plotly_chart(fig, use_container_width=True)
        tdf["keywords"] = tdf["keywords"].map(lambda k: ", ".join(k[:6]))
        tdf["examples"] = tdf["examples"].map(lambda e: " | ".join(e[:2]))
        st.dataframe(tdf.sort_values("size", ascending=False)[
            ["topic", "label", "size", "top_queue", "high_priority_rate", "keywords", "examples"]],
            use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m csi.nlp.topics` to discover themes.")

# ---------------------------------------------------------------- Live triage
with tab_triage:
    choice = st.selectbox("Load a sample ticket", ["(write your own)"] + list(SAMPLES))
    subj0, body0 = SAMPLES.get(choice, ("", ""))
    subject = st.text_input("Subject", subj0)
    body = st.text_area("Ticket body", body0, height=140)
    if st.button("Triage ticket", type="primary") and len(body) >= 10:
        with st.spinner("Running agent workflow..."):
            # kept in session state so the result survives the rerun triggered by the feedback form
            st.session_state["triage"] = service.triage(subject, body, source="dashboard")
    r = st.session_state.get("triage")
    if r:
        c, u, plan = r["classification"], r["urgency"], r["action_plan"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Queue", c["queue"]["label"], f"{c['queue']['confidence']:.0%} confidence", delta_color="off")
        m2.metric("Urgency", f"{u['score']}/100", u["tier"], delta_color="off")
        m3.metric("Sentiment", r["sentiment"]["label"], f"{r['sentiment']['score']:+.2f}", delta_color="off")
        m4.metric("First-response SLA", u["first_response_sla"])
        st.info(f"**Action plan:** assign to *{plan['assign_to_queue']}* · {plan['priority']} · "
                f"{plan['reply_mode']}" + (" · **escalated**" if plan["escalated"] else ""))
        st.markdown("**Why this urgency?** " + " · ".join(u["drivers"]))

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Suggested reply")
            st.text_area("Draft (edit before sending)", r["draft"]["reply"], height=260)
            g = r["guardrail"]
            st.caption(f"Generated by {r['draft']['provider']} · groundedness {g['groundedness']:.0%} · "
                       f"citations: {', '.join(g['citations']) or 'none'} · guardrail "
                       f"{'passed ✅' if g['passed'] else 'flagged ⚠️'}")
        with right:
            st.subheader("Retrieved sources")
            for h in r["kb_hits"]:
                with st.expander(f"{h['chunk_id']} · {h['section']} (sim {h['similarity']:.2f})"):
                    st.write(h["text"])
            for t in r["similar_tickets"]:
                with st.expander(f"{t['ticket_id']} · {t['subject'][:50]} (sim {t['similarity']:.2f})"):
                    st.write(t["answer"])
        st.subheader("Agent trace")
        st.dataframe(pd.DataFrame(r["trace"]), use_container_width=True, hide_index=True)

        st.subheader("Was this helpful?")
        with st.form(f"feedback-{r['triage_id']}", clear_on_submit=True):
            rating = st.radio("Draft reply", ["👍 Helpful", "👎 Not helpful"], horizontal=True)
            queues = list(settings.queues)
            right_queue = st.selectbox("Correct queue", queues, index=queues.index(c["queue"]["label"]))
            comment = st.text_input("Comment (optional)")
            if st.form_submit_button("Send feedback"):
                service.record_feedback(r["triage_id"], rating.startswith("👍"), right_queue, comment,
                                        r["draft"]["provider"])
                st.success("Thanks! Feedback saved and added to the evaluation set.")

# ---------------------------------------------------------------- Batch triage
with tab_batch:
    st.markdown("Upload a CSV with a `body` column (optional `subject`). Every ticket is classified, "
                "scored and sorted into a prioritised work queue; themes spiking versus history are flagged.")
    up = st.file_uploader("Tickets CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up).fillna("")
        if "body" not in df.columns:
            st.error("CSV needs a `body` column.")
        else:
            df = df.head(500)
            with st.spinner(f"Scoring {len(df)} tickets..."):
                res = pd.DataFrame(service.classify_batch(
                    df[["subject", "body"]].to_dict("records") if "subject" in df else
                    [{"body": b} for b in df["body"]], source="dashboard-batch"))
            res = res.sort_values("urgency", ascending=False)
            st.dataframe(res[["subject", "queue", "queue_confidence", "tier", "urgency", "sentiment",
                              "topic_label", "routing", "urgency_drivers"]], use_container_width=True, hide_index=True)
            st.download_button("Download prioritised queue (CSV)", res.to_csv(index=False), "triaged_tickets.csv")
            from csi.nlp.topics import get_topic_model
            em = pd.DataFrame(get_topic_model().emerging(res["topic"].tolist()))
            st.plotly_chart(px.bar(em.head(10), x="lift", y="label", orientation="h", color="emerging",
                                   title="Theme lift vs historical baseline (emerging = significant spike)"),
                            use_container_width=True)

# ---------------------------------------------------------------- Summarize
with tab_summary:
    thread = st.text_area("Paste a long ticket thread", height=220)
    if st.button("Summarize") and len(thread) > 20:
        s = service.summarize(thread)
        st.success(s["summary"])
        st.caption(f"via {s['provider']}")

# ---------------------------------------------------------------- Models
with tab_models:
    base = service.load_report("metrics_baseline.json")
    bert = service.load_report("metrics_distilbert.json")
    rows = []
    for name, rep in [("TF-IDF + LogReg (baseline)", base), ("Multi-task DistilBERT", bert)]:
        for task in ("queue", "priority", "type"):
            if task in rep:
                rows.append({"model": name, "task": task, "accuracy": rep[task]["accuracy"],
                             "macro F1": rep[task]["macro_f1"], "weighted F1": rep[task]["weighted_f1"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if not bert:
        st.caption("DistilBERT not trained yet in this deployment (see TODO.md) - the API serves the baseline.")
    rep = bert or base
    if rep:
        q = rep["queue"]
        cm = pd.DataFrame(q["confusion_matrix"], index=q["labels"], columns=q["labels"])
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  labels={"x": "Predicted", "y": "Actual"}, title="Queue confusion matrix (test)"),
                        use_container_width=True)
        cp = pd.DataFrame(q.get("coverage_precision", []))
        if not cp.empty:
            st.plotly_chart(px.line(cp, x="coverage", y="precision", markers=True, hover_data=["threshold"],
                                    title="Auto-routing trade-off: share of tickets routed automatically vs accuracy"),
                            use_container_width=True)
    ret = service.load_report("metrics_retrieval.json")
    if ret:
        st.subheader("Retrieval (similar resolved tickets)")
        st.dataframe(pd.DataFrame({m: v for m, v in ret.items() if m != "config"}).T, use_container_width=True)
    dq = service.load_report("data_quality.json")
    if dq:
        st.subheader("Data pipeline funnel")
        steps = ["raw_rows", "after_language_filter", "after_empty_or_short_body", "final_rows"]
        st.plotly_chart(px.funnel(x=[dq[s] for s in steps], y=steps), use_container_width=True)
