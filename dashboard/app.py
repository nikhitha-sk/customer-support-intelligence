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
    st.plotly_chart(fig, width="stretch")
    with st.expander("SQL behind this chart"):
        st.code(q_sql, language="sql")

    col1, col2 = st.columns(2)
    risk = sql("""SELECT queue, ROUND(AVG(priority='high')*100,1) AS pct_high, COUNT(*) AS tickets
                  FROM tickets GROUP BY queue ORDER BY pct_high DESC""")
    col1.plotly_chart(px.bar(risk, x="pct_high", y="queue", orientation="h",
                             title="% high-priority tickets by queue"), width="stretch")
    types = sql("SELECT type, COUNT(*) AS tickets FROM tickets GROUP BY type")
    col2.plotly_chart(px.pie(types, names="type", values="tickets", hole=0.5, title="Ticket types"),
                      width="stretch")

    # Sentiment is pre-computed by `python -m csi.nlp.sentiment`; older warehouses do not have the column.
    if "sentiment" in sql("SELECT * FROM tickets LIMIT 1").columns:
        s_sql = """SELECT queue, ROUND(AVG(sentiment = 'negative') * 100, 1) AS pct_negative,
                  ROUND(AVG(priority = 'high') * 100, 1) AS pct_high, COUNT(*) AS tickets
           FROM tickets GROUP BY queue ORDER BY pct_negative DESC"""
        sent_q = sql(s_sql)
        col3, col4 = st.columns(2)
        col3.plotly_chart(px.bar(sent_q, x="pct_negative", y="queue", orientation="h",
                                 title="% tickets with negative customer tone, by queue"),
                          width="stretch")
        topics_file = settings.paths.processed / "ticket_topics.parquet"
        report = service.load_report("topics.json")
        if topics_file.exists() and report:
            tt = pd.read_parquet(topics_file).merge(
                sql("SELECT ticket_id, sentiment FROM tickets"), on="ticket_id")
            labels = {t["topic"]: t["label"] for t in report["topics"]}
            by_theme = (tt.assign(negative=tt.sentiment.eq("negative")).groupby("topic")["negative"].mean()
                        .mul(100).round(1).sort_values().tail(10).rename(index=labels).reset_index())
            col4.plotly_chart(px.bar(by_theme, x="negative", y="topic", orientation="h",
                                     labels={"negative": "% negative", "topic": "theme"},
                                     title="Most negative complaint themes"), width="stretch")
        with st.expander("SQL behind the sentiment chart"):
            st.code(s_sql, language="sql")

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
                           "topic_label", "routing"]], width="stretch", hide_index=True)

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
        st.plotly_chart(fig, width="stretch")
        tdf["keywords"] = tdf["keywords"].map(lambda k: ", ".join(k[:6]))
        tdf["examples"] = tdf["examples"].map(lambda e: " | ".join(e[:2]))
        st.dataframe(tdf.sort_values("size", ascending=False)[
            ["topic", "label", "size", "top_queue", "high_priority_rate", "keywords", "examples"]],
            width="stretch", hide_index=True)
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
        st.dataframe(pd.DataFrame(r["trace"]), width="stretch", hide_index=True)

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
                              "topic_label", "routing", "route_to", "urgency_drivers"]], width="stretch", hide_index=True)
            st.download_button("Download prioritised queue (CSV)", res.to_csv(index=False), "triaged_tickets.csv")
            from csi.nlp.topics import get_topic_model
            em = pd.DataFrame(get_topic_model().emerging(res["topic"].tolist()))
            st.plotly_chart(px.bar(em.head(10), x="lift", y="label", orientation="h", color="emerging",
                                   title="Theme lift vs historical baseline (emerging = significant spike)"),
                            width="stretch")

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
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    from csi.models.predictor import DISTILBERT, choose_backend
    served = choose_backend()
    st.caption(f"Model served by the API and the agent: **{served}** "
               "(`CSI_CLASSIFIER=auto` serves the model with the better test macro F1 for queue routing).")
    # charts below describe the model that is actually served
    rep = bert if served == DISTILBERT else base
    if rep:
        q = rep["queue"]
        cm = pd.DataFrame(q["confusion_matrix"], index=q["labels"], columns=q["labels"])
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  labels={"x": "Predicted", "y": "Actual"}, title="Queue confusion matrix (test)"),
                        width="stretch")
        cal_report = service.load_report("metrics_calibration.json")
        cal = cal_report if served == DISTILBERT else {}
        # after calibration the confidence thresholds mean what they say, so prefer that curve
        cp = pd.DataFrame(cal.get("queue", {}).get("coverage_precision") or q.get("coverage_precision", []))
        if not cp.empty:
            st.plotly_chart(px.line(cp, x="coverage", y="precision", markers=True, hover_data=["threshold"],
                                    title="Auto-routing trade-off: share of tickets routed automatically vs accuracy"
                                          + (" (calibrated)" if cal else "")),
                            width="stretch")
        if cal_report:
            st.subheader("Confidence calibration of DistilBERT (temperature scaling)")
            st.caption("ECE = average gap between how sure the model is and how often it is right (lower is better).")
            st.dataframe(pd.DataFrame(cal_report).T[["temperature", "test_ece_before", "test_ece_after",
                                              "mean_confidence_before", "mean_confidence_after", "accuracy"]],
                         width="stretch")

    err = service.load_report("error_analysis.json")
    if err:
        st.subheader("Error analysis")
        best = err["leakage_check"]["model"]
        m = err["models"][best]
        e1, e2, e3 = st.columns(3)
        e1.metric("Errors inside Technical / IT / Product", f"{m['share_of_errors_inside_tech_group']:.0%}")
        e2.metric("Accuracy if the tech group is merged", f"{m['group_level_routing']['accuracy']:.0%}",
                  f"from {m['accuracy']:.0%}", delta_color="off")
        lk = err["leakage_check"]
        e3.metric("Accuracy on truly new tickets", f"{lk['accuracy_on_novel_tickets']:.0%}",
                  f"{lk['accuracy_on_near_copies']:.0%} on near-copies of training tickets", delta_color="off")
        st.dataframe(pd.DataFrame(m["top_confusions"])[["true", "predicted", "count", "share_of_true_class"]],
                     width="stretch", hide_index=True)

    urg = service.load_report("metrics_urgency.json")
    if urg:
        st.subheader("Urgency score vs human priority labels")
        st.caption("AUROC: 0.5 = random, 1.0 = perfect at ranking human 'high' tickets above the rest.")
        u1, u2 = st.columns(2)
        u1.dataframe(pd.Series(urg["auroc_vs_human_high_priority"], name="AUROC"), width="stretch")
        u2.dataframe(pd.DataFrame(urg["tiers"]), width="stretch", hide_index=True)

    replies = service.load_report("metrics_replies.json")
    if replies:
        st.subheader(f"Drafted replies per LLM provider ({replies['tickets']} test tickets)")
        st.dataframe(pd.DataFrame(replies["results"]).drop(columns=["example"]).set_index("provider"),
                     width="stretch")

    ret = service.load_report("metrics_retrieval.json")
    if ret:
        st.subheader("Retrieval (similar resolved tickets)")
        st.dataframe(pd.DataFrame({m: v for m, v in ret.items() if m != "config"}).T, width="stretch")
    dq = service.load_report("data_quality.json")
    if dq:
        st.subheader("Data pipeline funnel")
        steps = ["raw_rows", "after_language_filter", "after_empty_or_short_body", "final_rows"]
        st.plotly_chart(px.funnel(x=[dq[s] for s in steps], y=steps), width="stretch")
