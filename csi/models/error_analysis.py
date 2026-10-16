"""Error analysis and label-noise audit for queue routing.

Answers four questions:
  1. Which queues are confused with which (top confused pairs, with example tickets)?
  2. How much of the error is inside the "technical" group (Technical / IT / Product Support)?
  3. Would hierarchical routing help: route to the group first, then let the group's team pick?
  4. How noisy are the labels: how often do almost identical training tickets carry different queues?

Run:  python -m csi.models.error_analysis
"""
from __future__ import annotations

import json
import logging

import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from csi.config import settings

log = logging.getLogger("csi.error_analysis")

TECH_GROUP = ("Technical Support", "IT Support", "Product Support")
GROUP_LABEL = "Technical (group)"


def merge_group(labels) -> np.ndarray:
    return np.array([GROUP_LABEL if l in TECH_GROUP else l for l in labels])


def predictions(test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Queue predictions on the test set from every model that has been trained."""
    p = settings.paths
    out = {"tfidf-logreg": joblib.load(p.models / "baseline_queue.joblib").predict(test["text"])}
    pred_file = p.processed / "test_predictions.parquet"
    if pred_file.exists():
        pred = pd.read_parquet(pred_file).set_index("ticket_id").loc[test["ticket_id"]]
        out["distilbert-multitask"] = pred["pred_queue"].to_numpy()
    return out


def confusion_pairs(y: np.ndarray, pred: np.ndarray, test: pd.DataFrame, top: int = 8) -> list[dict]:
    wrong = pd.DataFrame({"true": y, "pred": pred, "subject": test["subject"].to_numpy(),
                          "body": test["body"].str[:160].to_numpy()})[y != pred]
    class_size = pd.Series(y).value_counts()
    rows = []
    for (t, p_), g in wrong.groupby(["true", "pred"]):
        rows.append({"true": t, "predicted": p_, "count": int(len(g)),
                     "share_of_true_class": round(len(g) / class_size[t], 3),
                     "inside_tech_group": t in TECH_GROUP and p_ in TECH_GROUP,
                     "examples": [f"{s} | {b}" for s, b in g[["subject", "body"]].head(3).values]})
    return sorted(rows, key=lambda r: -r["count"])[:top]


def label_noise(threshold: float = 0.95) -> dict:
    """Near-duplicate training tickets (cosine >= threshold) whose queues disagree."""
    p = settings.paths
    index = faiss.read_index(str(p.index / "tickets.faiss"))
    tickets = pd.read_parquet(p.index / "resolved_tickets.parquet", columns=["ticket_id", "queue", "subject"])
    vecs = index.reconstruct_n(0, index.ntotal)
    sims, ids = index.search(vecs, 2)  # neighbour 0 is the ticket itself
    nn_sim, nn_id = sims[:, 1], ids[:, 1]
    queues = tickets["queue"].to_numpy()
    close = nn_sim >= threshold
    disagree = close & (queues != queues[nn_id])
    pairs = pd.DataFrame({"a": queues[disagree], "b": queues[nn_id[disagree]]})
    pairs = pairs.apply(lambda r: " vs ".join(sorted([r.a, r.b])), axis=1).value_counts()
    return {
        "threshold_cosine": threshold,
        "tickets_with_near_duplicate": int(close.sum()),
        "near_duplicates_with_different_queue": int(disagree.sum()),
        "disagreement_rate": round(float(disagree.sum() / max(1, close.sum())), 3),
        "top_disagreeing_pairs": {k: int(v) for k, v in pairs.head(6).items()},
    }


def leakage_check(test: pd.DataFrame, pred: np.ndarray, threshold: float = 0.95) -> dict:
    """Test tickets that have a near-copy in the training data are easier than truly new tickets.
    Reports accuracy separately for 'seen-like' and 'novel' test tickets."""
    from csi.rag.embeddings import embed

    index = faiss.read_index(str(settings.paths.index / "tickets.faiss"))
    sims, _ = index.search(embed(test["text"].tolist()), 1)
    near = sims[:, 0] >= threshold
    y = test["queue"].to_numpy()
    return {
        "threshold_cosine": threshold,
        "test_tickets_with_near_copy_in_train": round(float(near.mean()), 3),
        "accuracy_on_near_copies": round(float((y[near] == pred[near]).mean()), 4) if near.any() else None,
        "accuracy_on_novel_tickets": round(float((y[~near] == pred[~near]).mean()), 4) if (~near).any() else None,
    }


def main() -> dict:
    p = settings.paths
    test = pd.read_parquet(p.processed / "tickets.parquet").query("split == 'test'").reset_index(drop=True)
    y = test["queue"].to_numpy()
    labels = list(settings.queues)
    merged_labels = sorted(set(merge_group(labels)))

    report: dict = {"tech_group": list(TECH_GROUP), "models": {}}
    for name, pred in predictions(test).items():
        errors = y != pred
        in_group = errors & np.isin(y, TECH_GROUP) & np.isin(pred, TECH_GROUP)
        ym, pm = merge_group(y), merge_group(pred)
        per_class = f1_score(y, pred, labels=labels, average=None, zero_division=0)
        report["models"][name] = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)), 4),
            "errors": int(errors.sum()),
            "share_of_errors_inside_tech_group": round(float(in_group.sum() / max(1, errors.sum())), 3),
            "per_queue_f1": {l: round(float(f), 3) for l, f in zip(labels, per_class)},
            "group_level_routing": {
                "classes": len(merged_labels),
                "accuracy": round(float(accuracy_score(ym, pm)), 4),
                "macro_f1": round(float(f1_score(ym, pm, labels=merged_labels, average="macro",
                                                 zero_division=0)), 4),
            },
            "top_confusions": confusion_pairs(y, pred, test),
        }
        m = report["models"][name]
        log.info("%s: acc %.3f -> %.3f with tech group merged; %.0f%% of errors inside the tech group",
                 name, m["accuracy"], m["group_level_routing"]["accuracy"],
                 100 * m["share_of_errors_inside_tech_group"])

    best = max(report["models"], key=lambda n: report["models"][n]["macro_f1"])
    report["leakage_check"] = {"model": best, **leakage_check(test, predictions(test)[best])}
    log.info("leakage check: %s", report["leakage_check"])
    report["label_noise"] = label_noise()
    log.info("label noise: %s", report["label_noise"])
    (p.reports / "error_analysis.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    main()
