"""Does the urgency score agree with people? Validates it against the human "high" priority labels.

Reports, on the held-out test set:
  * AUROC of the full score and of each component (model / sentiment / risk keywords),
  * how the SLA tiers line up with the human labels (share of truly high-priority tickets per tier),
  * a small weight search on the validation set, checked on the test set.

Needs sentiment for every ticket first:  python -m csi.nlp.sentiment
Run:  python -m csi.nlp.validate_urgency
"""
from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from csi.config import settings
from csi.models.predictor import get_classifier
from csi.nlp.urgency import SLA_TIERS, urgency_score

log = logging.getLogger("csi.urgency.validate")


def components(df: pd.DataFrame) -> pd.DataFrame:
    """Urgency score and its three parts for every ticket in df."""
    clf = get_classifier()
    preds = []
    for i in range(0, len(df), 64):
        preds += clf.predict(df["text"].iloc[i:i + 64].tolist())
    rows = []
    for pred, neg, text in zip(preds, df["sentiment_negative"], df["text"]):
        u = urgency_score(pred["priority"]["probabilities"], float(neg), text)
        rows.append({"score": u["score"], "tier": u["tier"], **u["components"]})
    return pd.DataFrame(rows, index=df.index)


def combine(c: pd.DataFrame, w_model: float, w_sent: float, w_kw: float) -> np.ndarray:
    """Re-weight the components. Defaults in urgency.py correspond to (1, 1, 1)."""
    return np.minimum(100, w_model * c["model"] + w_sent * c["sentiment"] + w_kw * c["keywords"])


def main() -> dict:
    p = settings.paths
    df = pd.read_parquet(p.processed / "tickets.parquet")
    sent = pd.read_parquet(p.processed / "ticket_sentiment.parquet")
    df = df.merge(sent[["ticket_id", "sentiment_negative"]], on="ticket_id")
    val, test = df[df.split == "val"], df[df.split == "test"]
    log.info("scoring %d val + %d test tickets with %s", len(val), len(test), get_classifier().backend)
    cv, ct = components(val), components(test)
    yv, yt = (val["priority"] == "high").to_numpy(), (test["priority"] == "high").to_numpy()

    auroc = {"urgency_score": roc_auc_score(yt, ct["score"])}
    for part in ("model", "sentiment", "keywords"):
        auroc[f"{part}_component_only"] = roc_auc_score(yt, ct[part])
    auroc = {k: round(float(v), 4) for k, v in auroc.items()}

    tiers = []
    for _, tier, sla in SLA_TIERS[::-1]:
        mask = (ct["tier"] == tier).to_numpy()
        tiers.append({"tier": tier, "first_response_sla": sla, "share_of_tickets": round(float(mask.mean()), 4),
                      "human_high_rate": round(float(yt[mask].mean()), 4) if mask.any() else None})

    grid = []
    for wm, ws, wk in itertools.product([1.0], [0, 0.5, 1, 2], [0, 0.5, 1, 2]):
        grid.append({"w_model": wm, "w_sentiment": ws, "w_keywords": wk,
                     "val_auroc": round(float(roc_auc_score(yv, combine(cv, wm, ws, wk))), 4)})
    best = max(grid, key=lambda g: g["val_auroc"])
    tuned_test = roc_auc_score(yt, combine(ct, best["w_model"], best["w_sentiment"], best["w_keywords"]))

    report = {
        "classifier_backend": get_classifier().backend,
        "test_tickets": int(len(test)),
        "human_high_rate": round(float(yt.mean()), 4),
        "auroc_vs_human_high_priority": auroc,
        "tiers": tiers,
        "weight_search": {"grid": grid, "best_on_val": best, "test_auroc_with_best": round(float(tuned_test), 4),
                          "test_auroc_with_default": auroc["urgency_score"]},
    }
    (p.reports / "metrics_urgency.json").write_text(json.dumps(report, indent=2))
    log.info("urgency validation: %s", json.dumps({k: v for k, v in report.items() if k != "weight_search"}))
    log.info("best weights on val %s -> test AUROC %.4f (default %.4f)", best, tuned_test, auroc["urgency_score"])
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    main()
