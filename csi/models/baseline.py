"""Baseline: TF-IDF (word + char n-grams) + class-weighted Logistic Regression, one model per task.

Run:  python -m csi.models.baseline
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from csi.config import settings
from csi.models.metrics import classification_metrics, coverage_precision_curve

log = logging.getLogger("csi.baseline")

TASKS = {
    "queue": list(settings.queues),
    "priority": list(settings.priorities),
    "type": list(settings.ticket_types),
}


def make_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                 sublinear_tf=True, stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=60000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("tfidf", features), ("clf", clf)])


def run() -> dict:
    p = settings.paths
    df = pd.read_parquet(p.processed / "tickets.parquet")
    train, test = df[df.split == "train"], df[df.split == "test"]

    results: dict[str, dict] = {}
    for task, labels in TASKS.items():
        start = time.time()
        model = make_pipeline().fit(train["text"], train[task])
        proba = model.predict_proba(test["text"])
        pred = model.classes_[proba.argmax(1)]
        m = classification_metrics(test[task], pred, labels)
        m["train_seconds"] = round(time.time() - start, 1)
        if task == "queue":
            m["coverage_precision"] = coverage_precision_curve(test[task], pred, proba.max(1))
        results[task] = m
        joblib.dump(model, p.models / f"baseline_{task}.joblib")
        log.info("baseline %-8s acc=%.3f macroF1=%.3f weightedF1=%.3f (%.0fs)",
                 task, m["accuracy"], m["macro_f1"], m["weighted_f1"], m["train_seconds"])

    (p.reports / "metrics_baseline.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    run()
