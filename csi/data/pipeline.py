"""Ingestion and cleaning pipeline: raw Hugging Face dump -> clean, split, warehouse-ready tickets.

Run:  python -m csi.data.pipeline
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3

import pandas as pd
from ftfy import fix_text
from sklearn.model_selection import train_test_split

from csi.config import settings
from csi.text import redact_pii

log = logging.getLogger("csi.pipeline")

PRIORITY_MAP = {
    "very_low": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "high",
}
TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]


def ingest(force: bool = False) -> pd.DataFrame:
    """Download the public dataset once and cache it as parquet."""
    p = settings.paths
    p.ensure()
    raw_file = p.raw / "tickets_raw.parquet"
    if raw_file.exists() and not force:
        log.info("Using cached raw data at %s", raw_file)
        return pd.read_parquet(raw_file)

    from datasets import load_dataset

    log.info("Downloading %s from Hugging Face...", settings.dataset_id)
    df = load_dataset(settings.dataset_id, split="train").to_pandas()
    df.to_parquet(raw_file, index=False)
    log.info("Saved %d raw rows to %s", len(df), raw_file)
    return df


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = fix_text(text)  # repairs mojibake such as "mÃ¶chte" -> "möchte"
    text = text.replace("\\n", "\n").replace("\\t", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedup_key(text: str) -> str:
    """Normalised fingerprint so trivially different copies collapse together."""
    norm = re.sub(r"[^a-z ]", "", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.md5(norm.encode()).hexdigest()


def _collect_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    for col in TAG_COLUMNS:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            for t in val.split(","):
                t = clean_text(t).strip()
                if t and t not in tags:
                    tags.append(t)
    return tags


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict[str, object] = {"raw_rows": int(len(raw))}
    df = raw.copy()

    df = df[df["language"] == settings.language]
    report["after_language_filter"] = int(len(df))

    df = df[df["queue"].isin(settings.queues)]
    report["after_queue_filter"] = int(len(df))

    for col in ("subject", "body", "answer"):
        df[col] = df[col].map(clean_text)
    report["mojibake_or_escape_fixes"] = int(
        (raw.loc[df.index, "body"].fillna("") != df["body"]).sum()
    )

    df = df[df["body"].str.len() >= 20]
    report["after_empty_or_short_body"] = int(len(df))

    df["priority"] = df["priority"].map(PRIORITY_MAP)
    df = df[df["priority"].notna() & df["type"].isin(settings.ticket_types)]
    report["after_label_validation"] = int(len(df))

    df["text"] = (df["subject"] + "\n" + df["body"]).str.strip()
    df["dedup_key"] = df["text"].map(_dedup_key)

    # Same text with conflicting queue labels is label noise: drop every copy.
    label_counts = df.groupby("dedup_key")["queue"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    report["conflicting_label_rows_dropped"] = int(df["dedup_key"].isin(conflicting).sum())
    df = df[~df["dedup_key"].isin(conflicting)]

    before = len(df)
    df = df.drop_duplicates("dedup_key")
    report["duplicates_dropped"] = int(before - len(df))

    pii_before = df["text"].copy()
    df["text"] = df["text"].map(redact_pii)
    df["answer"] = df["answer"].map(redact_pii)
    report["rows_with_pii_redacted"] = int((pii_before != df["text"]).sum())

    df["tags"] = df.apply(_collect_tags, axis=1)
    df["has_answer"] = df["answer"].str.len() > 0
    df["n_chars"] = df["text"].str.len()

    df = df.reset_index(drop=True)
    df.insert(0, "ticket_id", [f"T{i:06d}" for i in range(len(df))])
    keep = ["ticket_id", "subject", "body", "text", "answer", "queue", "priority", "type",
            "tags", "has_answer", "n_chars"]
    df = df[keep]

    report["final_rows"] = int(len(df))
    report["queue_distribution"] = df["queue"].value_counts().to_dict()
    report["priority_distribution"] = df["priority"].value_counts().to_dict()
    report["type_distribution"] = df["type"].value_counts().to_dict()
    report["imbalance_ratio_max_to_min_queue"] = round(
        df["queue"].value_counts().max() / df["queue"].value_counts().min(), 1
    )
    return df, report


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 80/10/10 split on queue so rare queues appear in every split."""
    train, temp = train_test_split(
        df, test_size=0.2, stratify=df["queue"], random_state=settings.seed
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["queue"], random_state=settings.seed
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def load_warehouse(df: pd.DataFrame) -> None:
    """Load the clean tickets into SQLite so the dashboard can answer questions with SQL."""
    con = sqlite3.connect(settings.paths.warehouse)
    out = df.drop(columns=["tags"]).assign(tags=df["tags"].map(lambda t: ", ".join(t)))
    out.to_sql("tickets", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_tickets_queue ON tickets(queue)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_tickets_split ON tickets(split)")
    con.commit()
    con.close()


def run(force_download: bool = False) -> pd.DataFrame:
    p = settings.paths
    p.ensure()
    raw = ingest(force=force_download)
    df, report = clean(raw)
    df = split(df)
    report["split_sizes"] = df["split"].value_counts().to_dict()

    df.to_parquet(p.processed / "tickets.parquet", index=False)
    load_warehouse(df)
    (p.reports / "data_quality.json").write_text(json.dumps(report, indent=2))
    log.info("Data quality report: %s", json.dumps(report, indent=2))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    run()
