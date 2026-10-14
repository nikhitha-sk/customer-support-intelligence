"""Small text utilities shared by the pipeline, the models and the API."""
from __future__ import annotations

import re

_PII_PATTERNS = [
    # Order matters: IPs and card numbers would otherwise be caught by the phone pattern.
    ("<email>", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("<ip>", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("<card>", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("<phone>", re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")),
]


def redact_pii(text: str) -> str:
    """Mask e-mails, card numbers, phone numbers and IPs before text is stored or sent to an LLM."""
    if not text:
        return text
    for token, pattern in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def build_ticket_text(subject: str | None, body: str) -> str:
    return f"{(subject or '').strip()}\n{body.strip()}".strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]
