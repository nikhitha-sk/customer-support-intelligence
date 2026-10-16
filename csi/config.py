"""Central configuration. Every setting can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    knowledge_base: Path = ROOT / "data" / "knowledge_base"
    artifacts: Path = ROOT / "artifacts"
    models: Path = ROOT / "artifacts" / "models"
    index: Path = ROOT / "artifacts" / "index"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"
    warehouse: Path = ROOT / "artifacts" / "support.db"

    def ensure(self) -> None:
        for p in (self.raw, self.processed, self.models, self.index, self.figures):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    dataset_id: str = _env("CSI_DATASET", "Tobi-Bueck/customer-support-tickets")
    language: str = "en"
    # The 10 operational support queues. The dataset also contains ~40 thin
    # "web category" labels from a different labelling scheme; we drop those.
    queues: tuple[str, ...] = (
        "Technical Support",
        "Product Support",
        "Customer Service",
        "IT Support",
        "Billing and Payments",
        "Returns and Exchanges",
        "Service Outages and Maintenance",
        "Sales and Pre-Sales",
        "Human Resources",
        "General Inquiry",
    )
    priorities: tuple[str, ...] = ("low", "medium", "high")
    ticket_types: tuple[str, ...] = ("Incident", "Request", "Problem", "Change")
    seed: int = 42

    # Models
    encoder_name: str = _env("CSI_ENCODER", "distilbert-base-uncased")
    embedding_model: str = _env("CSI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    sentiment_model: str = _env(
        "CSI_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    max_length: int = int(_env("CSI_MAX_LENGTH", "128"))

    # RAG
    chunk_size: int = int(_env("CSI_CHUNK_SIZE", "600"))
    chunk_overlap: int = int(_env("CSI_CHUNK_OVERLAP", "100"))
    top_k: int = int(_env("CSI_TOP_K", "4"))

    # Agent guardrails
    route_confidence_threshold: float = float(_env("CSI_ROUTE_THRESHOLD", "0.60"))
    escalation_urgency: int = int(_env("CSI_ESCALATION_URGENCY", "80"))

    # LLM provider: "auto" tries Groq, then Ollama, then the offline fallback.
    llm_provider: str = _env("CSI_LLM_PROVIDER", "auto")
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("CSI_GROQ_MODEL", "llama-3.1-8b-instant")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = _env("CSI_OLLAMA_MODEL", "gemma2:9b")
    llm_timeout: float = float(_env("CSI_LLM_TIMEOUT", "120"))

    api_url: str = _env("CSI_API_URL", "http://localhost:8000")
    # API protection: empty key = open API (local demo); rate limit 0 = unlimited.
    api_key: str = _env("CSI_API_KEY", "")
    rate_limit_per_minute: int = int(_env("CSI_RATE_LIMIT", "60"))
    paths: Paths = field(default_factory=Paths)


settings = Settings()
