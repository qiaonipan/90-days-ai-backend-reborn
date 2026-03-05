"""
Central configuration for the AI Infrastructure Diagnostic pipeline.
Reads from environment and optional .env file. Paths are resolved relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Root directory of the ai-network-diagnostics package
ROOT_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str) -> str:
    if not value:
        return value
    p = Path(value)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return str(p.resolve())


def _get_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


class Settings:
    """Application settings from environment and .env. Access via get_settings()."""

    def __init__(self) -> None:
        self.telemetry_path: str = _resolve_path(_get_str("TELEMETRY_PATH", "telemetry/telemetry_events.json"))
        self.historical_incidents_path: str = _resolve_path(
            _get_str("HISTORICAL_INCIDENTS_PATH", "knowledge_base/historical_incidents.json")
        )
        self.runbooks_path: str = _resolve_path(_get_str("RUNBOOKS_PATH", "knowledge_base/runbooks.json"))
        self.vector_db_dir: str = _resolve_path(_get_str("VECTOR_DB_DIR", "vector_db"))

        self.top_k_incidents: int = _get_int("TOP_K_INCIDENTS", 5)
        self.top_k_runbooks: int = _get_int("TOP_K_RUNBOOKS", 5)

        self.threshold_latency_ms: float = _get_float("THRESHOLD_LATENCY_MS", 100.0)
        self.threshold_packet_loss_pct: float = _get_float("THRESHOLD_PACKET_LOSS_PCT", 2.0)
        self.threshold_queue_depth_pct: float = _get_float("THRESHOLD_QUEUE_DEPTH_PCT", 80.0)

        self.embedding_model_name: str = _get_str("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.llm_model_name: str = _get_str("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY") or None
        if self.openai_api_key:
            self.openai_api_key = self.openai_api_key.strip()

        self.log_level: str = _get_str("LOG_LEVEL", "INFO").upper() or "INFO"

    @property
    def telemetry_path_resolved(self) -> Path:
        return Path(self.telemetry_path)

    @property
    def historical_incidents_path_resolved(self) -> Path:
        return Path(self.historical_incidents_path)

    @property
    def runbooks_path_resolved(self) -> Path:
        return Path(self.runbooks_path)

    @property
    def vector_db_dir_resolved(self) -> Path:
        return Path(self.vector_db_dir)


def get_settings() -> Settings:
    """Return the singleton settings instance."""
    return _settings


_settings: Settings = Settings()
