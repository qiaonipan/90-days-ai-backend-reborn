"""
Embedding and vector store build for the diagnostic RAG pipeline.
Uses sentence-transformers; persists embeddings and metadata under vector_db/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import get_settings

logger = logging.getLogger(__name__)

# Paths from config (resolved at call time)
def _paths() -> tuple[Path, Path, Path]:
    s = get_settings()
    return (
        s.vector_db_dir_resolved,
        s.historical_incidents_path_resolved,
        s.runbooks_path_resolved,
    )

EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.jsonl"
RUNBOOKS_EMBEDDINGS_FILE = "runbooks_embeddings.npy"
RUNBOOKS_METADATA_FILE = "runbooks_metadata.jsonl"


def _incident_to_text(incident: dict[str, Any]) -> str:
    """Single searchable string for one historical incident."""
    return (
        f"Symptom: {incident.get('symptom', '')} "
        f"Root cause: {incident.get('root_cause', '')} "
        f"Mitigation: {incident.get('mitigation', '')}"
    ).strip()


def load_historical_incidents() -> List[dict[str, Any]]:
    """Load historical incidents from knowledge_base/historical_incidents.json."""
    _, hist_path, _ = _paths()
    if not hist_path.exists():
        logger.warning("Historical incidents file not found: %s", hist_path)
        return []
    try:
        raw = hist_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            logger.warning("Historical incidents file is not a JSON array: %s", hist_path)
            return []
        return data
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in historical incidents file %s: %s", hist_path, e)
        return []


def build_vector_db() -> None:
    """
    Load historical incidents, embed each as one vector, persist to vector_db/.
    Overwrites existing embeddings and metadata.
    """
    s = get_settings()
    VECTOR_DB_DIR = s.vector_db_dir_resolved
    model_name = s.embedding_model_name

    incidents = load_historical_incidents()
    if not incidents:
        logger.warning("No historical incidents to embed; creating empty incident vector store")
        VECTOR_DB_DIR.mkdir(exist_ok=True)
        np.save(VECTOR_DB_DIR / EMBEDDINGS_FILE, np.empty((0, 384), dtype="float32"))
        (VECTOR_DB_DIR / METADATA_FILE).write_text("")
        return

    texts = [_incident_to_text(inc) for inc in incidents]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    VECTOR_DB_DIR.mkdir(exist_ok=True)
    np.save(VECTOR_DB_DIR / EMBEDDINGS_FILE, embeddings)
    with (VECTOR_DB_DIR / METADATA_FILE).open("w", encoding="utf-8") as f:
        for i, inc in enumerate(incidents):
            rec = {
                "id": i,
                "text": texts[i],
                "symptom": inc.get("symptom", ""),
                "root_cause": inc.get("root_cause", ""),
                "mitigation": inc.get("mitigation", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Built incident vector store with %d entries", len(incidents))


def ensure_vector_db() -> None:
    """Build vector_db if it does not exist or is empty."""
    s = get_settings()
    VECTOR_DB_DIR = s.vector_db_dir_resolved
    emb_path = VECTOR_DB_DIR / EMBEDDINGS_FILE
    if not emb_path.exists():
        build_vector_db()
        return
    emb = np.load(emb_path)
    if emb.size == 0:
        build_vector_db()


def _runbook_to_text(runbook: dict[str, Any]) -> str:
    """Single searchable string for one runbook."""
    return (
        f"Symptom: {runbook.get('symptom', '')} "
        f"Possible cause: {runbook.get('possible_cause', '')} "
        f"Mitigation: {runbook.get('mitigation', '')}"
    ).strip()


def load_runbooks() -> List[dict[str, Any]]:
    """Load runbooks from knowledge_base/runbooks.json."""
    _, _, runbooks_path = _paths()
    if not runbooks_path.exists():
        logger.warning("Runbooks file not found: %s", runbooks_path)
        return []
    try:
        raw = runbooks_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            logger.warning("Runbooks file is not a JSON array: %s", runbooks_path)
            return []
        return data
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in runbooks file %s: %s", runbooks_path, e)
        return []


def build_runbooks_vector_db() -> None:
    """Load runbooks, embed each, persist to vector_db/. Overwrites existing."""
    s = get_settings()
    VECTOR_DB_DIR = s.vector_db_dir_resolved
    model_name = s.embedding_model_name

    runbooks = load_runbooks()
    if not runbooks:
        logger.warning("No runbooks to embed; creating empty runbook vector store")
        VECTOR_DB_DIR.mkdir(exist_ok=True)
        np.save(VECTOR_DB_DIR / RUNBOOKS_EMBEDDINGS_FILE, np.empty((0, 384), dtype="float32"))
        (VECTOR_DB_DIR / RUNBOOKS_METADATA_FILE).write_text("")
        return

    texts = [_runbook_to_text(rb) for rb in runbooks]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    VECTOR_DB_DIR.mkdir(exist_ok=True)
    np.save(VECTOR_DB_DIR / RUNBOOKS_EMBEDDINGS_FILE, embeddings)
    with (VECTOR_DB_DIR / RUNBOOKS_METADATA_FILE).open("w", encoding="utf-8") as f:
        for i, rb in enumerate(runbooks):
            rec = {
                "id": i,
                "text": texts[i],
                "symptom": rb.get("symptom", ""),
                "possible_cause": rb.get("possible_cause", ""),
                "mitigation": rb.get("mitigation", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Built runbook vector store with %d entries", len(runbooks))


def ensure_runbooks_vector_db() -> None:
    """Build runbooks vector_db if it does not exist or is empty."""
    s = get_settings()
    VECTOR_DB_DIR = s.vector_db_dir_resolved
    emb_path = VECTOR_DB_DIR / RUNBOOKS_EMBEDDINGS_FILE
    if not emb_path.exists():
        build_runbooks_vector_db()
        return
    emb = np.load(emb_path)
    if emb.size == 0:
        build_runbooks_vector_db()


# Expose for retrieve.py
VECTOR_DB_DIR = get_settings().vector_db_dir_resolved
MODEL_NAME = get_settings().embedding_model_name
