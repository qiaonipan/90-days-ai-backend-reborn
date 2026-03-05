"""
Semantic search over the historical-incident vector store.
Loads embeddings and metadata, embeds the query, returns top-k similar incidents.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_engine.embed import (
    EMBEDDINGS_FILE,
    METADATA_FILE,
    MODEL_NAME,
    RUNBOOKS_EMBEDDINGS_FILE,
    RUNBOOKS_METADATA_FILE,
    VECTOR_DB_DIR,
)

logger = logging.getLogger(__name__)


def _load_store():
    """Load embeddings and metadata from vector_db/."""
    emb_path = VECTOR_DB_DIR / EMBEDDINGS_FILE
    meta_path = VECTOR_DB_DIR / METADATA_FILE
    if not emb_path.exists() or not meta_path.exists():
        logger.warning("Incident vector store missing or incomplete (no embeddings or metadata)")
        return None, []
    embeddings = np.load(emb_path)
    metadata = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    return embeddings, metadata


def retrieve(query: str, top_k: int = 5) -> List[dict[str, Any]]:
    """
    Embed the query (incident snapshot), compute similarity to stored incidents,
    return top_k results with text, symptom, root_cause, mitigation, and score.
    """
    embeddings, metadata = _load_store()
    if embeddings is None or embeddings.size == 0 or not metadata:
        logger.debug("Retrieving incidents: empty store, returning no results")
        return []

    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Cosine similarity (vectors already normalized) = dot product
    scores = np.dot(embeddings, q_vec.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        rec = metadata[idx].copy()
        rec["score"] = float(scores[idx])
        results.append(rec)
    return results


def format_retrieved_for_prompt(results: List[dict[str, Any]]) -> str:
    """Format retrieved incidents as a single string for the LLM prompt."""
    if not results:
        return "No similar historical incidents found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. Symptom: {r.get('symptom', '')}\n"
            f"   Root cause: {r.get('root_cause', '')}\n"
            f"   Mitigation: {r.get('mitigation', '')}"
        )
    return "\n\n".join(lines)


def _load_runbooks_store():
    """Load runbook embeddings and metadata from vector_db/."""
    emb_path = VECTOR_DB_DIR / RUNBOOKS_EMBEDDINGS_FILE
    meta_path = VECTOR_DB_DIR / RUNBOOKS_METADATA_FILE
    if not emb_path.exists() or not meta_path.exists():
        logger.warning("Runbook vector store missing or incomplete (no embeddings or metadata)")
        return None, []
    embeddings = np.load(emb_path)
    metadata = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    return embeddings, metadata


def retrieve_runbooks(query: str, top_k: int = 5) -> List[dict[str, Any]]:
    """Embed query, search runbook store, return top_k runbooks with score."""
    embeddings, metadata = _load_runbooks_store()
    if embeddings is None or embeddings.size == 0 or not metadata:
        logger.debug("Retrieving runbooks: empty store, returning no results")
        return []

    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    scores = np.dot(embeddings, q_vec.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        rec = metadata[idx].copy()
        rec["score"] = float(scores[idx])
        results.append(rec)
    return results


def format_runbooks_for_prompt(results: List[dict[str, Any]]) -> str:
    """Format retrieved runbooks for the LLM prompt."""
    if not results:
        return "No relevant runbooks found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. Symptom: {r.get('symptom', '')}\n"
            f"   Possible cause: {r.get('possible_cause', '')}\n"
            f"   Mitigation: {r.get('mitigation', '')}"
        )
    return "\n\n".join(lines)
