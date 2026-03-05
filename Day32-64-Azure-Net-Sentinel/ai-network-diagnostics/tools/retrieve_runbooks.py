"""
Runbook retrieval tool.
Uses the RAG engine to return top-k relevant runbooks for an incident snapshot.
"""

from __future__ import annotations

from typing import Any, List

from rag_engine import retrieve_runbooks


def retrieve_relevant_runbooks(incident_snapshot: str, top_k: int = 5) -> List[dict[str, Any]]:
    """Return top_k relevant runbooks with similarity scores."""
    return retrieve_runbooks(incident_snapshot, top_k=top_k)
