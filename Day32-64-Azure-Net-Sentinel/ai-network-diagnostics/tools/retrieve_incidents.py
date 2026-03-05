"""
Incident retrieval tool.
Wraps the RAG engine to return top-k similar historical incidents for an incident snapshot.
"""

from __future__ import annotations

from typing import Any, List

from rag_engine import format_retrieved_for_prompt, retrieve


def retrieve_similar_incidents(incident_snapshot: str, top_k: int = 5) -> List[dict[str, Any]]:
    """Return top_k similar historical incidents with similarity scores."""
    return retrieve(incident_snapshot, top_k=top_k)


def format_incidents_for_prompt(results: List[dict[str, Any]]) -> str:
    """Format retrieved incidents as a single string for the LLM prompt."""
    return format_retrieved_for_prompt(results)
