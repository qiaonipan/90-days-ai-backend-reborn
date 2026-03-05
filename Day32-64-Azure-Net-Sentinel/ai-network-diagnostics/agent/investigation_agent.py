"""
Investigation Agent: orchestrates multi-step diagnosis.
Receives incident snapshot and anomalous telemetry events; runs tools sequentially
then LLM reasoning to produce a diagnostic report.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from rag_engine import (
    ensure_vector_db,
    ensure_runbooks_vector_db,
    format_runbooks_for_prompt,
    agent_reason,
    DiagnosticReport,
)
from tools.analyze_telemetry import analyze_telemetry
from tools.retrieve_incidents import format_incidents_for_prompt, retrieve_similar_incidents
from tools.retrieve_runbooks import retrieve_relevant_runbooks

from logging_config import log_with_correlation

logger = logging.getLogger(__name__)


def run_investigation(
    incident_snapshot: str,
    telemetry_events: List[dict[str, Any]],
    top_k_incidents: int = 5,
    top_k_runbooks: int = 5,
    correlation_id: Optional[str] = None,
) -> tuple[DiagnosticReport, str]:
    """
    Execute the agent workflow:
    Step 1: Analyze telemetry
    Step 2: Retrieve similar incidents
    Step 3: Retrieve relevant runbooks
    Step 4: LLM reasoning
    Step 5: Return diagnostic report and telemetry analysis text (for printing).
    """
    # Ensure vector stores exist
    ensure_vector_db()
    ensure_runbooks_vector_db()

    # Step 1: Analyze telemetry patterns
    log_with_correlation(logger, logging.DEBUG, "Analyzing telemetry patterns.", correlation_id)
    telemetry_analysis = analyze_telemetry(telemetry_events)

    # Step 2: Retrieve similar historical incidents
    incident_results = retrieve_similar_incidents(incident_snapshot, top_k=top_k_incidents)
    incident_results_text = format_incidents_for_prompt(incident_results)
    log_with_correlation(
        logger, logging.INFO,
        "Retrieval: %d similar incidents.",
        correlation_id, len(incident_results),
    )

    # Step 3: Retrieve relevant runbooks
    runbook_results = retrieve_relevant_runbooks(incident_snapshot, top_k=top_k_runbooks)
    runbook_results_text = format_runbooks_for_prompt(runbook_results)
    log_with_correlation(
        logger, logging.INFO,
        "Retrieval: %d relevant runbooks.",
        correlation_id, len(runbook_results),
    )

    # Step 4: LLM reasoning
    log_with_correlation(logger, logging.DEBUG, "Running LLM reasoning.", correlation_id)
    report = agent_reason(
        incident_snapshot=incident_snapshot,
        telemetry_analysis=telemetry_analysis,
        incident_results=incident_results_text,
        runbook_results=runbook_results_text,
    )

    return report, telemetry_analysis
