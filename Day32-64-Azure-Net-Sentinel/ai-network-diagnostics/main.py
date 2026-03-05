"""
AI Infrastructure Diagnostic Assistant – agentic demo pipeline.

Flow: load telemetry → detect anomaly → incident snapshot → Investigation Agent
(tools: analyze telemetry, retrieve incidents, retrieve runbooks) → LLM reasoning → diagnostic report.
Run from this directory: python main.py
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import get_settings
from logging_config import log_with_correlation, setup_logging

from anomaly_detection.detect import detect_anomalies_from_stream
from incident_processing.snapshot import anomalies_to_snapshot
from agent.investigation_agent import run_investigation
from rag_engine import DiagnosticReport

logger = logging.getLogger(__name__)


def load_telemetry(events_path: Path) -> Optional[list]:
    """
    Load telemetry events from JSON file.
    Returns list of events, or None on missing file / invalid JSON (errors are logged).
    """
    if not events_path.exists():
        logger.error("Telemetry file not found: %s", events_path)
        return None
    try:
        raw = events_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            logger.error("Telemetry file is not a JSON array: %s", events_path)
            return None
        return data
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in telemetry file %s: %s", events_path, e)
        return None
    except OSError as e:
        logger.error("Failed to read telemetry file %s: %s", events_path, e)
        return None


def run_pipeline(
    events_path: Path,
    correlation_id: Optional[str] = None,
) -> Optional[DiagnosticReport]:
    """
    Load telemetry → detect anomalies → snapshot → run investigation agent → return report.
    """
    events = load_telemetry(events_path)
    if events is None:
        return None
    if not events:
        log_with_correlation(logger, logging.INFO, "No telemetry events in file.", correlation_id)
        return None

    log_with_correlation(logger, logging.INFO, "Telemetry loaded, %d events.", correlation_id, len(events))

    anomalies = detect_anomalies_from_stream(events)
    if not anomalies:
        log_with_correlation(logger, logging.INFO, "No anomalies detected in telemetry stream.", correlation_id)
        return None

    log_with_correlation(logger, logging.INFO, "Anomaly detected, %d anomalous events.", correlation_id, len(anomalies))

    incident_snapshot = anomalies_to_snapshot(anomalies)
    telemetry_events = [a.event for a in anomalies]
    log_with_correlation(logger, logging.INFO, "Incident snapshot created.", correlation_id)

    settings = get_settings()
    report, telemetry_analysis = run_investigation(
        incident_snapshot=incident_snapshot,
        telemetry_events=telemetry_events,
        top_k_incidents=settings.top_k_incidents,
        top_k_runbooks=settings.top_k_runbooks,
        correlation_id=correlation_id,
    )

    log_with_correlation(logger, logging.INFO, "Investigation complete, report generated.", correlation_id)
    print_report(telemetry_analysis, report)
    return report


def print_report(
    telemetry_analysis: str,
    report: DiagnosticReport,
) -> None:
    """Print telemetry summary and AI investigation report to stdout (unchanged format)."""
    print("Incident detected:\n")
    for line in telemetry_analysis.split("\n"):
        if line.strip() and line.strip() != "Telemetry analysis:":
            print(line)
    print()

    print("AI Investigation Report")
    print("-" * 48)
    print()
    print("Possible root causes:")
    for i, cause in enumerate(report.root_causes, 1):
        print(f"{i}. {cause}")
    print()
    print("Evidence:")
    for item in report.evidence:
        print(f"- {item}")
    print()
    print("Suggested mitigation:")
    for step in report.mitigation_steps:
        print(f"- {step}")
    print()
    print("Confidence:", report.confidence)
    print("-" * 48)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    correlation_id = str(uuid.uuid4())[:8]
    logger.info("Starting diagnostic pipeline, correlation_id=%s", correlation_id)

    telemetry_path = settings.telemetry_path_resolved
    report = run_pipeline(telemetry_path, correlation_id=correlation_id)
    if report is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
