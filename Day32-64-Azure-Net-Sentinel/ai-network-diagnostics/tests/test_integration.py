"""
Integration test: run pipeline end-to-end with small telemetry fixture.
LLM is mocked so no API key is required; tests are deterministic.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

# Import after conftest has added ROOT to path
from anomaly_detection.detect import detect_anomalies_from_stream
from incident_processing.snapshot import anomalies_to_snapshot
from rag_engine import DiagnosticReport, retrieve
from main import load_telemetry, print_report, run_pipeline

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TELEMETRY_SMALL = FIXTURES_DIR / "telemetry_small.json"


@pytest.fixture
def fixed_report() -> DiagnosticReport:
    """Deterministic report for mocked LLM (no API key)."""
    return DiagnosticReport(
        root_causes=["Switch buffer overflow (mocked)"],
        evidence=["high queue depth", "packet loss spike"],
        mitigation_steps=["Drain traffic", "Reset switch", "Verify ECMP routes"],
        confidence="medium",
    )


def test_snapshot_from_fixture_not_empty():
    """Load fixture, detect anomalies, build snapshot -> snapshot not empty."""
    events = load_telemetry(TELEMETRY_SMALL)
    assert events is not None
    assert len(events) >= 1
    anomalies = detect_anomalies_from_stream(events)
    assert len(anomalies) >= 1
    snapshot = anomalies_to_snapshot(anomalies)
    assert snapshot
    assert "tor-switch-01" in snapshot or "High latency" in snapshot or "Packet loss" in snapshot


def test_retrieval_returns_at_most_top_k():
    """Retrieval returns <= top_k items (vector store must exist from prior run)."""
    events = load_telemetry(TELEMETRY_SMALL)
    assert events is not None
    anomalies = detect_anomalies_from_stream(events)
    snapshot = anomalies_to_snapshot(anomalies)
    assert snapshot
    top_k = 5
    results = retrieve(snapshot, top_k=top_k)
    assert len(results) <= top_k


def test_pipeline_end_to_end_mocked_llm(fixed_report: DiagnosticReport):
    """Run full pipeline with mocked LLM; assert report structure and printed output."""
    with patch("agent.investigation_agent.agent_reason", return_value=fixed_report):
        report = run_pipeline(TELEMETRY_SMALL, correlation_id="test-cid")
    assert report is not None
    assert len(report.root_causes) >= 1
    assert len(report.mitigation_steps) >= 1
    assert "Possible root cause" in report.root_causes[0] or "mocked" in report.root_causes[0]
    assert report.confidence in ("low", "medium", "high")

    # Printed output contains section headers
    buf = io.StringIO()
    # Reuse same telemetry_analysis shape (minimal for assertion)
    telemetry_analysis = "Latency trend: increasing (120 -> 135 ms)\n\n"
    import sys
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        print_report(telemetry_analysis, report)
    finally:
        sys.stdout = old_stdout
    out = buf.getvalue()
    assert "Possible root cause" in out
    assert "Suggested mitigation" in out
