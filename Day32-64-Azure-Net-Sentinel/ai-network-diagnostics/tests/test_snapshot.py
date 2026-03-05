"""
Unit tests for incident_processing.snapshot: expected snapshot string content.
"""

from __future__ import annotations

from anomaly_detection.detect import AnomalyResult
from incident_processing.snapshot import event_to_snapshot, anomalies_to_snapshot


def test_event_to_snapshot_contains_device_and_latency():
    """Single anomaly with high latency produces snapshot with device and latency."""
    result = AnomalyResult(
        is_anomaly=True,
        event={
            "device": "tor-switch-01",
            "latency_ms": 120,
            "packet_loss_pct": 0,
            "queue_depth_pct": 30,
            "timestamp": "2026-03-04T12:00:00",
        },
        triggered_conditions=["latency_ms"],
    )
    snapshot = event_to_snapshot(result)
    assert "tor-switch-01" in snapshot
    assert "120" in snapshot
    assert "High latency" in snapshot
    assert "Traffic congestion suspected" in snapshot


def test_event_to_snapshot_packet_loss_and_queue():
    """Triggered packet_loss and queue_depth appear in snapshot."""
    result = AnomalyResult(
        is_anomaly=True,
        event={
            "device": "leaf-02",
            "latency_ms": 50,
            "packet_loss_pct": 5,
            "queue_depth_pct": 88,
            "timestamp": "2026-03-04T12:00:00",
        },
        triggered_conditions=["packet_loss_pct", "queue_depth_pct"],
    )
    snapshot = event_to_snapshot(result)
    assert "Packet loss" in snapshot
    assert "5%" in snapshot
    assert "Queue depth" in snapshot
    assert "88" in snapshot
    assert "Traffic congestion suspected" in snapshot


def test_anomalies_to_snapshot_empty_returns_no_anomalies():
    """Empty list returns fixed string."""
    assert anomalies_to_snapshot([]) == "No anomalies detected."


def test_anomalies_to_snapshot_single_same_as_event_to_snapshot():
    """Single result: same as event_to_snapshot for that result."""
    result = AnomalyResult(
        is_anomaly=True,
        event={"device": "tor-01", "latency_ms": 110, "packet_loss_pct": 0, "queue_depth_pct": 40, "timestamp": ""},
        triggered_conditions=["latency_ms"],
    )
    single = anomalies_to_snapshot([result])
    direct = event_to_snapshot(result)
    assert single == direct
    assert "tor-01" in single and "110" in single


def test_anomalies_to_snapshot_multiple_combined():
    """Multiple results are combined into one string."""
    results = [
        AnomalyResult(True, {"device": "tor-01", "latency_ms": 120, "packet_loss_pct": 0, "queue_depth_pct": 50, "timestamp": ""}, ["latency_ms"]),
        AnomalyResult(True, {"device": "tor-01", "latency_ms": 130, "packet_loss_pct": 3, "queue_depth_pct": 85, "timestamp": ""}, ["latency_ms", "packet_loss_pct", "queue_depth_pct"]),
    ]
    snapshot = anomalies_to_snapshot(results)
    assert "tor-01" in snapshot
    assert "120" in snapshot
    assert "130" in snapshot
    assert "Traffic congestion suspected" in snapshot
