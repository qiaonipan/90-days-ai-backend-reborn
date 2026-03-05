"""
Unit tests for anomaly_detection: events above/below thresholds.
Thresholds are fixed via conftest (THRESHOLD_* env vars).
"""

from __future__ import annotations

import pytest

from anomaly_detection.detect import detect_anomaly, detect_anomalies_from_stream


def test_detect_anomaly_below_thresholds_not_anomaly():
    """Event with latency 50, packet_loss 0, queue 50 should not be anomalous."""
    event = {
        "device": "tor-01",
        "latency_ms": 50,
        "packet_loss_pct": 0,
        "queue_depth_pct": 50,
        "timestamp": "2026-03-04T12:00:00",
    }
    result = detect_anomaly(event)
    assert result.is_anomaly is False
    assert result.triggered_conditions == []


def test_detect_anomaly_high_latency_triggered():
    """Latency > 100 triggers anomaly."""
    event = {
        "device": "tor-01",
        "latency_ms": 120,
        "packet_loss_pct": 0,
        "queue_depth_pct": 30,
        "timestamp": "2026-03-04T12:00:00",
    }
    result = detect_anomaly(event)
    assert result.is_anomaly is True
    assert "latency_ms" in result.triggered_conditions


def test_detect_anomaly_packet_loss_triggered():
    """Packet loss > 2% triggers anomaly."""
    event = {
        "device": "tor-01",
        "latency_ms": 50,
        "packet_loss_pct": 3,
        "queue_depth_pct": 30,
        "timestamp": "2026-03-04T12:00:00",
    }
    result = detect_anomaly(event)
    assert result.is_anomaly is True
    assert "packet_loss_pct" in result.triggered_conditions


def test_detect_anomaly_queue_depth_triggered():
    """Queue depth > 80% triggers anomaly."""
    event = {
        "device": "tor-01",
        "latency_ms": 50,
        "packet_loss_pct": 0,
        "queue_depth_pct": 85,
        "timestamp": "2026-03-04T12:00:00",
    }
    result = detect_anomaly(event)
    assert result.is_anomaly is True
    assert "queue_depth_pct" in result.triggered_conditions


def test_detect_anomaly_multiple_conditions():
    """Multiple thresholds exceeded: all conditions in triggered_conditions."""
    event = {
        "device": "tor-01",
        "latency_ms": 120,
        "packet_loss_pct": 5,
        "queue_depth_pct": 90,
        "timestamp": "2026-03-04T12:00:00",
    }
    result = detect_anomaly(event)
    assert result.is_anomaly is True
    assert set(result.triggered_conditions) == {"latency_ms", "packet_loss_pct", "queue_depth_pct"}


def test_detect_anomalies_from_stream_returns_only_anomalous():
    """Stream with mixed events returns only anomalous ones."""
    events = [
        {"device": "tor-01", "latency_ms": 50, "packet_loss_pct": 0, "queue_depth_pct": 30, "timestamp": "T1"},
        {"device": "tor-02", "latency_ms": 150, "packet_loss_pct": 0, "queue_depth_pct": 30, "timestamp": "T2"},
        {"device": "tor-03", "latency_ms": 50, "packet_loss_pct": 1, "queue_depth_pct": 30, "timestamp": "T3"},
    ]
    results = detect_anomalies_from_stream(events)
    assert len(results) == 1
    assert results[0].event["device"] == "tor-02"
    assert "latency_ms" in results[0].triggered_conditions
