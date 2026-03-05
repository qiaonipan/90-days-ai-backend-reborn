"""
Unit tests for tools.analyze_telemetry: trend detection (increasing latency, packet loss, queue depth).
"""

from __future__ import annotations

import pytest

from tools.analyze_telemetry import analyze_telemetry


def test_analyze_telemetry_empty_events():
    """Empty events returns fixed message."""
    out = analyze_telemetry([])
    assert "No telemetry events provided" in out


def test_analyze_telemetry_increasing_latency():
    """Events with increasing latency report 'increasing' and congestion message."""
    events = [
        {"device": "tor-01", "latency_ms": 80, "packet_loss_pct": 0, "queue_depth_pct": 40, "timestamp": "T1"},
        {"device": "tor-01", "latency_ms": 120, "packet_loss_pct": 0, "queue_depth_pct": 40, "timestamp": "T2"},
    ]
    out = analyze_telemetry(events)
    assert "Latency trend: increasing" in out
    assert "80" in out and "120" in out
    assert "network congestion or buffer pressure" in out


def test_analyze_telemetry_increasing_packet_loss():
    """Events with increasing packet loss report 'increasing' for packet loss."""
    events = [
        {"device": "tor-01", "latency_ms": 50, "packet_loss_pct": 1, "queue_depth_pct": 50, "timestamp": "T1"},
        {"device": "tor-01", "latency_ms": 50, "packet_loss_pct": 5, "queue_depth_pct": 50, "timestamp": "T2"},
    ]
    out = analyze_telemetry(events)
    assert "Packet loss trend: increasing" in out
    assert "1.0" in out or "1%" in out
    assert "5.0" in out or "5%" in out
    assert "network congestion or buffer pressure" in out


def test_analyze_telemetry_increasing_queue_depth():
    """Events with increasing queue depth report 'increasing' for queue depth."""
    events = [
        {"device": "tor-01", "latency_ms": 50, "packet_loss_pct": 0, "queue_depth_pct": 70, "timestamp": "T1"},
        {"device": "tor-01", "latency_ms": 50, "packet_loss_pct": 0, "queue_depth_pct": 90, "timestamp": "T2"},
    ]
    out = analyze_telemetry(events)
    assert "Queue depth trend: increasing" in out
    assert "70" in out and "90" in out
    assert "network congestion or buffer pressure" in out


def test_analyze_telemetry_stable_trends():
    """Stable or decreasing trends do not suggest congestion."""
    events = [
        {"device": "tor-01", "latency_ms": 50, "packet_loss_pct": 0, "queue_depth_pct": 30, "timestamp": "T1"},
        {"device": "tor-01", "latency_ms": 50, "packet_loss_pct": 0, "queue_depth_pct": 30, "timestamp": "T2"},
    ]
    out = analyze_telemetry(events)
    assert "Latency trend: stable" in out or "no data" in out or "50" in out
    assert "No strong upward trends" in out
