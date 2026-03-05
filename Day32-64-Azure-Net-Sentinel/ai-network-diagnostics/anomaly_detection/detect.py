"""
Lightweight anomaly detector for network telemetry.
Flags events that exceed latency, packet loss, or queue depth thresholds.
Thresholds are read from config (env or .env).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from config import get_settings


@dataclass
class AnomalyResult:
    """Result of anomaly check for a single telemetry event."""
    is_anomaly: bool
    event: dict[str, Any]
    triggered_conditions: List[str]


def detect_anomaly(event: dict[str, Any]) -> AnomalyResult:
    """
    Check if a telemetry event is anomalous based on configurable thresholds.
    Returns which conditions were triggered (latency, packet_loss, queue_depth).
    """
    s = get_settings()
    triggered: List[str] = []
    latency = event.get("latency_ms")
    packet_loss = event.get("packet_loss_pct")
    queue_depth = event.get("queue_depth_pct")

    if latency is not None and latency > s.threshold_latency_ms:
        triggered.append("latency_ms")
    if packet_loss is not None and packet_loss > s.threshold_packet_loss_pct:
        triggered.append("packet_loss_pct")
    if queue_depth is not None and queue_depth > s.threshold_queue_depth_pct:
        triggered.append("queue_depth_pct")

    return AnomalyResult(
        is_anomaly=len(triggered) > 0,
        event=event,
        triggered_conditions=triggered,
    )


def detect_anomalies_from_stream(events: List[dict[str, Any]]) -> List[AnomalyResult]:
    """Run anomaly detection on a list of telemetry events; return only anomalous ones."""
    results: List[AnomalyResult] = []
    for ev in events:
        r = detect_anomaly(ev)
        if r.is_anomaly:
            results.append(r)
    return results
