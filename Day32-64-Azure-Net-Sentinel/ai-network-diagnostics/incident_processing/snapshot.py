"""
Convert anomalous telemetry events into natural-language incident snapshots
suitable for embedding and retrieval.
"""

from __future__ import annotations

from typing import Any, List

from anomaly_detection.detect import AnomalyResult


def event_to_snapshot(result: AnomalyResult) -> str:
    """
    Turn a single anomaly result into a short incident description.
    Used as query-side text for vector search and as context for the LLM.
    """
    ev = result.event
    device = ev.get("device", "unknown device")
    latency = ev.get("latency_ms")
    packet_loss = ev.get("packet_loss_pct")
    queue_depth = ev.get("queue_depth_pct")
    ts = ev.get("timestamp", "")

    parts: List[str] = []
    if "latency_ms" in result.triggered_conditions and latency is not None:
        parts.append(f"High latency detected ({latency} ms) on {device}.")
    if "packet_loss_pct" in result.triggered_conditions and packet_loss is not None:
        parts.append(f"Packet loss increased to {packet_loss}%.")
    if "queue_depth_pct" in result.triggered_conditions and queue_depth is not None:
        parts.append(f"Queue depth reached {queue_depth}%.")
    if not parts:
        parts.append(f"Anomaly on {device} at {ts}.")
    else:
        parts.append("Traffic congestion suspected.")
    return " ".join(parts)


def anomalies_to_snapshot(results: List[AnomalyResult]) -> str:
    """
    Combine multiple anomaly results (e.g. for one device over time) into
    one incident snapshot string.
    """
    if not results:
        return "No anomalies detected."
    if len(results) == 1:
        return event_to_snapshot(results[0])
    parts = [event_to_snapshot(r) for r in results]
    return " ".join(parts)
