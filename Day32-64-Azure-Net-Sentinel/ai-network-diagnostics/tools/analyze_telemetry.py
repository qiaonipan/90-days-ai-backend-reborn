"""
Telemetry pattern analysis tool.
Analyzes trends (latency, packet loss, queue depth) from a list of telemetry events.
"""

from __future__ import annotations

from typing import Any, List


def _trend(values: List[float]) -> str:
    """Return 'increasing', 'decreasing', or 'stable' from a list of numbers."""
    if not values or len(values) < 2:
        return "stable"
    first, last = values[0], values[-1]
    if last > first:
        return "increasing"
    if last < first:
        return "decreasing"
    return "stable"


def _extract_series(events: List[dict[str, Any]], key: str) -> List[float]:
    """Extract numeric series from events; skip None/missing."""
    out = []
    for ev in events:
        v = ev.get(key)
        if v is not None:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
    return out


def analyze_telemetry(events: List[dict[str, Any]]) -> str:
    """
    Analyze latency, packet loss, and queue depth trends from telemetry events.
    Returns a multi-line report suitable for the LLM prompt.
    """
    if not events:
        return "No telemetry events provided."

    latency_series = _extract_series(events, "latency_ms")
    packet_loss_series = _extract_series(events, "packet_loss_pct")
    queue_depth_series = _extract_series(events, "queue_depth_pct")

    lines = ["Telemetry analysis:", ""]

    if latency_series:
        trend = _trend(latency_series)
        lo, hi = min(latency_series), max(latency_series)
        lines.append(f"Latency trend: {trend} ({lo:.0f} -> {hi:.0f} ms)")
    else:
        lines.append("Latency trend: no data")

    if packet_loss_series:
        trend = _trend(packet_loss_series)
        lo, hi = min(packet_loss_series), max(packet_loss_series)
        lines.append(f"Packet loss trend: {trend} ({lo:.1f}% -> {hi:.1f}%)")
    else:
        lines.append("Packet loss trend: no data")

    if queue_depth_series:
        trend = _trend(queue_depth_series)
        lo, hi = min(queue_depth_series), max(queue_depth_series)
        lines.append(f"Queue depth trend: {trend} ({lo:.0f}% -> {hi:.0f}%)")
    else:
        lines.append("Queue depth trend: no data")

    # Simple heuristic summary
    trends = []
    if latency_series and _trend(latency_series) == "increasing":
        trends.append("latency increasing")
    if packet_loss_series and _trend(packet_loss_series) == "increasing":
        trends.append("packet loss increasing")
    if queue_depth_series and _trend(queue_depth_series) == "increasing":
        trends.append("queue depth increasing")

    lines.append("")
    if trends:
        lines.append("This pattern suggests network congestion or buffer pressure.")
    else:
        lines.append("No strong upward trends in key metrics.")

    return "\n".join(lines)
