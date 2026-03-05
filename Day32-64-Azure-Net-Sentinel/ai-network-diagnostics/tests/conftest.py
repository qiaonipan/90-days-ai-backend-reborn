"""
Pytest configuration and shared fixtures for ai-network-diagnostics.
Ensures project root is on sys.path so imports work when running from any cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root (ai-network-diagnostics)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def pytest_configure(config):
    """Set default env for deterministic tests (no API keys)."""
    import os
    # Ensure thresholds are fixed so anomaly tests are deterministic
    os.environ.setdefault("THRESHOLD_LATENCY_MS", "100")
    os.environ.setdefault("THRESHOLD_PACKET_LOSS_PCT", "2")
    os.environ.setdefault("THRESHOLD_QUEUE_DEPTH_PCT", "80")
