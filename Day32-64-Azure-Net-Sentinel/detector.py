"""
Real-time anomaly trend detector for live_telemetry.csv.
Uses a 30-second sliding window slope analysis on optical power; triggers early gray failure
alerts when slope < -0.01 for 5 consecutive seconds.
"""

from __future__ import annotations

import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- Configuration ---
CSV_PATH = Path("live_telemetry.csv")
INCIDENTS_LOG_PATH = Path("detected_incidents.log")
WINDOW_SEC = 30
SLOPE_THRESHOLD = -0.01
CONSECUTIVE_SEC_ALERT = 5
REFRESH_INTERVAL_SEC = 1.0

# ANSI red (works on Windows 10+ terminals)
RED = "\033[31m"
RESET = "\033[0m"


def device_key(row: pd.Series) -> tuple[str, str, str]:
    return (str(row["region"]), str(row["dc"]), str(row["device_id"]))
def get_last_processed_count(path: Path) -> int:
    """Return current number of data rows in CSV (excluding header)."""
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # subtract header
def read_new_rows(path: Path, skip_rows: int) -> pd.DataFrame:
    """Read only newly appended, unprocessed rows to avoid memory overflow."""
    if not path.exists():
        return pd.DataFrame()
    try:
        total = get_last_processed_count(path)
        if total <= skip_rows:
            return pd.DataFrame()
        # Skip already processed rows; only read new rows
        df = pd.read_csv(
            path,
            skiprows=range(1, 1 + skip_rows),
            nrows=total - skip_rows,
            dtype={
                "timestamp": np.float64,
                "region": str,
                "dc": str,
                "device_id": str,
                "optical_power_dbm": np.float64,
            },
        )
        return df
    except Exception as e:
        print(f"[WARN] Failed to read CSV: {e}", file=sys.stderr)
        return pd.DataFrame()
def compute_slope(timestamps: np.ndarray, values: np.ndarray) -> float | None:
    """Compute slope via linear regression (optical_power_dbm vs time). Return None if not enough points or degenerate."""
    n = len(timestamps)
    if n < 2:
        return None
    try:
        coefs = np.polyfit(timestamps.astype(float), values.astype(float), 1)
        return float(coefs[0])
    except (np.linalg.LinAlgError, TypeError):
        return None


def trim_window_to_interval(
    window: deque[tuple[float, float]], now: float, interval_sec: float
) -> None:
    """Keep only points within [now - interval_sec, now] in the window."""
    cutoff = now - interval_sec
    while window and window[0][0] < cutoff:
        window.popleft()
def call_rag_agent(incident_data: dict[str, Any]) -> None:
    """Placeholder: send incident context to RAG system for deep diagnosis. Currently only prints a message."""
    print("Sending incident context to RAG system for deep diagnosis...")


def log_incident(
    path: Path,
    timestamp: float,
    device_id: str,
    region: str,
    slope: float,
    optical_power_dbm: float,
) -> None:
    """Append alert evidence into detected_incidents.log."""
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp\tdevice_id\tregion\tslope\toptical_power_dbm\n")
        f.write(
            f"{timestamp}\t{device_id}\t{region}\t{slope:.4f}\t{optical_power_dbm:.4f}\n"
        )


def run_detector() -> None:
    """Main loop: every second read new rows, update windows, compute slopes, and trigger alerts when slope stays below threshold."""
    last_processed = 0
    # device_key -> deque of (timestamp, optical_power_dbm), keeping at most 30 seconds of data
    windows: dict[tuple[str, str, str], deque[tuple[float, float]]] = {}
    # device_key -> how many consecutive seconds slope < SLOPE_THRESHOLD
    slope_streak: dict[tuple[str, str, str], int] = {}
    shutdown = [False]  # use a list so closure can mutate it

    def on_signal(*_args: object) -> None:
        shutdown[0] = True
        print("\nDetector is shutting down...")

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    print(f"Monitoring file: {CSV_PATH.resolve()}")
    print(f"Window size: {WINDOW_SEC}s, slope threshold: {SLOPE_THRESHOLD}, alerts after {CONSECUTIVE_SEC_ALERT} consecutive seconds below threshold")
    print(f"Evidence log: {INCIDENTS_LOG_PATH}, refresh interval: {REFRESH_INTERVAL_SEC}s")
    print("Press Ctrl+C to stop.\n")

    while not shutdown[0]:
        loop_start = time.time()

        df = read_new_rows(CSV_PATH, last_processed)
        if not df.empty:
            last_processed += len(df)
            for _, row in df.iterrows():
                key = device_key(row)
                ts = float(row["timestamp"])
                optical = float(row["optical_power_dbm"])
                if key not in windows:
                    windows[key] = deque()
                    slope_streak[key] = 0
                windows[key].append((ts, optical))

        now = time.time()

        for key, window in list(windows.items()):
            trim_window_to_interval(window, now, WINDOW_SEC)
            if len(window) < 2:
                slope_streak[key] = 0
                continue

            ts_arr = np.array([p[0] for p in window])
            val_arr = np.array([p[1] for p in window])
            slope = compute_slope(ts_arr, val_arr)
            if slope is None:
                slope_streak[key] = 0
                continue

            if slope < SLOPE_THRESHOLD:
                slope_streak[key] = slope_streak.get(key, 0) + 1
                if slope_streak[key] >= CONSECUTIVE_SEC_ALERT:
                    region, dc, device_id = key
                    optical_current = val_arr[-1]
                    msg = (
                        f"[ALERT] Early gray failure detected: device {device_id} in region {region}. "
                        f"Current slope: {slope:.4f}."
                    )
                    print(f"{RED}{msg}{RESET}")
                    log_incident(
                        INCIDENTS_LOG_PATH,
                        now,
                        device_id,
                        region,
                        slope,
                        optical_current,
                    )
                    incident_data = {
                        "timestamp": now,
                        "device_id": device_id,
                        "region": region,
                        "dc": dc,
                        "slope": slope,
                        "optical_power_dbm": optical_current,
                    }
                    call_rag_agent(incident_data)
                    slope_streak[key] = 0
            else:
                slope_streak[key] = 0

        elapsed = time.time() - loop_start
        sleep_time = max(0.0, REFRESH_INTERVAL_SEC - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    if not CSV_PATH.exists():
        print(f"File {CSV_PATH} not found. Please run telemetry_generator.py to generate telemetry first.", file=sys.stderr)
        sys.exit(1)
    try:
        run_detector()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
