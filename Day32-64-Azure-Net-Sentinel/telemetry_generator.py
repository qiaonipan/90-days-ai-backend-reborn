"""
Azure network telemetry generator with automated Gray Failure injection.
Simulates 2 regions × 2 DCs × 5 devices, 1s telemetry interval, 30s failure lottery.
"""

from __future__ import annotations

import random
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- Constants ---
REGIONS = ["eastus", "westus"]
DCS_PER_REGION = 2
DEVICES_PER_DC = 5
DEVICE_ROLE = "Leaf"
INTERFACE = "Eth1/1"

TELEMETRY_INTERVAL_SEC = 1.0
FAILURE_CHECK_INTERVAL_SEC = 30.0
FAILURE_PROBABILITY = 0.20
GRAY_FAILURE_DURATION_SEC = 300

# Normal state
OPTICAL_POWER_MEAN = -8.0
OPTICAL_POWER_STD = 0.05
ASIC_TEMP_MEAN = 55.0
ASIC_TEMP_STD = 2.0
UTIL_MIN, UTIL_MAX = 30.0, 50.0

# Gray failure
OPTICAL_DROP_PER_SEC = 0.02
OPTICAL_FLOOR = -11.0
ERROR_COUNT_MIN, ERROR_COUNT_MAX = 1, 10

CSV_PATH = Path("live_telemetry.csv")
GROUND_TRUTH_PATH = Path("ground_truth.log")

# Thread-safe state
_lock = threading.Lock()
_shutdown = False
# device_key -> (start_ts, initial_optical, cumulative_error_count) for active failures
_active_failures: dict[str, tuple[float, float, int]] = {}


def build_topology() -> list[dict]:
    """Build list of device descriptors: region, dc, device_id."""
    devices = []
    for r in REGIONS:
        for dc_idx in range(1, DCS_PER_REGION + 1):
            dc = f"dc{dc_idx}"
            for leaf_idx in range(1, DEVICES_PER_DC + 1):
                devices.append({
                    "region": r,
                    "dc": dc,
                    "device_id": f"Leaf-{leaf_idx:02d}",
                })
    return devices


def device_key(region: str, dc: str, device_id: str) -> str:
    return f"{region}|{dc}|{device_id}"


def get_optical_and_errors(
    key: str, now: float
) -> tuple[float, int]:
    """Return (optical_power_dbm, error_count) for a device; applies failure logic if active."""
    with _lock:
        entry = _active_failures.get(key)
    if not entry:
        return (float(np.random.normal(OPTICAL_POWER_MEAN, OPTICAL_POWER_STD)), 0)

    start_ts, initial_optical, cumulative_errors = entry
    elapsed = now - start_ts
    if elapsed >= GRAY_FAILURE_DURATION_SEC:
        with _lock:
            _active_failures.pop(key, None)
        return (float(np.random.normal(OPTICAL_POWER_MEAN, OPTICAL_POWER_STD)), 0)

    optical = initial_optical - OPTICAL_DROP_PER_SEC * elapsed
    optical = max(optical, OPTICAL_FLOOR)

    if optical <= OPTICAL_FLOOR:
        err_delta = random.randint(ERROR_COUNT_MIN, ERROR_COUNT_MAX)
        with _lock:
            if key in _active_failures:
                t = _active_failures[key]
                _active_failures[key] = (t[0], t[1], t[2] + err_delta)
                cumulative_errors = _active_failures[key][2]
            else:
                cumulative_errors = err_delta
        return (OPTICAL_FLOOR, cumulative_errors)
    return (round(optical, 4), 0)


def generate_row(dev: dict, now: float) -> dict:
    """Generate one telemetry row for a device."""
    key = device_key(dev["region"], dev["dc"], dev["device_id"])
    optical, err_count = get_optical_and_errors(key, now)

    return {
        "timestamp": now,
        "region": dev["region"],
        "dc": dev["dc"],
        "device_id": dev["device_id"],
        "device_role": DEVICE_ROLE,
        "interface": INTERFACE,
        "optical_power_dbm": optical,
        "error_count": err_count,
        "asic_temp": round(float(np.random.normal(ASIC_TEMP_MEAN, ASIC_TEMP_STD)), 2),
        "utilization_pct": round(random.uniform(UTIL_MIN, UTIL_MAX), 2),
    }


def append_telemetry_csv(rows: list[dict], path: Path = CSV_PATH) -> None:
    """Append rows to CSV; write header only if file is new."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        float_format="%.4f",
    )


def log_ground_truth(ts: float, device_id: str, region: str, dc: str, failure_type: str) -> None:
    """Append one line to ground_truth.log."""
    with open(GROUND_TRUTH_PATH, "a", encoding="utf-8") as f:
        f.write(f"{ts}\t{device_id}\t{region}\t{dc}\t{failure_type}\n")


def failure_scheduler(devices: list[dict]) -> None:
    """Every FAILURE_CHECK_INTERVAL_SEC seconds, 20% chance to start a Gray Failure on one random device."""
    global _shutdown
    while not _shutdown:
        time.sleep(FAILURE_CHECK_INTERVAL_SEC)
        if _shutdown:
            break
        if random.random() > FAILURE_PROBABILITY:
            continue

        # Pick a device that is not already in failure
        with _lock:
            busy = set(_active_failures.keys())
        candidates = [
            d for d in devices
            if device_key(d["region"], d["dc"], d["device_id"]) not in busy
        ]
        if not candidates:
            continue

        dev = random.choice(candidates)
        key = device_key(dev["region"], dev["dc"], dev["device_id"])
        now = time.time()
        # Initial optical at start of failure (still normal range)
        initial_optical = float(np.random.normal(OPTICAL_POWER_MEAN, OPTICAL_POWER_STD))

        with _lock:
            _active_failures[key] = (now, initial_optical, 0)

        print(
            f"[INJECTION] Starting Gray Failure on {dev['device_id']} in {dev['region']} "
            f"(dc={dev['dc']}) at t={now:.1f} for {GRAY_FAILURE_DURATION_SEC}s"
        )
        log_ground_truth(now, dev["device_id"], dev["region"], dev["dc"], "GrayFailure")


def telemetry_loop(devices: list[dict]) -> None:
    """Emit telemetry every TELEMETRY_INTERVAL_SEC and append to CSV."""
    global _shutdown
    next_ts = time.time()
    while not _shutdown:
        now = time.time()
        if now < next_ts:
            time.sleep(min(0.05, next_ts - now))
            continue
        next_ts = next_ts + TELEMETRY_INTERVAL_SEC
        if next_ts <= now:
            next_ts = now + TELEMETRY_INTERVAL_SEC

        rows = [generate_row(d, now) for d in devices]
        append_telemetry_csv(rows)


def main() -> None:
    global _shutdown
    devices = build_topology()
    print(
        f"Topology: {len(REGIONS)} regions × {DCS_PER_REGION} DCs × {DEVICES_PER_DC} devices = {len(devices)} devices"
    )
    print(f"Telemetry every {TELEMETRY_INTERVAL_SEC}s → {CSV_PATH}")
    print(f"Failure check every {FAILURE_CHECK_INTERVAL_SEC}s, P(inject)={FAILURE_PROBABILITY}, duration={GRAY_FAILURE_DURATION_SEC}s")
    print("Ground truth log:", GROUND_TRUTH_PATH)
    print("Press Ctrl+C to stop.\n")

    scheduler = threading.Thread(target=failure_scheduler, args=(devices,), daemon=True)
    scheduler.start()

    def on_signal(*_args: object) -> None:
        global _shutdown
        _shutdown = True
        print("\nShutting down...")

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    try:
        telemetry_loop(devices)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    finally:
        _shutdown = True


if __name__ == "__main__":
    main()
