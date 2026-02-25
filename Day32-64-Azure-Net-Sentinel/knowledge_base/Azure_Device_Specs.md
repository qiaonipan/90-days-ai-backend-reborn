## Azure Leaf Switch Optical & Telemetry Reference

**Audience**: Azure Network Engineering & NOC  
**Scope**: Leaf switches modeled in `Day32-64-Azure-Net-Sentinel` (simulated environment).

---

## 1. Logical Device Model

- **Role**: `Leaf`
- **Interface naming**: `Eth1/1` (single monitored optical uplink in this lab)
- **Telemetry fields of interest**:
  - `optical_power_dbm`
  - `error_count`
  - `asic_temp`
  - `utilization_pct`

These values are emitted once per second per device by `telemetry_generator.py` and consumed by `detector.py`.

---

## 2. Optical Power (Rx) – Expected Ranges

All values below refer to **received optical power** on the monitored interface.

- **Nominal operating band**
  - Target mean: **-8.0 dBm**
  - Normal short-term variation (noise): ±0.3 dB
  - Acceptable range for healthy links: **[-9.0 dBm, -7.0 dBm]**
- **Early warning band**
  - Degradation band: **[-10.5 dBm, -9.0 dBm]**
  - Links in this range:
    - Should be watched for **downward trend**.
    - Are more sensitive to additional loss (dirty connectors, minor bends).
- **Critical band**
  - Severe optical loss: **≤ -11.0 dBm**
  - This is the threshold used by `telemetry_generator.py` to start injecting errors.
  - In production, values at or below this level are strongly associated with:
    - Intermittent packet loss
    - High FEC activity
    - Elevated CRC/align errors

**Detector behavior**  
`detector.py` focuses on the **slope** of `optical_power_dbm` over a 30-second window, rather than just absolute values:

- If slope < **-0.01 dB/s** for **5 consecutive seconds**, it raises an early gray failure alert.

---

## 3. Error Counters

In the lab telemetry:

- `error_count` represents **cumulative error counter** for the interface during a gray failure episode.
  - Normal steady state: **0**
  - During failure (optical_power_dbm ≤ -11.0 dBm):
    - `error_count` increases by **1–10 per second**.

In production, this typically corresponds to:

- CRC errors
- FEC uncorrectable errors
- Alignment or framing errors

**Healthy link expectations**:

- Over a 5-minute window:
  - CRC/FEC errors **≈ 0** under normal load.
  - Any **sustained non-zero** rate should be investigated together with optical power and utilization.

---

## 4. ASIC Temperature

`asic_temp` represents the temperature of the main switching ASIC.

- **Nominal band**
  - Mean: **55°C**
  - Typical variation: ±5°C
  - Normal operating range: **45–65°C**
- **Warning band**
  - **> 70°C** sustained for more than 5 minutes
  - Investigate:
    - Fan speed / health
    - Airflow direction and rack placement
    - Hot-aisle / cold-aisle containment
- **Critical band**
  - **≥ 80°C**: immediate risk of thermal throttling or protective shutdown.

Note: In this lab, ASIC temperature is primarily used as a **correlative** signal; primary detection logic focuses on optical power.

---

## 5. Interface Utilization

`utilization_pct` represents the percentage of link bandwidth used.

- **Baseline in simulated environment**
  - Typical range: **30–50%**
  - Modeled as random swings within this band.
- **Operational interpretation**
  - Short-term spikes above 80% can be normal during bursts.
  - Sustained utilization > 85%:
    - Consider congestion or capacity planning issues.
    - Check for correlation with increased latency, drops, or ECN marks.
- **During optical degradation**
  - Utilization may:
    - Drop (traffic successfully re-routed away from bad link), or
    - Show increased retransmissions and effective throughput loss.

---

## 6. Summary of Normal Ranges

| Metric              | Normal Range            | Warning Range         | Critical Range      |
|---------------------|------------------------|-----------------------|---------------------|
| `optical_power_dbm` | -9.0 to -7.0 dBm       | -10.5 to -9.0 dBm     | ≤ -11.0 dBm         |
| `error_count`       | 0 (steady state)       | >0 but non-increasing | Rapidly increasing  |
| `asic_temp`         | 45–65°C                | 70–80°C               | ≥ 80°C              |
| `utilization_pct`   | 30–50% (baseline load) | >80% sustained        | >90% with symptoms  |

These ranges are calibrated for the **Azure Net Sentinel lab** and are intentionally conservative to emphasize **early detection** of gray failures.

