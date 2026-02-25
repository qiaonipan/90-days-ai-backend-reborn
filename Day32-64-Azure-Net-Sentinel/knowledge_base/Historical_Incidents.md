## Historical Incident: Humidity-Induced Bulk Optical Degradation

**Incident ID**: NET-HYDRO-2024-07-15-01  
**Category**: Environmental – Elevated Humidity / Condensation  
**Status**: Resolved – Mitigation and Long-Term Fix Implemented

---

## 1. Executive Summary

On 2024-07-15, Azure network operations observed a **cluster of gray failures** across multiple leaf switches in the `eastus` region, `dc2`.  
The failures manifested as **gradual optical power degradation** followed by rising error counters, impacting **two full rows** of racks.

Root cause was traced to **localized humidity control failure** in the cold aisle, leading to **condensation forming on optical connectors and patch panels**. After remediation of the environmental conditions and cleaning of affected optics, optical power and error rates returned to normal.

---

## 2. Timeline (UTC)

- **07:12** – First early-warning alerts from internal gray-failure detector for `Leaf-03` and `Leaf-04` in `eastus/dc2`.
- **07:19** – Additional alerts for `Leaf-01`, `Leaf-02`, `Leaf-05` in the same DC row; blast radius starts to look systemic.
- **07:25** – NOC escalates to L3 network on-call; incident formally opened.
- **07:32** – Quick CLI checks confirm:
  - Optical power drifting from -8 dBm toward **-11 dBm**
  - Error counters (CRC / FEC) increasing non-linearly
  - No correlated ASIC temperature or CPU anomalies
- **07:40** – Pattern recognition: similar slope and error patterns across **> 20 leaf ports** in adjacent racks.
- **07:48** – DC operations engaged for on-site inspection; humidity sensor data requested.
- **08:05** – DC ops report:
  - Humidity in affected cold aisle elevated by ~15% vs. baseline
  - Minor condensation observed on metalwork near top-of-rack patch panels
- **08:20** – Controlled shutdown and rerouting of non-critical traffic from the most affected links.
- **08:35** – Targeted cleaning and reseating of optics and patch cables in impacted racks.
- **08:50** – Optical power and error counters begin returning to normal ranges.
- **09:15** – All impacted links stable; no new alerts from gray-failure detector.
- **09:30** – Incident declared resolved; monitoring extended for 24 hours.

---

## 3. Technical Symptoms

Across the affected devices (`Leaf-01`–`Leaf-05` in multiple racks):

- **Optical power (`optical_power_dbm`)**
  - Baseline: around **-8.0 dBm**
  - Degradation: linear drift toward **-11.0 dBm** over 15–30 minutes
- **Error counters**
  - CRC/FEC counters remained at 0 initially
  - Once power crossed approximately **-10.5 dBm**, errors began to increase
  - After **-11.0 dBm**, error rates accelerated, leading to application-visible packet loss
- **ASIC temperature**
  - Remained within nominal band (50–60°C)
  - No evidence of thermal stress
- **Utilization**
  - Stable or slightly reduced as traffic was drained and rerouted

The combination of **stable device health** and **degrading optics across multiple adjacent racks** suggested a **non-device, non-firmware** root cause.

---

## 4. Root Cause Analysis

### 4.1 Immediate Cause

Condensation on optical connectors and patch panels due to **elevated humidity** in a localized cold aisle segment.

- Water molecules on connector end-faces introduce:
  - Additional insertion loss
  - Increased back reflection
  - Intermittent contact quality under thermal cycles

### 4.2 Contributing Factors

- **Humidity control misconfiguration**
  - A miscalibrated humidity sensor caused the local HVAC subsystem to **under-dehumidify** the affected zone.
  - The deviation was large enough to allow short periods of condensation during cooler night-time temperatures.
- **Monitoring gaps**
  - Environmental alerts for humidity were configured, but:
    - Thresholds were set too permissively.
    - Integration with network NOC dashboards was missing.
- **Physical layout**
  - Impacted racks were located near a chilled-water line with imperfect insulation.
  - Minor surface temperature changes increased the likelihood of condensation in that specific row.

---

## 5. Mitigation & Remediation

### 5.1 Short-Term Mitigation

- Rerouted traffic away from the most degraded links using:
  - ECMP rebalancing
  - Temporary interface shutdowns on severely impacted ports
- Performed:
  - On-site cleaning of optics and patch panels using approved fiber cleaning kits
  - Reseating of optics and patch cords on impacted links

### 5.2 Long-Term Fixes

- **HVAC & humidity control**
  - Recalibrated humidity sensors in the affected cold aisle.
  - Tightened humidity setpoints and alarm thresholds.
  - Added redundancy to humidity sensing for cross-validation.
- **Insulation improvements**
  - Enhanced insulation around the nearby chilled-water line.
  - Performed thermal imaging to validate uniform temperature across racks.
- **Fiber infrastructure**
  - Introduced periodic cleaning and inspection schedule for high-density patch panels.
  - Updated installation guidelines to emphasize:
    - Drip loops where appropriate
    - Avoiding routing near potential condensation points

---

## 6. Monitoring & Detection Enhancements

Following this incident, several improvements were made:

- **Correlation between environmental and optical metrics**
  - Added automated correlation logic:
    - If multiple gray failures are detected in the same DC + row AND humidity is elevated, raise a **single correlated incident**.
- **Knowledge base integration**
  - This incident is referenced by:
    - `SOP_Optical_Failure.md` (environmental checks section)
    - `Azure_Device_Specs.md` (optical thresholds and gray failure definitions)
- **Detector enhancements (design notes)**
  - Future iterations of `detector.py` are expected to:
    - Incorporate device-local environmental feeds (e.g., humidity sensors) where available.
    - Adjust alert thresholds when environmental risk factors are present.

---

## 7. Lessons Learned

- **Gray failures are often environmental**
  - When multiple devices show similar optical slopes in the same physical area, environment is a strong suspect.
- **Environmental telemetry must be first-class**
  - Humidity, temperature, and condensation risk must be surfaced alongside network metrics in operator dashboards.
- **Document and reuse patterns**
  - Capturing the **pattern** of this incident (optical slopes, spatial distribution, environmental anomalies) allows faster recognition and mitigation of similar events in the future.

This document serves as a template for documenting future humidity-related optical incidents and should be updated whenever similar patterns are observed.

