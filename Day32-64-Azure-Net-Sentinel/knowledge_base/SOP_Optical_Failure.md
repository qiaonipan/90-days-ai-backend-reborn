## SOP: Optical Power Degradation / Gray Failure Handling

**Audience**: Azure Network Operations – L2/L3  
**Scope**: Leaf switches with optical interfaces monitored by `telemetry_generator.py` and `detector.py`.

---

## 1. Detection & Initial Triage

- **Primary signals**
  - `detector.py` console: red `[ALERT] Early gray failure detected: ...`
  - `detected_incidents.log`: new entries for the same `device_id` / region
  - `live_telemetry.csv`: sustained downwards trend in `optical_power_dbm`, increasing `error_count`
- **Immediate checks**
  - Confirm whether alerts are:
    - **Single-link** (one `device_id` / DC)
    - **Clustered** (multiple devices in same rack / DC / region)
  - Check for **correlated signals**:
    - CRC/FEC error spikes
    - ASIC temperature anomalies
    - Utilization anomalies (sudden drops or bursts)

**Decision gate**  
If more than 5 devices in the same DC are impacted within 5 minutes, treat as a **potential environmental / bulk failure** and escalate to DC operations in parallel with the steps below.

---

## 2. Data Collection (T0 Snapshot)

For each impacted `device_id`:

- **Telemetry snapshot**
  - Export last 10 minutes from `live_telemetry.csv` filtered by:
    - `device_id`
    - `region`, `dc`
  - Persist as:  
    `knowledge_base/cases/{region}_{dc}_{device_id}_telemetry_T0.csv`
- **Interface state (if CLI accessible) – examples**
  - Vendor-agnostic commands (adjust to platform):
    - `show interfaces transceiver detail`
    - `show interfaces <port> counters errors`
    - `show logging last 200 | include <port>`
  - Collect at least:
    - Rx/Tx optical power (dBm)
    - LOS/LOF/LOM status
    - FEC, BER metrics
    - CRC / alignment errors
    - Interface admin/oper state
- **Environment & rack context**
  - Rack ID / row
  - Location of patch panels and MPO cassettes
  - Any **recent change** in:
    - Cabling
    - Optics replacement
    - DC maintenance (cooling / humidity / power work)

Store all raw artifacts under `knowledge_base/cases/<incident_id>/`.

---

## 3. Scope & Blast Radius Analysis

- **Single link vs. systemic**
  - Single device + single interface: likely **optic / fiber / port** level.
  - Multiple devices in same rack or row:
    - Check for shared **fiber trunks / patch panels**.
    - Check environmental telemetry (temperature, humidity) for that row.
- **Temporal pattern**
  - Gradual linear drop over minutes/hours:
    - Suggests **gray failure**: aging optics, fiber damage, or environmental factors.
  - Sudden drop with LOS:
    - Suggests **hard failure**: unplugged cable, broken connector, or power event.

If multiple gray failures start within a narrow time window in one area, strongly suspect **environmental** causes (e.g., humidity, condensation, or physical damage).

---

## 4. Optical Path Verification

Work from **device outward**:

1. **Local port verification**
   - Confirm admin state is `up`, speed/duplex match, correct optic type.
   - Check DOM / optical telemetry:
     - Rx power vs. `Azure_Device_Specs.md` normal range.
     - Tx power within spec.
     - Temperature of the optic module.
2. **Patch panel & fiber checks**
   - Verify patching diagram matches reality:
     - Correct port on both sides
     - No accidental cross-connects
   - Inspect fiber for:
     - Excessive bend radius
     - Pinched or crushed sections
     - Visible damage near tray exits
3. **Clean and reseat**
   - Use approved fiber cleaning tools (no dry tissue, no compressed air alone).
   - Reseat:
     - Optic module
     - Patch cable at both ends
   - Re-check telemetry after 2–3 minutes:
     - If `optical_power_dbm` returns to normal band and error counters stabilize, classify as **“contamination-induced gray failure”**.

---

## 5. Correlation with Device & ASIC Health

For each affected device:

- Check **ASIC temperature** vs. normal range:
  - If temperature significantly elevated, consider:
    - Localized airflow issues
    - Fan failures
    - Hot-aisle containment leaks
- Validate:
  - System-wide alarms / syslog (power supplies, fan trays)
  - Any recent firmware changes or reloads

If only optical metrics are degraded and ASIC / system health is nominal, prioritize **optic + fiber + environment** as root-cause candidates.

---

## 6. Remediation Actions

### 6.1 Low-Risk Immediate Actions

- Move traffic away using:
  - ECMP rebalancing
  - Interface shutdown on degraded link (if redundant paths exist)
- Rate-limit or drain flows over the affected interface where possible.

### 6.2 Component Replacement

Replace **one component at a time**:

1. Swap optic on the leaf side.
2. If issue persists, swap the fiber jumper.
3. If still present, inspect/replace the remote optic / port.

After each step:

- Re-run telemetry checks:
  - `optical_power_dbm` trend
  - `error_count`, FEC, CRC

Record in ticket:

- Which component was replaced
- Before/after optical readings

---

## 7. Environmental & Bulk Failure Handling

When multiple devices in the same DC / row exhibit similar optical degradation:

- Engage **DC operations** to check:
  - Humidity and temperature trends for the room and specific row
  - Recent HVAC changes (setpoint modifications, dehumidifier issues)
  - Reports of condensation or water ingress
- Compare with prior cases in `Historical_Incidents.md`, especially humidity-induced events.

If humidity or condensation is suspected:

- Prioritize **bulk inspection** of:
  - Top-of-rack fiber trays
  - Overhead fiber runners in the impacted row
  - Any locations near chilled-water or condensation-prone areas

---

## 8. Recovery Verification

After remediation:

- Confirm for each previously affected `device_id`:
  - `optical_power_dbm` back within normal range from `Azure_Device_Specs.md`
  - `error_count`, FEC, CRC return to baseline
  - No new alerts in `detector.py` for at least **30 minutes**
- Validate end-to-end service:
  - Latency and packet loss from synthetic probes
  - Any customer-impact KPIs (if applicable)

---

## 9. Post-Incident Documentation

Create / update an incident record with:

- Summary:
  - Region, DC, rack(s), impacted device_ids
  - Start/end times, detection source
- Root cause classification:
  - Optic contamination
  - Fiber damage
  - Humidity / condensation
  - Hardware failure
  - Other (with detail)
- Lessons learned:
  - Monitoring gaps
  - Documentation or diagram issues
  - Automation improvements (e.g., better early detection thresholds)

If the incident matches previously seen patterns (e.g., humidity-induced bulk loss), link the record to `Historical_Incidents.md` and update that document with new findings.

