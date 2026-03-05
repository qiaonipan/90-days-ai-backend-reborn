# AI Infrastructure Diagnostic Assistant

AI-assisted infrastructure diagnostic system for hyperscale telemetry. Pipeline: **Telemetry → anomaly detection → RAG (incident + runbook retrieval) → agent reasoning → mitigation suggestions.** The system uses tools (telemetry analysis, vector search over incidents and runbooks) and LLM reasoning to produce diagnostic hypotheses. Backend prototype only (no UI); not a chatbot.

## Architecture

```
Telemetry anomaly
    ↓
Incident snapshot
    ↓
AI Investigation Agent
    ↓
Agent tools:
    - analyze_telemetry (latency / packet loss / queue depth trends)
    - retrieve_similar_incidents (vector search over historical incidents)
    - retrieve_relevant_runbooks (vector search over runbooks)
    ↓
LLM reasoning (hypotheses, evidence, mitigation)
    ↓
Diagnostic report
```

## Pipeline

1. **Telemetry** – Load events from `telemetry/telemetry_events.json` (device, latency_ms, packet_loss_pct, queue_depth_pct, timestamp).
2. **Anomaly detection** – Threshold rules flag events (e.g. latency > 100 ms, packet loss > 2%, queue depth > 80%).
3. **Incident snapshot** – Anomalous events are converted into a natural-language description.
4. **Investigation Agent** – Orchestrates:
   - **Step 1:** Analyze telemetry (trends: increasing/stable/decreasing for latency, packet loss, queue depth).
   - **Step 2:** Retrieve top-k similar historical incidents from the vector store.
   - **Step 3:** Retrieve top-k relevant runbooks from the runbook vector store.
   - **Step 4:** LLM reasoning with incident snapshot + telemetry analysis + incidents + runbooks → hypotheses, evidence, mitigation.
   - **Step 5:** Produce a structured diagnostic report.
5. **Output** – Possible root causes (ranked), supporting evidence, suggested mitigation steps, confidence level.

## Project structure

```
ai-network-diagnostics/
  telemetry/
    telemetry_events.json       # Mock telemetry events
  anomaly_detection/
    detect.py                   # Threshold-based anomaly detector
  incident_processing/
    snapshot.py                 # Telemetry → natural-language snapshot
  agent/
    investigation_agent.py      # Orchestrates tools and LLM reasoning
  tools/
    analyze_telemetry.py       # Latency / packet loss / queue depth trend analysis
    retrieve_incidents.py     # Wrapper for historical incident vector search
    retrieve_runbooks.py       # Wrapper for runbook vector search
  knowledge_base/
    historical_incidents.json   # Symptom / root cause / mitigation
    runbooks.json               # Symptom / possible_cause / mitigation
  rag_engine/
    embed.py                    # Build incident + runbook vector DBs
    retrieve.py                 # Semantic search (incidents + runbooks)
    reasoning.py                # LLM prompt and report parsing for the agent
  vector_db/                    # Generated: embeddings + metadata (no git)
  main.py                       # Demo: load → detect → snapshot → agent → report
  README.md
```

## Requirements

- Python 3.10+
- `sentence-transformers` (local embeddings; no API key required for retrieval)
- `python-dotenv` (optional; for loading `.env`)
- Optional: `OPENAI_API_KEY` for LLM reasoning; without it, the report is built from the top retrieved incidents and runbooks

Install from the parent project:

```bash
pip install -r ../requirements.txt
```

## How to run

1. **Copy environment template** (optional but recommended for LLM and overrides):
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set at least `OPENAI_API_KEY` if you want LLM-generated hypotheses. See `.env.example` for all supported variables (paths, thresholds, model names, `LOG_LEVEL`).

2. **Run the pipeline** from the `ai-network-diagnostics` directory:
   ```bash
   cd ai-network-diagnostics
   python main.py
   ```

Logs go to stderr; the diagnostic report is printed to stdout. Exit code 0 on success, 1 if no telemetry or no anomalies.

## Run the demo

The script will:

1. Load telemetry and detect anomalies.
2. Build an incident snapshot.
3. Run the **Investigation Agent**: analyze telemetry, retrieve incidents, retrieve runbooks, run LLM reasoning.
4. Print the incident summary (trends), then the **AI Investigation Report** (root causes, evidence, suggested mitigation, confidence).

## Configuration

All settings are centralized in `config.py`. Values are read from the environment and from an optional `.env` file in the project directory. See `.env.example` for documented variables.

- **Paths** – `TELEMETRY_PATH`, `HISTORICAL_INCIDENTS_PATH`, `RUNBOOKS_PATH`, `VECTOR_DB_DIR` (relative to the project root or absolute).
- **Retrieval** – `TOP_K_INCIDENTS`, `TOP_K_RUNBOOKS`.
- **Anomaly thresholds** – `THRESHOLD_LATENCY_MS`, `THRESHOLD_PACKET_LOSS_PCT`, `THRESHOLD_QUEUE_DEPTH_PCT`.
- **Models** – `EMBEDDING_MODEL_NAME`, `LLM_MODEL_NAME`; `OPENAI_API_KEY` for the LLM (optional).
- **Logging** – `LOG_LEVEL` (e.g. `DEBUG`, `INFO`).

## Testing

From the `ai-network-diagnostics` directory, install dev dependencies and run tests (no API keys required; LLM is mocked in integration tests):

```bash
pip install -r requirements-dev.txt
pytest -q
```

Unit tests cover anomaly detection (above/below thresholds), incident snapshot content, and telemetry trend analysis. The integration test runs the pipeline end-to-end with a small fixture and a mocked LLM.

## Design notes

- **Tool-based agent** – The agent does not loop or chat; it runs a fixed sequence: analyze → retrieve incidents → retrieve runbooks → reason → report.
- **Reusable RAG** – Incidents and runbooks each have their own vector store; same embedding model and retrieval pattern.
- **Deterministic fallback** – Without an LLM, the report is derived from the closest historical incident and runbook.
