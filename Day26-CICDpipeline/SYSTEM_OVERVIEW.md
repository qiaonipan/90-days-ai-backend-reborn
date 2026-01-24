# System Overview: Production-Grade Anomaly Diagnosis

## Problem

This system helps SREs move from raw logs to **interpretable anomaly hypotheses** by combining statistical signal detection with LLM-based pattern reasoning.

Traditional log analysis tools either:
- Drown operators in noise (showing thousands of logs with no prioritization)
- Provide black-box AI summaries without explainable evidence
- Require manual pattern matching that doesn't scale

**The approach**: This system reduces the signal-to-noise ratio first through unsupervised anomaly detection, then applies RAG-based reasoning only on a high-confidence subset. This enables **human-in-the-loop** decision-making with clear, evidence-backed hypotheses rather than overwhelming raw data or opaque AI conclusions.

**Key Value Propositions:**
- **Interpretable**: Every diagnosis includes statistical patterns (error distribution, time concentration, common features) before LLM reasoning
- **Hypothesis-driven**: Provides structured root cause hypotheses with confidence scores and alternatives
- **Human-in-the-loop**: SREs can validate patterns, review evidence, and make informed decisions

---

## Design Philosophy

### Why "Signal → RAG", Not Direct RAG?

This system first detects anomaly signals using **time-series statistics** to reduce noise, then applies RAG only on a **high-signal subset** (100-300 logs), preventing LLM hallucination on irrelevant logs.

**The Three-Stage Pipeline:**

1. **Stage 0: Unsupervised Signal Detection**
   - Parse logs into structured format (timestamp, level, component, message)
   - Extract message templates (generalize dynamic parts: numbers, IPs, paths)
   - Aggregate by 5-minute windows
   - Detect spikes: error rate > historical p95, volume change > 3x, new templates
   - Score anomalies with error-prioritized weighting (80% error count, 20% volume spike)
   - **Output**: Top-3 suspicious time windows with template signatures

2. **Stage 1: Signal-Driven Candidate Retrieval**
   - Use `REGEXP_LIKE` to match log signatures within time windows
   - Filter by error keywords (ERROR, FATAL, Exception, 404, 500)
   - Merge candidates from all signals, deduplicate, limit to 300 logs
   - **Output**: Small, high-signal-to-noise subset (typically 100-300 logs)

3. **Stage 2: RAG Explanation Layer**
   - Perform statistical analysis on candidate subset:
     - Error type distribution
     - Time concentration patterns
     - Common keywords/components
     - Message pattern frequencies
   - Optionally rerank using hybrid search (vector + BM25) if user query provided
   - Feed statistical summary + representative samples to LLM
   - Force structured JSON output: `root_cause`, `confidence`, `evidence`, `alternatives`, `next_steps`
   - **Output**: Interpretable diagnosis with pattern-based evidence

**Why This Works:**
- **Noise Reduction**: Statistical filtering eliminates 95%+ of irrelevant logs before LLM processing
- **Cost Efficiency**: LLM only processes 100-300 logs instead of thousands
- **Accuracy**: High-signal subset prevents hallucination on unrelated logs
- **Interpretability**: Statistical patterns provide explainable foundation for LLM reasoning

---

## What This System Explicitly Does NOT Do

This system does **not** aim to replace alerting systems.

It does **not** automatically label incidents as ground truth.

It prioritizes **signal reduction and interpretability** over black-box accuracy.

**Explicit Boundaries:**

1. **Not a Real-Time Alerting System**
   - The system analyzes uploaded log batches, not streaming logs
   - No automatic incident creation or escalation
   - Designed for post-incident analysis and proactive investigation

2. **Not a Ground Truth Labeler**
   - All diagnoses are hypotheses with confidence scores
   - The system provides alternatives and evidence, not definitive answers
   - SRE judgment is required to validate and act on findings

3. **Not a Black-Box AI**
   - Every diagnosis includes statistical patterns (error distribution, time concentration, keywords)
   - Evidence is traceable to specific log entries
   - Confidence scores reflect pattern strength, not arbitrary AI certainty

4. **Not a General-Purpose Log Search**
   - Focused on anomaly diagnosis, not ad-hoc log exploration
   - Optimized for error patterns, not general information retrieval
   - Requires anomaly signals to be detected first

5. **Not a Replacement for Domain Expertise**
   - System provides structured hypotheses, not final decisions
   - SREs must interpret patterns in context of their infrastructure
   - Designed to augment, not replace, human expertise

---

## System Flow

### Upload & Signal Detection (Stage 0)

1. **Log Upload**
   - User uploads log file (.log, .txt, .gz, max 5000 lines)
   - System parses logs into structured format (timestamp, level, component, message)
   - Extracts message templates (replaces dynamic parts with placeholders)

2. **Anomaly Signal Extraction**
   - Aggregates logs into 5-minute time windows
   - Calculates error rates, volume changes, template frequencies
   - Detects spikes: error rate > historical p95, volume change > 3x, new templates
   - Scores windows with error-prioritized weighting (80% error count, 20% volume spike)
   - Stores top-3 anomaly signals in database

3. **Embedding & Indexing**
   - Generates embeddings for all log entries (text-embedding-3-large)
   - Stores logs with timestamps in Oracle Autonomous Database 26ai
   - Builds BM25 index for keyword search

### Diagnosis Request (Stage 1 + Stage 2)

1. **Signal-Driven Retrieval (Stage 1)**
   - Retrieves top anomaly signals from database
   - For each signal:
     - Converts signature to regex pattern (handles placeholders: `<NUM>`, `<IP>`, etc.)
     - Uses `REGEXP_LIKE` to match logs within time window + 5 minutes
     - Filters by error keywords (ERROR, FATAL, Exception, 404, 500)
     - Limits to 200 logs per signal
   - Merges and deduplicates candidates, limits to 300 total

2. **Statistical Analysis**
   - Analyzes candidate subset:
     - Error type distribution (ERROR, FATAL, Exception types)
     - Time concentration (hourly patterns, time span)
     - Common keywords (top 10 by frequency)
     - Component distribution (top 5 components)
     - Message patterns (top 5 templates)
   - Generates statistical summary

3. **Hybrid Reranking (Optional)**
   - If user query provided:
     - Performs vector search on candidate subset
     - Performs BM25 search on candidate subset
     - Fuses scores (70% vector, 30% BM25)
     - Reranks to top 100 most relevant logs

4. **RAG Explanation (Stage 2)**
   - Feeds statistical summary + representative log samples (50 logs) to LLM
   - Prompts LLM to:
     - First summarize overall patterns (error distribution, time concentration, common features)
     - Then provide root cause hypothesis based on patterns
     - Include evidence as pattern descriptions (not individual log lines)
   - Forces structured JSON output:
     - `pattern_summary`: Statistical pattern summaries
     - `root_cause`: Hypothesis based on patterns
     - `confidence`: 0.0-1.0 based on pattern strength
     - `evidence`: Pattern descriptions (e.g., "45% of logs show ERROR type X")
     - `alternatives`: Alternative explanations
     - `next_steps`: Actionable troubleshooting steps

5. **Response Formatting**
   - Formats diagnosis with colored sections (Root Cause, Evidence, Alternatives, Next Steps)
   - Displays retrieved log evidence with similarity scores
   - Presents structured, interpretable diagnosis to user

---

## Key Technical Decisions

### Why Oracle Autonomous Database 26ai?

- Native vector search capabilities (no external vector DB needed)
- Hybrid search: vector + BM25 in single query
- Production-grade reliability and scalability
- SQL-based, familiar to most engineers

### Why Three-Stage Pipeline?

- **Stage 0 (Unsupervised)**: Fast, no LLM cost, identifies suspicious windows
- **Stage 1 (Signal-Driven)**: Reduces noise by 95%+, creates high-signal subset
- **Stage 2 (RAG)**: Expensive LLM only processes 100-300 logs, not thousands

### Why Statistical Analysis Before LLM?

- Provides interpretable foundation for LLM reasoning
- Prevents hallucination on irrelevant patterns
- Enables pattern-based evidence (not just log citations)
- Reduces LLM token costs (statistics + samples, not all logs)

### Why Error-Prioritized Weighting?

- Production incidents are primarily error-driven
- Volume spikes without errors are often benign
- 80% error / 20% volume weighting aligns with SRE priorities

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 0: Signal Detection                │
│  Log Upload → Parse → Template Extract → Time-Series →     │
│  Anomaly Scoring → Top-3 Signals Stored                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Stage 1: Signal-Driven Retrieval               │
│  Load Signals → REGEXP_LIKE Match → Error Filter →         │
│  Deduplicate → 300 High-Signal Logs                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Stage 2: Statistical Analysis + RAG               │
│  Analyze Patterns → (Optional) Hybrid Rerank →              │
│  LLM Diagnosis → Structured JSON → Formatted Display        │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

- **Signal Reduction**: 95%+ noise reduction (from thousands to 100-300 logs)
- **Cost Efficiency**: LLM processes <1% of total logs
- **Interpretability**: Every diagnosis includes statistical patterns
- **Accuracy**: Pattern-based evidence prevents hallucination
- **Human-in-the-Loop**: SREs can validate and act on structured hypotheses

