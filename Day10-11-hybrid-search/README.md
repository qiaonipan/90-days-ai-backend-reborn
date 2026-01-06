# Oracle 26ai Cloud Vector Semantic Search (RAG Prototype)

> **A production-style semantic search and RAG system built on real HDFS logs — exploring both the capabilities and the limits of AI-assisted operational diagnosis.**

---

## 📌 Project Overview

This project is a **full-cycle Retrieval-Augmented Generation (RAG) prototype** that performs **semantic search over real HDFS production logs** using:

- OpenAI text embeddings  
- Oracle Autonomous Database 26ai native vector search  
- FastAPI backend  
- Lightweight HTML / JavaScript frontend  

Users can ask **natural language questions** such as:

- *“What caused the block to be missing?”*  
- *“Why did the DataNode stop responding?”*

The system retrieves the **most semantically relevant logs** and generates an **evidence-based AI summary** grounded strictly in retrieved data.

This project was built as part of my **reskilling journey after a layoff**, with a strong emphasis on **realistic enterprise constraints rather than toy AI demos**.

---

## 📌 Motivation

Most AI demos assume:
- Clean datasets  
- Explicit labels  
- Clear cause–effect relationships  

**Production logs are the opposite.**

They are:
- Noisy  
- State-heavy  
- Often missing explicit causal explanations  

This project intentionally uses **1000+ real HDFS production logs** to explore a critical question:

> **What can semantic search and RAG realistically do — and where do they break — in real operational environments?**

---

## 📌 System Architecture

**Query Flow**

1. User submits a natural language query  
2. Query is converted into an embedding via OpenAI  
3. Oracle 26ai performs vector similarity search over stored log embeddings  
4. Top-K relevant logs are retrieved with similarity scores  
5. An LLM generates a concise summary based strictly on retrieved evidence  
6. Frontend renders:
   - AI summary  
   - Ranked log evidence  
   - Similarity scores  
```
User Query
↓
OpenAI Embedding
↓
Oracle 26ai Vector Search
↓
Top-K Relevant Logs
↓
LLM Evidence-Based Summary
↓
Frontend Rendering
```

---

## 📌 What Works Well

### 1. End-to-End RAG Pipeline (Production-Oriented)

- No notebooks  
- No mock data  
- No in-memory toy vector stores  

This is a **real service-style system** with:
- Persistent vector storage  
- Clear API contracts  
- Frontend–backend integration  

**Qiaoni note:**  
> “I prioritized system completeness and production realism over isolated model experiments.”

---

### 2. Honest AI Behavior (No Hallucination)

When the retrieved logs **do not contain enough causal evidence**, the AI:
- Explicitly states uncertainty  
- Avoids inventing root causes  
- References only retrieved logs  

**Qiaoni note:**  
> “I treated missing information as a first-class signal instead of hiding it with hallucination.”

---

### 3. Semantic Retrieval Over Noisy Logs

Despite noisy, unlabeled data, the system consistently clusters:
- HDFS block lifecycle events  
- DataNode-related behavior  
- Storage and network-adjacent signals  

This validates the **embedding + vector search foundation**.

---

## 📌 Known Limitations (By Design)

### 1. State ≠ Cause

Most production logs describe **what happened**, not **why it happened**.

As a result:
- Queries asking *“why”* often retrieve state transitions (e.g. `addStoredBlock`)  
- The system can summarize context but cannot always infer true root cause  

This reflects **real observability constraints**, not a modeling bug.

---

### 2. Semantic Distribution Is Skewed

- ~80% of logs represent normal INFO-level operations  
- Failure and corruption signals are sparse  

Pure semantic similarity retrieval tends to favor **frequent states over rare failures**.

---

## 📌 Key Insight

> **Semantic search does not create causality — it amplifies whatever semantic signals already exist in the data.**

This project demonstrates that improving AI-assisted diagnosis often requires:
- Redesigning data semantics  
- Enriching failure signals  
- Explicitly modeling uncertainty  

—not simply switching models or embeddings.

**Qiaoni-ready phrasing:**  
> “The biggest improvement opportunity wasn’t the model, but how failure semantics are represented in the data.”

---

## 📌 Planned V2: Failure-Aware & Causal-Oriented Retrieval

The next iteration focuses on bridging the gap between **state retrieval** and **causal diagnosis**.

### V2 Goals

#### 1. Failure-Aware Semantic Enrichment
- Extract and up-weight logs related to:
  - Missing blocks  
  - Corruption  
  - Under-replication  
  - Node failures  
- Introduce a curated **failure-centric sub-corpus**

#### 2. Two-Stage Retrieval
- Stage 1: Vector similarity recall  
- Stage 2: Failure / causality-aware reranking (lightweight rules or LLM-based)

#### 3. Confidence & Coverage Signals
- Explicitly indicate:
  - Whether causal evidence exists  
  - Whether the answer is speculative or evidence-backed  

#### 4. Timeline Reconstruction (Optional)
- Group logs by time to expose:
  - Pre-failure signals  
  - Cascading effects  

**Qiaoni note:**  
> “V2 shifts the problem from ‘better embeddings’ to ‘better semantic representation of failures.’”

---

## 📌 Tech Stack

- **Backend:** FastAPI (Python)  
- **Vector Database:** Oracle Autonomous Database 26ai  
- **Embeddings & LLM:** OpenAI  
- **Frontend:** Vanilla HTML / JavaScript  
- **Data:** 1000+ real HDFS production logs  

---

## 📌 Final Note

This project is intentionally **not a perfect AI diagnosis system**.

It is a realistic exploration of:
- What semantic search and RAG can do  
- Where they fail  
- Why those failures matter  

> The most valuable outcome was not higher accuracy,  
> but a clearer understanding of **where intelligence must be designed, not assumed**.
