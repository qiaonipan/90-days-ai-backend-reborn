# AI 网络诊断助手 — 架构与执行流程说明

本文档帮助快速理解 **ai-network-diagnostics** 项目的整体目标、架构与执行流程。

---

## 1️⃣ 项目整体目标

**这个项目在做什么：**

- 面向**超大规模网络基础设施**的 **AI 辅助诊断系统**：从遥测数据出发，自动检测异常、检索历史事件与预案，再通过 LLM 推理给出根因假设与处置建议。
- **解决的问题**：把「遥测异常 → 人工查历史、查 runbook → 写报告」的流程，压缩成一条自动化 pipeline，缩短 MTTR（平均修复时间），并保证输出结构化（根因、证据、缓解步骤、置信度）。
- **模拟的真实系统**：类似 hyperscale 数据中心里的 **AI Infrastructure Diagnostics**——对 TOR/Spine 等网络设备的延迟、丢包、队列深度等遥测做异常检测，再结合历史 incident 与 runbook 做 RAG + Agent 推理，输出诊断报告（无 UI，仅后端流水线）。

---

## 2️⃣ 系统整体架构

整体是一条**单向 pipeline**，各阶段依次执行：

```
Telemetry（遥测）
    ↓
Anomaly detection（异常检测）
    ↓
Incident snapshot（事件快照 / 自然语言描述）
    ↓
AI Investigation Agent（调查智能体）
    ↓
Agent 内部：Tools
    - analyze_telemetry（遥测趋势：延迟 / 丢包 / 队列深度）
    - retrieve_similar_incidents（历史 incident 向量检索）
    - retrieve_relevant_runbooks（runbook 向量检索）
    ↓
LLM reasoning（根因假设 + 证据 + 缓解步骤 + 置信度）
    ↓
Diagnostic report（诊断报告）
```

**各阶段作用简述：**

| 阶段 | 作用 |
|------|------|
| **Telemetry** | 输入：设备遥测事件（device, latency_ms, packet_loss_pct, queue_depth_pct, timestamp）。来自 JSON 文件，模拟真实遥测流。 |
| **Anomaly detection** | 根据配置的阈值（如延迟>100ms、丢包>2%、队列>80%）标记异常事件，只保留「超标」的事件供后续使用。 |
| **Incident snapshot** | 将异常事件转成一段自然语言描述（例如「High latency detected (120 ms) on tor-switch-23. Packet loss increased to 3%. …」），用作检索 query 和 LLM 的上下文。 |
| **Investigation Agent** | 编排工具调用与推理：先分析遥测趋势，再检索历史 incident 和 runbook，最后把 snapshot + 趋势 + 检索结果交给 LLM。 |
| **Tools** | 三个工具：遥测趋势分析（上升/稳定/下降）；基于 incident snapshot 的语义检索（历史 incident）；基于同一 snapshot 的 runbook 检索。 |
| **LLM reasoning** | 根据当前事件描述、遥测分析、相似历史事件和 runbook，生成结构化诊断报告（根因、证据、缓解、置信度）。无 API Key 时使用「检索结果」构造 fallback 报告。 |
| **Diagnostic report** | 输出：可能的根因（排序）、支持证据、建议缓解步骤、置信度（low/medium/high）。 |

---

## 3️⃣ 关键模块解释

### telemetry/

- **职责**：存放**模拟遥测数据**（JSON 数组）。
- **输入**：无（人工维护或脚本生成的 `telemetry_events.json`）。
- **输出**：被 `main.py` 的 `load_telemetry()` 读取，得到 `List[dict]`，每个 dict 含 `device`, `latency_ms`, `packet_loss_pct`, `queue_depth_pct`, `timestamp`。
- **在系统中的位置**：Pipeline 的**数据源**；真实系统中会替换为 Kafka/时序库/API 等。

### anomaly_detection/

- **职责**：基于阈值的**轻量异常检测**（无 ML 模型）。
- **输入**：单条或一批遥测事件（dict）。
- **输出**：`AnomalyResult`（是否异常、触发的条件：latency_ms / packet_loss_pct / queue_depth_pct）；`detect_anomalies_from_stream()` 只返回异常事件列表。
- **在系统中的位置**：紧接在「加载遥测」之后，过滤出需要调查的异常事件；阈值来自 `config`（环境变量 / .env）。

### incident_processing/

- **职责**：把异常检测结果转成**自然语言 incident snapshot**，便于向量检索和 LLM 理解。
- **输入**：`List[AnomalyResult]`。
- **输出**：一个字符串（例如多句「High latency …」「Packet loss …」「Queue depth …」拼接）。
- **在系统中的位置**：异常事件 → 文本描述，作为 Agent 的「当前事件」输入和 RAG 的 query。

### agent/

- **职责**：**调查智能体**，固定顺序调用工具并调用 LLM，不循环、不对话。
- **输入**：incident_snapshot（字符串）、telemetry_events（异常事件列表）、top_k_incidents、top_k_runbooks、correlation_id。
- **输出**：`(DiagnosticReport, telemetry_analysis)`：诊断报告 + 遥测分析文本（供打印）。
- **在系统中的位置**：Pipeline 的「大脑」：先确保向量库存在，再执行 analyze_telemetry → retrieve incidents → retrieve runbooks → agent_reason。

### tools/

- **职责**：为 Agent 提供三个**工具**：
  - **analyze_telemetry**：对遥测事件序列做趋势分析（latency / packet_loss / queue_depth 的 increasing / stable / decreasing），并给出一句启发式总结（如「network congestion or buffer pressure」）。
  - **retrieve_incidents**：封装 RAG 的 `retrieve()`，用 snapshot 查历史 incident，返回 top_k 条并格式化为 prompt 文本。
  - **retrieve_runbooks**：封装 RAG 的 `retrieve_runbooks()`，用 snapshot 查 runbook，返回 top_k 条并格式化为 prompt 文本。
- **输入**：analyze_telemetry 输入为事件列表；两个 retrieve 输入为 incident_snapshot 字符串和 top_k。
- **输出**：analyze_telemetry 为多行字符串；两个 retrieve 为格式化后的字符串（供 LLM 使用）。
- **在系统中的位置**：Agent 的「手」：提供证据与上下文，不直接做决策，决策在 rag_engine.reasoning 的 LLM/fallback 中完成。

### knowledge_base/

- **职责**：存放**静态知识**：
  - `historical_incidents.json`：历史事件，每条含 symptom、root_cause、mitigation。
  - `runbooks.json`：预案，每条含 symptom、possible_cause、mitigation。
- **输入**：被 `rag_engine.embed` 读取。
- **输出**：嵌入后写入 `vector_db/`（embeddings + metadata），供检索使用。
- **在系统中的位置**：RAG 的**数据源**；真实系统中可能来自 CMDB、运维库、工单系统等。

### rag_engine/

- **职责**：**RAG 全流程**：建库、检索、推理。
  - **embed**：读取 knowledge_base，用 sentence-transformers 做 embedding，写入 `vector_db/`（incident 与 runbook 各一套）；提供 `ensure_vector_db` / `ensure_runbooks_vector_db` 按需建库。
  - **retrieve**：用 snapshot 作为 query，做向量相似度检索，返回 top_k 条 incident 或 runbook（带 score）；并提供 `format_retrieved_for_prompt` / `format_runbooks_for_prompt`。
  - **reasoning**：`agent_reason()` 将 snapshot + 遥测分析 + 检索结果拼成 prompt，调 OpenAI；若无 API Key 或调用失败，则从检索结果中解析根因与缓解步骤做 fallback。输出为 `DiagnosticReport`。
- **输入**：embed 输入为配置文件中的路径；retrieve 输入为 query 字符串和 top_k；reasoning 输入为 snapshot、telemetry_analysis、incident_results 文本、runbook_results 文本。
- **输出**：embed 写入磁盘；retrieve 返回 List[dict] 或格式化字符串；reasoning 返回 `DiagnosticReport`。
- **在系统中的位置**：**检索 + 推理** 的核心；Agent 只编排，真正「查资料」和「写报告」都在这里。

### main.py

- **职责**：**入口脚本**：加载配置与日志、读取遥测、跑整条 pipeline、打印报告。
- **在系统中的位置**：用户执行 `python main.py` 的入口；内部依次调用 load_telemetry → detect_anomalies_from_stream → anomalies_to_snapshot → run_investigation → print_report。

### config.py

- **职责**：**集中配置**：从环境变量和可选的 `.env` 读取路径、阈值、top_k、模型名、API Key、日志级别等；提供 `get_settings()` 单例。
- **在系统中的位置**：所有模块需要路径或阈值时都通过 `get_settings()` 获取，便于部署与测试时覆盖。

---

## 4️⃣ main.py 运行流程

1. **初始化**：`main()` 调用 `get_settings()`、`setup_logging()`，生成 `correlation_id`，解析 `telemetry_path`。
2. **读取遥测**：`run_pipeline(telemetry_path)` 内调用 `load_telemetry(events_path)`，得到 `events: List[dict]`；若文件不存在或非 JSON 数组则返回 None并 exit(1)。
3. **异常检测**：`detect_anomalies_from_stream(events)`，只保留超过阈值的异常事件列表 `anomalies`；若无异常则返回 None 并 exit(1)。
4. **生成 incident snapshot**：`incident_snapshot = anomalies_to_snapshot(anomalies)`，得到自然语言描述；同时 `telemetry_events = [a.event for a in anomalies]` 供后续分析。
5. **启动 Investigation Agent**：`run_investigation(incident_snapshot, telemetry_events, top_k_incidents, top_k_runbooks, correlation_id)`。
6. **Agent 内部步骤**：  
   - 确保 incident / runbook 向量库存在（`ensure_vector_db` / `ensure_runbooks_vector_db`）；  
   - **Step 1**：`analyze_telemetry(telemetry_events)` → 遥测趋势文本；  
   - **Step 2**：`retrieve_similar_incidents(incident_snapshot, top_k)` → 格式化 incident 文本；  
   - **Step 3**：`retrieve_relevant_runbooks(incident_snapshot, top_k)` → 格式化 runbook 文本；  
   - **Step 4**：`agent_reason(snapshot, telemetry_analysis, incident_results, runbook_results)` → `DiagnosticReport`。  
7. **输出报告**：`print_report(telemetry_analysis, report)` 将遥测摘要和「Possible root causes / Evidence / Suggested mitigation / Confidence」打印到 stdout；`run_pipeline` 返回 `report`。

---

## 5️⃣ Agent 是如何工作的

`investigation_agent.run_investigation()` 采用**固定顺序、无循环**的编排：

1. **确保向量库**：若 `vector_db/` 下没有 incident 或 runbook 的 embedding，则从 knowledge_base 加载并构建。
2. **Telemetry analysis**：`analyze_telemetry(telemetry_events)` 对异常事件的 latency_ms、packet_loss_pct、queue_depth_pct 做趋势判断（increasing/stable/decreasing），并附加一句启发式总结（如「network congestion or buffer pressure」），得到 `telemetry_analysis` 字符串。
3. **Retrieve incidents**：`retrieve_similar_incidents(incident_snapshot, top_k)` 内部调用 `rag_engine.retrieve()`，用 snapshot 的 embedding 在 incident 向量库中做相似度检索，返回 top_k 条；再 `format_incidents_for_prompt()` 转成给 LLM 看的文本。
4. **Retrieve runbooks**：`retrieve_relevant_runbooks(incident_snapshot, top_k)` 同理，用 `rag_engine.retrieve_runbooks()` 和 `format_runbooks_for_prompt()`。
5. **Combine evidence**：不在这里做融合，只是把 snapshot、telemetry_analysis、incident 文本、runbook 文本一起交给 `agent_reason()`。
6. **Run LLM reasoning**：`agent_reason()` 用固定模板拼好 prompt，调用 OpenAI；若未配置 API Key 或调用失败，则从「相似历史 incident」文本中解析 root_cause 和 mitigation，构造 fallback 的 `DiagnosticReport`。
7. **返回**：`(report, telemetry_analysis)` 给 main，用于打印。

---

## 6️⃣ RAG 在哪里使用

- **Embedding 什么数据**：  
  - **历史 incident**：`knowledge_base/historical_incidents.json` 中每条 incident 的 symptom + root_cause + mitigation 拼成一段文本，用 sentence-transformers 编码，存到 `vector_db/embeddings.npy` 与 `metadata.jsonl`。  
  - **Runbook**：`knowledge_base/runbooks.json` 中每条 runbook 的 symptom + possible_cause + mitigation 拼成一段文本，编码后存到 `vector_db/runbooks_embeddings.npy` 与 `runbooks_metadata.jsonl`。
- **Vector search 用在哪里**：  
  - 用**当前 incident snapshot**（自然语言）作为 query，分别对 incident 库和 runbook 库做**余弦相似度**检索，各取 top_k，得到「与当前症状最相似的历史事件」和「最相关的预案」。
- **Retrieval 返回什么**：  
  - 每条 incident：`symptom`, `root_cause`, `mitigation`, `score`；  
  - 每条 runbook：`symptom`, `possible_cause`, `mitigation`, `score`。  
  再通过 `format_retrieved_for_prompt` / `format_runbooks_for_prompt` 转成多段「Symptom: … Root cause: … Mitigation: …」的文本。
- **LLM 如何使用这些上下文**：  
  - `agent_reason()` 的 prompt 模板中显式包含：当前 incident、遥测分析、相似历史 incident 文本、相关 runbook 文本；LLM 基于这四块内容生成根因、证据、缓解步骤和置信度。无 LLM 时，fallback 直接从「相似历史 incident」的 Root cause / Mitigation 字段抽取并填进 `DiagnosticReport`。

---

## 7️⃣ 一个完整运行示例

**假设一条遥测事件：**  
`latency_ms=120`, `packet_loss_pct=3`, `queue_depth_pct=85`, `device=tor-switch-23`。

1. **加载**：从 `telemetry/telemetry_events.json` 读到包含该条的事件列表。
2. **异常检测**：阈值默认 100ms / 2% / 80%，该条三项均超标 → 判为异常，进入 `anomalies`。
3. **Snapshot**：生成类似「High latency detected (120 ms) on tor-switch-23. Packet loss increased to 3%. Queue depth reached 85%. Traffic congestion suspected.」
4. **Agent**：  
   - 遥测分析：若序列中延迟/丢包/队列呈上升趋势，输出「Latency trend: increasing …」「This pattern suggests network congestion or buffer pressure.»  
   - 用上述 snapshot 做向量检索，得到若干条相似历史 incident 和 runbook（如「Switch buffer overflow」「Drain traffic and reset switch」）。  
   - 将 snapshot + 遥测分析 + 检索结果填入 prompt，调用 LLM（或走 fallback）。
5. **报告**：输出例如根因「Switch buffer overflow / Micro-burst」，证据「high queue depth, packet loss spike」，缓解「Drain traffic, reset switch, verify ECMP routes」，置信度「medium」。

---

## 8️⃣ 哪些部分是 demo / mock

- **Mock 数据**：  
  - **遥测**：`telemetry/telemetry_events.json` 是静态 JSON，模拟多条设备遥测；真实系统会对接 Kafka、Prometheus、Azure Monitor 等流式或拉取数据。  
  - **历史 incident / runbook**：`knowledge_base/*.json` 是手工维护的小规模数据，真实系统可能来自 CMDB、故障库、运维知识库，且规模大、需增量更新与版本管理。
- **简化逻辑**：  
  - **异常检测**：仅规则阈值，无时序模型、无多指标联合；生产可能用 ML 或统计算法。  
  - **Agent**：固定顺序调用工具、无多轮对话、无工具重试或分支；生产可能引入规划、反思或更多工具。  
  - **LLM**：单次调用、简单解析；生产可能用结构化输出（JSON schema）、多步推理或专用 fine-tune 模型。  
- **向量库**：本地 `vector_db/`、sentence-transformers 本地 embedding，适合 demo；生产可能用 Azure AI Search、Pinecone 等托管向量库与专用 embedding 服务。

---

## 9️⃣ 如果我是面试官，我应该怎么介绍这个项目

用 **5 句话**概括价值：

1. **AI 辅助基础设施诊断**：从遥测到根因与缓解的端到端自动化，面向 hyperscale 网络设备（如 TOR/Spine）。  
2. **基于阈值的异常检测**：快速筛出超标事件（延迟、丢包、队列深度），为后续分析缩小范围。  
3. **RAG 驱动推理**：用当前事件描述做语义检索，拉取相似历史 incident 与 runbook，为 LLM 提供可解释证据。  
4. **Agent 式调查流水线**：固定编排「遥测分析 → 检索 incident → 检索 runbook → LLM 推理」，工具化、可扩展。  
5. **结构化诊断报告**：输出根因（排序）、证据、缓解步骤与置信度，便于运维决策与审计，且无 API Key 时仍有 fallback 报告。

---

## 10️⃣ 用简单语言总结

这个项目**本质上是在模拟「超大规模网络基础设施的 AI 运维诊断系统」**：  
把设备遥测当成输入，先用简单规则找出「异常」，再把异常转成一段人话描述；用这段描述去**语义检索**历史故障和预案，并结合**遥测趋势分析**，一起喂给 **LLM**（或 fallback 逻辑）生成**根因假设和处置建议**。  

这种架构在 hyperscale 场景下的价值在于：**把「查历史、对 runbook、写报告」变成一条可复现、可测试的 pipeline**，减少人工排查时间，同时保留「证据来源」（相似 incident / runbook）和「置信度」，便于在真实环境中与告警、工单、变更系统集成，并逐步替换或增强为更复杂的异常检测与多步推理模型。
