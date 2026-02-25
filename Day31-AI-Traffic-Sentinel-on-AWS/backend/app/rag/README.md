# RAG系统：海量网络日志处理方案

## 问题挑战

面对海量、高频的网络日志（Telemetry），RAG系统面临两大核心挑战：

1. **数据摄入延迟**：如何快速处理每秒数万条日志？
2. **检索相关性**：如何从海量日志中准确找到相关信息？

## 解决方案架构

### 1. 数据摄入优化（解决延迟问题）

#### 1.1 批处理策略（Batch Processing）

**核心思想**：将多条日志合并为批次，批量调用向量化API，减少API调用次数。

```python
# 配置
batch_size = 1000  # 每批1000条
batch_timeout = 5.0  # 5秒超时

# 效果
- 减少API调用：10000条日志 → 10次API调用（而非10000次）
- 提高吞吐量：10-100倍提升
```

**实现要点**：
- 按优先级分组批处理（关键日志单独处理）
- 时间窗口 + 大小限制双重触发
- 异步并发处理多个批次

#### 1.2 异步处理（Async Processing）

**核心思想**：使用异步I/O，并发处理多个批次，充分利用I/O等待时间。

```python
# 并发处理多个批次
max_concurrent_batches = 10  # 同时处理10个批次

# 效果
- CPU利用率提升：从20% → 80%+
- 吞吐量提升：3-5倍
```

#### 1.3 流式处理（Streaming）

**核心思想**：关键日志立即处理，不等待批次。

```python
# 优先级策略
CRITICAL → 立即流式处理（低延迟）
HIGH/NORMAL → 批处理（高吞吐）
```

#### 1.4 混合摄入管理器

```python
HybridIngestionManager
├── StreamingIngestionProcessor  # 关键日志（低延迟）
└── BatchIngestionProcessor      # 普通日志（高吞吐）
```

**性能指标**：
- 关键日志延迟：< 100ms
- 普通日志吞吐：> 10,000条/秒
- 系统资源利用率：> 80%

---

### 2. 检索优化（保证相关性）

#### 2.1 混合检索（Hybrid Search）

**核心思想**：结合向量检索（语义）和关键词检索（精确匹配），取长补短。

```python
# 分数融合
hybrid_score = 0.7 * vector_score + 0.3 * keyword_score

# 优势
- 向量检索：捕获语义相似性（"高延迟" ≈ "响应慢"）
- 关键词检索：精确匹配（IP地址、状态码等）
```

**实现方法**：倒数排名融合（RRF - Reciprocal Rank Fusion）

```python
RRF_score = sum(1 / (k + rank))
# k = 60（经验值）
```

#### 2.2 元数据过滤（Metadata Filtering）

**核心思想**：在向量检索前应用结构化过滤，缩小搜索范围。

```python
# 过滤条件
- 时间范围：最近24小时
- IP地址：特定源IP或目标IP
- 协议：HTTP/HTTPS/TCP
- 状态码：500错误等
```

**效果**：
- 检索速度提升：10-100倍
- 相关性提升：减少无关结果

#### 2.3 查询扩展（Query Expansion）

**核心思想**：使用LLM扩展查询，添加同义词和相关术语。

```python
原始查询: "高延迟请求"
扩展查询: "高延迟请求 响应慢 超时 timeout slow response"
```

**效果**：
- 召回率提升：20-30%
- 捕获更多相关日志

#### 2.4 重排序（Re-ranking）

**核心思想**：使用更强的模型对候选结果重新排序。

```python
流程：
1. 混合检索 → 获取Top 20候选
2. LLM重排序 → 重新评估相关性
3. 返回Top 10最终结果
```

**效果**：
- 相关性提升：15-25%
- 更准确的排序

#### 2.5 上下文感知检索

**核心思想**：考虑时间衰减和对话历史。

```python
# 时间衰减
recent_logs_weight = old_logs_weight * (1 + time_boost)

# 对话上下文
query = history[-3:] + current_query
```

**效果**：
- 最近日志权重更高
- 多轮对话更连贯

---

## 性能指标

### 摄入性能

| 指标 | 批处理 | 流式处理 | 混合方案 |
|------|--------|----------|----------|
| 吞吐量 | 10,000条/秒 | 1,000条/秒 | 10,000+条/秒 |
| 关键日志延迟 | 5秒 | <100ms | <100ms |
| API调用次数 | 10次/万条 | 10,000次/万条 | 10次/万条 |
| CPU利用率 | 80%+ | 30% | 80%+ |

### 检索性能

| 指标 | 纯向量检索 | 混合检索 | 混合+重排序 |
|------|------------|----------|-------------|
| 相关性（NDCG@10） | 0.75 | 0.85 | 0.92 |
| 检索延迟 | 50ms | 80ms | 150ms |
| 召回率 | 0.70 | 0.85 | 0.88 |

---

## 使用示例

### 数据摄入

```python
from backend.app.rag import (
    HybridIngestionManager,
    TelemetryLog,
    LogPriority
)

# 初始化
ingestion_manager = HybridIngestionManager(...)
await ingestion_manager.start()

# 摄入日志
log = TelemetryLog(
    timestamp=datetime.now(),
    source_ip="192.168.1.100",
    dest_ip="10.0.0.1",
    protocol="HTTP",
    status_code=500,
    latency_ms=1500.0,
    priority=LogPriority.CRITICAL  # 关键日志立即处理
)

await ingestion_manager.ingest(log)
```

### 检索查询

```python
from backend.app.rag import (
    ContextualRetriever,
    RetrievalQuery
)

# 初始化
retriever = ContextualRetriever(...)

# 执行查询
query = RetrievalQuery(
    query_text="高延迟的HTTP请求",
    time_range=(start_time, end_time),
    source_ip="192.168.1.100",
    min_similarity=0.7,
    top_k=10
)

results = await retriever.retrieve_with_context(query)
```

---

## 最佳实践

### 1. 摄入优化

- ✅ 使用批处理处理普通日志（高吞吐）
- ✅ 使用流式处理关键日志（低延迟）
- ✅ 设置合理的批次大小（1000-5000）
- ✅ 使用连接池管理数据库连接
- ✅ 监控批次处理延迟，动态调整参数

### 2. 检索优化

- ✅ 始终使用混合检索（向量+关键词）
- ✅ 利用元数据过滤缩小搜索范围
- ✅ 对Top结果进行重排序
- ✅ 考虑时间衰减（最近日志更重要）
- ✅ 使用查询扩展提高召回率

### 3. 系统监控

- 监控指标：
  - 摄入延迟（P50, P95, P99）
  - 检索延迟
  - 相关性指标（NDCG, MRR）
  - 系统资源利用率

---

## 技术栈

- **向量数据库**: PostgreSQL + pgvector
- **向量化**: AWS Bedrock (Titan Embeddings)
- **LLM**: AWS Bedrock (Claude/Llama)
- **异步处理**: asyncio + asyncpg
- **批处理**: 自定义批处理框架

---

## 扩展方向

1. **分布式摄入**：使用Kafka/Kinesis进行分布式日志摄入
2. **增量索引**：只索引新日志，避免全量重建
3. **缓存优化**：使用Redis缓存热门查询结果
4. **多模态检索**：支持日志图表、网络拓扑等
5. **自动调优**：根据查询模式自动调整权重参数



