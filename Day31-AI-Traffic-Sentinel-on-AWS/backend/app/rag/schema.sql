-- RAG系统数据库Schema
-- PostgreSQL + pgvector扩展

-- 启用pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 网络遥测日志表
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    source_ip INET NOT NULL,
    dest_ip INET NOT NULL,
    protocol VARCHAR(20) NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms FLOAT NOT NULL,
    bytes_sent BIGINT NOT NULL,
    bytes_received BIGINT NOT NULL,
    path TEXT,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- 向量嵌入（使用pgvector）
    embedding vector(1536),  -- Bedrock Titan Embeddings维度
    
    -- 索引
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 向量相似度搜索索引（HNSW算法，高性能）
CREATE INDEX IF NOT EXISTS telemetry_logs_embedding_idx 
ON telemetry_logs 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 时间范围索引（用于时间过滤）
CREATE INDEX IF NOT EXISTS telemetry_logs_timestamp_idx 
ON telemetry_logs (timestamp DESC);

-- IP地址索引（用于IP过滤）
CREATE INDEX IF NOT EXISTS telemetry_logs_source_ip_idx 
ON telemetry_logs (source_ip);
CREATE INDEX IF NOT EXISTS telemetry_logs_dest_ip_idx 
ON telemetry_logs (dest_ip);

-- 协议和状态码索引（用于过滤）
CREATE INDEX IF NOT EXISTS telemetry_logs_protocol_idx 
ON telemetry_logs (protocol);
CREATE INDEX IF NOT EXISTS telemetry_logs_status_code_idx 
ON telemetry_logs (status_code);

-- 全文搜索索引（用于关键词检索）
CREATE INDEX IF NOT EXISTS telemetry_logs_fulltext_idx 
ON telemetry_logs 
USING gin(to_tsvector('english', 
    COALESCE(path, '') || ' ' ||
    COALESCE(user_agent, '') || ' ' ||
    source_ip::text || ' ' || dest_ip::text || ' ' ||
    protocol
));

-- 复合索引（优化常见查询模式）
CREATE INDEX IF NOT EXISTS telemetry_logs_time_status_idx 
ON telemetry_logs (timestamp DESC, status_code);

CREATE INDEX IF NOT EXISTS telemetry_logs_time_source_idx 
ON telemetry_logs (timestamp DESC, source_ip);

-- 分区表（可选，用于超大规模数据）
-- 按月分区，提高查询性能
-- CREATE TABLE telemetry_logs_2024_01 PARTITION OF telemetry_logs
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 性能优化视图
CREATE OR REPLACE VIEW telemetry_logs_summary AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    source_ip,
    protocol,
    status_code,
    COUNT(*) as log_count,
    AVG(latency_ms) as avg_latency,
    MAX(latency_ms) as max_latency,
    SUM(bytes_sent) as total_bytes_sent,
    SUM(bytes_received) as total_bytes_received
FROM telemetry_logs
GROUP BY DATE_TRUNC('hour', timestamp), source_ip, protocol, status_code;

-- 清理旧数据函数（数据保留策略）
CREATE OR REPLACE FUNCTION cleanup_old_logs(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM telemetry_logs
    WHERE timestamp < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 示例：清理30天前的数据
-- SELECT cleanup_old_logs(30);



