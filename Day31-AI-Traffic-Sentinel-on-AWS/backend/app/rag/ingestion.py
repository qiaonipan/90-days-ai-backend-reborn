"""
RAG 数据摄入优化模块
解决海量、高频网络日志的摄入延迟问题

核心策略：
1. 批处理（Batch Processing）- 减少向量化API调用
2. 异步处理（Async Processing）- 提高并发吞吐量
3. 流式处理（Streaming）- 实时摄入关键日志
4. 索引优化（Index Optimization）- 增量更新而非全量重建
5. 优先级队列（Priority Queue）- 重要日志优先处理
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

import numpy as np
from pgvector.asyncpg import register_vector
import asyncpg
from boto3 import client as boto_client

logger = logging.getLogger(__name__)


class LogPriority(Enum):
    """日志优先级"""
    CRITICAL = 1  # 立即处理
    HIGH = 2      # 高优先级批次
    NORMAL = 3    # 正常批次
    LOW = 4       # 低优先级批次


@dataclass
class TelemetryLog:
    """网络遥测日志"""
    timestamp: datetime
    source_ip: str
    dest_ip: str
    protocol: str
    status_code: int
    bytes_sent: int
    bytes_received: int
    latency_ms: float
    user_agent: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: LogPriority = LogPriority.NORMAL


class BatchIngestionProcessor:
    """
    批处理摄入器 - 核心延迟优化策略
    
    策略：
    - 时间窗口批处理：按时间窗口（如5秒）收集日志
    - 大小限制批处理：达到阈值（如1000条）立即处理
    - 优先级批处理：不同优先级分别批处理
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        batch_timeout: float = 5.0,
        max_concurrent_batches: int = 10,
        db_pool: Optional[asyncpg.Pool] = None,
        bedrock_client: Optional[Any] = None
    ):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        
        # 按优先级分组的批次缓冲区
        self.batch_buffers: Dict[LogPriority, List[TelemetryLog]] = defaultdict(list)
        self.batch_locks: Dict[LogPriority, asyncio.Lock] = {
            priority: asyncio.Lock() for priority in LogPriority
        }
        self.last_batch_time: Dict[LogPriority, float] = defaultdict(float)
        
        # 数据库连接池
        self.db_pool = db_pool
        self.bedrock_client = bedrock_client
        
        # 批处理任务
        self.batch_tasks: List[asyncio.Task] = []
        self.running = False
        
    async def start(self):
        """启动批处理处理器"""
        self.running = True
        # 为每个优先级启动定时批处理任务
        for priority in LogPriority:
            task = asyncio.create_task(self._periodic_batch_processor(priority))
            self.batch_tasks.append(task)
        logger.info("批处理处理器已启动")
    
    async def stop(self):
        """停止批处理处理器"""
        self.running = False
        # 等待所有任务完成
        for task in self.batch_tasks:
            task.cancel()
        await asyncio.gather(*self.batch_tasks, return_exceptions=True)
        # 处理剩余日志
        await self._flush_all_batches()
        logger.info("批处理处理器已停止")
    
    async def ingest(self, log: TelemetryLog):
        """
        异步摄入单条日志
        
        策略：
        - 立即处理CRITICAL优先级日志
        - 其他优先级加入批次缓冲区
        """
        if log.priority == LogPriority.CRITICAL:
            # 关键日志立即处理，不等待批次
            await self._process_single_critical(log)
        else:
            async with self.batch_locks[log.priority]:
                self.batch_buffers[log.priority].append(log)
                
                # 检查是否达到批次大小
                if len(self.batch_buffers[log.priority]) >= self.batch_size:
                    await self._process_batch(log.priority)
    
    async def _periodic_batch_processor(self, priority: LogPriority):
        """定时批处理任务"""
        while self.running:
            try:
                await asyncio.sleep(self.batch_timeout)
                async with self.batch_locks[priority]:
                    if self.batch_buffers[priority]:
                        await self._process_batch(priority)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"批处理任务错误 [{priority}]: {e}")
    
    async def _process_batch(self, priority: LogPriority):
        """处理一个批次"""
        if not self.batch_buffers[priority]:
            return
        
        logs = self.batch_buffers[priority].copy()
        self.batch_buffers[priority].clear()
        self.last_batch_time[priority] = time.time()
        
        # 异步处理批次，不阻塞
        asyncio.create_task(self._vectorize_and_store_batch(logs, priority))
    
    async def _process_single_critical(self, log: TelemetryLog):
        """立即处理关键日志"""
        await self._vectorize_and_store_batch([log], LogPriority.CRITICAL)
    
    async def _vectorize_and_store_batch(
        self,
        logs: List[TelemetryLog],
        priority: LogPriority
    ):
        """
        批量向量化并存储
        
        优化点：
        1. 批量调用Bedrock API（减少API调用次数）
        2. 批量数据库插入（减少数据库往返）
        3. 并发处理多个批次
        """
        try:
            # 1. 批量生成文本表示
            texts = [self._log_to_text(log) for log in logs]
            
            # 2. 批量向量化（一次API调用处理多条）
            vectors = await self._batch_embed(texts)
            
            # 3. 批量插入数据库
            await self._batch_insert(logs, vectors)
            
            logger.info(
                f"成功处理批次 [{priority}]: {len(logs)} 条日志, "
                f"耗时: {time.time() - self.last_batch_time[priority]:.2f}s"
            )
        except Exception as e:
            logger.error(f"批次处理失败 [{priority}]: {e}")
            # 失败重试逻辑（可加入死信队列）
    
    def _log_to_text(self, log: TelemetryLog) -> str:
        """将日志转换为文本表示（用于向量化）"""
        return (
            f"时间: {log.timestamp.isoformat()}, "
            f"源IP: {log.source_ip}, 目标IP: {log.dest_ip}, "
            f"协议: {log.protocol}, 状态码: {log.status_code}, "
            f"延迟: {log.latency_ms}ms, "
            f"发送: {log.bytes_sent}bytes, 接收: {log.bytes_received}bytes"
            + (f", 路径: {log.path}" if log.path else "")
            + (f", UA: {log.user_agent}" if log.user_agent else "")
        )
    
    async def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化
        
        优化：使用Bedrock的批量API或并发调用
        """
        if not self.bedrock_client:
            # 模拟向量化（实际应调用Bedrock）
            return [[0.0] * 1536 for _ in texts]
        
        # 实际实现：调用AWS Bedrock Embedding API
        # 可以使用批量API或并发调用多个单次API
        try:
            # 示例：并发调用（如果Bedrock不支持批量）
            tasks = [
                self._single_embed(text) for text in texts
            ]
            vectors = await asyncio.gather(*tasks)
            return vectors
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            raise
    
    async def _single_embed(self, text: str) -> List[float]:
        """单次向量化（实际应调用Bedrock）"""
        # 实际实现应调用AWS Bedrock
        # 这里返回模拟向量
        await asyncio.sleep(0.01)  # 模拟API延迟
        return [0.0] * 1536
    
    async def _batch_insert(
        self,
        logs: List[TelemetryLog],
        vectors: List[List[float]]
    ):
        """批量插入数据库"""
        if not self.db_pool:
            logger.warning("数据库连接池未配置")
            return
        
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            
            # 批量插入（使用executemany优化）
            values = []
            for log, vector in zip(logs, vectors):
                values.append((
                    log.timestamp,
                    log.source_ip,
                    log.dest_ip,
                    log.protocol,
                    log.status_code,
                    log.latency_ms,
                    log.bytes_sent,
                    log.bytes_received,
                    log.path,
                    log.user_agent,
                    log.metadata,
                    np.array(vector)  # pgvector格式
                ))
            
            await conn.executemany(
                """
                INSERT INTO telemetry_logs (
                    timestamp, source_ip, dest_ip, protocol,
                    status_code, latency_ms, bytes_sent, bytes_received,
                    path, user_agent, metadata, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                values
            )
    
    async def _flush_all_batches(self):
        """刷新所有批次（关闭时调用）"""
        for priority in LogPriority:
            async with self.batch_locks[priority]:
                if self.batch_buffers[priority]:
                    await self._process_batch(priority)


class StreamingIngestionProcessor:
    """
    流式摄入处理器 - 用于实时关键日志
    
    策略：
    - 使用Kafka/Kinesis等流式处理
    - 实时向量化和索引
    - 与批处理并行运行
    """
    
    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        bedrock_client: Optional[Any] = None
    ):
        self.db_pool = db_pool
        self.bedrock_client = bedrock_client
    
    async def ingest_stream(self, log: TelemetryLog):
        """流式摄入单条日志"""
        # 实时向量化
        text = self._log_to_text(log)
        vector = await self._single_embed(text)
        
        # 立即插入（使用连接池，不阻塞）
        await self._insert_single(log, vector)
    
    def _log_to_text(self, log: TelemetryLog) -> str:
        """同BatchIngestionProcessor"""
        return (
            f"时间: {log.timestamp.isoformat()}, "
            f"源IP: {log.source_ip}, 目标IP: {log.dest_ip}, "
            f"协议: {log.protocol}, 状态码: {log.status_code}, "
            f"延迟: {log.latency_ms}ms"
        )
    
    async def _single_embed(self, text: str) -> List[float]:
        """单次向量化"""
        # 实际调用Bedrock
        await asyncio.sleep(0.01)
        return [0.0] * 1536
    
    async def _insert_single(self, log: TelemetryLog, vector: List[float]):
        """单条插入"""
        if not self.db_pool:
            return
        
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(
                """
                INSERT INTO telemetry_logs (
                    timestamp, source_ip, dest_ip, protocol,
                    status_code, latency_ms, bytes_sent, bytes_received,
                    path, user_agent, metadata, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                log.timestamp, log.source_ip, log.dest_ip, log.protocol,
                log.status_code, log.latency_ms, log.bytes_sent,
                log.bytes_received, log.path, log.user_agent,
                log.metadata, np.array(vector)
            )


class HybridIngestionManager:
    """
    混合摄入管理器
    
    策略：
    - 关键日志：流式处理（低延迟）
    - 普通日志：批处理（高吞吐）
    - 自动负载均衡
    """
    
    def __init__(
        self,
        batch_processor: BatchIngestionProcessor,
        stream_processor: StreamingIngestionProcessor
    ):
        self.batch_processor = batch_processor
        self.stream_processor = stream_processor
    
    async def ingest(self, log: TelemetryLog):
        """根据优先级选择处理方式"""
        if log.priority == LogPriority.CRITICAL:
            # 关键日志流式处理
            await self.stream_processor.ingest_stream(log)
        else:
            # 普通日志批处理
            await self.batch_processor.ingest(log)
    
    async def start(self):
        """启动所有处理器"""
        await self.batch_processor.start()
    
    async def stop(self):
        """停止所有处理器"""
        await self.batch_processor.stop()



