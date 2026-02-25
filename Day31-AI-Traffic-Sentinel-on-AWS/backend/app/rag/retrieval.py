"""
RAG 检索优化模块
保证检索相关性和准确性

核心策略：
1. 混合检索（Hybrid Search）- 向量检索 + 关键词检索
2. 元数据过滤（Metadata Filtering）- 时间范围、IP、协议等
3. 重排序（Re-ranking）- 使用更强大的模型重新排序
4. 查询扩展（Query Expansion）- 增强查询语义
5. 上下文窗口优化（Context Window）- 选择最相关的片段
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging

import numpy as np
from pgvector.asyncpg import register_vector
import asyncpg
from boto3 import client as boto_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQuery:
    """检索查询"""
    query_text: str
    time_range: Optional[Tuple[datetime, datetime]] = None
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    protocol: Optional[str] = None
    status_code: Optional[int] = None
    min_similarity: float = 0.7
    top_k: int = 10


@dataclass
class RetrievedLog:
    """检索到的日志"""
    log_id: int
    timestamp: datetime
    source_ip: str
    dest_ip: str
    protocol: str
    status_code: int
    latency_ms: float
    similarity_score: float
    hybrid_score: float  # 混合检索分数
    text_content: str


class HybridRetriever:
    """
    混合检索器
    
    策略：
    - 向量检索（语义相似度）
    - 关键词检索（BM25/全文搜索）
    - 分数融合（Reciprocal Rank Fusion）
    """
    
    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        bedrock_client: Optional[Any] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        self.db_pool = db_pool
        self.bedrock_client = bedrock_client
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    async def retrieve(
        self,
        query: RetrievalQuery
    ) -> List[RetrievedLog]:
        """
        混合检索
        
        流程：
        1. 查询扩展
        2. 并行执行向量检索和关键词检索
        3. 元数据过滤
        4. 分数融合
        5. 重排序（可选）
        """
        # 1. 查询扩展
        expanded_query = await self._expand_query(query.query_text)
        
        # 2. 并行检索
        vector_results, keyword_results = await asyncio.gather(
            self._vector_search(expanded_query, query),
            self._keyword_search(query.query_text, query)
        )
        
        # 3. 分数融合
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results
        )
        
        # 4. 元数据过滤（在融合后应用，避免过早过滤）
        filtered_results = self._apply_metadata_filters(
            fused_results,
            query
        )
        
        # 5. 重排序（可选，使用更强的模型）
        if len(filtered_results) > 0:
            reranked_results = await self._rerank(
                query.query_text,
                filtered_results
            )
            return reranked_results[:query.top_k]
        
        return filtered_results[:query.top_k]
    
    async def _expand_query(self, query_text: str) -> str:
        """
        查询扩展
        
        策略：
        - 使用LLM生成同义词和相关术语
        - 提取关键实体（IP、协议等）
        """
        if not self.bedrock_client:
            return query_text
        
        try:
            # 实际实现：调用Bedrock生成扩展查询
            # 示例提示词：
            # "扩展以下网络日志查询，添加同义词和相关术语：{query_text}"
            expanded = await self._llm_expand(query_text)
            return expanded
        except Exception as e:
            logger.warning(f"查询扩展失败，使用原查询: {e}")
            return query_text
    
    async def _llm_expand(self, query: str) -> str:
        """使用LLM扩展查询"""
        # 实际实现：调用Bedrock
        await asyncio.sleep(0.1)
        return query  # 简化实现
    
    async def _vector_search(
        self,
        query_text: str,
        query: RetrievalQuery
    ) -> List[Tuple[int, float]]:
        """
        向量检索
        
        使用pgvector的相似度搜索
        """
        if not self.db_pool:
            return []
        
        # 向量化查询
        query_vector = await self._embed_query(query_text)
        
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
            
            # 构建SQL查询（带元数据过滤）
            sql = """
                SELECT 
                    id, timestamp, source_ip, dest_ip, protocol,
                    status_code, latency_ms, bytes_sent, bytes_received,
                    path, user_agent, metadata,
                    1 - (embedding <=> $1::vector) as similarity
                FROM telemetry_logs
                WHERE 1 - (embedding <=> $1::vector) >= $2
            """
            params = [np.array(query_vector), query.min_similarity]
            param_idx = 3
            
            # 添加元数据过滤条件
            filters = []
            if query.time_range:
                filters.append(f"timestamp BETWEEN ${param_idx} AND ${param_idx + 1}")
                params.extend(query.time_range)
                param_idx += 2
            if query.source_ip:
                filters.append(f"source_ip = ${param_idx}")
                params.append(query.source_ip)
                param_idx += 1
            if query.dest_ip:
                filters.append(f"dest_ip = ${param_idx}")
                params.append(query.dest_ip)
                param_idx += 1
            if query.protocol:
                filters.append(f"protocol = ${param_idx}")
                params.append(query.protocol)
                param_idx += 1
            if query.status_code:
                filters.append(f"status_code = ${param_idx}")
                params.append(query.status_code)
                param_idx += 1
            
            if filters:
                sql += " AND " + " AND ".join(filters)
            
            sql += " ORDER BY similarity DESC LIMIT $%d" % param_idx
            params.append(query.top_k * 2)  # 获取更多结果用于融合
            
            rows = await conn.fetch(sql, *params)
            
            results = []
            for row in rows:
                results.append((
                    row['id'],
                    float(row['similarity'])
                ))
            
            return results
    
    async def _keyword_search(
        self,
        query_text: str,
        query: RetrievalQuery
    ) -> List[Tuple[int, float]]:
        """
        关键词检索
        
        使用PostgreSQL全文搜索（tsvector/tsquery）
        """
        if not self.db_pool:
            return []
        
        async with self.db_pool.acquire() as conn:
            # 构建全文搜索查询
            sql = """
                SELECT 
                    id, timestamp, source_ip, dest_ip, protocol,
                    status_code, latency_ms,
                    ts_rank_cd(
                        to_tsvector('english', 
                            COALESCE(path, '') || ' ' ||
                            COALESCE(user_agent, '') || ' ' ||
                            source_ip || ' ' || dest_ip || ' ' ||
                            protocol
                        ),
                        plainto_tsquery('english', $1)
                    ) as rank
                FROM telemetry_logs
                WHERE to_tsvector('english', 
                    COALESCE(path, '') || ' ' ||
                    COALESCE(user_agent, '') || ' ' ||
                    source_ip || ' ' || dest_ip || ' ' ||
                    protocol
                ) @@ plainto_tsquery('english', $1)
            """
            
            params = [query_text]
            param_idx = 2
            
            # 添加元数据过滤
            filters = []
            if query.time_range:
                filters.append(f"timestamp BETWEEN ${param_idx} AND ${param_idx + 1}")
                params.extend(query.time_range)
                param_idx += 2
            if query.source_ip:
                filters.append(f"source_ip = ${param_idx}")
                params.append(query.source_ip)
                param_idx += 1
            if query.dest_ip:
                filters.append(f"dest_ip = ${param_idx}")
                params.append(query.dest_ip)
                param_idx += 1
            if query.protocol:
                filters.append(f"protocol = ${param_idx}")
                params.append(query.protocol)
                param_idx += 1
            if query.status_code:
                filters.append(f"status_code = ${param_idx}")
                params.append(query.status_code)
                param_idx += 1
            
            if filters:
                sql += " AND " + " AND ".join(filters)
            
            sql += " ORDER BY rank DESC LIMIT $%d" % param_idx
            params.append(query.top_k * 2)
            
            rows = await conn.fetch(sql, *params)
            
            results = []
            for row in rows:
                # 归一化rank分数到[0, 1]
                normalized_score = min(float(row['rank']) * 10, 1.0)
                results.append((
                    row['id'],
                    normalized_score
                ))
            
            return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]]
    ) -> Dict[int, float]:
        """
        倒数排名融合（RRF）
        
        公式：RRF_score = sum(1 / (k + rank))
        k通常为60
        """
        k = 60
        fused_scores: Dict[int, float] = {}
        
        # 向量检索结果
        for rank, (log_id, score) in enumerate(vector_results, 1):
            rrf_score = 1.0 / (k + rank)
            fused_scores[log_id] = fused_scores.get(log_id, 0) + \
                self.vector_weight * rrf_score
        
        # 关键词检索结果
        for rank, (log_id, score) in enumerate(keyword_results, 1):
            rrf_score = 1.0 / (k + rank)
            fused_scores[log_id] = fused_scores.get(log_id, 0) + \
                self.keyword_weight * rrf_score
        
        return fused_scores
    
    def _apply_metadata_filters(
        self,
        fused_scores: Dict[int, float],
        query: RetrievalQuery
    ) -> List[Tuple[int, float]]:
        """应用元数据过滤（已在SQL中应用，这里做二次验证）"""
        # 大部分过滤已在SQL中完成
        # 这里可以添加额外的业务逻辑过滤
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_results
    
    async def _rerank(
        self,
        query_text: str,
        candidates: List[Tuple[int, float]]
    ) -> List[RetrievedLog]:
        """
        重排序
        
        使用更强的LLM模型对候选结果重新排序
        提高最终相关性
        """
        if not self.bedrock_client or len(candidates) == 0:
            # 如果没有重排序，直接返回
            return await self._fetch_log_details(candidates)
        
        try:
            # 获取候选日志详情
            candidate_logs = await self._fetch_log_details(candidates)
            
            # 使用LLM重排序
            reranked = await self._llm_rerank(query_text, candidate_logs)
            return reranked
        except Exception as e:
            logger.warning(f"重排序失败，使用原始排序: {e}")
            return await self._fetch_log_details(candidates)
    
    async def _llm_rerank(
        self,
        query_text: str,
        candidate_logs: List[RetrievedLog]
    ) -> List[RetrievedLog]:
        """
        使用LLM重排序
        
        策略：
        - 构建提示词，包含查询和候选日志
        - 让LLM按相关性排序
        - 返回重排序后的结果
        """
        # 实际实现：调用Bedrock进行重排序
        # 这里简化处理，实际应使用cross-encoder模型
        await asyncio.sleep(0.1)
        return candidate_logs  # 简化实现
    
    async def _fetch_log_details(
        self,
        candidates: List[Tuple[int, float]]
    ) -> List[RetrievedLog]:
        """获取日志详情"""
        if not self.db_pool or not candidates:
            return []
        
        log_ids = [log_id for log_id, _ in candidates]
        score_map = {log_id: score for log_id, score in candidates}
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    id, timestamp, source_ip, dest_ip, protocol,
                    status_code, latency_ms, bytes_sent, bytes_received,
                    path, user_agent, metadata
                FROM telemetry_logs
                WHERE id = ANY($1)
                """,
                log_ids
            )
            
            results = []
            for row in rows:
                # 构建文本内容
                text_content = (
                    f"时间: {row['timestamp']}, "
                    f"源IP: {row['source_ip']}, 目标IP: {row['dest_ip']}, "
                    f"协议: {row['protocol']}, 状态码: {row['status_code']}, "
                    f"延迟: {row['latency_ms']}ms"
                )
                
                results.append(RetrievedLog(
                    log_id=row['id'],
                    timestamp=row['timestamp'],
                    source_ip=row['source_ip'],
                    dest_ip=row['dest_ip'],
                    protocol=row['protocol'],
                    status_code=row['status_code'],
                    latency_ms=row['latency_ms'],
                    similarity_score=score_map.get(row['id'], 0.0),
                    hybrid_score=score_map.get(row['id'], 0.0),
                    text_content=text_content
                ))
            
            # 按分数排序
            results.sort(key=lambda x: x.hybrid_score, reverse=True)
            return results
    
    async def _embed_query(self, query_text: str) -> List[float]:
        """向量化查询"""
        if not self.bedrock_client:
            return [0.0] * 1536
        
        # 实际实现：调用Bedrock Embedding API
        await asyncio.sleep(0.01)
        return [0.0] * 1536


class ContextualRetriever:
    """
    上下文感知检索器
    
    策略：
    - 时间窗口感知（最近日志权重更高）
    - 异常检测增强（异常日志优先）
    - 多轮对话上下文（考虑历史查询）
    """
    
    def __init__(
        self,
        base_retriever: HybridRetriever,
        time_decay_factor: float = 0.1
    ):
        self.base_retriever = base_retriever
        self.time_decay_factor = time_decay_factor
    
    async def retrieve_with_context(
        self,
        query: RetrievalQuery,
        conversation_history: Optional[List[str]] = None
    ) -> List[RetrievedLog]:
        """
        带上下文的检索
        
        策略：
        1. 扩展查询（包含历史上下文）
        2. 基础检索
        3. 时间衰减调整
        4. 异常检测增强
        """
        # 1. 构建上下文增强查询
        enhanced_query = self._build_contextual_query(
            query,
            conversation_history
        )
        
        # 2. 基础检索
        results = await self.base_retriever.retrieve(enhanced_query)
        
        # 3. 应用时间衰减
        results = self._apply_time_decay(results)
        
        # 4. 异常检测增强（如果有异常日志，提升其权重）
        results = await self._enhance_anomalies(results)
        
        return results
    
    def _build_contextual_query(
        self,
        query: RetrievalQuery,
        history: Optional[List[str]]
    ) -> RetrievalQuery:
        """构建上下文增强查询"""
        if not history:
            return query
        
        # 将历史上下文添加到查询文本
        context_text = " ".join(history[-3:])  # 最近3轮
        enhanced_text = f"{context_text} {query.query_text}"
        
        return RetrievalQuery(
            query_text=enhanced_text,
            time_range=query.time_range,
            source_ip=query.source_ip,
            dest_ip=query.dest_ip,
            protocol=query.protocol,
            status_code=query.status_code,
            min_similarity=query.min_similarity,
            top_k=query.top_k
        )
    
    def _apply_time_decay(
        self,
        results: List[RetrievedLog]
    ) -> List[RetrievedLog]:
        """应用时间衰减（最近日志权重更高）"""
        if not results:
            return results
        
        now = datetime.now()
        for result in results:
            age_hours = (now - result.timestamp).total_seconds() / 3600
            # 时间衰减：exp(-decay_factor * age_hours)
            time_boost = np.exp(-self.time_decay_factor * age_hours)
            result.hybrid_score *= (1 + time_boost * 0.2)  # 最多提升20%
        
        # 重新排序
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results
    
    async def _enhance_anomalies(
        self,
        results: List[RetrievedLog]
    ) -> List[RetrievedLog]:
        """异常检测增强"""
        # 检测异常日志（高延迟、错误状态码等）
        for result in results:
            is_anomaly = (
                result.latency_ms > 1000 or  # 高延迟
                result.status_code >= 500 or  # 服务器错误
                result.status_code == 0  # 连接失败
            )
            if is_anomaly:
                result.hybrid_score *= 1.3  # 提升30%权重
        
        # 重新排序
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results



