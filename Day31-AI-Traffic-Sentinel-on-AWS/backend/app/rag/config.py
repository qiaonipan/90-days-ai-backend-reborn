"""
RAG系统配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IngestionConfig:
    """摄入配置"""
    # 批处理配置
    batch_size: int = 1000  # 批次大小
    batch_timeout: float = 5.0  # 批次超时（秒）
    max_concurrent_batches: int = 10  # 最大并发批次
    
    # 优先级阈值
    critical_latency_threshold: float = 1000.0  # 关键日志延迟阈值（ms）
    critical_status_codes: list = None  # 关键状态码列表
    
    # 数据库配置
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "traffic_sentinel"
    db_user: str = "postgres"
    db_password: str = ""
    db_pool_size: int = 20  # 连接池大小
    db_max_overflow: int = 10
    
    # Bedrock配置
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.titan-embed-text-v1"
    bedrock_embedding_dim: int = 1536
    
    def __post_init__(self):
        if self.critical_status_codes is None:
            self.critical_status_codes = [500, 502, 503, 504, 0]


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 混合检索权重
    vector_weight: float = 0.7  # 向量检索权重
    keyword_weight: float = 0.3  # 关键词检索权重
    
    # 默认检索参数
    default_top_k: int = 10  # 默认返回数量
    default_min_similarity: float = 0.7  # 默认最小相似度
    
    # 重排序配置
    enable_reranking: bool = True  # 是否启用重排序
    rerank_model: str = "claude-3-sonnet"  # 重排序模型
    
    # 时间衰减
    time_decay_factor: float = 0.1  # 时间衰减因子
    
    # 查询扩展
    enable_query_expansion: bool = True  # 是否启用查询扩展
    
    # 数据库配置（同摄入）
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "traffic_sentinel"
    db_user: str = "postgres"
    db_password: str = ""
    db_pool_size: int = 20
    
    # Bedrock配置（同摄入）
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.titan-embed-text-v1"
    bedrock_embedding_dim: int = 1536



