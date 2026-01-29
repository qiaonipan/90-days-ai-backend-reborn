"""
使用CrossEncoder进行查询-文档相关性评分的重排序服务
"""

from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from functools import lru_cache
from utils.logging_config import logger


class RerankerService:
    """使用CrossEncoder重排序搜索结果的服务"""

    _model: CrossEncoder = None
    _model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def get_model(cls) -> CrossEncoder:
        """获取CrossEncoder模型单例"""
        if cls._model is None:
            logger.info(f"Loading CrossEncoder model: {cls._model_name}")
            try:
                cls._model = CrossEncoder(cls._model_name)
                logger.info("CrossEncoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model: {e}", exc_info=True)
                raise
        return cls._model

    @classmethod
    def warmup(cls):
        """使用虚拟数据预热模型以避免冷启动延迟"""
        try:
            model = cls.get_model()
            # 使用虚拟查询-文档对进行预热
            dummy_pairs = [
                ["What is the error?", "This is a test log entry with error message"],
                ["Why did it fail?", "The system encountered a failure due to timeout"],
            ]
            model.predict(dummy_pairs)
            logger.info("CrossEncoder model warmed up successfully")
        except Exception as e:
            logger.warning(f"Reranker warmup failed: {e}", exc_info=True)

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        基于查询-文档相关性重排序文档

        Args:
            query: 用户查询字符串
            documents: 包含'text'字段的文档字典列表
            top_k: 重排序后返回的top结果数量

        Returns:
            添加了'rerank_score'的重排序文档列表
        """
        if not documents:
            return []

        try:
            model = self.get_model()

            # 为CrossEncoder准备查询-文档对
            pairs = [[query, doc.get("text", "")] for doc in documents]

            # 获取重排序分数
            rerank_scores = model.predict(pairs)

            # 为每个文档添加rerank_score
            for doc, score in zip(documents, rerank_scores):
                doc["rerank_score"] = float(score)

            # 按rerank_score降序排序并返回top_k
            reranked = sorted(
                documents, key=lambda x: x.get("rerank_score", 0.0), reverse=True
            )[:top_k]

            logger.debug(
                f"Reranked {len(documents)} documents, returning top {len(reranked)}"
            )

            return reranked

        except Exception as e:
            logger.error(f"Error in reranking: {e}", exc_info=True)
            # 回退：如果重排序失败，返回原始文档
            return documents[:top_k]

