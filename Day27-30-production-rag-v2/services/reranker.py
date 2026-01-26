"""
Reranker service using CrossEncoder for query-document relevance scoring
"""

from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from functools import lru_cache
from utils.logging_config import logger


class RerankerService:
    """Service for reranking search results using CrossEncoder"""

    _model: CrossEncoder = None
    _model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def get_model(cls) -> CrossEncoder:
        """Get CrossEncoder model singleton"""
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
        """Warmup the model with dummy data to avoid cold start delay"""
        try:
            model = cls.get_model()
            # Warmup with dummy query-document pairs
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
        Rerank documents based on query-document relevance

        Args:
            query: User query string
            documents: List of document dicts with 'text' field
            top_k: Number of top results to return after reranking

        Returns:
            List of reranked documents with 'rerank_score' added
        """
        if not documents:
            return []

        try:
            model = self.get_model()

            # Prepare query-document pairs for CrossEncoder
            pairs = [[query, doc.get("text", "")] for doc in documents]

            # Get rerank scores
            rerank_scores = model.predict(pairs)

            # Add rerank_score to each document
            for doc, score in zip(documents, rerank_scores):
                doc["rerank_score"] = float(score)

            # Sort by rerank_score descending and return top_k
            reranked = sorted(
                documents, key=lambda x: x.get("rerank_score", 0.0), reverse=True
            )[:top_k]

            logger.debug(
                f"Reranked {len(documents)} documents, returning top {len(reranked)}"
            )

            return reranked

        except Exception as e:
            logger.error(f"Error in reranking: {e}", exc_info=True)
            # Fallback: return original documents if reranking fails
            return documents[:top_k]

