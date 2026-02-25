"""
候选日志检索和混合搜索服务
"""

import array
import math
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from database.connection import db_pool
from config import settings
from utils.logging_config import logger
from services.reranker import RerankerService


class RetrievalService:
    """日志检索和搜索服务"""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self._bm25_index = None
        self._all_texts = []

    def reload_bm25(self):
        """从数据库重新加载BM25索引"""
        with db_pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM docs")
            all_logs = cursor.fetchall()
            self._all_texts = [log[0].strip() for log in all_logs if log[0]]

            if len(self._all_texts) > 0:
                tokenized_corpus = [text.split() for text in self._all_texts]
                self._bm25_index = BM25Okapi(tokenized_corpus)
            else:
                self._bm25_index = BM25Okapi([[""]])
                logger.warning(
                    "No logs found in database. BM25 initialized with empty corpus."
                )

    def retrieve_candidate_logs(self, suspicious_signals: List[Dict]) -> List[Dict]:
        """
        基于异常信号使用REGEXP_LIKE检索候选日志。
        返回一个小的、高信噪比的日志子集（最多300条）。
        """
        all_candidates = []

        with db_pool.acquire() as conn:
            cursor = conn.cursor()

            for sig in suspicious_signals:
                try:
                    window_start = pd.to_datetime(sig["window_start"])
                    window_end = window_start + pd.Timedelta(minutes=5)

                    signature = sig["signature"][:200] if sig.get("signature") else ""
                    if not signature:
                        signature = sig.get("template_id", "")

                    regex_pattern = re.escape(signature)
                    regex_pattern = regex_pattern.replace("&lt;NUM&gt;", r"\d+")
                    regex_pattern = regex_pattern.replace(
                        "&lt;IP&gt;", r"\d+\.\d+\.\d+\.\d+"
                    )
                    regex_pattern = regex_pattern.replace(
                        "&lt;BLOCK_ID&gt;", r"blk_[-\w]+"
                    )
                    regex_pattern = regex_pattern.replace("&lt;PATH&gt;", r"/[^\s]+")
                    regex_pattern = regex_pattern.replace("<NUM>", r"\d+")
                    regex_pattern = regex_pattern.replace("<IP>", r"\d+\.\d+\.\d+\.\d+")
                    regex_pattern = regex_pattern.replace("<BLOCK_ID>", r"blk_[-\w]+")
                    regex_pattern = regex_pattern.replace("<PATH>", r"/[^\s]+")

                    regex_pattern_escaped = regex_pattern.replace("'", "''")
                    window_end_plus_5min = window_end + pd.Timedelta(minutes=5)

                    try:
                        cursor.execute(
                            """
                            SELECT id, text, ts
                            FROM docs
                            WHERE REGEXP_LIKE(text, :pattern, 'i')
                              AND REGEXP_LIKE(text, 'ERROR|FATAL|Exception| 404 | 500 ', 'i')
                              AND ts IS NOT NULL
                              AND ts BETWEEN :window_start AND :window_end_plus_5min
                            ORDER BY ts
                            FETCH FIRST 200 ROWS ONLY
                        """,
                            pattern=regex_pattern_escaped,
                            window_start=window_start,
                            window_end_plus_5min=window_end_plus_5min,
                        )
                    except Exception as sql_err:
                        logger.warning(
                            f"REGEXP_LIKE failed, using LIKE fallback: {sql_err}"
                        )
                        signature_like = (
                            signature.replace("'", "''")
                            .replace("%", "\\%")
                            .replace("_", "\\_")
                        )
                        cursor.execute(
                            """
                            SELECT id, text, ts
                            FROM docs
                            WHERE text LIKE :pattern ESCAPE '\\'
                              AND (UPPER(text) LIKE '%ERROR%' OR UPPER(text) LIKE '%FATAL%' OR UPPER(text) LIKE '%EXCEPTION%' OR text LIKE '% 404 %' OR text LIKE '% 500 %')
                              AND (ts IS NULL OR ts BETWEEN :window_start AND :window_end_plus_5min)
                            ORDER BY id
                            FETCH FIRST 200 ROWS ONLY
                        """,
                            pattern=f"%{signature_like}%",
                            window_start=window_start,
                            window_end_plus_5min=window_end_plus_5min,
                        )

                    rows = cursor.fetchall()
                    for row in rows:
                        all_candidates.append(
                            {
                                "id": row[0],
                                "text": row[1],
                                "ts": (
                                    row[2].isoformat()
                                    if row[2] and hasattr(row[2], "isoformat")
                                    else (str(row[2]) if row[2] else None)
                                ),
                            }
                        )
                except Exception as e:
                    logger.error(
                        f"Error retrieving candidates for signal {sig.get('template_id', 'unknown')}: {e}"
                    )
                    continue

        seen_ids = set()
        unique_candidates = []
        for cand in all_candidates:
            if cand["id"] not in seen_ids:
                seen_ids.add(cand["id"])
                unique_candidates.append(cand)

        return unique_candidates[:300]

    def hybrid_search(
        self, query: str, top_k: int, use_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        执行混合搜索，结合向量搜索和BM25，可选重排序
        
        Args:
            query: 用户查询字符串
            top_k: 返回结果数量
            use_rerank: 是否使用重排序（默认：True）
        
        Returns:
            包含retrieved_logs和distances的字典
        """
        try:
            # 步骤1：获取初始候选（重排序时top_k=30，否则top_k*2）
            initial_top_k = 30 if use_rerank else top_k * 2
            
            embedding_response = self.openai_client.embeddings.create(
                model=settings.openai_model, input=query
            )
            query_embedding = embedding_response.data[0].embedding
            query_vector = array.array("f", query_embedding)

            with db_pool.acquire() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
                    FROM docs
                    ORDER BY distance ASC
                    FETCH FIRST :k ROWS ONLY
                """,
                    query_vec=query_vector,
                    k=initial_top_k * 2,
                )
                vector_results = cursor.fetchall()

            bm25_results = []
            if self._bm25_index and len(self._all_texts) > 0:
                tokenized_query = query.split()
                bm25_scores = self._bm25_index.get_scores(tokenized_query)

                if bm25_scores.max() > 0:
                    bm25_results = sorted(
                        range(len(bm25_scores)),
                        key=lambda i: bm25_scores[i],
                        reverse=True,
                    )[: initial_top_k * 2]
                    bm25_results = [
                        (self._all_texts[i], bm25_scores[i]) for i in bm25_results
                    ]

            fused_scores = {}
            text_to_distance = {}

            distances = [d for _, d in vector_results]
            if not distances:
                return {"retrieved_logs": [], "distances": []}

            min_distance = min(distances)
            max_distance = max(distances)

            for text, distance in vector_results:
                text_to_distance[text] = distance
                vector_score_exp = math.exp(-distance * 1.2)

                if max_distance > min_distance:
                    normalized = (distance - min_distance) / (
                        max_distance - min_distance
                    )
                    vector_score_linear = 1.0 - (normalized * 0.5)
                else:
                    vector_score_linear = 1.0

                vector_score = 0.6 * vector_score_exp + 0.4 * vector_score_linear

                if distance == min_distance:
                    vector_score = max(vector_score, 0.7)

                fused_scores[text] = fused_scores.get(text, 0) + vector_score * 0.7

            if bm25_results:
                max_bm25 = max([score for _, score in bm25_results])
                min_bm25 = min([score for _, score in bm25_results])
                for text, score in bm25_results:
                    if max_bm25 > min_bm25:
                        normalized = (score - min_bm25) / (max_bm25 - min_bm25)
                        bm25_norm = 0.5 + (normalized * 0.5)
                    else:
                        bm25_norm = 1.0 if score > 0 else 0.0
                    fused_scores[text] = fused_scores.get(text, 0) + bm25_norm * 0.3

            # 步骤2：获取初始混合搜索结果
            initial_results = sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )[:initial_top_k]

            # 准备用于重排序的文档
            candidate_docs = []
            for text, hybrid_score in initial_results:
                original_distance = text_to_distance.get(text, None)
                candidate_docs.append(
                    {
                        "text": text,
                        "hybrid_score": round(hybrid_score, 4),
                        "distance": (
                            round(original_distance, 4)
                            if original_distance is not None
                            else None
                        ),
                    }
                )

            # 步骤3：如果启用，进行重排序
            if use_rerank and len(candidate_docs) > 0:
                try:
                    reranker = RerankerService()
                    # 重排序并获取前10条作为LLM上下文，但尊重用户的top_k
                    reranked_docs = reranker.rerank(query, candidate_docs, top_k=max(10, top_k))
                    # 只取用户请求的top_k
                    final_docs = reranked_docs[:top_k]
                except Exception as e:
                    logger.warning(f"Reranking failed, using hybrid search results: {e}")
                    final_docs = candidate_docs[:top_k]
            else:
                final_docs = candidate_docs[:top_k]

            # 步骤4：格式化最终结果
            retrieved_logs = []
            for rank, doc in enumerate(final_docs, 1):
                result = {
                    "rank": rank,
                    "text": doc["text"],
                    "hybrid_score": doc["hybrid_score"],
                    "distance": doc.get("distance"),
                }
                # 如果可用，添加rerank_score
                if "rerank_score" in doc:
                    result["rerank_score"] = round(doc["rerank_score"], 4)
                retrieved_logs.append(result)

            return {"retrieved_logs": retrieved_logs, "distances": distances}

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}", exc_info=True)
            raise
