"""
Retrieval service for candidate log retrieval and hybrid search
"""
import array
import math
import re
import pandas as pd
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from database.connection import db_pool
from config import settings
from utils.logging_config import logger


class RetrievalService:
    """Service for retrieving and searching logs"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self._bm25_index = None
        self._all_texts = []
    
    def reload_bm25(self):
        """Reload BM25 index from database"""
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
                logger.warning("No logs found in database. BM25 initialized with empty corpus.")
    
    def retrieve_candidate_logs(self, suspicious_signals: List[Dict]) -> List[Dict]:
        """
        Retrieve candidate logs based on anomaly signals using REGEXP_LIKE.
        Returns a small, high-signal-to-noise subset of logs (max 300).
        """
        all_candidates = []
        
        with db_pool.acquire() as conn:
            cursor = conn.cursor()
            
            for sig in suspicious_signals:
                try:
                    window_start = pd.to_datetime(sig['window_start'])
                    window_end = window_start + pd.Timedelta(minutes=5)
                    
                    signature = sig['signature'][:200] if sig.get('signature') else ""
                    if not signature:
                        signature = sig.get('template_id', '')
                    
                    regex_pattern = re.escape(signature)
                    regex_pattern = regex_pattern.replace('&lt;NUM&gt;', r'\d+')
                    regex_pattern = regex_pattern.replace('&lt;IP&gt;', r'\d+\.\d+\.\d+\.\d+')
                    regex_pattern = regex_pattern.replace('&lt;BLOCK_ID&gt;', r'blk_[-\w]+')
                    regex_pattern = regex_pattern.replace('&lt;PATH&gt;', r'/[^\s]+')
                    regex_pattern = regex_pattern.replace('<NUM>', r'\d+')
                    regex_pattern = regex_pattern.replace('<IP>', r'\d+\.\d+\.\d+\.\d+')
                    regex_pattern = regex_pattern.replace('<BLOCK_ID>', r'blk_[-\w]+')
                    regex_pattern = regex_pattern.replace('<PATH>', r'/[^\s]+')
                    
                    regex_pattern_escaped = regex_pattern.replace("'", "''")
                    window_end_plus_5min = window_end + pd.Timedelta(minutes=5)
                    
                    try:
                        cursor.execute("""
                            SELECT id, text, ts
                            FROM docs
                            WHERE REGEXP_LIKE(text, :pattern, 'i')
                              AND REGEXP_LIKE(text, 'ERROR|FATAL|Exception| 404 | 500 ', 'i')
                              AND ts IS NOT NULL
                              AND ts BETWEEN :window_start AND :window_end_plus_5min
                            ORDER BY ts
                            FETCH FIRST 200 ROWS ONLY
                        """, pattern=regex_pattern_escaped, window_start=window_start, window_end_plus_5min=window_end_plus_5min)
                    except Exception as sql_err:
                        logger.warning(f"REGEXP_LIKE failed, using LIKE fallback: {sql_err}")
                        signature_like = signature.replace("'", "''").replace('%', '\\%').replace('_', '\\_')
                        cursor.execute("""
                            SELECT id, text, ts
                            FROM docs
                            WHERE text LIKE :pattern ESCAPE '\\'
                              AND (UPPER(text) LIKE '%ERROR%' OR UPPER(text) LIKE '%FATAL%' OR UPPER(text) LIKE '%EXCEPTION%' OR text LIKE '% 404 %' OR text LIKE '% 500 %')
                              AND (ts IS NULL OR ts BETWEEN :window_start AND :window_end_plus_5min)
                            ORDER BY id
                            FETCH FIRST 200 ROWS ONLY
                        """, pattern=f'%{signature_like}%', window_start=window_start, window_end_plus_5min=window_end_plus_5min)
                    
                    rows = cursor.fetchall()
                    for row in rows:
                        all_candidates.append({
                            'id': row[0],
                            'text': row[1],
                            'ts': row[2].isoformat() if row[2] and hasattr(row[2], 'isoformat') else (str(row[2]) if row[2] else None)
                        })
                except Exception as e:
                    logger.error(f"Error retrieving candidates for signal {sig.get('template_id', 'unknown')}: {e}")
                    continue
        
        seen_ids = set()
        unique_candidates = []
        for cand in all_candidates:
            if cand['id'] not in seen_ids:
                seen_ids.add(cand['id'])
                unique_candidates.append(cand)
        
        return unique_candidates[:300]
    
    def hybrid_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Perform hybrid search combining vector and BM25"""
        try:
            embedding_response = self.openai_client.embeddings.create(
                model=settings.openai_model,
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            query_vector = array.array("f", query_embedding)
            
            with db_pool.acquire() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
                    FROM docs
                    ORDER BY distance ASC
                    FETCH FIRST :k ROWS ONLY
                """, query_vec=query_vector, k=top_k * 2)
                vector_results = cursor.fetchall()
            
            bm25_results = []
            if self._bm25_index and len(self._all_texts) > 0:
                tokenized_query = query.split()
                bm25_scores = self._bm25_index.get_scores(tokenized_query)
                
                if bm25_scores.max() > 0:
                    bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
                    bm25_results = [(self._all_texts[i], bm25_scores[i]) for i in bm25_results]
            
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
                    normalized = (distance - min_distance) / (max_distance - min_distance)
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
            
            final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            retrieved_logs = []
            for rank, (text, hybrid_score) in enumerate(final_results, 1):
                original_distance = text_to_distance.get(text, None)
                retrieved_logs.append({
                    "rank": rank,
                    "text": text,
                    "hybrid_score": round(hybrid_score, 4),
                    "distance": round(original_distance, 4) if original_distance is not None else None
                })
            
            return {
                "retrieved_logs": retrieved_logs,
                "distances": distances
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}", exc_info=True)
            raise
