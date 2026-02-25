"""
基于RAG的异常诊断服务
"""

import array
import json
import re
from typing import List, Dict, Any, Optional
from services.pattern_analysis import PatternAnalysisService
from services.retrieval import RetrievalService
from database.connection import db_pool
from config import settings
from utils.logging_config import logger


class DiagnosisService:
    """执行基于RAG的诊断服务"""

    def __init__(self, openai_client, retrieval_service: RetrievalService):
        self.openai_client = openai_client
        self.retrieval_service = retrieval_service
        self.pattern_analysis = PatternAnalysisService()

    def diagnose_anomaly(
        self, candidate_logs: List[Dict], original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        对小的、高信号日志子集执行基于RAG的诊断。
        如果提供了查询，在LLM之前执行混合向量+文本搜索重排序。
        """
        if not candidate_logs:
            return {
                "pattern_summary": {
                    "error_distribution_summary": "No logs available for analysis",
                    "time_concentration_summary": "No logs available for analysis",
                    "common_features_summary": "No logs available for analysis",
                },
                "root_cause": "No candidate logs found for analysis.",
                "confidence": 0.0,
                "evidence": [],
                "alternatives": [],
                "next_steps": [],
            }

        log_subset = candidate_logs[:300]

        if original_query:
            try:
                query_embedding = self.openai_client.embeddings.create(
                    model=settings.openai_model, input=original_query
                )
                query_vector = array.array("f", query_embedding.data[0].embedding)

                candidate_ids = [log["id"] for log in candidate_logs]
                if candidate_ids:
                    params = {"query_vec": query_vector}
                    placeholders = []
                    for i, cid in enumerate(candidate_ids):
                        param_name = f"id_{i}"
                        params[param_name] = cid
                        placeholders.append(f":{param_name}")

                    placeholders_str = ",".join(placeholders)

                    with db_pool.acquire() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            f"""
                            SELECT id, text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
                            FROM docs
                            WHERE id IN ({placeholders_str})
                            ORDER BY distance ASC
                            FETCH FIRST 100 ROWS ONLY
                        """,
                            params,
                        )
                        vector_results = cursor.fetchall()

                    candidate_texts = {log["id"]: log["text"] for log in candidate_logs}
                    bm25_scores = {}
                    if len(candidate_texts) > 0:
                        tokenized_query = original_query.split()
                        for log_id, text in candidate_texts.items():
                            tokens = text.split()
                            score = sum(
                                1
                                for token in tokenized_query
                                if token.lower() in [t.lower() for t in tokens]
                            )
                            if score > 0:
                                bm25_scores[log_id] = score

                    fused_scores = {}
                    text_to_distance = {}
                    for row in vector_results:
                        log_id, text, distance = row[0], row[1], row[2]
                        text_to_distance[log_id] = distance
                        vector_score = 1.0 / (1.0 + distance)
                        bm25_score = (
                            bm25_scores.get(log_id, 0) / max(bm25_scores.values())
                            if bm25_scores
                            else 0
                        )
                        fused_scores[log_id] = vector_score * 0.7 + bm25_score * 0.3

                    sorted_ids = sorted(
                        fused_scores.items(), key=lambda x: x[1], reverse=True
                    )[:100]
                    reranked_ids = {log_id for log_id, _ in sorted_ids}
                    log_subset = [
                        log for log in candidate_logs if log["id"] in reranked_ids
                    ]
                    if len(log_subset) < 100:
                        remaining = [
                            log
                            for log in candidate_logs
                            if log["id"] not in reranked_ids
                        ]
                        log_subset.extend(remaining[: 100 - len(log_subset)])
            except Exception as e:
                logger.warning(f"Reranking failed, using original candidates: {e}")

        stats = self.pattern_analysis.analyze_log_patterns(log_subset)

        log_samples = log_subset[:50]
        log_texts = [log["text"] for log in log_samples]
        context_samples = "\n".join(log_texts)

        query_part = f"\n\nUser question: {original_query}" if original_query else ""

        stats_summary = f"""
Statistical Analysis of {stats.get('total_logs', len(log_subset))} Anomaly Logs:

1. Error Type Distribution:
{json.dumps(stats.get('error_distribution', {}), indent=2)}

2. Time Concentration:
{json.dumps(stats.get('time_concentration', {}), indent=2)}

3. Top Keywords (Frequency):
{json.dumps(stats.get('top_keywords', {}), indent=2)}

4. Component Distribution:
{json.dumps(stats.get('component_distribution', {}), indent=2)}

5. Common Message Patterns (Template Frequency):
{json.dumps(stats.get('common_patterns', {}), indent=2)}
"""

        prompt = f"""You are a senior SRE analyzing a high-signal subset of anomaly logs (pre-filtered by statistical methods).

CRITICAL: You must analyze the OVERALL PATTERNS, not just individual log lines.

{stats_summary}

Representative Log Samples (first 50 of {len(log_subset)} total logs):
{context_samples}{query_part}

Your task:
1. Analyze the STATISTICAL PATTERNS above (error distribution, time concentration, keywords, components, message patterns)
2. Identify the MOST LIKELY ROOT CAUSE based on overall patterns, not individual log lines
3. Provide evidence as PATTERN DESCRIPTIONS (e.g., "45% of logs show ERROR type X", "Logs cluster in hour Y", "Keyword Z appears in 60% of messages")
4. Do NOT just cite individual log lines - summarize the patterns

You MUST respond with ONLY a valid JSON object (no markdown, no code blocks, no additional text):
{{
  "pattern_summary": {{
    "error_distribution_summary": "Summary of dominant error types and their proportions (e.g., 'ERROR type X accounts for Y% of all errors')",
    "time_concentration_summary": "Summary of time patterns (e.g., 'Errors cluster in hour Z, spanning W minutes')",
    "common_features_summary": "Summary of shared characteristics (e.g., 'Keyword A appears in X% of logs, Component B is involved in Y% of cases')"
  }},
  "root_cause": "string describing the most likely root cause hypothesis based on overall patterns",
  "confidence": 0.85,
  "evidence": ["pattern description 1 (e.g., 'ERROR type X appears in Y% of logs')", "pattern description 2", "pattern description 3"],
  "alternatives": ["alternative explanation 1", "alternative explanation 2"],
  "next_steps": ["actionable step 1", "actionable step 2", "actionable step 3"]
}}

Requirements:
- pattern_summary: Summarize error distribution, time concentration, and common features from the statistical analysis above.
- root_cause: Must be based on STATISTICAL PATTERNS, not individual logs. Describe the overall pattern.
- confidence: Float 0.0-1.0 based on pattern strength
- evidence: Array of 3-5 PATTERN DESCRIPTIONS (e.g., "X% of logs show Y", "Time concentration in Z hour", "Keyword W appears N times")
- alternatives: Array of 1-2 alternative pattern-based explanations
- next_steps: Array of 3-5 concrete, actionable troubleshooting/fix recommendations

Do not cite individual log lines. Focus on summarizing the overall patterns from the statistics.
"""

        try:
            completion = self.openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a production SRE. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            analysis_text = completion.choices[0].message.content.strip()

            if analysis_text.startswith("```"):
                analysis_text = re.sub(
                    r"^```(?:json)?\s*", "", analysis_text, flags=re.MULTILINE
                )
                analysis_text = re.sub(
                    r"```\s*$", "", analysis_text, flags=re.MULTILINE
                )

            diagnosis_json = json.loads(analysis_text)

            pattern_summary = diagnosis_json.get("pattern_summary", {})
            root_cause_raw = diagnosis_json.get("root_cause", "Analysis completed")

            is_access_log = any(
                "access" in str(log.get("text", "")).lower()
                or "nginx" in str(log.get("text", "")).lower()
                for log in log_subset[:10]
            )

            if is_access_log:
                scope_prefix = "From access logs alone, "
            else:
                scope_prefix = "Based on the available logs alone, "

            if root_cause_raw and not (
                root_cause_raw.startswith("Based on")
                or root_cause_raw.startswith("From")
            ):
                root_cause = (
                    scope_prefix + root_cause_raw[0].lower() + root_cause_raw[1:]
                    if len(root_cause_raw) > 0
                    else root_cause_raw
                )
            else:
                root_cause = root_cause_raw

            confidence = float(diagnosis_json.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            evidence = diagnosis_json.get("evidence", [])
            alternatives = diagnosis_json.get("alternatives", [])
            next_steps = diagnosis_json.get("next_steps", [])

            return {
                "pattern_summary": {
                    "error_distribution_summary": (
                        pattern_summary.get("error_distribution_summary", "")
                        if isinstance(pattern_summary, dict)
                        else ""
                    ),
                    "time_concentration_summary": (
                        pattern_summary.get("time_concentration_summary", "")
                        if isinstance(pattern_summary, dict)
                        else ""
                    ),
                    "common_features_summary": (
                        pattern_summary.get("common_features_summary", "")
                        if isinstance(pattern_summary, dict)
                        else ""
                    ),
                },
                "root_cause": root_cause,
                "confidence": round(confidence, 2),
                "evidence": evidence[:5] if isinstance(evidence, list) else [],
                "alternatives": (
                    alternatives[:3] if isinstance(alternatives, list) else []
                ),
                "next_steps": next_steps[:5] if isinstance(next_steps, list) else [],
                "raw_analysis": analysis_text,
                "candidate_count": len(log_subset),
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "pattern_summary": {
                    "error_distribution_summary": "Failed to parse LLM response",
                    "time_concentration_summary": "Failed to parse LLM response",
                    "common_features_summary": "Failed to parse LLM response",
                },
                "root_cause": "Failed to parse LLM response. Raw output: "
                + analysis_text[:200],
                "confidence": 0.0,
                "evidence": [],
                "alternatives": [],
                "next_steps": ["Review raw_analysis field for manual interpretation"],
                "raw_analysis": analysis_text,
                "candidate_count": len(log_subset),
            }
        except Exception as e:
            logger.error(f"Error in diagnosis: {e}", exc_info=True)
            raise
