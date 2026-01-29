"""
搜索API路由
"""

from fastapi import APIRouter, Depends, HTTPException
from api.models import QueryRequest
from api.dependencies import get_retrieval_service, get_openai_client
from services.retrieval import RetrievalService
from openai import OpenAI
from config import settings
from utils.logging_config import logger

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("")
def hybrid_search(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    openai_client: OpenAI = Depends(get_openai_client),
):
    """执行混合搜索，结合向量搜索、BM25和重排序"""
    try:
        search_results = retrieval_service.hybrid_search(request.query, request.top_k)

        if not search_results["retrieved_logs"]:
            return {
                "query": request.query,
                "ai_summary": "No logs found in database.",
                "retrieved_logs": [],
            }

        retrieved_logs = search_results["retrieved_logs"]
        
        # 使用前10条日志作为LLM上下文（重排序后的结果），但返回用户请求的所有top_k结果
        context_logs = retrieved_logs[:10]
        context = "\n\n".join(log["text"] for log in context_logs)
        
        prompt = f"""Analyze the following log entries to diagnose the issue described in the user's question.

User question: {request.query}

Retrieved relevant logs:
{context}

Provide a structured analysis with the following sections:
1. Root Cause: Identify the underlying cause (not just symptoms)
2. Evidence: Cite specific log entries that support your analysis
3. Alternatives: List other possible explanations if the root cause is uncertain
4. Next Steps: Suggest concrete troubleshooting actions

Format your response clearly and concisely."""

        completion = openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        summary = completion.choices[0].message.content.strip()
        
        # 检查是否使用了重排序（结果中存在rerank_score）
        has_rerank = any("rerank_score" in log for log in retrieved_logs)
        note = (
            "Results from hybrid retrieval (vector 70% + BM25 30%) with CrossEncoder reranking"
            if has_rerank
            else "Results from hybrid retrieval (vector 70% + BM25 30%)"
        )

        return {
            "query": request.query,
            "ai_summary": summary,
            "retrieved_logs": retrieved_logs,
            "note": note,
        }
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
