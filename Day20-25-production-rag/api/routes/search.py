"""
Search API routes
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
    openai_client: OpenAI = Depends(get_openai_client)
):
    """Perform hybrid search combining vector and BM25"""
    try:
        search_results = retrieval_service.hybrid_search(request.query, request.top_k)
        
        if not search_results["retrieved_logs"]:
            return {
                "query": request.query,
                "ai_summary": "No logs found in database.",
                "retrieved_logs": []
            }
        
        retrieved_logs = search_results["retrieved_logs"]
        
        context = "\n\n".join(log["text"] for log in retrieved_logs)
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
            temperature=0.3
        )

        summary = completion.choices[0].message.content.strip()
        
        return {
            "query": request.query,
            "ai_summary": summary,
            "retrieved_logs": retrieved_logs,
            "note": "Results from hybrid retrieval (vector 70% + BM25 30%)"
        }
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
