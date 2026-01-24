"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel
from typing import Optional, List


class QueryRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = "What caused the block to be missing?"
    top_k: int = 3


class DiagnosisRequest(BaseModel):
    """Request model for diagnosis endpoint"""
    query: Optional[str] = None
    signal_ids: Optional[List[int]] = None
