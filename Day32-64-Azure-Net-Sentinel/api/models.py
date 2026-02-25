"""
API请求和响应的Pydantic模型
"""

from pydantic import BaseModel
from typing import Optional, List


class QueryRequest(BaseModel):
    """搜索端点的请求模型"""

    query: str = "What caused the block to be missing?"
    top_k: int = 3


class DiagnosisRequest(BaseModel):
    """诊断端点的请求模型"""

    query: Optional[str] = None
    signal_ids: Optional[List[int]] = None
