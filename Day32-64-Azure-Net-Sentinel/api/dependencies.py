"""
FastAPI依赖注入的依赖项
"""

from functools import lru_cache
from openai import OpenAI
from services.retrieval import RetrievalService
from services.diagnosis import DiagnosisService
from services.signal_detection import SignalDetectionService
from config import settings


@lru_cache()
def get_openai_client() -> OpenAI:
    """获取OpenAI客户端单例"""
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache()
def get_signal_detection_service() -> SignalDetectionService:
    """获取信号检测服务单例"""
    return SignalDetectionService()


def get_retrieval_service() -> RetrievalService:
    """获取检索服务实例"""
    openai_client = get_openai_client()
    service = RetrievalService(openai_client)
    service.reload_bm25()
    return service


def get_diagnosis_service() -> DiagnosisService:
    """获取诊断服务实例"""
    openai_client = get_openai_client()
    retrieval_service = get_retrieval_service()
    return DiagnosisService(openai_client, retrieval_service)
