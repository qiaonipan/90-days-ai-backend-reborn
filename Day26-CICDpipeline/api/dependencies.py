"""
FastAPI dependencies for dependency injection
"""
from functools import lru_cache
from openai import OpenAI
from services.retrieval import RetrievalService
from services.diagnosis import DiagnosisService
from services.signal_detection import SignalDetectionService
from config import settings


@lru_cache()
def get_openai_client() -> OpenAI:
    """Get OpenAI client singleton"""
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache()
def get_signal_detection_service() -> SignalDetectionService:
    """Get signal detection service singleton"""
    return SignalDetectionService()


def get_retrieval_service() -> RetrievalService:
    """Get retrieval service instance"""
    openai_client = get_openai_client()
    service = RetrievalService(openai_client)
    service.reload_bm25()
    return service


def get_diagnosis_service() -> DiagnosisService:
    """Get diagnosis service instance"""
    openai_client = get_openai_client()
    retrieval_service = get_retrieval_service()
    return DiagnosisService(openai_client, retrieval_service)
