"""
诊断API路由
"""

from fastapi import APIRouter, Depends, HTTPException
from api.models import DiagnosisRequest
from api.dependencies import (
    get_diagnosis_service,
    get_retrieval_service,
    get_signal_detection_service,
)
from services.diagnosis import DiagnosisService
from services.retrieval import RetrievalService
from services.signal_detection import SignalDetectionService
from database.connection import db_pool
from utils.logging_config import logger

router = APIRouter(prefix="/diagnose", tags=["Diagnosis"])


@router.post("")
def diagnose_anomalies(
    request: DiagnosisRequest = DiagnosisRequest(),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    signal_service: SignalDetectionService = Depends(get_signal_detection_service),
):
    """使用信号驱动的检索和RAG执行异常诊断"""
    try:
        with db_pool.acquire() as conn:
            cursor = conn.cursor()

            if request.signal_ids and len(request.signal_ids) > 0:
                signal_ids_tuple = tuple(request.signal_ids)
                if len(signal_ids_tuple) == 1:
                    cursor.execute(
                        """
                        SELECT window_start, template_id, signature, count, score
                        FROM anomaly_signals
                        WHERE id = :1
                        ORDER BY score DESC
                    """,
                        signal_ids_tuple[0],
                    )
                else:
                    placeholders = ",".join(
                        [f":{i+1}" for i in range(len(signal_ids_tuple))]
                    )
                    cursor.execute(
                        f"""
                        SELECT window_start, template_id, signature, count, score
                        FROM anomaly_signals
                        WHERE id IN ({placeholders})
                        ORDER BY score DESC
                    """,
                        signal_ids_tuple,
                    )
            else:
                cursor.execute("""
                    SELECT window_start, template_id, signature, count, score
                    FROM anomaly_signals
                    ORDER BY score DESC
                    FETCH FIRST 3 ROWS ONLY
                """)

            signal_rows = cursor.fetchall()

            if not signal_rows:
                return {
                    "signals": [],
                    "candidate_count": 0,
                    "diagnosis": {
                        "root_cause": "No significant anomalies detected in the uploaded logs.",
                        "confidence": 0.0,
                        "evidence": [],
                        "alternatives": [],
                        "next_steps": [
                            "Upload logs to enable anomaly detection",
                            "Check if logs contain error patterns",
                        ],
                    },
                    "candidate_samples": [],
                    "message": "No anomaly signals found. System is operating normally or no errors detected.",
                }

            suspicious_signals = []
            for row in signal_rows:
                suspicious_signals.append(
                    {
                        "window_start": (
                            row[0].isoformat()
                            if hasattr(row[0], "isoformat")
                            else str(row[0])
                        ),
                        "template_id": row[1],
                        "signature": row[2],
                        "count": row[3],
                        "score": float(row[4]),
                    }
                )

            candidate_logs = retrieval_service.retrieve_candidate_logs(
                suspicious_signals
            )

            if not candidate_logs:
                return {
                    "signals": suspicious_signals,
                    "candidate_count": 0,
                    "diagnosis": {
                        "root_cause": "No candidate logs found matching the anomaly signals.",
                        "confidence": 0.0,
                        "evidence": [],
                        "alternatives": [],
                        "next_steps": [
                            "Check if logs match the signature patterns",
                            "Verify anomaly signal time windows",
                        ],
                    },
                    "candidate_samples": [],
                    "message": "No candidate logs matched the anomaly signal patterns.",
                }

            diagnosis = diagnosis_service.diagnose_anomaly(
                candidate_logs, request.query
            )

            return {
                "signals": suspicious_signals,
                "candidate_count": len(candidate_logs),
                "diagnosis": diagnosis,
                "candidate_samples": candidate_logs[:10],
                "message": f"Diagnosis complete. Analyzed {len(candidate_logs)} candidate logs from {len(suspicious_signals)} anomaly signals.",
            }
    except Exception as e:
        logger.error(f"Diagnosis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")
