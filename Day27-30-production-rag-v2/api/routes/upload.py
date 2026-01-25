"""
Upload API routes
"""

import array
import gzip
import json
import re
import time
import asyncio
import pandas as pd
from collections import Counter
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from api.dependencies import (
    get_openai_client,
    get_signal_detection_service,
    get_retrieval_service,
)
from services.signal_detection import SignalDetectionService
from services.retrieval import RetrievalService
from database.connection import db_pool
from config import settings
from utils.logging_config import logger

router = APIRouter(prefix="/upload", tags=["Upload"])

upload_progress = {"total": 0, "processed": 0, "status": "idle", "start_time": None}


@router.get("/progress")
def get_progress():
    """Get upload progress status"""
    if upload_progress["total"] == 0:
        return {"progress": 0, "status": "idle"}
    progress = (upload_progress["processed"] / upload_progress["total"]) * 100
    elapsed = (
        time.time() - upload_progress["start_time"]
        if upload_progress["start_time"]
        else 0
    )
    return {
        "progress": round(progress, 1),
        "processed": upload_progress["processed"],
        "total": upload_progress["total"],
        "status": upload_progress["status"],
        "elapsed_seconds": round(elapsed, 1),
    }


@router.post("")
async def upload_logs(
    file: UploadFile = File(...),
    openai_client=Depends(get_openai_client),
    signal_service: SignalDetectionService = Depends(get_signal_detection_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Upload and process log files"""
    global upload_progress

    async def progress_generator():
        global upload_progress
        try:
            if not file.filename:
                yield '{"status": "error", "message": "No file provided", "progress": 0}\n'
                return

            logger.info(f"Received upload request for file: {file.filename}")

            yield '{"status": "parsing", "progress": 0}\n'

            content = await file.read()
            if not content:
                yield '{"status": "error", "message": "File is empty", "progress": 0}\n'
                return

            try:
                if file.filename.endswith(".gz"):
                    content = gzip.decompress(content)
                text_content = content.decode("utf-8", errors="ignore")
            except Exception as e:
                yield f'{{"status": "error", "message": "Failed to decode file: {str(e)}", "progress": 0}}\n'
                return

            lines = text_content.split("\n")
            log_entries = [
                line.strip()
                for line in lines
                if line.strip() and len(line.strip()) > 20
            ]

            logger.info(f"Parsed {len(log_entries)} valid log entries from file")

            if len(log_entries) > settings.max_upload_entries:
                log_entries = log_entries[: settings.max_upload_entries]
                logger.info(f"Limited to {settings.max_upload_entries} entries")

            if not log_entries:
                yield '{"status": "error", "message": "No valid log entries found", "progress": 0}\n'
                return

            total = len(log_entries)
            start_time = time.time()
            upload_progress = {
                "total": total,
                "processed": 0,
                "status": "processing",
                "start_time": start_time,
            }

            yield '{"status": "analyzing", "progress": 5, "detailed_message": "Extracting anomaly signals..."}\n'
            anomaly_signals = signal_service.extract_anomaly_signals(log_entries)

            anomaly_count = 0
            if anomaly_signals:
                with db_pool.acquire() as conn:
                    cursor = conn.cursor()
                    signal_data = []
                    for signal in anomaly_signals:
                        signal_data.append(
                            (
                                pd.to_datetime(signal["window_start"]),
                                signal["template_id"],
                                signal["signature"],
                                signal["count"],
                                signal["score"],
                            )
                        )

                    cursor.executemany(
                        """INSERT INTO anomaly_signals (window_start, template_id, signature, count, score)
                           VALUES (:1, :2, :3, :4, :5)""",
                        signal_data,
                    )
                    conn.commit()
                    anomaly_count = len(anomaly_signals)
                    logger.info(f"Stored {anomaly_count} anomaly signals")

            yield f'{{"status": "analyzing", "progress": 7, "detailed_message": "Extracted {anomaly_count} anomaly signals"}}\n'

            yield '{"status": "truncating", "progress": 10}\n'
            with db_pool.acquire() as conn:
                cursor = conn.cursor()
                cursor.execute("TRUNCATE TABLE docs")
                conn.commit()

            yield f'{{"status": "embedding", "progress": 15, "processed": 0, "total": {total}}}\n'

            batch_size = settings.embedding_batch_size
            all_embeddings = []
            for i in range(0, len(log_entries), batch_size):
                batch = log_entries[i : i + batch_size]
                response = openai_client.embeddings.create(
                    model=settings.openai_model, input=batch
                )
                batch_embs = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embs)

                processed = min(i + batch_size, len(log_entries))
                progress = 15 + int((processed / total) * 70)
                upload_progress["processed"] = processed
                yield f'{{"status": "embedding", "progress": {progress}, "processed": {processed}, "total": {total}}}\n'
                await asyncio.sleep(0)

            yield '{"status": "inserting", "progress": 90}\n'
            data = []
            for text, emb in zip(log_entries, all_embeddings):
                vec = array.array("f", emb)
                ts = None
                ts_match = re.match(r"^(\d{6})\s+(\d{6})", text)
                if ts_match:
                    try:
                        ts = pd.to_datetime(
                            f"20{ts_match.group(1)} {ts_match.group(2)}",
                            format="%Y%m%d %H%M%S",
                        )
                    except (ValueError, TypeError):
                        ts = None
                data.append((text, vec, ts))

            with db_pool.acquire() as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    "INSERT INTO docs (text, embedding, ts) VALUES (:1, :2, :3)", data
                )
                conn.commit()

            yield '{"status": "reloading", "progress": 95}\n'
            retrieval_service.reload_bm25()
            logger.info("BM25 index reloaded successfully")

            yield '{"status": "generating_summary", "progress": 98}\n'

            summary_lines = []
            if anomaly_signals and len(anomaly_signals) > 0:
                summary_lines.append(
                    f"{len(anomaly_signals)} high-confidence anomaly window(s) detected"
                )

                component_counter = Counter()
                template_counter = Counter()

                for signal in anomaly_signals:
                    if "templates" in signal:
                        for template_id, count in signal["templates"].items():
                            template_counter[template_id] += count

                    signature = signal.get("signature", "")
                    sig_match = re.search(r"(\w+):\s+", signature)
                    if sig_match:
                        component_counter[sig_match.group(1)] += signal.get("count", 0)

                if component_counter:
                    top_components = ", ".join(
                        [comp for comp, _ in component_counter.most_common(3)]
                    )
                    summary_lines.append(
                        f"Errors concentrated on component(s): {top_components}"
                    )
                else:
                    error_components = []
                    for line in log_entries:
                        match = re.match(
                            r"^(\d{6})\s+(\d{6})\s+\d+\s+(ERROR|FATAL)\s+(\w+):", line
                        )
                        if match:
                            error_components.append(match.group(4))
                    if error_components:
                        top_components = ", ".join(
                            [
                                comp
                                for comp, _ in Counter(error_components).most_common(3)
                            ]
                        )
                        summary_lines.append(
                            f"Errors concentrated on component(s): {top_components}"
                        )

                if template_counter:
                    top_templates = ", ".join(
                        [
                            f"T{str(tid)[:6]}"
                            for tid, _ in template_counter.most_common(3)
                        ]
                    )
                    summary_lines.append(f"Repeated patterns: {top_templates}")
                else:
                    template_ids = [
                        sig.get("template_id", "")
                        for sig in anomaly_signals
                        if sig.get("template_id")
                    ]
                    if template_ids:
                        top_templates = ", ".join(
                            [f"T{str(tid)[:6]}" for tid in template_ids[:3]]
                        )
                        summary_lines.append(f"Repeated patterns: {top_templates}")
            else:
                summary_lines.append("Log analysis complete.")
                summary_lines.append(
                    "No system-level anomaly windows were detected across the uploaded logs."
                )
                summary_lines.append(
                    "The system is ready to analyze specific patterns or questions you want to investigate."
                )

            preliminary_summary = "Preliminary Diagnosis Summary\n" + "\n".join(
                f"• {line}" for line in summary_lines
            )

            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Processing complete! Total time: {total_time:.1f} seconds")

            upload_progress["status"] = "complete"
            upload_progress["processed"] = total

            yield json.dumps(
                {
                    "status": "complete",
                    "progress": 100,
                    "chunks_loaded": len(log_entries),
                    "processing_time_seconds": round(total_time, 1),
                    "anomaly_signals_count": anomaly_count,
                    "preliminary_summary": preliminary_summary,
                    "message": f"Upload successful! Loaded {len(log_entries)} log entries. Detected {anomaly_count} anomaly signals.",
                }
            ) + "\n"

        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            upload_progress["status"] = "error"
            yield f'{{"status": "error", "message": "Upload failed: {str(e)}", "progress": 0}}\n'

    return StreamingResponse(progress_generator(), media_type="text/event-stream")
