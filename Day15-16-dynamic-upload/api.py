import os
import array
import gzip
import math
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import oracledb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi  # BM25 keyword search
from concurrent.futures import ThreadPoolExecutor, as_completed  
import time
import asyncio

upload_progress = {"total": 0, "processed": 0, "status": "idle", "start_time": None}

# =========================
# Load environment variables
# =========================
load_dotenv()

# =========================
# OpenAI & Oracle config
# =========================
client = OpenAI()

username = os.getenv("ORACLE_USERNAME")
password = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_DSN")
wallet_path = os.getenv("ORACLE_WALLET_PATH")

# =========================
# Oracle DB connection
# =========================
connection = oracledb.connect(
    user=username,
    password=password,
    dsn=dsn,
    config_dir=wallet_path,
    wallet_location=wallet_path,
    wallet_password=password
)
cursor = connection.cursor()

# =========================
# Load all logs for BM25 (global initialization once)
# =========================
def reload_bm25():
    """Reload BM25 index from database"""
    global all_texts, bm25
    cursor.execute("SELECT text FROM docs")
    all_logs = cursor.fetchall()
    all_texts = [log[0].strip() for log in all_logs if log[0]]
    
    # BM25 initialization (tokenization)
    # Handle empty corpus to avoid ZeroDivisionError
    if len(all_texts) > 0:
        tokenized_corpus = [text.split() for text in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        # Initialize with empty corpus if no data exists
        bm25 = BM25Okapi([[""]])
        print("Warning: No logs found in database. BM25 initialized with empty corpus.")

# Initial load
reload_bm25()

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Oracle 26ai Cloud Vector Semantic Search API (RAG + Hybrid Search)",
    description="""
    Enterprise-grade semantic search system with hybrid retrieval capabilities.
    
    Features:
    • OpenAI embeddings with Oracle 26ai native vector search
    • BM25 keyword search fusion
    • Retrieval-Augmented Generation (RAG)
    • Real-time file upload with streaming progress
    • Dynamic log analysis and diagnosis
    
    Access the web interface at the root endpoint (/).
    """,
    version="1.2.0 (Streaming Upload + Hybrid Search)",
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/", tags=["Frontend"])
def redirect_to_frontend():
    return RedirectResponse(url="/static/index.html")

@app.get("/favicon.ico", tags=["Frontend"])
def favicon():
    return Response(status_code=204)

# =========================
# Progress tracking endpoint
# =========================
@app.get("/progress", tags=["File Upload"])
def get_progress():
    """Get upload progress status"""
    if upload_progress["total"] == 0:
        return {"progress": 0, "status": "idle"}
    progress = (upload_progress["processed"] / upload_progress["total"]) * 100
    elapsed = time.time() - upload_progress["start_time"] if upload_progress["start_time"] else 0
    return {
        "progress": round(progress, 1),
        "processed": upload_progress["processed"],
        "total": upload_progress["total"],
        "status": upload_progress["status"],
        "elapsed_seconds": round(elapsed, 1)
    }

# =========================
# Request model
# =========================
class QueryRequest(BaseModel):
    query: str = "What caused the block to be missing?"
    top_k: int = 3

# =========================
# Hybrid Search API
# =========================
@app.post("/search", tags=["RAG Full Cycle + Hybrid"])
def hybrid_search(request: QueryRequest):
    # Step 1: Query embedding
    embedding_response = client.embeddings.create(
        model="text-embedding-3-large",
        input=request.query
    )
    query_embedding = embedding_response.data[0].embedding
    query_vector = array.array("f", query_embedding)

    # Step 2: Vector search
    cursor.execute(
        """
        SELECT text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
        FROM docs
        ORDER BY distance ASC
        FETCH FIRST :k ROWS ONLY
        """,
        query_vec=query_vector,
        k=request.top_k * 2
    )
    vector_results = cursor.fetchall()

    # Step 3: BM25 keyword search
    bm25_results = []
    if len(all_texts) > 0:
        tokenized_query = request.query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        if bm25_scores.max() == 0:
            bm25_results = []
        else:
            bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:request.top_k * 2]
            bm25_results = [(all_texts[i], bm25_scores[i]) for i in bm25_results]

    # Step 4: Fusion scoring with improved normalization
    fused_scores = {}
    text_to_distance = {}

    # For text-embedding-3-large, cosine distances typically range from 0 to ~2
    # Use more aggressive normalization to get scores in 0.5-0.95 range
    distances = [d for _, d in vector_results]
    if not distances:
        return {"query": request.query, "ai_summary": "No logs found in database.", "retrieved_logs": []}
    
    min_distance = min(distances)
    max_distance = max(distances)
    
    for text, distance in vector_results:
        text_to_distance[text] = distance
        
        # Method 1: Exponential decay with adjusted scaling
        # For distance 0.0-0.5: exp(-distance*1.5) gives 0.47-1.0
        # For distance 0.5-1.0: exp(-distance*1.5) gives 0.22-0.47
        # Scale to ensure top results get 0.7-0.95 range
        vector_score_exp = math.exp(-distance * 1.2)
        
        # Method 2: Min-max normalization with compression
        # Compress the range to 0.5-1.0 instead of 0-1
        if max_distance > min_distance:
            normalized = (distance - min_distance) / (max_distance - min_distance)
            # Invert and compress: best (0) -> 1.0, worst (1) -> 0.5
            vector_score_linear = 1.0 - (normalized * 0.5)
        else:
            vector_score_linear = 1.0
        
        # Combine: 60% exponential (more sensitive), 40% linear (stable baseline)
        vector_score = 0.6 * vector_score_exp + 0.4 * vector_score_linear
        
        # Ensure minimum score for top results
        if distance == min_distance:
            vector_score = max(vector_score, 0.7)  # Top result at least 0.7
        
        fused_scores[text] = fused_scores.get(text, 0) + vector_score * 0.7

    if bm25_results:
        max_bm25 = max([score for _, score in bm25_results])
        min_bm25 = min([score for _, score in bm25_results])
        for text, score in bm25_results:
            # BM25 normalization: compress to 0.5-1.0 range
            if max_bm25 > min_bm25:
                normalized = (score - min_bm25) / (max_bm25 - min_bm25)
                bm25_norm = 0.5 + (normalized * 0.5)  # Range: 0.5-1.0
            else:
                bm25_norm = 1.0 if score > 0 else 0.0
            fused_scores[text] = fused_scores.get(text, 0) + bm25_norm * 0.3

    final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:request.top_k]

    retrieved_logs = []
    for rank, (text, hybrid_score) in enumerate(final_results, 1):
        original_distance = text_to_distance.get(text, None)
        retrieved_logs.append({
            "rank": rank,
            "text": text,
            "hybrid_score": round(hybrid_score, 4),
            "distance": round(original_distance, 4) if original_distance is not None else None
        })

    # Step 5: RAG generation 
    if retrieved_logs:
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

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        summary = completion.choices[0].message.content.strip()
    else:
        summary = "No highly relevant logs found for this query."

    return {
        "query": request.query,
        "ai_summary": summary,
        "retrieved_logs": retrieved_logs,
        "note": "Results from hybrid retrieval (vector 70% + BM25 30%)"
    }

# =========================
# File Upload API (Streaming response with real-time progress)
# =========================
@app.post("/upload", tags=["File Upload"])
async def upload_logs(file: UploadFile = File(...)):
    async def progress_generator():
        try:
            if not file.filename:
                yield '{"status": "error", "message": "No file provided", "progress": 0}\n'
                return
            
            print(f"Received upload request for file: {file.filename}")
            
            # Step 1: Parse file
            yield '{"status": "parsing", "progress": 0}\n'
            
            content = await file.read()
            if not content:
                yield '{"status": "error", "message": "File is empty", "progress": 0}\n'
                return
            
            # Decode (support gz)
            try:
                if file.filename.endswith('.gz'):
                    content = gzip.decompress(content)
                text_content = content.decode('utf-8', errors='ignore')
            except Exception as e:
                yield f'{{"status": "error", "message": "Failed to decode file: {str(e)}", "progress": 0}}\n'
                return
            
            lines = text_content.split('\n')
            log_entries = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
            
            print(f"Parsed {len(log_entries)} valid log entries from file")
            
            if len(log_entries) > 5000:
                log_entries = log_entries[:5000]
                print("Limited to 5000 entries")
            
            if not log_entries:
                yield '{"status": "error", "message": "No valid log entries found", "progress": 0}\n'
                return
            
            total = len(log_entries)
            start_time = time.time()
            
            # Step 2: Truncate table
            yield '{"status": "truncating", "progress": 5}\n'
            cursor.execute("TRUNCATE TABLE docs")
            connection.commit()
            
            # Step 3: Batch generate embeddings (max 1000 entries per batch)
            yield '{"status": "embedding", "progress": 10, "processed": 0, "total": ' + str(total) + '}\n'
            
            batch_size = 1000  # OpenAI allows max 1000 entries per API call
            all_embeddings = []
            for i in range(0, len(log_entries), batch_size):
                batch = log_entries[i:i+batch_size]
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch
                )
                batch_embs = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embs)
                
                # Send progress update (10% to 80% for embedding)
                processed = min(i + batch_size, len(log_entries))
                progress = 10 + int((processed / total) * 70)  # 10% to 80%
                yield f'{{"status": "embedding", "progress": {progress}, "processed": {processed}, "total": {total}}}\n'
                await asyncio.sleep(0)  # Yield control
            
            # Step 4: Batch insert
            yield '{"status": "inserting", "progress": 85}\n'
            data = []
            for text, emb in zip(log_entries, all_embeddings):
                vec = array.array("f", emb)
                data.append((text, vec))
            
            cursor.executemany(
                "INSERT INTO docs (text, embedding) VALUES (:1, :2)",
                data
            )
            connection.commit()
            
            # Step 5: Reload BM25
            yield '{"status": "reloading", "progress": 95}\n'
            reload_bm25()
            print("BM25 index reloaded successfully")
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing complete! Total time: {total_time:.1f} seconds")
            
            # Final success message
            yield f'{{"status": "complete", "progress": 100, "chunks_loaded": {len(log_entries)}, "processing_time_seconds": {round(total_time, 1)}}}\n'
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Upload error: {traceback.format_exc()}")
            yield f'{{"status": "error", "message": "Upload failed: {error_msg}", "progress": 0}}\n'
    
    return StreamingResponse(progress_generator(), media_type="text/event-stream")