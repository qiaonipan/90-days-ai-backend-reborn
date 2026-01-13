import os
import array
import gzip
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import oracledb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi  # BM25 keyword search
from concurrent.futures import ThreadPoolExecutor, as_completed  
import time

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
    <b>Qiaoni's reskilling project after layoff · RAG Full Cycle + Hybrid Search Demo</b><br><br>

    This service demonstrates an enterprise-style semantic search system with hybrid retrieval:
    • OpenAI embeddings + Oracle 26ai native vector search
    • BM25 keyword search fusion
    • Retrieval-Augmented Generation (RAG)<br><br>

    👇 Try it via /
    """,
    version="1.1.0 (Hybrid Search)",
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
        model="text-embedding-3-small",
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

    # Step 4: Fusion scoring
    fused_scores = {}
    text_to_distance = {}

    distances = [d for _, d in vector_results]
    max_distance = max(distances) if distances else 1
    for text, distance in vector_results:
        text_to_distance[text] = distance
        vector_score = 1 - (distance / max_distance if max_distance > 0 else 0)
        fused_scores[text] = fused_scores.get(text, 0) + vector_score * 0.7

    if bm25_results:
        max_bm25 = max([score for _, score in bm25_results])
        for text, score in bm25_results:
            bm25_norm = score / max_bm25 if max_bm25 > 0 else 0
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
        prompt = f"""
You are a senior distributed systems SRE with 10+ years experience diagnosing HDFS production issues.

User question: {request.query}

Retrieved relevant logs (chronological order):
{context}

Task:
1. Identify the most likely root cause (not just symptoms)
2. Explain WHY it happened, citing specific log evidence
3. List possible alternatives if uncertain
4. Suggest next troubleshooting steps
5. Answer in English, structured and concise

Output format:
- Root Cause: ...
- Evidence: ...
- Alternatives: ...
- Next Steps: ...
"""

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
# File Upload API (Batch processing with explicit IDs and progress tracking)
# =========================
@app.post("/upload", tags=["File Upload"])
async def upload_logs(file: UploadFile = File(...)):
    global upload_progress
    try:
        if not file.filename:
            return JSONResponse(status_code=400, content={"message": "No file provided", "chunks_loaded": 0})
        
        print(f"Received upload request for file: {file.filename}")
        
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"message": "File is empty", "chunks_loaded": 0})
        
        # Decode (support gz)
        try:
            if file.filename.endswith('.gz'):
                content = gzip.decompress(content)
            text_content = content.decode('utf-8', errors='ignore')
        except Exception as e:
            return JSONResponse(status_code=400, content={"message": f"Failed to decode file: {str(e)}", "chunks_loaded": 0})
        
        lines = text_content.split('\n')
        log_entries = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
        
        print(f"Parsed {len(log_entries)} valid log entries from file")
        
        if len(log_entries) > 5000:
            log_entries = log_entries[:5000]
            print("Limited to 5000 entries")
        
        if not log_entries:
            return JSONResponse(status_code=400, content={"message": "No valid log entries found", "chunks_loaded": 0})
        
        # Initialize progress tracking
        upload_progress = {
            "total": len(log_entries),
            "processed": 0,
            "status": "processing",
            "start_time": time.time()
        }
        
        print(f"Starting to process {len(log_entries)} log entries")
        
        # Step 1: Truncate table
        cursor.execute("TRUNCATE TABLE docs")
        
        # Step 2: Batch generate embeddings (max 1000 entries per batch)
        batch_size = 1000  # OpenAI allows max 1000 entries per API call
        all_embeddings = []
        for i in range(0, len(log_entries), batch_size):
            batch = log_entries[i:i+batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)
            
            # Update progress
            upload_progress["processed"] = min(i + batch_size, len(log_entries))
            print(f"Generated embeddings for {upload_progress['processed']}/{len(log_entries)} entries...")
        
        # Step 3: Batch insert (explicit IDs starting from 1, no conflicts)
        data = []
        for i, (text, emb) in enumerate(zip(log_entries, all_embeddings), 1):
            vec = array.array("f", emb)
            data.append((i, text, vec))
        
        cursor.executemany(
            "INSERT INTO docs (id, text, embedding) VALUES (:1, :2, :3)",
            data
        )
        
        connection.commit()
        end_time = time.time()
        total_time = end_time - upload_progress["start_time"]
        print(f"Processing complete! Total time: {total_time:.1f} seconds")
        
        # Reload BM25
        reload_bm25()
        print("BM25 index reloaded successfully")
        
        # Mark progress as complete
        upload_progress["status"] = "complete"
        upload_progress["processed"] = upload_progress["total"]
        
        return {
            "message": f"Upload successful! Loaded {len(log_entries)} log entries (took {total_time:.1f} seconds), ready to query.",
            "chunks_loaded": len(log_entries),
            "processing_time_seconds": round(total_time, 1)
        }
        
    except Exception as e:
        upload_progress["status"] = "error"
        import traceback
        print(f"Upload error: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"Upload failed: {str(e)}", "chunks_loaded": 0})