import os
import array
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
from openai import OpenAI
import oracledb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi  # BM25 keyword search

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
cursor.execute("SELECT text FROM docs")
all_logs = cursor.fetchall()
all_texts = [log[0].strip() for log in all_logs if log[0]]

# BM25 initialization (tokenization)
tokenized_corpus = [text.split() for text in all_texts]
bm25 = BM25Okapi(tokenized_corpus)

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
    # Return 204 No Content to avoid 404 errors
    return Response(status_code=204)

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
    tokenized_query = request.query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Fix: Check if all bm25_scores are 0
    if bm25_scores.max() == 0:
        bm25_results = []
    else:
        bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:request.top_k * 2]
        bm25_results = [(all_texts[i], bm25_scores[i]) for i in bm25_results]

    # Step 4: Fusion scoring
    fused_scores = {}

    # Vector results
    distances = [d for _, d in vector_results]
    max_distance = max(distances) if distances else 1
    for text, distance in vector_results:
        vector_score = 1 - (distance / max_distance if max_distance > 0 else 0)
        fused_scores[text] = fused_scores.get(text, 0) + vector_score * 0.7

    # BM25 results
    if bm25_results:
        max_bm25 = max([score for _, score in bm25_results])
        for text, score in bm25_results:
            bm25_norm = score / max_bm25 if max_bm25 > 0 else 0
            fused_scores[text] = fused_scores.get(text, 0) + bm25_norm * 0.3

    # Final sorting
    final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:request.top_k]

    retrieved_logs = []
    for rank, (text, hybrid_score) in enumerate(final_results, 1):
        retrieved_logs.append({
            "rank": rank,
            "text": text,
            "hybrid_score": round(hybrid_score, 4)
        })

    # Step 5: RAG generation 
    if retrieved_logs:
        context = "\n\n".join(log["text"] for log in retrieved_logs)
        prompt = f"""
You are a professional distributed systems operations expert.

User question:
{request.query}

Relevant logs retrieved via hybrid search:
{context}

Please provide a brief one-sentence explanation in English about what these logs typically indicate. 
Keep it concise and actionable (e.g., "This usually indicates network timeout or node overload...").
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