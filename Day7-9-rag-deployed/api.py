import os
import array
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
import oracledb
from dotenv import load_dotenv

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
# FastAPI app
# =========================
app = FastAPI(
    title="Oracle 26ai Cloud Vector Semantic Search API (RAG prototype)",
    description="""
    <b>Qiaoni's reskilling project after layoff · RAG Full Cycle Demo</b><br><br>

    This service demonstrates an enterprise-style semantic search system:
    • OpenAI embeddings
    • Oracle 26ai native vector search
    • Retrieval-Augmented Generation (RAG)<br><br>

    👇 Try it via /docs
    """,
    version="1.0.0",
)

# =========================
# Static frontend
# =========================
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", tags=["Frontend"])
def redirect_to_frontend():
    return RedirectResponse(url="/static/index.html")


# =========================
# Request model
# =========================
class QueryRequest(BaseModel):
    query: str = "What caused the block to be missing?"
    top_k: int = 3


# =========================
# Core semantic search API
# =========================
@app.post("/search", tags=["RAG Full Cycle"])
def semantic_search(request: QueryRequest):
    """
    1. Embed query
    2. Vector search in Oracle
    3. RAG summarization
    """

    # ---------- Step 1: Query embedding ----------
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.query
    )
    query_embedding = embedding_response.data[0].embedding
    query_vector = array.array("f", query_embedding)

    # ---------- Step 2: Vector search ----------
    cursor.execute(
        """
        SELECT text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
        FROM docs
        ORDER BY distance ASC
        FETCH FIRST :top_k ROWS ONLY
        """,
        query_vec=query_vector,
        top_k=request.top_k
    )

    results = cursor.fetchall()

    retrieved_logs = []
    for rank, (text, distance) in enumerate(results, start=1):
        similarity = round(1 - distance / 2, 4)
        retrieved_logs.append({
            "rank": rank,
            "text": text.strip(),
            "similarity": similarity
        })

    # ---------- Step 3: RAG generation ----------
    if retrieved_logs:
        context = "\n\n".join(log["text"] for log in retrieved_logs[:3])

        prompt = f"""
You are a professional distributed systems operations expert.

User question:
{request.query}

Below are the most relevant entries retrieved from HDFS production logs:
{context}

Please:
1. Provide the most likely root cause directly
2. Cite key logs as evidence
3. If there is uncertainty, explain the possibilities
4. Answer in English
"""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        summary = completion.choices[0].message.content.strip()
    else:
        summary = "No records highly relevant to this question were found in the current log set."

    return {
        "query": request.query,
        "ai_summary": summary,
        "retrieved_logs": retrieved_logs,
        "note": "AI summary is based on real log vector retrieval results and is traceable"
    }
