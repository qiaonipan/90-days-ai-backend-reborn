import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import openai
import oracledb
import array
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI and Oracle configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
username = os.getenv("ORACLE_USERNAME")
password = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_DSN")
wallet_path = os.getenv("ORACLE_WALLET_PATH")

# Connect to Oracle database (connect once at startup)
connection = oracledb.connect(
    user=username,
    password=password,
    dsn=dsn,
    config_dir=wallet_path,
    wallet_location=wallet_path,
    wallet_password=password
)
cursor = connection.cursor()

# FastAPI application
app = FastAPI(
    title="Oracle 26ai Cloud Vector Semantic Search API (RAG prototype)",
    description="""
    <b>Qiaoni's reskilling project after layoff · Day 4 major milestone</b> <br><br>

    This is an enterprise-grade semantic search service prototype:<br>
    • Generate text embeddings with OpenAI<br>
    • Store them in Oracle Cloud 26ai native vector database<br>
    • Support querying logs/documents with <b>natural language</b> to quickly find the most relevant content<br><br>

    <b>Real-world enterprise use cases:</b><br>
    • Ops: "Why did the k8s pod crash?" → return the most relevant error logs<br>
    • Product: "What are French users most dissatisfied about?" → return the most relevant feedback<br>
    • Security: "Any signs similar to known attacks?" → return similar logs<br><br>

    👇 Try queries below!
    """,
    version="1.0.0",
    contact={
        "name": "Reskilling Developer",
        "url": "https://github.com/qiaonipan99-dotcom/90-days-ai-backend-reborn",
    }
)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
@app.get("/")
def redirect_to_frontend():
    return RedirectResponse(url="/static/index.html")

# Request body model - add defaults and example
class QueryRequest(BaseModel):
    query: str = "What makes Qiaoni a great contributor to a team project?"
    top_k: int = 3

    class Config:
        schema_extra = {
            "example": {
                "query": "Which milk tea experience made me feel the happiest?",
                "top_k": 3
            }
        }

# Friendly welcome page for the API root
@app.get("/", tags=["Welcome"])
def root():
    return {
        "title": "Welcome to the Oracle 26ai Cloud Vector Semantic Search Service 🚀",
        "message": "This is an enterprise-grade RAG (Retrieval-Augmented Generation) prototype that understands natural language and accurately finds the most relevant content from logs/documents.",
        "quick_start": [
            "1. Click /docs on the left to open the interactive docs",
            "2. Find the POST /search endpoint",
            "3. Click 'Try it out' and enter your question",
            "4. Click Execute to see the most relevant results immediately!"
        ],
        "demo_queries": [
            "What makes Qiaoni a great contributor to a team project?",
            "Which milk tea experience made me feel the happiest?",
            "Which coffee made me feel most energized?"
        ],
        "docs": "/docs",
        "redoc": "/redoc",
        "github": "https://github.com/qiaonipan99-dotcom/90-days-ai-backend-reborn"  # ← change to yours
    }

# Core search endpoint
@app.post("/search", tags=["Semantic Search"])
def semantic_search(request: QueryRequest):
    # Generate query embedding
    embedding_list = openai.embeddings.create(
        model="text-embedding-3-small",
        input=request.query
    ).data[0].embedding
    query_embedding = array.array('f', embedding_list)

    # Execute Oracle vector distance query
    cursor.execute("""
        SELECT text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
        FROM docs
        ORDER BY distance ASC
        FETCH FIRST :top_k ROWS ONLY
    """, query_vec=query_embedding, top_k=request.top_k)

    results = cursor.fetchall()

    # Format returned results
    formatted = []
    for rank, (text, distance) in enumerate(results, 1):
        similarity = round(1 - distance / 2, 4)
        formatted.append({
            "rank": rank,
            "text": text.strip(),
            "similarity": similarity,
            "distance": round(distance, 6)
        })

    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": formatted,
        "tip": "Higher similarity (closer to 1) indicates greater relevance."
    }
