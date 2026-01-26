"""
FastAPI application main module
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from api.routes import search, diagnose, upload
from database.connection import init_tables, db_pool
from utils.logging_config import logger
from services.reranker import RerankerService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    try:
        db_pool.initialize()
        init_tables()
        # Warmup reranker model to avoid cold start delay
        RerankerService.warmup()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    try:
        db_pool.close()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


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
    """,
    version="2.0.0 (Refactored Architecture)",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.include_router(search.router)
app.include_router(diagnose.router)
app.include_router(upload.router)


@app.get("/", tags=["Frontend"])
def redirect_to_frontend():
    """Redirect root to frontend"""
    return RedirectResponse(url="/static/index.html")


@app.get("/favicon.ico", tags=["Frontend"])
def favicon():
    """Favicon endpoint"""
    return Response(status_code=204)


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    try:
        with db_pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            cursor.fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
