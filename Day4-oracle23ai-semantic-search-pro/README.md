# Oracle 26ai Cloud Vector Semantic Search Demo (RAG prototype)

**A reskilling project by a developer after layoff · Day 4 milestone** 

An enterprise-grade semantic search service prototype built on Oracle Cloud Always Free Autonomous Database 26ai.

Core capability:  
Query logs/documents using natural language; the system understands intent and returns the most relevant content from the cloud database.

## Why this project?
- Traditional keyword search (grep, ELK) can miss semantically similar content that uses different words.
- Vector semantic search can truly "understand" query intent and significantly improve issue localization.
- Real-world enterprise scenarios:
  - Ops: diagnose k8s/Hadoop logs — ask "Why did the pod restart?" → instantly return relevant error logs
  - User feedback analysis: ask "What are the most common complaints?" → return similar comments
  - Security audits: quickly locate logs similar to known attack patterns

## Tech stack
- OpenAI `text-embedding-3-small` (produces 1536-dim vectors)
- Oracle Autonomous Database 26ai (native VECTOR type + VECTOR_DISTANCE computation)
- FastAPI (REST API + Swagger docs)
- Simple frontend HTML/JS (framework-free, easy to understand)

## Quick start (local)

1. Clone the repo and install dependencies
   ```bash
   git clone https://github.com/qiaonipan99-dotcom/90-days-ai-backend-reborn.git
   cd 04-oracle23ai-semantic-search-pro
   pip install -r requirements.txt
   ```
2. Configure `.env` (copy `.env.example` and fill in your keys and Oracle info)
3. Insert sample data
   ```bash
   python insert_logs.py
   ```
4. Start the service
   ```bash
   python -m uvicorn api:app --reload
   ```
5. Open http://127.0.0.1:8000 in your browser
   → Use the polished frontend, enter a question, and search!

## Screenshots
![Project welcome page](screenshots/UI.png)

![Example search results](screenshots/prompts_and_results.png)

## API (advanced)
- Swagger interactive docs: http://127.0.0.1:8000/docs
- Example curl:
   ```bash
   curl -X POST http://127.0.0.1:8000/search \
      -H "Content-Type: application/json" \
      -d '{"query": "Which milk tea experience made me feel the happiest?", "top_k": 3}'
   ```

## Current data and future roadmap
- The demo contains 19 sample entries (Qiaoni qualities + milk tea/coffee experiences) to quickly illustrate semantic search.
- Next steps:
   - Ingest large-scale real system logs (Loghub/HDFS/k8s)
   - Support millions of vectors with indexing and acceleration
   - Add hybrid search (vector + keyword)

## Afterword
After being laid off, I found that AI + vector databases are becoming essential skills for backend engineers.

Instead of worrying, take action. This project is a small milestone in my reskilling journey — an end-to-end enterprise RAG prototype built from scratch.

I hope it helps others who are transitioning as well. If you find it useful, please Star ⭐ the repo or open an Issue/PR to help improve the project.

— A developer reskilling after a layoff (Qiaoni)

2025-12-30