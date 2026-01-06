# Oracle 26ai Vector Semantic Search Demo

This is a small hands-on project (Day 3) by a developer who is retraining after being laid off. At the end of 2025 I learned AI and vector databases and built this demo.

The project migrated a local OpenAI embedding-based vector search to the Oracle Cloud Always Free Autonomous Database 26ai (latest version), which supports a native VECTOR type and VECTOR_DISTANCE calculations. The demo implements a full end-to-end semantic search workflow.

Goal: drop in a few text snippets → generate embeddings → query with natural language → Oracle returns the most similar results.

## Highlights
- Uses OpenAI `text-embedding-3-small` to produce 1536-dimensional vectors
- Vectors are stored directly in Oracle 26ai in a `VECTOR(1536, FLOAT32)` column
- Oracle computes vector distances natively (VECTOR_DISTANCE) on the server — no local distance computation required
- Supports natural language semantic search (RAG prototype)
- Privacy-first: keys, passwords, and Wallet path are managed through `.env` and not uploaded to GitHub

## Demo datasets (3 small collections, 19 items total)
1. Personal qualities describing Qiaoni (9 items)
2. Milk tea experiences (5 items)
3. Coffee experiences (5 items)

After running, you can query in natural English, for example:
- “What makes Qiaoni a great contributor to a team project?”
- “Which milk tea experience made me feel the happiest?”
- “Which coffee made me feel most energized?”

## How to run locally

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables (copy `.env.example` to `.env` and fill in)
   ```bash
   cp .env.example .env
   ```
Required values:
- `OPENAI_API_KEY`: your OpenAI API key
- `ORACLE_USERNAME`: typically `ADMIN`
- `ORACLE_PASSWORD`: the database administrator password
- `ORACLE_DSN`: the service name from `tnsnames.ora` in the Wallet (e.g. `vectorsearch26ai_high`)
- `ORACLE_WALLET_PATH`: path to the extracted Wallet folder
3. Insert data
   ```bash
   python insert_logs.py
   ```
4. Run semantic search
   ```bash
   python search.py
   ```

## Local version vs Oracle 26ai cloud version

| Item | Local (in-memory) version | Oracle 26ai cloud version |
|------|---------------------------|---------------------------|
| Data storage | Stored in program memory during runtime; lost on shutdown | Persistently stored in Oracle Cloud database |
| Cost | Free | Always Free (Oracle Always Free tier) |
| Scale | A few hundred items may fill memory | Supports tens of thousands to millions of vectors, auto-scalable |
| Query performance | All computation locally, very fast for small sets | Native cloud vector distance computations, stable and scalable |
| Data privacy | Data stored on local machine | Oracle enterprise-grade encryption + managed governance |
| Production readiness | Toy-level | Enterprise-ready (backups, monitoring, security/compliance) |
| Deployment complexity | No deployment needed | Requires Wallet and cloud configuration (worked through in this project) |

Conclusion: the local version is fine for quick experiments, but for real, production-ready and sustainable AI applications, cloud-native vector databases like Oracle 26ai are the right choice.

## Tech stack
- Python 3.11+
- OpenAI API (`text-embedding-3-small`)
- `oracledb` (Thin mode)
- Oracle Autonomous Database 26ai (Always Free)
- `python-dotenv` (environment variable management)

## Afterword
After being laid off I realized that AI and vector databases are quickly becoming essential skills for backend engineers.
Instead of worrying, take action. This project is a small step in my reskilling journey — I hope it helps others who are also transitioning.
If you find it useful, please Star ⭐ or Fork and try it with larger datasets and more complex queries.

— A developer retraining after a layoff, 2025-12-29
