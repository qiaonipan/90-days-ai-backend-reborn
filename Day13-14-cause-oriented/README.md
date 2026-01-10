# AI-Powered Log Diagnosis Assistant: Oracle 26ai RAG + Hybrid Search
# Click to play: https://nine0-days-ai-backend-reborn.onrender.com
**An enterprise-grade RAG system with hybrid retrieval (vector + BM25) for intelligent log diagnosis using real HDFS production logs.**
![RAG Result](screenshots/cause-oriented_RAG.png)
*Example query result showing AI diagnosis and retrieved log evidence with similarity scores*



## 📌 Project Overview

This project is a **full-cycle Retrieval-Augmented Generation (RAG) system** that performs **hybrid semantic search** over real HDFS production logs using:

- **OpenAI text embeddings** (text-embedding-3-small)
- **Oracle Autonomous Database 26ai** native vector search
- **BM25 keyword search** for lexical matching
- **Hybrid fusion** (70% vector + 30% BM25)
- **FastAPI backend** with RESTful API
- **Modern chat-style frontend** with accordion UI

Users can ask **natural language questions** such as:

- *"What caused the block to be missing?"*
- *"Why did the DataNode stop responding?"*
- *"PacketResponder terminating"*

The system retrieves the **top 3 most relevant logs** using hybrid search and generates a **concise AI diagnosis** based strictly on retrieved evidence.
---

## 🚀 Key Features

### 1. Hybrid Search Architecture
- **Vector Search (70%)**: Semantic similarity using OpenAI embeddings + Oracle 26ai
- **BM25 Keyword Search (30%)**: Traditional lexical matching for exact term matches
- **Fusion Scoring**: Combines both approaches for optimal retrieval

### 2. Modern Chat Interface
- Real-time chat-style UI
- Collapsible accordion for log evidence
- Similarity scores with intuitive display
- Keyword highlighting in results

### 3. Production-Ready
- Real HDFS production logs (1000+ entries)
- Persistent vector storage in Oracle database
- No mock data or toy examples
- Enterprise-grade error handling

---

## 📋 Prerequisites

- **Python 3.8+**
- **Oracle Autonomous Database 26ai** (with vector search enabled)
- **OpenAI API Key**
- **Oracle Wallet** for database connection

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/qiaonipan/90-days-ai-backend-reborn.git
cd Day10-11-hybrid-search
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Oracle Database Configuration
ORACLE_USERNAME=your_oracle_username
ORACLE_PASSWORD=your_oracle_password
ORACLE_DSN=your_oracle_dsn
ORACLE_WALLET_PATH=path_to_your_wallet_directory
```

### 4. Prepare database table

Ensure your Oracle database has the `docs` table:

```sql
CREATE TABLE docs (
    id NUMBER PRIMARY KEY,
    text CLOB,
    embedding VECTOR(1536, FLOAT32)
);
```

---

## 🚀 Quick Start

### Step 1: Insert data into database

```bash
python insert_logs.py
```

This script will:
- Read HDFS logs from `data/HDFS_2k.log`
- Generate embeddings using OpenAI
- Insert data into Oracle database
- Process ~1000 log entries (takes a few minutes)

### Step 2: Start the API server

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Or using uvicorn directly:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Access the application

- **Frontend UI**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Static Files**: http://localhost:8000/static/index.html

---

## 📖 Usage

### Web Interface

1. Open http://localhost:8000/ in your browser
2. Type your question in the chat input (e.g., "What caused the block to be missing?")
3. Press Enter or click "Send"
4. View the AI diagnosis and top 3 relevant logs with similarity scores

### API Endpoint

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What caused the block to be missing?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "query": "What caused the block to be missing?",
  "ai_summary": "This usually indicates network timeout or node overload...",
  "retrieved_logs": [
    {
      "rank": 1,
      "text": "081110 145404 34 INFO dfs.DataNode$PacketResponder: ...",
      "hybrid_score": 0.892,
      "distance": 0.108
    },
    ...
  ],
  "note": "Results from hybrid retrieval (vector 70% + BM25 30%)"
}
```

---

## 🏗️ System Architecture

```
User Query
    ↓
OpenAI Embedding (text-embedding-3-small)
    ↓
┌─────────────────────────────────────┐
│      Hybrid Search Engine           │
│  ┌─────────────┐  ┌──────────────┐ │
│  │   Vector    │  │    BM25      │ │
│  │  Search 70% │  │  Search 30%  │ │
│  └─────────────┘  └──────────────┘ │
│         ↓              ↓            │
│      Fusion Scoring                 │
└─────────────────────────────────────┘
    ↓
Top 3 Relevant Logs
    ↓
LLM Evidence-Based Summary (GPT-4o-mini)
    ↓
Frontend Rendering (Chat UI)
```

---

## 🔧 Technical Details

### Hybrid Search Algorithm

1. **Vector Search**: 
   - Uses Oracle `VECTOR_DISTANCE` function
   - Returns top 2×k candidates
   - Normalizes scores to 0-1 range

2. **BM25 Search**:
   - Tokenizes query and corpus
   - Calculates BM25 scores
   - Returns top 2×k candidates

3. **Fusion**:
   - Vector score × 0.7 + BM25 score × 0.3
   - Ranks by fused score
   - Returns top k results

### Similarity Score Calculation

- **Formula**: `similarity = 1 - distance`
- **Range**: 0.0 - 1.0 (higher is better)
- **Display**: Typically shows 0.5 - 0.9 for relevant results

### Frontend Features

- **Chat Interface**: Real-time message display
- **Accordion UI**: Collapsible log evidence panels
- **Keyword Highlighting**: Auto-highlights query terms
- **Responsive Design**: Adapts to different screen sizes

---

## 📁 Project Structure

```
Day10-11-hybrid-search/
├── api.py                 # FastAPI backend with hybrid search
├── insert_logs.py         # Data insertion script
├── search.py              # Simple search test script
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── data/
│   └── HDFS_2k.log      # HDFS production logs
├── static/
│   └── index.html       # Frontend chat interface
└── screenshots/
    └── RAG_result.png   # Example screenshots
```

---

## 🎯 Key Improvements Over Previous Versions

### Hybrid Search (Day 10-11)
- ✅ Combined vector + BM25 search
- ✅ Improved retrieval accuracy
- ✅ Better handling of exact keyword matches

### UI Enhancements
- ✅ Modern chat-style interface
- ✅ Accordion for log evidence
- ✅ Intuitive similarity scores
- ✅ Responsive design

### Code Quality
- ✅ All comments in English
- ✅ Clean code structure
- ✅ Error handling
- ✅ Type hints and documentation

---

## 🔍 Example Queries

Try these queries to see the system in action:

- `"What caused the block to be missing"`
- `"DataNode failed"`
- `"PacketResponder terminating"`
- `"Why did replication fail"`
- `"Network timeout issues"`

---

## 🐛 Troubleshooting

### Issue: `uvicorn` command not found

**Solution**: Use `python -m uvicorn` instead:
```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Issue: Database connection failed

**Solution**: 
- Check `.env` file configuration
- Verify Oracle wallet path
- Ensure network access to Oracle database

### Issue: OpenAI API errors

**Solution**:
- Verify `OPENAI_API_KEY` in `.env`
- Check API quota and billing

### Issue: No logs found

**Solution**:
- Run `insert_logs.py` to populate database
- Check database connection
- Verify `docs` table has data

---

## 📊 Performance

- **Query Response Time**: ~2-3 seconds (including LLM generation)
- **Vector Search**: < 100ms (Oracle 26ai native)
- **BM25 Search**: < 50ms (in-memory)
- **LLM Generation**: ~1-2 seconds (GPT-4o-mini)

---

## 🛣️ Roadmap

### Planned Improvements

- [ ] Failure-aware semantic enrichment
- [ ] Two-stage retrieval with reranking
- [ ] Confidence and coverage signals
- [ ] Timeline reconstruction
- [ ] Multi-language support
- [ ] Advanced filtering options

---

## 📝 License

This project is part of a personal reskilling journey and is provided as-is for educational purposes.

---

## 🙏 Acknowledgments

- **Oracle 26ai** for native vector search capabilities
- **OpenAI** for embeddings and LLM
- **FastAPI** for the excellent web framework
- Real HDFS production logs for realistic testing

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ as part of a reskilling journey after layoff**
