# AI-Powered Log Anomaly Diagnosis System: Oracle 26ai RAG + Hybrid Search + Signal-Driven Diagnosis

> **Enterprise-grade RAG system combining hybrid retrieval (vector + BM25), anomaly signal detection, and reranking technology for intelligent log diagnosis. Features dynamic file upload, real-time progress tracking, and batch processing optimization.**

## 📌 Project Overview

This project is a **production-grade anomaly diagnosis system** that helps SREs move from raw logs to **interpretable anomaly hypotheses** by combining statistical signal detection with LLM-based pattern reasoning, achieving intelligent noise-to-signal filtering.

> **📖 For system design philosophy and problem statement, see [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)**

**Key Features:**

- **Three-stage diagnosis pipeline**: Signal detection → Signal-driven retrieval → RAG diagnosis
- **OpenAI text embeddings** (text-embedding-3-large, 3072 dimensions)
- **Oracle Autonomous Database 26ai** native vector search
- **BM25 keyword search** for exact lexical matching
- **Hybrid fusion** (70% vector + 30% BM25)
- **CrossEncoder reranking** to improve retrieval accuracy
- **Automatic anomaly signal detection** based on time-series statistics
- **Dynamic file upload** - Upload custom log files via web interface
- **Real-time progress tracking** - Monitor processing progress with live updates
- **Batch processing optimization** - Efficient batch embedding generation (1000 entries per batch)
- **FastAPI backend** RESTful API
- **Modern chat-style frontend** with accordion UI
- **Complete test coverage** Unit tests, integration tests, API tests
- **Evaluation framework** Supporting signal detection accuracy, noise reduction rate, and other metrics
- **CI/CD pipeline** Code linting and test automation

Users can use the system in the following ways:

1. **Upload custom log files** (.log, .txt, .gz) up to 5000 entries
2. **Monitor real-time progress** View progress during processing
3. **Execute anomaly diagnosis** Automatically detect anomaly signals and generate diagnosis reports
4. **Natural language queries** For example:
   - *"What caused the block to be missing?"*
   - *"Why did the DataNode stop responding?"*
   - *"Why did PacketResponder terminate?"*

The system retrieves the **most relevant logs** through hybrid search and generates **concise AI diagnosis** based on retrieved evidence.

---

## 🚀 Key Features

### 1. Three-Stage Diagnosis Pipeline

#### Stage 0: Anomaly Signal Detection
- **Log parsing** Parse logs into structured format (timestamp, level, component, message)
- **Template extraction** Use Drain3 to extract message templates (generalize dynamic parts: numbers, IPs, paths)
- **Time window aggregation** Aggregate logs by 5-minute windows
- **Anomaly detection** Detect anomaly spikes:
  - Error rate > historical p95
  - Log volume change > 3x
  - New template appearance
- **Anomaly scoring** Error-prioritized weighting (80% error count, 20% volume spike)
- **Output** Store Top-3 suspicious time windows and their template signatures

#### Stage 1: Signal-Driven Retrieval
- **Signature matching** Use `REGEXP_LIKE` to match log signatures within time windows
- **Error filtering** Filter logs containing error keywords (ERROR, FATAL, Exception, 404, 500)
- **Candidate merging** Merge candidate logs from all signals, deduplicate, limit to 300 entries
- **Output** High signal-to-noise ratio subset (typically 100-300 logs)

#### Stage 2: RAG Diagnosis Layer
- **Statistical analysis** Perform statistical analysis on candidate subset:
  - Error type distribution
  - Time concentration patterns
  - Common keywords/components
  - Message pattern frequencies
- **Optional reranking** If user query provided, use hybrid search reranking
- **LLM diagnosis** Feed statistical summary + representative samples to LLM
- **Structured output** Force JSON format: `root_cause`, `confidence`, `evidence`, `alternatives`, `next_steps`
- **Output** Interpretable diagnosis report with pattern-based evidence

### 2. Dynamic File Upload

- **Web-based upload interface** - Drag and drop or click to select files
- **Multiple file format support** - Supports .log, .txt, and .gz (compressed) files
- **Automatic processing** - Uploaded files are automatically parsed, embedded, and indexed
- **Data management** - New uploads automatically replace existing data
- **Size limits** - Maximum 5000 log entries per file
- **Streaming response** - Real-time progress updates without polling

### 3. Real-Time Progress Tracking

- **Real-time progress updates** - Real-time progress bar showing processing status
- **Progress API endpoint** - `/progress` endpoint provides current status
- **Detailed status information** - Shows processed/total entries, elapsed time, and current stage
- **Batch progress** - Tracks embedding generation progress per batch
- **Streaming** - Uses Server-Sent Events (SSE) to push progress in real-time

### 4. Hybrid Search Architecture

- **Vector Search (70%)**: Semantic similarity search using OpenAI embeddings + Oracle 26ai
- **BM25 Keyword Search (30%)**: Traditional lexical matching for exact term matching
- **Fusion Scoring**: Combines both methods for optimal retrieval
- **CrossEncoder Reranking**: Uses `sentence-transformers` CrossEncoder model for query-document relevance reranking

### 5. Batch Processing Optimization

- **Efficient embedding generation** - Processes up to 1000 entries per batch (OpenAI API limit)
- **Automatic batching** - Large files are automatically split into batches
- **Progress tracking** - Real-time updates during batch processing
- **Optimized database insertion** - Bulk insert operations for better performance

### 6. Modern Chat Interface

- Real-time chat-style UI
- Collapsible accordion for log evidence display
- Intuitive similarity score display
- Keyword highlighting in results
- Responsive design for different screen sizes

### 7. Production-Ready

- Real HDFS production logs support
- Persistent vector storage in Oracle database
- Enterprise-grade error handling
- Empty corpus protection (prevents ZeroDivisionError)
- Structured logging
- Health check endpoint

### 8. Multi-Format Log Support

- **HDFS logs**: Primary format with optimized parsing
- **Kubernetes logs**: Support for RFC3339 timestamps, JSON format, and Pod/Namespace metadata
- **Nginx logs**: Standard web server log format
- **Generic logs**: Auto-detection of timestamps and log levels
- **Compressed files**: Automatic .gz decompression

### 9. Evaluation Framework

- **Signal detection accuracy**: Precision, recall, F1 score
- **Noise reduction rate**: Percentage of logs filtered out
- **Cost efficiency**: Token usage comparison with baseline methods
- **Root cause diagnosis accuracy**: Keyword overlap and semantic similarity metrics
- **Processing time**: Performance metrics for each stage

### 10. Test Coverage

- **Unit tests** (`@pytest.mark.unit`): Test independent functions and modules
- **Integration tests** (`@pytest.mark.integration`): Test component interactions
- **API tests** (`@pytest.mark.api`): Test API endpoints
- **Code coverage**: Generate coverage reports using pytest-cov

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
cd Day27-30-production-rag-v2
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

# Optional: Log level
LOG_LEVEL=INFO
```

### 4. Prepare database tables

Ensure your Oracle database has the following tables:

```sql
-- Documents table (stores logs and vector embeddings)
CREATE TABLE docs (
    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    text CLOB,
    embedding VECTOR(3072, FLOAT32),
    ts TIMESTAMP
);

-- Anomaly signals table (stores detected anomaly signals)
CREATE TABLE anomaly_signals (
    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    window_start TIMESTAMP,
    template_id NUMBER,
    signature VARCHAR2(4000),
    count NUMBER,
    score NUMBER
);
```

**⚠️ Important**: If upgrading from `text-embedding-3-small` (1536 dimensions) to `text-embedding-3-large` (3072 dimensions), you need to modify the table:

```sql
-- Drop existing table and recreate (if you can lose existing data)
DROP TABLE docs;
CREATE TABLE docs (
    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    text CLOB,
    embedding VECTOR(3072, FLOAT32),
    ts TIMESTAMP
);

-- OR alter the column (if your Oracle version supports it)
ALTER TABLE docs MODIFY embedding VECTOR(3072, FLOAT32);
```

---

## 🚀 Quick Start

### Option 1: Use Dynamic Upload (Recommended)

1. **Start the API server**:
```bash
cd Day27-30-production-rag-v2
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the web interface**:
   - Open http://localhost:8000/ in your browser
   - Upload your log file using the upload area
   - Monitor real-time progress
   - Start querying or diagnosing once upload completes

### Option 2: Pre-load Data (Optional)

If you want to pre-load sample data:

```bash
python insert_logs.py
```

This script will:
- Read HDFS logs from `data/HDFS_2k.log`
- Generate embeddings using OpenAI
- Insert data into Oracle database
- Process approximately 1000 log entries (takes a few minutes)

### Access Points

- **Frontend UI**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Progress Endpoint**: http://localhost:8000/progress
- **Health Check**: http://localhost:8000/health

---

## 📖 Usage

### Web Interface

#### Upload Custom Log Files

1. Open http://localhost:8000/ in your browser
2. **Upload your log file**:
   - Drag and drop a file onto the upload area, or
   - Click "Select File" to choose a file
   - Supported formats: `.log`, `.txt`, `.gz`
   - Maximum: 5000 log entries per file
3. **Monitor progress**:
   - Watch the real-time progress bar
   - View processing status and elapsed time
   - Wait for "Upload successful!" message
4. **Execute diagnosis**:
   - Click "Diagnose Anomalies" button to automatically detect anomalies
   - Or enter a question to query logs (e.g., "What caused the block to be missing?")
   - View AI diagnosis and top relevant logs with similarity scores

#### Query Existing Data

If data is already loaded:
1. Type your question in the chat input box
2. Press Enter or click "Send"
3. View the AI diagnosis and retrieved log evidence

### API Endpoints

#### Upload File

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_log_file.log"
```

**Response** (streaming):
```json
{"status": "parsing", "progress": 0}
{"status": "analyzing", "progress": 5, "detailed_message": "Extracting anomaly signals..."}
{"status": "embedding", "progress": 15, "processed": 0, "total": 5000}
{"status": "complete", "progress": 100, "chunks_loaded": 5000, "processing_time_seconds": 21.1}
```

#### Check Progress

```bash
curl "http://localhost:8000/progress"
```

**Response**:
```json
{
  "progress": 50.0,
  "processed": 2500,
  "total": 5000,
  "status": "processing",
  "elapsed_seconds": 10.5
}
```

#### Anomaly Diagnosis

```bash
curl -X POST "http://localhost:8000/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What caused the anomalies?",
    "signal_ids": [1, 2, 3]
  }'
```

**Response**:
```json
{
  "signals": [
    {
      "window_start": "2024-01-02T10:30:00",
      "template_id": 123,
      "signature": "dfs.DataNode$PacketResponder: Received block <NUM>",
      "count": 45,
      "score": 0.89
    }
  ],
  "candidate_count": 150,
  "diagnosis": {
    "root_cause": "Network timeout causing block replication failures",
    "confidence": 0.85,
    "evidence": ["45% of logs show ERROR type X", "Time concentration: 10:30-10:35"],
    "alternatives": ["Node overload", "Disk I/O issues"],
    "next_steps": ["Check network connectivity", "Review DataNode status"]
  },
  "candidate_samples": [...],
  "message": "Diagnosis complete. Analyzed 150 candidate logs from 3 anomaly signals."
}
```

#### Search Query

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What caused the block to be missing?",
    "top_k": 3
  }'
```

**Response**:
```json
{
  "query": "What caused the block to be missing?",
  "ai_summary": "This usually indicates network timeout or node overload...",
  "retrieved_logs": [
    {
      "rank": 1,
      "text": "081110 145404 34 INFO dfs.DataNode$PacketResponder: ...",
      "hybrid_score": 0.892,
      "distance": 0.108,
      "rerank_score": 0.95
    },
    ...
  ],
  "note": "Results from hybrid retrieval (vector 70% + BM25 30%) with CrossEncoder reranking"
}
```

---

## 🏗️ System Architecture

> **📖 For detailed system design philosophy, problem statement, and technical decisions, see [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)**

The system uses a **three-stage pipeline** for production-grade anomaly diagnosis:

```
┌─────────────────────────────────────────────────────────┐
│          Stage 0: Anomaly Signal Detection               │
│  Log Upload → Parse → Template Extract → Time-Series →  │
│  Anomaly Scoring → Store Top-3 Signals                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│       Stage 1: Signal-Driven Retrieval                  │
│  Load Signals → REGEXP_LIKE Match → Error Filter →     │
│  Deduplicate → 300 High-Signal Logs                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│    Stage 2: Statistical Analysis + RAG Diagnosis        │
│  Analyze Patterns → (Optional) Hybrid Rerank →          │
│  LLM Diagnosis → Structured JSON → Formatted Display     │
└─────────────────────────────────────────────────────────┘
```

### File Upload Flow

```
┌─────────────────────────────────────────┐
│         File Upload Flow                 │
│  User uploads .log/.txt/.gz file        │
│    ↓                                     │
│  Parse and validate (max 5000 entries)  │
│    ↓                                     │
│  Anomaly signal extraction (Stage 0)    │
│    ↓                                     │
│  Batch generate embeddings (1000/batch) │
│    ↓                                     │
│  Bulk insert into Oracle database       │
│    ↓                                     │
│  Reload BM25 index                       │
│    ↓                                     │
│  Ready for queries                       │
└─────────────────────────────────────────┘
```

### Query Flow

```
┌─────────────────────────────────────────┐
│         Query Flow                       │
│  User Query                              │
│    ↓                                     │
│  OpenAI Embedding (text-embedding-3-large)│
│    ↓                                     │
│  ┌─────────────────────────────────────┐ │
│  │      Hybrid Search Engine           │ │
│  │  ┌─────────────┐  ┌──────────────┐ │ │
│  │  │   Vector    │  │    BM25      │ │ │
│  │  │  Search 70% │  │  Search 30%  │ │ │
│  │  └─────────────┘  └──────────────┘ │ │
│  │         ↓              ↓            │ │
│  │      Fusion Scoring                 │ │
│  └─────────────────────────────────────┘ │
│    ↓                                     │
│  CrossEncoder Reranking                  │
│    ↓                                     │
│  Top K Relevant Logs                     │
│    ↓                                     │
│  LLM Evidence Summary (GPT-4o-mini)     │
│    ↓                                     │
│  Frontend Rendering (Chat UI)            │
└─────────────────────────────────────────┘
```

### Modular Architecture

```
Day27-30-production-rag-v2/
├── api/                      # FastAPI backend (modular architecture)
│   ├── main.py              # FastAPI app initialization
│   ├── dependencies.py      # Dependency injection
│   ├── models.py            # Pydantic models
│   └── routes/              # API route handlers
│       ├── search.py        # /search endpoint
│       ├── upload.py        # /upload endpoint
│       └── diagnose.py      # /diagnose endpoint
├── services/                 # Business logic services
│   ├── signal_detection.py  # Anomaly signal detection
│   ├── retrieval.py        # Hybrid search and retrieval
│   ├── diagnosis.py        # RAG diagnosis
│   ├── pattern_analysis.py  # Log pattern analysis
│   └── reranker.py         # CrossEncoder reranking
├── database/                 # Database layer
│   └── connection.py        # Database connection pool
├── utils/                    # Utility modules
│   └── logging_config.py    # Logging configuration
├── evaluation/               # Evaluation framework
│   ├── evaluate.py         # Main evaluation script
│   ├── metrics.py          # Metrics calculation
│   └── metrics_en.py       # Metrics calculation (English version)
├── tests/                    # Test suite
│   ├── test_api.py         # API tests
│   ├── test_retrieval.py   # Retrieval tests
│   └── test_signal_detection.py  # Signal detection tests
├── config.py                # Application configuration
├── insert_logs.py           # Optional: Pre-load data script
├── search.py                # Optional: Simple search test script
├── requirements.txt         # Python dependencies
├── pytest.ini              # pytest configuration
├── ruff.toml               # Code linting configuration
└── static/
    └── index.html          # Frontend (upload UI + chat interface)
```

---

## 🔧 Technical Details

### Hybrid Search Algorithm

1. **Vector Search**:
   - Uses Oracle `VECTOR_DISTANCE` function
   - Returns Top 2×k candidates
   - Normalizes scores to 0-1 range

2. **BM25 Search**:
   - Tokenizes query and corpus
   - Calculates BM25 scores
   - Returns Top 2×k candidates

3. **Fusion**:
   - Vector score × 0.7 + BM25 score × 0.3
   - Ranks by fused score
   - Returns Top k results

4. **Reranking**:
   - Uses CrossEncoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
   - Calculates query-document relevance scores
   - Reranks by rerank score
   - Returns Top k results

### Similarity Score Calculation

- **Formula**: `similarity = 1 - distance`
- **Range**: 0.0 - 1.0 (higher is better)
- **Display**: Relevant results typically show 0.5 - 0.9

### Supported Log Formats

The system automatically detects and parses multiple log formats:

1. **HDFS Logs** (Primary):
   ```
   081110 145404 34 ERROR dfs.DataNode$PacketResponder: Received block blk_12345
   ```
   Format: `YYMMDD HHMMSS LEVEL COMPONENT: MESSAGE`

2. **Kubernetes Logs**:
   - RFC3339 timestamp format: `2024-01-02T10:30:45.123Z INFO [component] message`
   - JSON format: `{"timestamp":"2024-01-02T10:30:45Z","level":"ERROR","pod":"my-pod","namespace":"default","message":"..."}`
   - Automatically extracts Pod and Namespace metadata

3. **Nginx Logs**:
   ```
   [02/Jan/2024:10:30:45] ERROR nginx: connection refused
   ```

4. **Generic Logs**:
   - ISO timestamp: `2024-01-02 10:30:45` or `2024-01-02T10:30:45`
   - Auto-detects log levels: ERROR, WARN, INFO, DEBUG, FATAL, CRITICAL
   - Extracts components from brackets: `[component]`

### File Upload & Processing

1. **File Upload**:
   - Accepts `.log`, `.txt`, and `.gz` (compressed) files
   - Validates file format and size
   - Parses log entries (minimum 20 characters per entry)
   - Limits to 5000 entries per file
   - Automatic format detection and parsing

2. **Anomaly Signal Extraction**:
   - Uses Drain3 to extract message templates
   - Aggregates by 5-minute windows
   - Detects error rate spikes, volume changes, new templates
   - Calculates anomaly scores (80% error, 20% volume)
   - Stores Top-3 anomaly signals

3. **Batch Embedding Generation**:
   - Splits log entries into batches of 1000 (OpenAI API limit)
   - Generates embeddings for each batch
   - Updates progress after each batch completion
   - Combines all embeddings for database insertion

4. **Database Operations**:
   - Truncates existing table before insert
   - Uses explicit IDs starting from 1
   - Bulk insert for optimal performance
   - Automatic BM25 index reload after insertion

5. **Progress Tracking**:
   - Global `upload_progress` dictionary tracks state
   - `/progress` endpoint provides real-time status
   - Frontend receives real-time updates via SSE
   - Shows processed/total entries and elapsed time

### Frontend Features

- **Chat Interface**: Real-time message display
- **Accordion UI**: Collapsible log evidence panels
- **Keyword Highlighting**: Auto-highlights query terms
- **Responsive Design**: Adapts to different screen sizes
- **File Upload UI**: Drag-and-drop or click to select
- **Progress Display**: Real-time progress bar and status updates

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific types of tests
pytest -m unit          # Only unit tests
pytest -m api          # Only API tests
pytest -m integration  # Only integration tests

# View test coverage
pytest --cov=. --cov-report=term-missing
pytest --cov=. --cov-report=html  # Generate HTML report, open htmlcov/index.html
```

### Test Structure

- **Unit Tests** (`tests/test_signal_detection.py`): Test signal detection logic
- **Integration Tests** (`tests/test_retrieval.py`): Test retrieval services
- **API Tests** (`tests/test_api.py`): Test API endpoints

---

## 📊 Evaluation

### Running Evaluation

```bash
python evaluation/evaluate.py
```

This will:
- Load logs from `data/HDFS_2k.log`
- Load ground truth from `data/benchmark/hdfs_ground_truth.json`
- Run evaluation and generate metrics
- Save results to `evaluation/results/evaluation_results.json`

### Evaluation Metrics

- **Signal Detection**: Precision, recall, F1 score
- **Noise Reduction**: Percentage of logs filtered out (target: 95%+)
- **Cost Efficiency**: Token usage comparison with baseline methods (direct RAG)
- **Root Cause Diagnosis**: Keyword overlap, Jaccard similarity

---

## 🎯 Key Improvements Over Previous Versions

### Architecture Refactoring (Day 20-25)
- ✅ **Modular Architecture** - Refactored from monolithic api.py to modular structure
- ✅ **Dependency Injection** - Services injected via FastAPI Depends
- ✅ **Connection Pool** - Database connections managed via pool
- ✅ **Configuration Management** - Centralized settings via Pydantic
- ✅ **Structured Logging** - Structured logging replaces print statements
- ✅ **Testability** - Services can be tested independently

### Three-Stage Diagnosis (Day 17-20)
- ✅ **Anomaly Signal Detection** - Automatic anomaly detection based on time-series statistics
- ✅ **Signal-Driven Retrieval** - Reduces from thousands of logs to 100-300 high-signal logs
- ✅ **Statistical Analysis** - Pattern analysis before LLM processing
- ✅ **Structured Diagnosis** - JSON format output with root cause, confidence, evidence, alternatives

### Reranking Feature (Day 27-30)
- ✅ **CrossEncoder Reranking** - Uses sentence-transformers to improve retrieval accuracy
- ✅ **Model Warmup** - Pre-warms model at startup to avoid cold-start latency
- ✅ **Optional Reranking** - Can optionally use reranking after hybrid search

### Dynamic Upload (Day 15-16)
- ✅ **Web-based File Upload** - No need to run scripts manually
- ✅ **Real-time Progress Tracking** - Monitor processing via SSE with real-time updates
- ✅ **Batch Processing Optimization** - Efficient embedding generation (1000 entries per batch)
- ✅ **Multiple File Format Support** - .log, .txt, .gz (compressed)
- ✅ **Automatic Data Management** - New uploads automatically replace existing data
- ✅ **Progress API Endpoint** - `/progress` for real-time status monitoring
- ✅ **Empty Corpus Protection** - Prevents errors when database is empty

### Hybrid Search (Day 10-11)
- ✅ Combined vector + BM25 search
- ✅ Improved retrieval accuracy
- ✅ Better handling of exact keyword matches

### UI Enhancements
- ✅ Modern chat-style interface
- ✅ Accordion for log evidence
- ✅ Intuitive similarity scores
- ✅ Responsive design
- ✅ File upload interface with drag-and-drop

### Code Quality
- ✅ All comments in Chinese
- ✅ Clean code structure
- ✅ Comprehensive error handling
- ✅ Type hints and documentation
- ✅ Production-ready error messages
- ✅ Ruff code linting
- ✅ Complete test coverage

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
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
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
- Upload a log file via the web interface (recommended)
- Or run `insert_logs.py` to populate database
- Check database connection
- Verify `docs` table has data

### Issue: Upload fails or progress not updating

**Solution**:
- Check file format (must be .log, .txt, or .gz)
- Ensure file has valid log entries (each entry must be at least 20 characters)
- Check file size (max 5000 entries)
- Verify `/progress` endpoint is accessible
- Check browser console for errors

### Issue: CrossEncoder model loading failed

**Solution**:
- Ensure `sentence-transformers` is installed: `pip install sentence-transformers`
- Check network connection (first run requires downloading the model)
- Check log files for detailed error information

---

## 📊 Performance

### Query Performance
- **Query Response Time**: ~2-3 seconds (including LLM generation)
- **Vector Search**: < 100ms (Oracle 26ai native)
- **BM25 Search**: < 50ms (in-memory)
- **Reranking**: < 200ms (CrossEncoder)
- **LLM Generation**: ~1-2 seconds (GPT-4o-mini)

### Upload Performance
- **File Parsing**: < 1 second for 5000 entries
- **Anomaly Signal Extraction**: ~2-3 seconds for 5000 entries
- **Embedding Generation**: ~15-25 seconds for 5000 entries (batch processing)
- **Database Insertion**: ~1-2 seconds for 5000 entries (bulk insert)
- **BM25 Index Reload**: < 1 second
- **Total Upload Time**: ~20-30 seconds for 5000 entries

### Signal Detection Performance
- **Template Extraction**: ~1-2 seconds for 5000 entries
- **Time Window Aggregation**: < 1 second
- **Anomaly Scoring**: < 1 second
- **Total Signal Detection Time**: ~2-4 seconds for 5000 entries

---

## 🛣️ Roadmap

### Completed (Day 27-30)
- [x] Architecture refactoring to modular structure
- [x] Three-stage diagnosis pipeline
- [x] CrossEncoder reranking
- [x] Complete test coverage
- [x] Evaluation framework
- [x] Code quality tools (Ruff)
- [x] Streaming upload progress tracking

### Planned Improvements

- [ ] Incremental upload (append instead of replace)
- [ ] Multiple file upload support
- [ ] Upload history and file management
- [ ] Failure-aware semantic enrichment
- [ ] Two-stage retrieval with reranking
- [ ] Confidence and coverage signals
- [ ] Timeline reconstruction
- [ ] Multi-language support
- [ ] Advanced filtering options
- [ ] Visualization dashboard

---

## 📝 License

This project is part of a personal reskilling journey and is provided as-is for educational purposes.

---

## 🙏 Acknowledgments

- **Oracle 26ai** for native vector search capabilities
- **OpenAI** for embeddings and LLM
- **FastAPI** for the excellent web framework
- **sentence-transformers** for reranking models
- Real HDFS production logs for realistic testing

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ as part of a reskilling journey after layoff**
