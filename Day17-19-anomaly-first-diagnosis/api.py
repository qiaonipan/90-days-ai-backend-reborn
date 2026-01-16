import os
import array
import gzip
import math
import re
import json
from datetime import datetime
from collections import Counter
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
import pandas as pd

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
# Initialize anomaly_signals table
# =========================
def init_anomaly_signals_table():
    """Create anomaly_signals table if it doesn't exist"""
    try:
        # Check if table exists
        cursor.execute("""
            SELECT COUNT(*) FROM user_tables WHERE table_name = 'ANOMALY_SIGNALS'
        """)
        exists = cursor.fetchone()[0] > 0
        
        if not exists:
            cursor.execute("""
                CREATE TABLE anomaly_signals (
                    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    window_start TIMESTAMP,
                    template_id VARCHAR2(100),
                    signature CLOB,
                    count NUMBER,
                    score NUMBER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.commit()
            print("Anomaly signals table created")
        else:
            print("Anomaly signals table already exists")
    except Exception as e:
        print(f"Anomaly signals table initialization: {str(e)}")

def ensure_docs_ts_column():
    """Add ts (TIMESTAMP) column to docs table if it doesn't exist"""
    try:
        # Check if ts column exists
        cursor.execute("""
            SELECT COUNT(*) FROM user_tab_columns 
            WHERE table_name = 'DOCS' AND column_name = 'TS'
        """)
        exists = cursor.fetchone()[0] > 0
        
        if not exists:
            cursor.execute("""
                ALTER TABLE docs ADD (ts TIMESTAMP)
            """)
            connection.commit()
            print("Added ts column to docs table")
        else:
            print("ts column already exists in docs table")
    except Exception as e:
        print(f"Docs table ts column check: {str(e)}")

# Initialize tables on startup
init_anomaly_signals_table()
ensure_docs_ts_column()

# =========================
# Anomaly signal extraction (Stage 0: Unsupervised)
# =========================
def extract_anomaly_signals(log_entries):
    """
    Extract anomaly signals from logs using Drain parser and time-series analysis.
    
    Returns list of top-3 anomaly windows with templates and scores.
    """
    try:
        # Parse logs into DataFrame
        # HDFS format: YYMMDD HHMMSS <number> <LEVEL> <component>: <message>
        parsed_logs = []
        for line in log_entries:
            # Match HDFS log format
            match = re.match(r'^(\d{6})\s+(\d{6})\s+\d+\s+(\w+)\s+(.+?):\s+(.+)$', line)
            if match:
                date_str, time_str, level, component, message = match.groups()
                # Convert to datetime (YYMMDD HHMMSS)
                try:
                    ts = pd.to_datetime(f"20{date_str} {time_str}", format='%Y%m%d %H%M%S')
                except:
                    ts = pd.Timestamp.now()  # Fallback
                parsed_logs.append({
                    'ts': ts,
                    'level': level,
                    'component': component,
                    'message': message,
                    'raw': line
                })
        
        if len(parsed_logs) < 10:
            return []  # Not enough logs for analysis
        
        df = pd.DataFrame(parsed_logs)
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.sort_values('ts')
        
        # Extract templates using simplified pattern matching
        # Replace numbers, IPs, IDs with placeholders to create templates
        def extract_template(message):
            # Replace numbers with <NUM>
            template = re.sub(r'\d+', '<NUM>', message)
            # Replace IP addresses with <IP>
            template = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', template)
            # Replace block IDs (blk_...) with <BLOCK_ID>
            template = re.sub(r'blk_[-\w]+', '<BLOCK_ID>', template)
            # Replace file paths with <PATH>
            template = re.sub(r'/[^\s]+', '<PATH>', template)
            return template
        
        df['template'] = df['message'].apply(extract_template)
        # Create template_id from hash of template
        df['template_id'] = df['template'].apply(lambda x: f"T{abs(hash(x)) % 100000}")
        
        # Filter error logs: ERROR, FATAL, or containing error keywords (404, 500, Exception)
        # More efficient: check level and message directly
        df['is_error'] = df['level'].isin(['ERROR', 'FATAL']) | df['message'].str.contains('ERROR|FATAL|Exception| 404 | 500 ', case=False, na=False, regex=True)
        
        # Set ts as index for resampling
        df_indexed = df.set_index('ts')
        
        # Aggregate by 5-minute windows
        window_size = '5min'
        error_df = df_indexed[df_indexed['is_error']]
        error_counts = error_df.resample(window_size).size()
        volume_counts = df_indexed.resample(window_size).size()
        
        # Calculate error rate
        error_rate = (error_counts / volume_counts).fillna(0)
        
        # Detect spikes: volume change > 3x or error rate > historical p95
        volume_pct_change = volume_counts.pct_change().fillna(0)
        
        # Historical percentile (use rolling window if available)
        if len(error_rate) > 10:
            error_p95 = error_rate.rolling(min(288, len(error_rate)), min_periods=1).quantile(0.95)
        else:
            error_p95 = error_rate.quantile(0.95) if len(error_rate) > 0 else pd.Series([0])
        
        # Identify anomaly windows
        volume_spikes = volume_pct_change > 3.0
        error_spikes = error_rate > error_p95
        
        # Combine anomaly conditions
        anomaly_windows = (volume_spikes | error_spikes)
        suspicious_windows = anomaly_windows[anomaly_windows].index.tolist()
        
        # Get top-3 most suspicious windows
        top_windows = suspicious_windows[:3] if len(suspicious_windows) >= 3 else suspicious_windows
        
        # Extract templates for each suspicious window
        results = []
        for window_start in top_windows:
            window_end = window_start + pd.Timedelta(minutes=5)
            window_logs = df[(df['ts'] >= window_start) & (df['ts'] < window_end)]
            
            if len(window_logs) == 0:
                continue
            
            # Count templates in this window - ONLY count templates containing error keywords
            error_window_logs = window_logs[window_logs['is_error']]
            if len(error_window_logs) == 0:
                continue  # Skip windows with no error logs
            
            template_counts = error_window_logs['template_id'].value_counts().head(5)
            
            # Calculate anomaly score with error-prioritized weighting
            window_volume = len(window_logs)
            window_errors = len(error_window_logs)
            # Normalize error count by max error count in dataset (0-1 range)
            max_error_count = error_counts.max() if len(error_counts) > 0 else 1
            error_count_score = min(window_errors / max(max_error_count, 1), 1.0)  # Normalize to 0-1
            
            # Score: error_count * 0.8 + volume_spike * 0.2 (prioritize errors)
            volume_spike_score = 1.0 if window_start in volume_spikes[volume_spikes].index else 0.0
            anomaly_score = (error_count_score * 0.8) + (volume_spike_score * 0.2)
            
            # Get top template signature (from error logs only)
            top_template_id = template_counts.index[0] if len(template_counts) > 0 else "UNKNOWN"
            top_template_logs = error_window_logs[error_window_logs['template_id'] == top_template_id]
            signature = top_template_logs['message'].iloc[0] if len(top_template_logs) > 0 else ""
            
            results.append({
                "window_start": window_start.isoformat(),
                "template_id": top_template_id,
                "signature": signature[:500],  # Limit length
                "count": int(template_counts.iloc[0]) if len(template_counts) > 0 else window_volume,
                "score": round(anomaly_score, 4),
                "templates": {str(k): int(v) for k, v in template_counts.head(5).items()}
            })
        
        return results
        
    except Exception as e:
        import traceback
        print(f"Error in anomaly signal extraction: {traceback.format_exc()}")
        return []

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

class DiagnosisRequest(BaseModel):
    query: str = None  # Optional user query
    signal_ids: list = None  # Optional: specific signal IDs to analyze, if None uses top signals

# =========================
# Stage 1: Signal-driven candidate retrieval (no LLM)
# =========================
def retrieve_candidate_logs(suspicious_signals):
    """
    Retrieve candidate logs based on anomaly signals using REGEXP_LIKE.
    Returns a small, high-signal-to-noise subset of logs (max 300).
    """
    all_candidates = []
    
    for sig in suspicious_signals:
        try:
            # Parse window_start from signal
            window_start = pd.to_datetime(sig['window_start'])
            window_end = window_start + pd.Timedelta(minutes=5)
            
            # Extract signature pattern for matching
            signature = sig['signature'][:200] if sig.get('signature') else ""
            
            if not signature:
                # If no signature, use template_id as fallback
                signature = sig.get('template_id', '')
            
            # Convert signature to regex pattern
            # Escape special regex characters, but treat <NUM>, <IP>, <BLOCK_ID>, <PATH> as .+?
            regex_pattern = signature
            # Escape special regex chars except our placeholders
            regex_pattern = re.escape(regex_pattern)
            # Replace escaped placeholders with regex patterns
            regex_pattern = regex_pattern.replace('&lt;NUM&gt;', r'\d+')
            regex_pattern = regex_pattern.replace('&lt;IP&gt;', r'\d+\.\d+\.\d+\.\d+')
            regex_pattern = regex_pattern.replace('&lt;BLOCK_ID&gt;', r'blk_[-\w]+')
            regex_pattern = regex_pattern.replace('&lt;PATH&gt;', r'/[^\s]+')
            # Also handle unescaped versions (if any)
            regex_pattern = regex_pattern.replace('<NUM>', r'\d+')
            regex_pattern = regex_pattern.replace('<IP>', r'\d+\.\d+\.\d+\.\d+')
            regex_pattern = regex_pattern.replace('<BLOCK_ID>', r'blk_[-\w]+')
            regex_pattern = regex_pattern.replace('<PATH>', r'/[^\s]+')
            
            # Escape single quotes for Oracle (REGEXP_LIKE uses bind variables)
            regex_pattern_escaped = regex_pattern.replace("'", "''")
            
            # Query logs matching the signature pattern within time window
            # Use REGEXP_LIKE with case-insensitive flag
            # Add error log filter to improve signal-to-noise ratio
            # Calculate window_end + 5 minutes in Python for reliable Oracle binding
            window_end_plus_5min = window_end + pd.Timedelta(minutes=5)
            try:
                cursor.execute("""
                    SELECT id, text, ts
                    FROM docs
                    WHERE REGEXP_LIKE(text, :pattern, 'i')
                      AND REGEXP_LIKE(text, 'ERROR|FATAL|Exception| 404 | 500 ', 'i')
                      AND ts IS NOT NULL
                      AND ts BETWEEN :window_start AND :window_end_plus_5min
                    ORDER BY ts
                    FETCH FIRST 200 ROWS ONLY
                """, pattern=regex_pattern_escaped, window_start=window_start, window_end_plus_5min=window_end_plus_5min)
            except Exception as sql_err:
                # Fallback: if REGEXP_LIKE fails or ts column issues, use LIKE
                print(f"REGEXP_LIKE failed, using LIKE fallback: {str(sql_err)}")
                signature_like = signature.replace("'", "''").replace('%', '\\%').replace('_', '\\_')
                cursor.execute("""
                    SELECT id, text, ts
                    FROM docs
                    WHERE text LIKE :pattern ESCAPE '\\'
                      AND (UPPER(text) LIKE '%ERROR%' OR UPPER(text) LIKE '%FATAL%' OR UPPER(text) LIKE '%EXCEPTION%' OR text LIKE '% 404 %' OR text LIKE '% 500 %')
                      AND (ts IS NULL OR ts BETWEEN :window_start AND :window_end_plus_5min)
                    ORDER BY id
                    FETCH FIRST 200 ROWS ONLY
                """, pattern=f'%{signature_like}%', window_start=window_start, window_end_plus_5min=window_end_plus_5min)
            
            rows = cursor.fetchall()
            for row in rows:
                all_candidates.append({
                    'id': row[0],
                    'text': row[1],
                    'ts': row[2].isoformat() if row[2] and hasattr(row[2], 'isoformat') else (str(row[2]) if row[2] else None)
                })
        except Exception as e:
            print(f"Error retrieving candidates for signal {sig.get('template_id', 'unknown')}: {str(e)}")
            continue
    
    # Remove duplicates by id
    seen_ids = set()
    unique_candidates = []
    for cand in all_candidates:
        if cand['id'] not in seen_ids:
            seen_ids.add(cand['id'])
            unique_candidates.append(cand)
    
    return unique_candidates[:300]  # Limit to 300 logs total

# =========================
# Statistical analysis of candidate logs
# =========================
def analyze_log_patterns(candidate_logs):
    """
    Analyze candidate logs to extract statistical patterns:
    - Error type distribution
    - Time concentration
    - Common keywords/patterns
    """
    if not candidate_logs:
        return {}
    
    # Parse logs into structured format
    parsed_logs = []
    for log in candidate_logs:
        text = log['text']
        # HDFS format: YYMMDD HHMMSS <number> <LEVEL> <component>: <message>
        match = re.match(r'^(\d{6})\s+(\d{6})\s+\d+\s+(\w+)\s+(.+?):\s+(.+)$', text)
        if match:
            date_str, time_str, level, component, message = match.groups()
            try:
                ts = pd.to_datetime(f"20{date_str} {time_str}", format='%Y%m%d %H%M%S')
            except:
                ts = None
            parsed_logs.append({
                'level': level,
                'component': component,
                'message': message,
                'text': text,
                'ts': ts
            })
    
    if not parsed_logs:
        return {}
    
    df = pd.DataFrame(parsed_logs)
    
    # 1. Error type distribution
    error_types = {}
    for text in df['text']:
        # Extract error patterns
        error_match = re.search(r'(ERROR|WARN|FATAL|Exception|Error|Failed|Timeout)', text, re.IGNORECASE)
        if error_match:
            error_type = error_match.group(1).upper()
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Extract exception types
        exception_match = re.search(r'(\w+Exception|\w+Error)', text)
        if exception_match:
            exc_type = exception_match.group(1)
            error_types[exc_type] = error_types.get(exc_type, 0) + 1
    
    top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # 2. Time concentration (if timestamps available)
    time_concentration = {}
    if df['ts'].notna().any():
        df_with_ts = df[df['ts'].notna()].copy()
        if len(df_with_ts) > 0:
            # Group by hour
            df_with_ts['hour'] = df_with_ts['ts'].dt.hour
            hour_counts = df_with_ts['hour'].value_counts().head(3)
            time_concentration = {f"{hour}:00": int(count) for hour, count in hour_counts.items()}
            
            # Time span
            time_span = (df_with_ts['ts'].max() - df_with_ts['ts'].min()).total_seconds() / 60  # minutes
            time_concentration['span_minutes'] = round(time_span, 1)
    
    # 3. Common keywords/patterns
    all_words = []
    for message in df['message']:
        # Extract meaningful words (length > 3, not numbers)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', message)
        all_words.extend([w.lower() for w in words])
    
    word_counts = Counter(all_words)
    top_keywords = word_counts.most_common(10)
    
    # 4. Component distribution
    component_counts = df['component'].value_counts().head(5).to_dict()
    
    # 5. Common message patterns (template extraction)
    message_patterns = {}
    for message in df['message']:
        # Replace numbers, IPs, IDs with placeholders
        pattern = re.sub(r'\d+', '<NUM>', message)
        pattern = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', pattern)
        pattern = re.sub(r'blk_[-\w]+', '<BLOCK_ID>', pattern)
        pattern = re.sub(r'/[^\s]+', '<PATH>', pattern)
        message_patterns[pattern] = message_patterns.get(pattern, 0) + 1
    
    top_patterns = sorted(message_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "total_logs": len(candidate_logs),
        "error_distribution": {k: int(v) for k, v in top_errors},
        "time_concentration": time_concentration,
        "top_keywords": {k: int(v) for k, v in top_keywords},
        "component_distribution": {k: int(v) for k, v in component_counts.items()},
        "common_patterns": {pattern: int(count) for pattern, count in top_patterns}
    }

# =========================
# Stage 2: RAG diagnosis on high-signal subset
# =========================
def diagnose_anomaly(candidate_logs, original_query=None):
    """
    Perform RAG-based diagnosis on a small, high-signal subset of logs.
    If query is provided, performs hybrid vector + text search rerank before LLM.
    Forces LLM to output strict JSON format.
    """
    if not candidate_logs:
        return {
            "pattern_summary": {
                "error_distribution_summary": "No logs available for analysis",
                "time_concentration_summary": "No logs available for analysis",
                "common_features_summary": "No logs available for analysis"
            },
            "root_cause": "No candidate logs found for analysis.",
            "confidence": 0.0,
            "evidence": [],
            "alternatives": [],
            "next_steps": []
        }
    
    # If query provided, perform hybrid rerank on candidates
    if original_query:
        # Generate query embedding
        query_embedding = client.embeddings.create(
            model="text-embedding-3-large",
            input=original_query
        )
        query_vector = array.array("f", query_embedding.data[0].embedding)
        
        # Get candidate IDs
        candidate_ids = [log['id'] for log in candidate_logs]
        if not candidate_ids:
            log_subset = candidate_logs[:300]
        else:
            # Hybrid search on candidate subset
            # Build parameter dict for Oracle
            params = {'query_vec': query_vector}
            placeholders = []
            for i, cid in enumerate(candidate_ids):
                param_name = f'id_{i}'
                params[param_name] = cid
                placeholders.append(f':{param_name}')
            
            placeholders_str = ','.join(placeholders)
            
            # Vector search on candidates
            cursor.execute(f"""
                SELECT id, text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
                FROM docs
                WHERE id IN ({placeholders_str})
                ORDER BY distance ASC
                FETCH FIRST 100 ROWS ONLY
            """, params)
            vector_results = cursor.fetchall()
            
            # BM25 search on candidates
            candidate_texts = {log['id']: log['text'] for log in candidate_logs}
            bm25_scores = {}
            if len(candidate_texts) > 0:
                tokenized_query = original_query.split()
                for log_id, text in candidate_texts.items():
                    tokens = text.split()
                    score = sum(1 for token in tokenized_query if token.lower() in [t.lower() for t in tokens])
                    if score > 0:
                        bm25_scores[log_id] = score
            
            # Fusion scoring
            fused_scores = {}
            text_to_distance = {}
            for row in vector_results:
                log_id, text, distance = row[0], row[1], row[2]
                text_to_distance[log_id] = distance
                vector_score = 1.0 / (1.0 + distance)  # Normalize
                bm25_score = bm25_scores.get(log_id, 0) / max(bm25_scores.values()) if bm25_scores else 0
                fused_scores[log_id] = vector_score * 0.7 + bm25_score * 0.3
            
            # Sort by fused score and get top logs
            sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            reranked_ids = {log_id for log_id, _ in sorted_ids}
            log_subset = [log for log in candidate_logs if log['id'] in reranked_ids]
            # If rerank didn't cover all, add remaining
            if len(log_subset) < 100:
                remaining = [log for log in candidate_logs if log['id'] not in reranked_ids]
                log_subset.extend(remaining[:100 - len(log_subset)])
    else:
        # No query, use candidates as-is
        log_subset = candidate_logs[:300]
    
    # Step 1: Statistical analysis of candidate logs
    stats = analyze_log_patterns(log_subset)
    
    # Step 2: Prepare log samples (not all logs, but representative samples)
    log_samples = log_subset[:50]  # Use first 50 as samples
    log_texts = [log['text'] for log in log_samples]
    context_samples = "\n".join(log_texts)
    
    # Build prompt with statistical analysis and pattern-based reasoning
    query_part = f"\n\nUser question: {original_query}" if original_query else ""
    
    stats_summary = f"""
Statistical Analysis of {stats.get('total_logs', len(log_subset))} Anomaly Logs:

1. Error Type Distribution:
{json.dumps(stats.get('error_distribution', {}), indent=2)}

2. Time Concentration:
{json.dumps(stats.get('time_concentration', {}), indent=2)}

3. Top Keywords (Frequency):
{json.dumps(stats.get('top_keywords', {}), indent=2)}

4. Component Distribution:
{json.dumps(stats.get('component_distribution', {}), indent=2)}

5. Common Message Patterns (Template Frequency):
{json.dumps(stats.get('common_patterns', {}), indent=2)}
"""
    
    prompt = f"""You are a senior SRE analyzing a high-signal subset of anomaly logs (pre-filtered by statistical methods).

CRITICAL: You must analyze the OVERALL PATTERNS, not just individual log lines.

{stats_summary}

Representative Log Samples (first 50 of {len(log_subset)} total logs):
{context_samples}{query_part}

Your task:
1. Analyze the STATISTICAL PATTERNS above (error distribution, time concentration, keywords, components, message patterns)
2. Identify the MOST LIKELY ROOT CAUSE based on overall patterns, not individual log lines
3. Provide evidence as PATTERN DESCRIPTIONS (e.g., "45% of logs show ERROR type X", "Logs cluster in hour Y", "Keyword Z appears in 60% of messages")
4. Do NOT just cite individual log lines - summarize the patterns

You MUST respond with ONLY a valid JSON object (no markdown, no code blocks, no additional text):
{{
  "pattern_summary": {{
    "error_distribution_summary": "Summary of dominant error types and their proportions (e.g., 'ERROR type X accounts for Y% of all errors')",
    "time_concentration_summary": "Summary of time patterns (e.g., 'Errors cluster in hour Z, spanning W minutes')",
    "common_features_summary": "Summary of shared characteristics (e.g., 'Keyword A appears in X% of logs, Component B is involved in Y% of cases')"
  }},
  "root_cause": "string describing the most likely root cause hypothesis based on overall patterns",
  "confidence": 0.85,
  "evidence": ["pattern description 1 (e.g., 'ERROR type X appears in Y% of logs')", "pattern description 2", "pattern description 3"],
  "alternatives": ["alternative explanation 1", "alternative explanation 2"],
  "next_steps": ["actionable step 1", "actionable step 2", "actionable step 3"]
}}

Requirements:
- pattern_summary: Summarize error distribution, time concentration, and common features from the statistical analysis above.
- root_cause: Must be based on STATISTICAL PATTERNS, not individual logs. Describe the overall pattern.
- confidence: Float 0.0-1.0 based on pattern strength
- evidence: Array of 3-5 PATTERN DESCRIPTIONS (e.g., "X% of logs show Y", "Time concentration in Z hour", "Keyword W appears N times")
- alternatives: Array of 1-2 alternative pattern-based explanations
- next_steps: Array of 3-5 concrete, actionable troubleshooting/fix recommendations

Do not cite individual log lines. Focus on summarizing the overall patterns from the statistics.
"""
    
    # Call LLM with JSON mode
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a production SRE. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    analysis_text = completion.choices[0].message.content.strip()
    
    # Parse JSON response
    try:
        # Remove markdown code blocks if present
        if analysis_text.startswith('```'):
            analysis_text = re.sub(r'^```(?:json)?\s*', '', analysis_text, flags=re.MULTILINE)
            analysis_text = re.sub(r'```\s*$', '', analysis_text, flags=re.MULTILINE)
        
        diagnosis_json = json.loads(analysis_text)
        
        # Validate and extract fields
        pattern_summary = diagnosis_json.get('pattern_summary', {})
        root_cause = diagnosis_json.get('root_cause', 'Analysis completed')
        confidence = float(diagnosis_json.get('confidence', 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
        evidence = diagnosis_json.get('evidence', [])
        alternatives = diagnosis_json.get('alternatives', [])
        next_steps = diagnosis_json.get('next_steps', [])
        
        return {
            "pattern_summary": {
                "error_distribution_summary": pattern_summary.get('error_distribution_summary', '') if isinstance(pattern_summary, dict) else '',
                "time_concentration_summary": pattern_summary.get('time_concentration_summary', '') if isinstance(pattern_summary, dict) else '',
                "common_features_summary": pattern_summary.get('common_features_summary', '') if isinstance(pattern_summary, dict) else ''
            },
            "root_cause": root_cause,
            "confidence": round(confidence, 2),
            "evidence": evidence[:5] if isinstance(evidence, list) else [],
            "alternatives": alternatives[:3] if isinstance(alternatives, list) else [],
            "next_steps": next_steps[:5] if isinstance(next_steps, list) else [],
            "raw_analysis": analysis_text,
            "candidate_count": len(log_subset)
        }
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        # Fallback: return structured error
        return {
            "pattern_summary": {
                "error_distribution_summary": "Failed to parse LLM response",
                "time_concentration_summary": "Failed to parse LLM response",
                "common_features_summary": "Failed to parse LLM response"
            },
            "root_cause": "Failed to parse LLM response. Raw output: " + analysis_text[:200],
            "confidence": 0.0,
            "evidence": [],
            "alternatives": [],
            "next_steps": ["Review raw_analysis field for manual interpretation"],
            "raw_analysis": analysis_text,
            "candidate_count": len(log_subset)
        }

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
# Request model for diagnosis
# =========================
class DiagnosisRequest(BaseModel):
    query: str = None  # Optional user query
    signal_ids: list = None  # Optional: specific signal IDs to analyze, if None uses top signals

# =========================
# Anomaly Diagnosis API (Stage 1 + Stage 2)
# =========================
@app.post("/diagnose", tags=["Anomaly Diagnosis"])
def diagnose_anomalies(request: DiagnosisRequest = DiagnosisRequest()):
    """
    Stage 1: Retrieve candidate logs from anomaly signals (no LLM)
    Stage 2: Perform RAG diagnosis on high-signal subset
    """
    try:
        # Get top anomaly signals from database
        if request.signal_ids and len(request.signal_ids) > 0:
            # Use specific signal IDs if provided
            signal_ids_tuple = tuple(request.signal_ids)
            if len(signal_ids_tuple) == 1:
                # Handle single ID case
                cursor.execute("""
                    SELECT window_start, template_id, signature, count, score
                    FROM anomaly_signals
                    WHERE id = :1
                    ORDER BY score DESC
                """, signal_ids_tuple[0])
            else:
                # Multiple IDs
                placeholders = ','.join([f':{i+1}' for i in range(len(signal_ids_tuple))])
                cursor.execute(f"""
                    SELECT window_start, template_id, signature, count, score
                    FROM anomaly_signals
                    WHERE id IN ({placeholders})
                    ORDER BY score DESC
                """, signal_ids_tuple)
        else:
            # Get top 3 signals by score
            cursor.execute("""
                SELECT window_start, template_id, signature, count, score
                FROM anomaly_signals
                ORDER BY score DESC
                FETCH FIRST 3 ROWS ONLY
            """)
        
        signal_rows = cursor.fetchall()
        
        if not signal_rows:
            return {
                "signals": [],
                "candidate_count": 0,
                "diagnosis": {
                    "root_cause": "No significant anomalies detected in the uploaded logs.",
                    "confidence": 0.0,
                    "evidence": [],
                    "alternatives": [],
                    "next_steps": ["Upload logs to enable anomaly detection", "Check if logs contain error patterns"]
                },
                "candidate_samples": [],
                "message": "No anomaly signals found. System is operating normally or no errors detected."
            }
        
        # Convert to signal format
        suspicious_signals = []
        for row in signal_rows:
            suspicious_signals.append({
                'window_start': row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0]),
                'template_id': row[1],
                'signature': row[2],
                'count': row[3],
                'score': float(row[4])
            })
        
        # Stage 1: Retrieve candidate logs (no LLM)
        candidate_logs = retrieve_candidate_logs(suspicious_signals)
        
        if not candidate_logs:
            return {
                "signals": suspicious_signals,
                "candidate_count": 0,
                "diagnosis": {
                    "root_cause": "No candidate logs found matching the anomaly signals.",
                    "confidence": 0.0,
                    "evidence": [],
                    "alternatives": [],
                    "next_steps": ["Check if logs match the signature patterns", "Verify anomaly signal time windows"]
                },
                "candidate_samples": [],
                "message": "No candidate logs matched the anomaly signal patterns."
            }
        
        # Stage 2: RAG diagnosis on high-signal subset
        diagnosis = diagnose_anomaly(candidate_logs, request.query)
        
        return {
            "signals": suspicious_signals,
            "candidate_count": len(candidate_logs),
            "diagnosis": diagnosis,
            "candidate_samples": candidate_logs[:10],  # Top 10 candidate logs for reference
            "message": f"Diagnosis complete. Analyzed {len(candidate_logs)} candidate logs from {len(suspicious_signals)} anomaly signals."
        }
        
    except Exception as e:
        import traceback
        print(f"Diagnosis error: {traceback.format_exc()}")
        return {
            "error": f"Diagnosis failed: {str(e)}",
            "diagnosis": None
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
            
            # Step 2: Extract anomaly signals immediately after parsing
            yield '{"status": "analyzing", "progress": 5, "detailed_message": "Extracting anomaly signals..."}\n'
            anomaly_signals = extract_anomaly_signals(log_entries)
            
            # Store anomaly signals in database
            anomaly_count = 0
            if anomaly_signals:
                signal_data = []
                for signal in anomaly_signals:
                    signal_data.append((
                        pd.to_datetime(signal['window_start']),
                        signal['template_id'],
                        signal['signature'],
                        signal['count'],
                        signal['score']
                    ))
                
                cursor.executemany(
                    """INSERT INTO anomaly_signals (window_start, template_id, signature, count, score)
                       VALUES (:1, :2, :3, :4, :5)""",
                    signal_data
                )
                connection.commit()
                anomaly_count = len(anomaly_signals)
                print(f"Stored {anomaly_count} anomaly signals")
            
            yield f'{{"status": "analyzing", "progress": 7, "detailed_message": "Extracted {anomaly_count} anomaly signals"}}\n'
            
            # Step 3: Truncate docs table
            yield '{"status": "truncating", "progress": 10}\n'
            cursor.execute("TRUNCATE TABLE docs")
            connection.commit()
            
            # Step 4: Batch generate embeddings (max 1000 entries per batch)
            yield '{"status": "embedding", "progress": 15, "processed": 0, "total": ' + str(total) + '}\n'
            
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
                
                # Send progress update (15% to 85% for embedding)
                processed = min(i + batch_size, len(log_entries))
                progress = 15 + int((processed / total) * 70)  # 15% to 85%
                yield f'{{"status": "embedding", "progress": {progress}, "processed": {processed}, "total": {total}}}\n'
                await asyncio.sleep(0)  # Yield control
            
            # Step 5: Batch insert into docs (with timestamp parsing)
            yield '{"status": "inserting", "progress": 90}\n'
            data = []
            for text, emb in zip(log_entries, all_embeddings):
                vec = array.array("f", emb)
                # Parse timestamp from log text (HDFS format: YYMMDD HHMMSS)
                ts = None
                ts_match = re.match(r'^(\d{6})\s+(\d{6})', text)
                if ts_match:
                    try:
                        ts = pd.to_datetime(f"20{ts_match.group(1)} {ts_match.group(2)}", format='%Y%m%d %H%M%S')
                    except:
                        ts = None
                data.append((text, vec, ts))
            
            cursor.executemany(
                "INSERT INTO docs (text, embedding, ts) VALUES (:1, :2, :3)",
                data
            )
            connection.commit()
            
            # Step 6: Reload BM25
            yield '{"status": "reloading", "progress": 95}\n'
            reload_bm25()
            print("BM25 index reloaded successfully")
            
            # Step 7: Generate preliminary summary
            yield '{"status": "generating_summary", "progress": 98}\n'
            
            # Calculate factual patterns based on anomaly_signals + simple statistics
            summary_lines = []
            
            if anomaly_signals and len(anomaly_signals) > 0:
                summary_lines.append(f"{len(anomaly_signals)} high-confidence anomaly window(s) detected")
                
                # Extract top components from error logs in anomaly windows
                component_counter = Counter()
                template_counter = Counter()
                
                for signal in anomaly_signals:
                    # Get templates from signal
                    if 'templates' in signal:
                        for template_id, count in signal['templates'].items():
                            template_counter[template_id] += count
                    
                    # Extract components from logs matching the signature
                    # Parse component from signature or search in log_entries
                    signature = signal.get('signature', '')
                    # Try to extract component from signature (HDFS format)
                    sig_match = re.search(r'(\w+):\s+', signature)
                    if sig_match:
                        component_counter[sig_match.group(1)] += signal.get('count', 0)
                
                # Get top components
                if component_counter:
                    top_components = ', '.join([comp for comp, _ in component_counter.most_common(3)])
                    summary_lines.append(f"Errors concentrated on component(s): {top_components}")
                else:
                    # Fallback: parse components from all error logs
                    error_components = []
                    for line in log_entries:
                        match = re.match(r'^(\d{6})\s+(\d{6})\s+\d+\s+(ERROR|FATAL)\s+(\w+):', line)
                        if match:
                            error_components.append(match.group(4))
                    if error_components:
                        top_components = ', '.join([comp for comp, _ in Counter(error_components).most_common(3)])
                        summary_lines.append(f"Errors concentrated on component(s): {top_components}")
                
                # Get top templates
                if template_counter:
                    top_templates = ', '.join([f"T{str(tid)[:6]}" for tid, _ in template_counter.most_common(3)])
                    summary_lines.append(f"Repeated patterns: {top_templates}")
                else:
                    # Fallback: use template_id from signals
                    template_ids = [sig.get('template_id', '') for sig in anomaly_signals if sig.get('template_id')]
                    if template_ids:
                        top_templates = ', '.join([f"T{str(tid)[:6]}" for tid in template_ids[:3]])
                        summary_lines.append(f"Repeated patterns: {top_templates}")
            else:
                summary_lines.append("Log analysis complete.")
                summary_lines.append("No system-level anomaly windows were detected across the uploaded logs.")
                summary_lines.append("The system is ready to analyze specific patterns or questions you want to investigate.")
            
            # Format summary with bullet points
            preliminary_summary = "Preliminary Diagnosis Summary\n" + "\n".join(f"• {line}" for line in summary_lines)
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing complete! Total time: {total_time:.1f} seconds")
            
            # Final success message with anomaly signals count and preliminary summary
            yield json.dumps({
                "status": "complete",
                "progress": 100,
                "chunks_loaded": len(log_entries),
                "processing_time_seconds": round(total_time, 1),
                "anomaly_signals_count": anomaly_count,
                "preliminary_summary": preliminary_summary,
                "message": f"Upload successful! Loaded {len(log_entries)} log entries. Detected {anomaly_count} anomaly signals."
            }) + '\n'
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Upload error: {traceback.format_exc()}")
            yield f'{{"status": "error", "message": "Upload failed: {error_msg}", "progress": 0}}\n'
    
    return StreamingResponse(progress_generator(), media_type="text/event-stream")