"""
Pytest configuration and shared fixtures
"""
import os
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ORACLE_USERNAME", "test_user")
os.environ.setdefault("ORACLE_PASSWORD", "test_pass")
os.environ.setdefault("ORACLE_DSN", "test_dsn")
os.environ.setdefault("ORACLE_WALLET_PATH", "/tmp/test_wallet")

# Create mock database connection objects
_global_mock_conn = MagicMock()
_global_mock_cursor = MagicMock()
_global_mock_conn.cursor.return_value = _global_mock_cursor

# Create mock connection pool
_global_mock_pool = MagicMock()
_global_mock_pool.acquire.return_value = _global_mock_conn
_global_mock_pool.release = MagicMock()

# Patch oracledb.create_pool BEFORE importing api module
_patcher = patch('oracledb.create_pool', return_value=_global_mock_pool)
_patcher.start()

# Now safe to import api (database connection is mocked)
from api.main import app
import api.main
from database.connection import db_pool

# Set mocked connection and cursor in api module
from fastapi.testclient import TestClient

# Reset db_pool before each test
db_pool._pool = _global_mock_pool
db_pool._initialized = True


@pytest.fixture(scope="session")
def mock_db_connection():
    """Mock Oracle database connection"""
    return _global_mock_conn, _global_mock_cursor


@pytest.fixture(scope="session")
def mock_openai_client():
    """Mock OpenAI client"""
    mock_client = MagicMock()
    
    # Mock embedding response
    mock_embedding_response = MagicMock()
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 3072  # text-embedding-3-large dimension
    mock_embedding_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    # Mock chat completion response
    mock_chat_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"pattern_summary": {"error_distribution_summary": "Test", "time_concentration_summary": "Test", "common_features_summary": "Test"}, "root_cause": "Test diagnosis", "confidence": 0.8, "evidence": [], "alternatives": [], "next_steps": []}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_chat_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_response
    
    return mock_client


@pytest.fixture(scope="function")
def sample_log_entries():
    """Sample log entries for testing"""
    return [
        "081110 145404 34 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862906 terminating",
        "081110 145404 35 ERROR dfs.DataNode$PacketResponder: Exception in PacketResponder 0 for block blk_-1608999687919862906",
        "081110 145405 36 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862907 terminating",
        "081110 145406 37 WARN dfs.DataNode$DataXceiver: DataXceiver error processing WRITE_BLOCK operation",
        "081110 145407 38 ERROR dfs.DataNode$DataXceiver: Exception in DataXceiver for block blk_-1608999687919862908"
    ]


@pytest.fixture(scope="function")
def sample_log_file(sample_log_entries, tmp_path):
    """Create a temporary log file for testing"""
    log_file = tmp_path / "test_logs.log"
    log_file.write_text("\n".join(sample_log_entries))
    return str(log_file)


@pytest.fixture(scope="function", autouse=True)
def reset_db_pool():
    """Reset database pool before each test"""
    db_pool._pool = _global_mock_pool
    db_pool._initialized = True
    yield
    # Cleanup after test if needed
    db_pool._initialized = False


@pytest.fixture(scope="function")
def app_with_mocks(mock_db_connection, mock_openai_client):
    """Create FastAPI app with mocked dependencies"""
    mock_conn, mock_cursor = mock_db_connection
    
    # Clear lru_cache before setting up
    from api.dependencies import get_openai_client, get_retrieval_service
    get_openai_client.cache_clear()
    
    # Create a new RetrievalService with mocked client to avoid real API calls
    from services.retrieval import RetrievalService
    mock_retrieval_service = RetrievalService(mock_openai_client)
    # Mock reload_bm25 to avoid database calls
    mock_retrieval_service.reload_bm25 = lambda: None
    
    # Use FastAPI dependency_overrides instead of patch
    app.dependency_overrides[get_openai_client] = lambda: mock_openai_client
    app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_service
    
    yield app
    
    # Clean up after test
    app.dependency_overrides.clear()
    get_openai_client.cache_clear()


@pytest.fixture(scope="function")
def client(app_with_mocks):
    """Test client for API endpoints"""
    return TestClient(app_with_mocks)


@pytest.fixture(scope="function")
def empty_db_cursor(mock_db_connection):
    """Mock cursor with empty database"""
    _, mock_cursor = mock_db_connection
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)  # For COUNT queries
    return mock_cursor


@pytest.fixture(scope="function")
def populated_db_cursor(mock_db_connection, sample_log_entries):
    """Mock cursor with sample data"""
    _, mock_cursor = mock_db_connection
    
    # Mock fetchall for docs table
    mock_docs = [
        (1, sample_log_entries[0], None),
        (2, sample_log_entries[1], None),
        (3, sample_log_entries[2], None),
    ]
    mock_cursor.fetchall.return_value = mock_docs
    
    # Mock fetchone for COUNT queries
    mock_cursor.fetchone.return_value = (len(mock_docs),)
    
    return mock_cursor
