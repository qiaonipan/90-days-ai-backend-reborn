"""
Pytest配置和共享fixture
"""

import os
import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

# 在任何导入之前模拟环境变量
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ORACLE_USERNAME", "test_user")
os.environ.setdefault("ORACLE_PASSWORD", "test_pass")
os.environ.setdefault("ORACLE_DSN", "test_dsn")
os.environ.setdefault("ORACLE_WALLET_PATH", "/tmp/test_wallet")

# 创建模拟数据库连接对象
_global_mock_conn = MagicMock()
_global_mock_cursor = MagicMock()
_global_mock_conn.cursor.return_value = _global_mock_cursor

# 创建模拟连接池
_global_mock_pool = MagicMock()
_global_mock_pool.acquire.return_value = _global_mock_conn
_global_mock_pool.release = MagicMock()

# 在导入api模块之前修补oracledb.create_pool
_patcher = patch("oracledb.create_pool", return_value=_global_mock_pool)
_patcher.start()

# 现在可以安全地导入api（数据库连接已被模拟）
from api.main import app
import api.main
from database.connection import db_pool

# 在api模块中设置模拟的连接和游标
from fastapi.testclient import TestClient

# 在每个测试之前重置db_pool
db_pool._pool = _global_mock_pool
db_pool._initialized = True


@pytest.fixture(scope="session")
def mock_db_connection():
    """模拟Oracle数据库连接"""
    return _global_mock_conn, _global_mock_cursor


@pytest.fixture(scope="session")
def mock_openai_client():
    """模拟OpenAI客户端"""
    mock_client = MagicMock()

    # 模拟embedding响应
    mock_embedding_response = MagicMock()
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 3072  # text-embedding-3-large维度
    mock_embedding_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_embedding_response

    # 模拟chat completion响应
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
    """用于测试的示例日志条目"""
    return [
        "081110 145404 34 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862906 terminating",
        "081110 145404 35 ERROR dfs.DataNode$PacketResponder: Exception in PacketResponder 0 for block blk_-1608999687919862906",
        "081110 145405 36 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862907 terminating",
        "081110 145406 37 WARN dfs.DataNode$DataXceiver: DataXceiver error processing WRITE_BLOCK operation",
        "081110 145407 38 ERROR dfs.DataNode$DataXceiver: Exception in DataXceiver for block blk_-1608999687919862908",
    ]


@pytest.fixture(scope="function")
def sample_log_file(sample_log_entries, tmp_path):
    """创建用于测试的临时日志文件"""
    log_file = tmp_path / "test_logs.log"
    log_file.write_text("\n".join(sample_log_entries))
    return str(log_file)


@pytest.fixture(scope="function", autouse=True)
def reset_db_pool():
    """在每个测试之前重置数据库连接池"""
    db_pool._pool = _global_mock_pool
    db_pool._initialized = True
    yield
    # 测试后清理（如果需要）
    db_pool._initialized = False


@pytest.fixture(scope="function")
def app_with_mocks(mock_db_connection, mock_openai_client):
    """创建带有模拟依赖项的FastAPI应用"""
    mock_conn, mock_cursor = mock_db_connection

    # 在设置之前清除lru_cache
    from api.dependencies import get_openai_client, get_retrieval_service

    get_openai_client.cache_clear()

    # 创建带有模拟客户端的新RetrievalService以避免真实API调用
    from services.retrieval import RetrievalService

    mock_retrieval_service = RetrievalService(mock_openai_client)
    # 模拟reload_bm25以避免数据库调用
    mock_retrieval_service.reload_bm25 = lambda: None

    # 使用FastAPI dependency_overrides而不是patch
    app.dependency_overrides[get_openai_client] = lambda: mock_openai_client
    app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_service

    yield app

    # 测试后清理
    app.dependency_overrides.clear()
    get_openai_client.cache_clear()


@pytest.fixture(scope="function")
def client(app_with_mocks):
    """API端点的测试客户端"""
    return TestClient(app_with_mocks)


@pytest.fixture(scope="function")
def empty_db_cursor(mock_db_connection):
    """带有空数据库的模拟游标"""
    _, mock_cursor = mock_db_connection
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)  # 用于COUNT查询
    return mock_cursor


@pytest.fixture(scope="function")
def populated_db_cursor(mock_db_connection, sample_log_entries):
    """带有示例数据的模拟游标"""
    _, mock_cursor = mock_db_connection

    # 为docs表模拟fetchall
    mock_docs = [
        (1, sample_log_entries[0], None),
        (2, sample_log_entries[1], None),
        (3, sample_log_entries[2], None),
    ]
    mock_cursor.fetchall.return_value = mock_docs

    # 为COUNT查询模拟fetchone
    mock_cursor.fetchone.return_value = (len(mock_docs),)

    return mock_cursor
