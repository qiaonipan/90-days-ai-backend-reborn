"""
API端点测试
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.mark.api
class TestUploadEndpoint:
    """测试 /upload 端点"""

    def test_upload_no_file(self, client):
        """测试无文件上传"""
        response = client.post("/upload")
        assert response.status_code == 422  # 验证错误

    def test_upload_invalid_file(self, client, tmp_path):
        """测试无效文件上传"""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("")

        with open(invalid_file, "rb") as f:
            response = client.post(
                "/upload", files={"file": ("invalid.txt", f, "text/plain")}
            )

        # 空文件应返回错误
        assert response.status_code in [200, 400]  # 流式响应或错误

    def test_upload_valid_log_file(self, client, sample_log_file, mock_db_connection):
        """测试有效日志文件上传"""
        mock_conn, mock_cursor = mock_db_connection

        # 模拟数据库操作
        mock_cursor.execute.return_value = None
        mock_cursor.executemany.return_value = None
        mock_cursor.fetchall.return_value = []

        with open(sample_log_file, "rb") as f:
            response = client.post(
                "/upload", files={"file": ("test.log", f, "text/plain")}
            )

        # 流式响应应返回200
        assert response.status_code == 200


@pytest.mark.api
class TestSearchEndpoint:
    """测试 /search 端点"""

    def test_search_empty_query(self, client):
        """测试空查询搜索"""
        response = client.post("/search", json={"query": "", "top_k": 3})
        # 应优雅地处理空查询
        assert response.status_code in [200, 400]

    def test_search_valid_query(self, client, populated_db_cursor, mock_openai_client):
        """测试有效查询搜索"""
        # 模拟向量搜索结果
        populated_db_cursor.fetchall.return_value = [
            (
                "081110 145404 34 INFO dfs.DataNode$PacketResponder: PacketResponder 0",
                0.1,
            ),
            ("081110 145404 35 ERROR dfs.DataNode$PacketResponder: Exception", 0.2),
        ]

        # 模拟检索服务返回示例结果
        from services.retrieval import RetrievalService

        mock_retrieval = RetrievalService(mock_openai_client)
        mock_retrieval.reload_bm25 = lambda: None

        # 模拟hybrid_search返回测试数据（兼容use_rerank参数）
        mock_retrieval.hybrid_search = lambda query, top_k, use_rerank=True: {
            "retrieved_logs": [
                {
                    "rank": 1,
                    "text": "081110 145404 34 INFO dfs.DataNode$PacketResponder: PacketResponder 0",
                    "hybrid_score": 0.9,
                    "distance": 0.1,
                },
                {
                    "rank": 2,
                    "text": "081110 145404 35 ERROR dfs.DataNode$PacketResponder: Exception",
                    "hybrid_score": 0.8,
                    "distance": 0.2,
                },
            ],
            "distances": [0.1, 0.2],
        }

        # 覆盖依赖项
        from api.dependencies import get_retrieval_service
        from api.main import app

        app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval

        try:
            response = client.post(
                "/search",
                json={"query": "What caused the block to be missing?", "top_k": 3},
            )

            assert response.status_code == 200
            data = response.json()
            assert "query" in data
            assert "retrieved_logs" in data
            assert "ai_summary" in data
        finally:
            # 清理覆盖
            app.dependency_overrides.pop(get_retrieval_service, None)


@pytest.mark.api
class TestDiagnoseEndpoint:
    """测试 /diagnose 端点"""

    def test_diagnose_no_signals(self, client, empty_db_cursor):
        """测试当不存在异常信号时的诊断"""
        empty_db_cursor.fetchall.return_value = []

        response = client.post("/diagnose", json={})

        assert response.status_code == 200
        data = response.json()
        assert "diagnosis" in data
        assert "signals" in data
        assert len(data["signals"]) == 0


@pytest.mark.api
class TestProgressEndpoint:
    """测试 /progress 端点"""

    def test_progress_idle(self, client):
        """测试无上传进行时的进度"""
        # 在测试前重置上传进度状态
        from api.routes import upload

        upload.upload_progress = {
            "total": 0,
            "processed": 0,
            "status": "idle",
            "start_time": None,
        }

        response = client.get("/upload/progress")

        assert response.status_code == 200
        data = response.json()
        assert "progress" in data
        assert "status" in data
        assert data["status"] == "idle" or data["progress"] == 0


@pytest.mark.api
class TestRootEndpoint:
    """测试根端点"""

    def test_root_redirect(self, client):
        """测试根端点重定向到前端"""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in [200, 307, 308]  # 重定向或提供静态文件

    def test_favicon(self, client):
        """测试favicon端点"""
        response = client.get("/favicon.ico")
        assert response.status_code == 204  # 无内容
