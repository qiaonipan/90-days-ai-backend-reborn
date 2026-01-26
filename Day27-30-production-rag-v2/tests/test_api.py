"""
API endpoint tests
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.mark.api
class TestUploadEndpoint:
    """Test /upload endpoint"""

    def test_upload_no_file(self, client):
        """Test upload without file"""
        response = client.post("/upload")
        assert response.status_code == 422  # Validation error

    def test_upload_invalid_file(self, client, tmp_path):
        """Test upload with invalid file"""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("")

        with open(invalid_file, "rb") as f:
            response = client.post(
                "/upload", files={"file": ("invalid.txt", f, "text/plain")}
            )

        # Should return error for empty file
        assert response.status_code in [200, 400]  # Streaming response or error

    def test_upload_valid_log_file(self, client, sample_log_file, mock_db_connection):
        """Test upload with valid log file"""
        mock_conn, mock_cursor = mock_db_connection

        # Mock database operations
        mock_cursor.execute.return_value = None
        mock_cursor.executemany.return_value = None
        mock_cursor.fetchall.return_value = []

        with open(sample_log_file, "rb") as f:
            response = client.post(
                "/upload", files={"file": ("test.log", f, "text/plain")}
            )

        # Streaming response should return 200
        assert response.status_code == 200


@pytest.mark.api
class TestSearchEndpoint:
    """Test /search endpoint"""

    def test_search_empty_query(self, client):
        """Test search with empty query"""
        response = client.post("/search", json={"query": "", "top_k": 3})
        # Should handle empty query gracefully
        assert response.status_code in [200, 400]

    def test_search_valid_query(self, client, populated_db_cursor, mock_openai_client):
        """Test search with valid query"""
        # Mock vector search results
        populated_db_cursor.fetchall.return_value = [
            (
                "081110 145404 34 INFO dfs.DataNode$PacketResponder: PacketResponder 0",
                0.1,
            ),
            ("081110 145404 35 ERROR dfs.DataNode$PacketResponder: Exception", 0.2),
        ]

        # Mock retrieval service to return sample results
        from services.retrieval import RetrievalService

        mock_retrieval = RetrievalService(mock_openai_client)
        mock_retrieval.reload_bm25 = lambda: None

        # Mock hybrid_search to return test data (compatible with use_rerank parameter)
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

        # Override dependencies
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
            # Clean up override
            app.dependency_overrides.pop(get_retrieval_service, None)


@pytest.mark.api
class TestDiagnoseEndpoint:
    """Test /diagnose endpoint"""

    def test_diagnose_no_signals(self, client, empty_db_cursor):
        """Test diagnose when no anomaly signals exist"""
        empty_db_cursor.fetchall.return_value = []

        response = client.post("/diagnose", json={})

        assert response.status_code == 200
        data = response.json()
        assert "diagnosis" in data
        assert "signals" in data
        assert len(data["signals"]) == 0


@pytest.mark.api
class TestProgressEndpoint:
    """Test /progress endpoint"""

    def test_progress_idle(self, client):
        """Test progress when no upload in progress"""
        # Reset upload progress state before test
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
    """Test root endpoint"""

    def test_root_redirect(self, client):
        """Test root endpoint redirects to frontend"""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in [200, 307, 308]  # Redirect or serve static

    def test_favicon(self, client):
        """Test favicon endpoint"""
        response = client.get("/favicon.ico")
        assert response.status_code == 204  # No content
