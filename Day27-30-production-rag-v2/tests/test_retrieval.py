"""
Retrieval and search unit tests
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestCandidateRetrieval:
    """Test candidate log retrieval"""

    def test_retrieve_candidates_empty_signals(self):
        """Test retrieval with empty signals"""
        from services.retrieval import RetrievalService

        mock_client = MagicMock()
        service = RetrievalService(mock_client)

        result = service.retrieve_candidate_logs([])
        assert result == []


@pytest.mark.unit
class TestPatternAnalysis:
    """Test log pattern analysis"""

    def test_analyze_log_patterns_empty(self):
        """Test pattern analysis with empty logs"""
        from services.pattern_analysis import PatternAnalysisService

        service = PatternAnalysisService()
        result = service.analyze_log_patterns([])
        assert result == {}

    def test_analyze_log_patterns_with_logs(self):
        """Test pattern analysis with sample logs"""
        from services.pattern_analysis import PatternAnalysisService

        service = PatternAnalysisService()
        candidate_logs = [
            {
                "id": 1,
                "text": "081110 145404 34 ERROR dfs.DataNode$PacketResponder: Exception",
                "ts": None,
            },
            {
                "id": 2,
                "text": "081110 145405 35 ERROR dfs.DataNode$DataXceiver: Timeout",
                "ts": None,
            },
        ]

        result = service.analyze_log_patterns(candidate_logs)

        assert isinstance(result, dict)
        assert "total_logs" in result
        assert "error_distribution" in result
        assert result["total_logs"] == 2
