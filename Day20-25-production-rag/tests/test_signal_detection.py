"""
Signal detection unit tests
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestAnomalySignalExtraction:
    """Test anomaly signal extraction logic"""
    
    def test_extract_signals_empty_logs(self):
        """Test with empty log list"""
        from services.signal_detection import SignalDetectionService
        
        service = SignalDetectionService()
        result = service.extract_anomaly_signals([])
        assert result == []
    
    def test_extract_signals_insufficient_logs(self):
        """Test with insufficient logs (< 10)"""
        from services.signal_detection import SignalDetectionService
        
        service = SignalDetectionService()
        logs = ["081110 145404 34 INFO dfs.DataNode: Test message"] * 5
        result = service.extract_anomaly_signals(logs)
        assert result == []
    
    def test_extract_signals_valid_hdfs_logs(self, sample_log_entries):
        """Test with valid HDFS format logs"""
        from services.signal_detection import SignalDetectionService
        
        service = SignalDetectionService()
        logs = sample_log_entries * 3  # Ensure we have enough logs
        
        result = service.extract_anomaly_signals(logs)
        
        # Should return list (may be empty if no anomalies detected)
        assert isinstance(result, list)
        assert len(result) <= 3  # Top 3 signals
