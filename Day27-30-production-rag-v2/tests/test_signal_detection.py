"""
信号检测单元测试
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestAnomalySignalExtraction:
    """测试异常信号提取逻辑"""

    def test_extract_signals_empty_logs(self):
        """测试空日志列表"""
        from services.signal_detection import SignalDetectionService

        service = SignalDetectionService()
        result = service.extract_anomaly_signals([])
        assert result == []

    def test_extract_signals_insufficient_logs(self):
        """测试日志不足（< 10）"""
        from services.signal_detection import SignalDetectionService

        service = SignalDetectionService()
        logs = ["081110 145404 34 INFO dfs.DataNode: Test message"] * 5
        result = service.extract_anomaly_signals(logs)
        assert result == []

    def test_extract_signals_valid_hdfs_logs(self, sample_log_entries):
        """测试有效的HDFS格式日志"""
        from services.signal_detection import SignalDetectionService

        service = SignalDetectionService()
        logs = sample_log_entries * 3  # 确保有足够的日志

        result = service.extract_anomaly_signals(logs)

        # 应返回列表（如果未检测到异常可能为空）
        assert isinstance(result, list)
        assert len(result) <= 3  # Top 3信号
