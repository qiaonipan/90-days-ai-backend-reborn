"""
日志模式提取的模式分析服务
"""

import re
import pandas as pd
from typing import List, Dict, Any
from collections import Counter


class PatternAnalysisService:
    """分析日志模式的服务"""

    def analyze_log_patterns(self, candidate_logs: List[Dict]) -> Dict[str, Any]:
        """
        分析候选日志以提取统计模式：
        - 错误类型分布
        - 时间集中度
        - 常见关键词/模式
        """
        if not candidate_logs:
            return {}

        parsed_logs = []
        for log in candidate_logs:
            text = log["text"]
            match = re.match(r"^(\d{6})\s+(\d{6})\s+\d+\s+(\w+)\s+(.+?):\s+(.+)$", text)
            if match:
                date_str, time_str, level, component, message = match.groups()
                try:
                    ts = pd.to_datetime(
                        f"20{date_str} {time_str}", format="%Y%m%d %H%M%S"
                    )
                except (ValueError, TypeError):
                    ts = None
                parsed_logs.append(
                    {
                        "level": level,
                        "component": component,
                        "message": message,
                        "text": text,
                        "ts": ts,
                    }
                )

        if not parsed_logs:
            return {}

        df = pd.DataFrame(parsed_logs)

        error_types = {}
        for text in df["text"]:
            error_match = re.search(
                r"(ERROR|WARN|FATAL|Exception|Error|Failed|Timeout)",
                text,
                re.IGNORECASE,
            )
            if error_match:
                error_type = error_match.group(1).upper()
                error_types[error_type] = error_types.get(error_type, 0) + 1

            exception_match = re.search(r"(\w+Exception|\w+Error)", text)
            if exception_match:
                exc_type = exception_match.group(1)
                error_types[exc_type] = error_types.get(exc_type, 0) + 1

        top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]

        time_concentration = {}
        if df["ts"].notna().any():
            df_with_ts = df[df["ts"].notna()].copy()
            if len(df_with_ts) > 0:
                df_with_ts["hour"] = df_with_ts["ts"].dt.hour
                hour_counts = df_with_ts["hour"].value_counts().head(3)
                time_concentration = {
                    f"{hour}:00": int(count) for hour, count in hour_counts.items()
                }

                time_span = (
                    df_with_ts["ts"].max() - df_with_ts["ts"].min()
                ).total_seconds() / 60
                time_concentration["span_minutes"] = round(time_span, 1)

        all_words = []
        for message in df["message"]:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", message)
            all_words.extend([w.lower() for w in words])

        word_counts = Counter(all_words)
        top_keywords = word_counts.most_common(10)

        component_counts = df["component"].value_counts().head(5).to_dict()

        message_patterns = {}
        for message in df["message"]:
            pattern = re.sub(r"\d+", "<NUM>", message)
            pattern = re.sub(r"\d+\.\d+\.\d+\.\d+", "<IP>", pattern)
            pattern = re.sub(r"blk_[-\w]+", "<BLOCK_ID>", pattern)
            pattern = re.sub(r"/[^\s]+", "<PATH>", pattern)
            message_patterns[pattern] = message_patterns.get(pattern, 0) + 1

        top_patterns = sorted(
            message_patterns.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_logs": len(candidate_logs),
            "error_distribution": {k: int(v) for k, v in top_errors},
            "time_concentration": time_concentration,
            "top_keywords": {k: int(v) for k, v in top_keywords},
            "component_distribution": {k: int(v) for k, v in component_counts.items()},
            "common_patterns": {pattern: int(count) for pattern, count in top_patterns},
        }
