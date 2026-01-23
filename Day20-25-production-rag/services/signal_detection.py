"""
Anomaly signal detection service
"""
import re
import pandas as pd
from typing import List, Dict, Any
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from utils.logging_config import logger


class SignalDetectionService:
    """Service for detecting anomaly signals from logs"""
    
    def __init__(self):
        self.template_miner = self._init_template_miner()
    
    def _init_template_miner(self) -> TemplateMiner:
        """Initialize Drain3 template miner"""
        config = TemplateMinerConfig()
        config.drain_depth = 4
        config.drain_sim_th = 0.5
        config.drain_max_children = 100
        config.masking = [
            r'blk_-?\d+',
            r'\d+\.\d+\.\d+\.\d+',
            r'/\S+',
            r'\d+',
        ]
        return TemplateMiner(config=config)
    
    def extract_anomaly_signals(self, log_entries: List[str]) -> List[Dict[str, Any]]:
        """
        Extract anomaly signals from logs using Drain parser and time-series analysis.
        
        Returns list of top-3 anomaly windows with templates and scores.
        """
        try:
            parsed_logs = []
            hdfs_format_count = 0
            
            for line in log_entries:
                ts = None
                level = None
                component = None
                message = line
                
                match = re.match(r'^(\d{6})\s+(\d{6})\s+\d+\s+(\w+)\s+(.+?):\s+(.+)$', line)
                if match:
                    date_str, time_str, level, component, message = match.groups()
                    hdfs_format_count += 1
                    try:
                        ts = pd.to_datetime(f"20{date_str} {time_str}", format='%Y%m%d %H%M%S')
                    except (ValueError, TypeError):
                        ts = None
                else:
                    nginx_match = re.search(r'\[(\d{2}/\w{3}/\d{4}):(\d{2}:\d{2}:\d{2})', line)
                    if nginx_match:
                        date_str, time_str = nginx_match.groups()
                    try:
                        ts = pd.to_datetime(f"{date_str} {time_str}", format='%d/%b/%Y %H:%M:%S')
                    except (ValueError, TypeError):
                        ts = None
                    
                    iso_match = re.search(r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})', line)
                    if iso_match and ts is None:
                        try:
                            ts = pd.to_datetime(iso_match.group(1))
                        except (ValueError, TypeError):
                            ts = None
                    
                    level_match = re.search(r'\b(ERROR|WARN|INFO|DEBUG|FATAL|CRITICAL)\b', line, re.IGNORECASE)
                    if level_match:
                        level = level_match.group(1).upper()
                
                result = self.template_miner.add_log_message(line)
                cluster_id = result['cluster_id']
                event_template = result['template_mined'] if result['template_mined'] else line
                
                if ts is None:
                    ts = pd.Timestamp.now()
                
                parsed_logs.append({
                    'ts': ts,
                    'level': level or 'UNKNOWN',
                    'component': component or 'UNKNOWN',
                    'message': message,
                    'raw': line,
                    'event_id': cluster_id,
                    'event_template': event_template
                })
            
            logger.info(f"Parsed {len(parsed_logs)} logs (HDFS: {hdfs_format_count})")
            
            if len(parsed_logs) < 10:
                logger.warning(f"Only {len(parsed_logs)} logs parsed, need at least 10")
                return []
            
            df = pd.DataFrame(parsed_logs)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts')
            
            df['template_id'] = df['event_id'].apply(lambda x: f"T{x}" if x is not None else "T0")
            df['template'] = df['event_template']
            
            df['is_error'] = (
                df['level'].isin(['ERROR', 'FATAL', 'CRITICAL']) | 
                df['message'].str.contains('ERROR|FATAL|Exception|error|fail|failed|failure', case=False, na=False, regex=True) |
                df['raw'].str.contains(r'\b(?:4\d{2}|5\d{2})\b', na=False, regex=True) |
                df['raw'].str.contains(r'\b(?:404|500|502|503|504)\b', na=False, regex=True)
            )
            
            error_count = df['is_error'].sum()
            logger.info(f"Detected {error_count} error logs out of {len(df)} total")
            
            df_indexed = df.set_index('ts')
            window_size = '5min'
            error_df = df_indexed[df_indexed['is_error']]
            error_counts = error_df.resample(window_size).size()
            volume_counts = df_indexed.resample(window_size).size()
            
            error_rate = (error_counts / volume_counts).fillna(0)
            volume_pct_change = volume_counts.pct_change().fillna(0)
            
            if len(error_rate) > 10:
                error_p95 = error_rate.rolling(min(288, len(error_rate)), min_periods=1).quantile(0.95)
            else:
                error_p95 = error_rate.quantile(0.95) if len(error_rate) > 0 else pd.Series([0])
            
            volume_spikes = volume_pct_change > 3.0
            error_spikes = error_rate > error_p95
            anomaly_windows = (volume_spikes | error_spikes)
            suspicious_windows = anomaly_windows[anomaly_windows].index.tolist()
            
            top_windows = suspicious_windows[:3] if len(suspicious_windows) >= 3 else suspicious_windows
            
            logger.info(f"Found {len(suspicious_windows)} suspicious windows, returning top {len(top_windows)}")
            
            results = []
            for window_start in top_windows:
                window_end = window_start + pd.Timedelta(minutes=5)
                window_logs = df[(df['ts'] >= window_start) & (df['ts'] < window_end)]
                
                if len(window_logs) == 0:
                    continue
                
                error_window_logs = window_logs[window_logs['is_error']]
                if len(error_window_logs) == 0:
                    continue
                
                template_counts = error_window_logs['template_id'].value_counts().head(5)
                
                window_volume = len(window_logs)
                window_errors = len(error_window_logs)
                max_error_count = error_counts.max() if len(error_counts) > 0 else 1
                error_count_score = min(window_errors / max(max_error_count, 1), 1.0)
                
                volume_spike_score = 1.0 if window_start in volume_spikes[volume_spikes].index else 0.0
                anomaly_score = (error_count_score * 0.8) + (volume_spike_score * 0.2)
                
                top_template_id = template_counts.index[0] if len(template_counts) > 0 else "UNKNOWN"
                top_template_logs = error_window_logs[error_window_logs['template_id'] == top_template_id]
                signature = top_template_logs['template'].iloc[0] if len(top_template_logs) > 0 else ""
                
                results.append({
                    "window_start": window_start.isoformat(),
                    "template_id": top_template_id,
                    "signature": signature[:500],
                    "count": int(template_counts.iloc[0]) if len(template_counts) > 0 else window_volume,
                    "score": round(anomaly_score, 4),
                    "templates": {str(k): int(v) for k, v in template_counts.head(5).items()}
                })
            
            if len(results) == 0:
                logger.info("No anomaly signals detected")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in anomaly signal extraction: {e}", exc_info=True)
            return []
