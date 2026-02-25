# Benchmark Dataset

## Directory Structure

```
benchmark/
├── hdfs_ground_truth.json    # HDFS log ground truth annotations
├── nginx_ground_truth.json    # Nginx log ground truth (to be created)
└── README_EN.md              # This file
```

## Ground Truth Format

Each ground truth file should follow this structure:

```json
{
  "dataset_name": "HDFS_2k",
  "description": "HDFS log anomaly detection ground truth",
  "signals": [
    {
      "window_start": "2008-11-10T14:54:04",
      "window_end": "2008-11-10T14:59:04",
      "template_id": "T123",
      "signature": "ERROR dfs.DataNode$PacketResponder: ...",
      "count": 45,
      "is_anomaly": true
    }
  ],
  "root_cause": "DataNode network timeout causing block replication failures",
  "evidence_logs": [
    "081110 145404 34 ERROR dfs.DataNode$PacketResponder: ...",
    "..."
  ],
  "annotator": "human",
  "annotation_date": "2024-01-01"
}
```

## Annotation Guidelines

### 1. Anomaly Signal Annotation
- **window_start/window_end**: Anomaly time window (5-minute window)
- **template_id**: Log template ID (if using Drain3)
- **signature**: Characteristic pattern of anomaly logs
- **count**: Number of occurrences of this pattern in the window
- **is_anomaly**: Whether this is a true anomaly (true/false)

### 2. Root Cause Annotation
- **root_cause**: One-sentence description of the root cause
- Should be specific and actionable, e.g.:
  - ✅ "DataNode network timeout causing block replication failures"
  - ❌ "System error" (too vague)

### 3. Evidence Log Annotation
- **evidence_logs**: Key log entries supporting the root cause
- Should contain 3-10 most relevant logs

## Steps to Create Ground Truth

1. **Analyze log file**
   - Use system to detect anomaly signals
   - Manually verify each signal as true anomaly

2. **Annotate anomaly windows**
   - Determine time range of anomalies
   - Record characteristic patterns

3. **Determine root cause**
   - Identify root cause based on log analysis
   - Describe concisely

4. **Select evidence logs**
   - Choose logs that best support root cause
   - Typically 3-10 logs

## Current Status

- [x] HDFS ground truth (example, needs completion)
- [ ] Nginx ground truth
- [ ] Kubernetes ground truth
- [ ] Application log ground truth

## Notes

- Ground truth should be annotated by experienced SREs or system administrators
- Update ground truth regularly to reflect new anomaly patterns
- Maintain annotation consistency
