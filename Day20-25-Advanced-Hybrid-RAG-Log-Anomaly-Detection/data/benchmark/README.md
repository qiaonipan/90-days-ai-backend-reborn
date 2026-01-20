# Benchmark数据集说明

## 目录结构

```
benchmark/
├── hdfs_ground_truth.json    # HDFS日志的ground truth标注
├── nginx_ground_truth.json    # Nginx日志的ground truth标注（待创建）
└── README.md                  # 本文件
```

## Ground Truth格式

每个ground truth文件应包含以下结构：

```json
{
  "dataset_name": "HDFS_2k",
  "description": "HDFS日志异常检测ground truth",
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

## 标注指南

### 1. 异常信号标注
- **window_start/window_end**: 异常时间窗口（5分钟窗口）
- **template_id**: 日志模板ID（如果使用Drain3）
- **signature**: 异常日志的特征模式
- **count**: 该模式在窗口内的出现次数
- **is_anomaly**: 是否为真实异常（true/false）

### 2. 根因标注
- **root_cause**: 用一句话描述根本原因
- 应该具体、可操作，例如：
  - ✅ "DataNode network timeout causing block replication failures"
  - ❌ "系统错误"（太模糊）

### 3. 证据日志标注
- **evidence_logs**: 支持根因的关键日志条目
- 应该包含3-10条最相关的日志

## 创建Ground Truth的步骤

1. **分析日志文件**
   - 使用系统检测异常信号
   - 人工验证每个信号是否为真实异常

2. **标注异常窗口**
   - 确定异常发生的时间范围
   - 记录异常模式的特征

3. **确定根因**
   - 基于日志分析确定根本原因
   - 用简洁的语言描述

4. **选择证据日志**
   - 选择最能支持根因的日志条目
   - 通常3-10条即可

## 当前状态

- [x] HDFS ground truth（示例，需要完善）
- [ ] Nginx ground truth
- [ ] Kubernetes ground truth
- [ ] 应用日志ground truth

## 注意事项

- Ground truth应该由有经验的SRE或系统管理员标注
- 定期更新ground truth以反映新的异常模式
- 保持标注的一致性
