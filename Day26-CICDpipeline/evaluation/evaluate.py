"""
主评估脚本
用于评估异常诊断系统的性能
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    calculate_signal_detection_accuracy,
    calculate_noise_reduction_rate,
    calculate_cost_efficiency,
    calculate_root_cause_accuracy,
    calculate_processing_time_metrics,
    generate_evaluation_summary,
)

# 导入主系统的API（需要根据实际结构调整）
try:
    from api import extract_anomaly_signals, retrieve_candidate_logs, diagnose_anomaly
except ImportError:
    print(
        "Warning: Could not import from api.py. Make sure you're running from the project root."
    )
    print("You may need to adjust the import path.")


def load_ground_truth(ground_truth_path: str) -> Dict[str, Any]:
    """
    加载ground truth数据

    Args:
        ground_truth_path: ground truth JSON文件路径

    Returns:
        ground truth数据字典
    """
    if not os.path.exists(ground_truth_path):
        print(f"Warning: Ground truth file not found: {ground_truth_path}")
        return {"signals": [], "root_causes": []}

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_system(
    log_entries: List[str], ground_truth: Dict[str, Any], user_query: str = None
) -> Dict[str, Any]:
    """
    评估系统性能

    Args:
        log_entries: 日志条目列表
        ground_truth: ground truth数据
        user_query: 可选的用户查询

    Returns:
        评估结果字典
    """
    print("=" * 60)
    print("Starting System Evaluation")
    print("=" * 60)

    start_time = time.time()
    all_metrics = {}

    # Step 1: 信号检测评估
    print("\n[1/4] Evaluating Signal Detection...")
    signal_start = time.time()
    detected_signals = extract_anomaly_signals(log_entries)
    signal_time = time.time() - signal_start

    ground_truth_signals = ground_truth.get("signals", [])
    signal_metrics = calculate_signal_detection_accuracy(
        detected_signals, ground_truth_signals
    )
    all_metrics["signal_detection"] = signal_metrics
    print(f"  ✓ Detected {len(detected_signals)} signals")
    print(
        f"  ✓ Precision: {signal_metrics['precision']:.2%}, Recall: {signal_metrics['recall']:.2%}"
    )

    # Step 2: 候选检索评估
    print("\n[2/4] Evaluating Candidate Retrieval...")
    retrieval_start = time.time()
    candidate_logs = (
        retrieve_candidate_logs(detected_signals) if detected_signals else []
    )
    retrieval_time = time.time() - retrieval_start

    total_logs = len(log_entries)
    candidate_count = len(candidate_logs)
    noise_metrics = calculate_noise_reduction_rate(total_logs, candidate_count)
    all_metrics["noise_reduction"] = noise_metrics
    print(f"  ✓ Reduced from {total_logs} to {candidate_count} logs")
    print(f"  ✓ Noise reduction: {noise_metrics['noise_reduction_rate']:.2f}%")

    # Step 3: RAG诊断评估
    print("\n[3/4] Evaluating RAG Diagnosis...")
    rag_start = time.time()
    diagnosis = diagnose_anomaly(candidate_logs, user_query) if candidate_logs else {}
    rag_time = time.time() - rag_start

    # 估算token使用（实际应该从API响应中获取）
    # 假设：每条候选日志50 tokens + prompt 200 tokens
    estimated_tokens = (candidate_count * 50) + 200

    cost_metrics = calculate_cost_efficiency(
        total_logs, candidate_count, estimated_tokens
    )
    all_metrics["cost_efficiency"] = cost_metrics

    # 根因准确率评估
    predicted_root_cause = diagnosis.get("root_cause", "")
    ground_truth_root_cause = ground_truth.get("root_cause", "")
    root_cause_metrics = calculate_root_cause_accuracy(
        predicted_root_cause, ground_truth_root_cause
    )
    all_metrics["root_cause"] = root_cause_metrics
    print(
        f"  ✓ Root cause keyword overlap: {root_cause_metrics['keyword_overlap']:.2%}"
    )

    # Step 4: 处理时间评估
    total_time = time.time() - start_time
    time_metrics = calculate_processing_time_metrics(
        signal_time, retrieval_time, rag_time, total_time
    )
    all_metrics["processing_time"] = time_metrics

    # 生成摘要
    summary = generate_evaluation_summary(all_metrics)
    print("\n" + summary)

    return {
        "metrics": all_metrics,
        "summary": summary,
        "detected_signals": detected_signals,
        "diagnosis": diagnosis,
    }


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """
    保存评估结果

    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 准备保存的数据（移除不可序列化的部分）
    save_data = {
        "metrics": results["metrics"],
        "summary": results["summary"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Evaluation results saved to: {output_path}")


def main():
    """
    主函数
    """
    # 配置路径
    project_root = Path(__file__).parent.parent
    log_file = project_root / "data" / "HDFS_2k.log"
    ground_truth_file = project_root / "data" / "benchmark" / "hdfs_ground_truth.json"
    output_file = project_root / "evaluation" / "results" / "evaluation_results.json"

    # 检查日志文件
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        print("Please ensure HDFS_2k.log exists in the data/ directory.")
        return

    # 加载日志
    print(f"Loading logs from: {log_file}")
    with open(log_file, "r", encoding="utf-8") as f:
        log_entries = [
            line.strip() for line in f if line.strip() and len(line.strip()) > 20
        ]

    print(f"Loaded {len(log_entries)} log entries")

    # 加载ground truth
    ground_truth = load_ground_truth(str(ground_truth_file))
    if not ground_truth.get("signals"):
        print(f"\nWarning: No ground truth signals found in {ground_truth_file}")
        print("Creating empty ground truth for evaluation...")
        print("(You should create ground truth annotations for accurate evaluation)")

    # 运行评估
    results = evaluate_system(
        log_entries, ground_truth, user_query="What caused the anomalies in the logs?"
    )

    # 保存结果
    save_evaluation_results(results, str(output_file))

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
