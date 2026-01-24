"""
评估指标计算模块
"""
from typing import List, Dict, Any


def calculate_signal_detection_accuracy(
    detected_signals: List[Dict],
    ground_truth_signals: List[Dict]
) -> Dict[str, float]:
    """
    计算信号检测准确率
    
    Args:
        detected_signals: 系统检测到的异常信号列表
        ground_truth_signals: 真实异常信号列表（ground truth）
    
    Returns:
        包含准确率、召回率、F1的字典
    """
    if not ground_truth_signals:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "detected_count": len(detected_signals),
            "ground_truth_count": 0
        }
    
    # 简单的匹配逻辑：如果检测到的信号时间窗口与ground truth重叠
    # 这里需要根据实际数据结构调整
    true_positives = 0
    false_positives = 0
    
    for detected in detected_signals:
        detected_start = detected.get('window_start')
        # 检查是否与任何ground truth信号匹配
        matched = False
        for gt in ground_truth_signals:
            gt_start = gt.get('window_start')
            # 简单的时间窗口匹配（5分钟窗口内）
            # 实际应该使用更精确的时间匹配逻辑
            if detected_start and gt_start:
                # 这里需要根据实际时间格式解析和比较
                matched = True
                break
        if matched:
            true_positives += 1
        else:
            false_positives += 1
    
    false_negatives = len(ground_truth_signals) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detected_count": len(detected_signals),
        "ground_truth_count": len(ground_truth_signals)
    }


def calculate_noise_reduction_rate(
    total_logs: int,
    candidate_logs: int
) -> Dict[str, Any]:
    """
    计算噪声减少率
    
    Args:
        total_logs: 总日志数
        candidate_logs: 候选日志数（信号检测后）
    
    Returns:
        噪声减少率统计
    """
    if total_logs == 0:
        return {
            "noise_reduction_rate": 0.0,
            "total_logs": 0,
            "candidate_logs": 0,
            "filtered_logs": 0
        }
    
    filtered_logs = total_logs - candidate_logs
    reduction_rate = (filtered_logs / total_logs) * 100
    
    return {
        "noise_reduction_rate": round(reduction_rate, 2),
        "total_logs": total_logs,
        "candidate_logs": candidate_logs,
        "filtered_logs": filtered_logs,
        "reduction_ratio": f"{candidate_logs}/{total_logs}"
    }


def calculate_cost_efficiency(
    total_logs: int,
    candidate_logs: int,
    llm_tokens_used: int,
    baseline_tokens_used: int = None
) -> Dict[str, Any]:
    """
    计算成本效率
    
    Args:
        total_logs: 总日志数
        candidate_logs: 候选日志数
        llm_tokens_used: LLM使用的token数
        baseline_tokens_used: Baseline方法使用的token数（可选）
    
    Returns:
        成本效率统计
    """
    # 估算：如果直接处理所有日志需要多少token
    # 假设每条日志平均50 tokens
    estimated_baseline_tokens = baseline_tokens_used if baseline_tokens_used else (total_logs * 50)
    
    cost_reduction = ((estimated_baseline_tokens - llm_tokens_used) / estimated_baseline_tokens) * 100 if estimated_baseline_tokens > 0 else 0.0
    
    return {
        "llm_tokens_used": llm_tokens_used,
        "estimated_baseline_tokens": estimated_baseline_tokens,
        "cost_reduction_percent": round(cost_reduction, 2),
        "tokens_per_log": round(llm_tokens_used / candidate_logs, 2) if candidate_logs > 0 else 0,
        "efficiency_ratio": f"{llm_tokens_used}/{estimated_baseline_tokens}"
    }


def calculate_root_cause_accuracy(
    predicted_root_cause: str,
    ground_truth_root_cause: str,
    use_semantic_similarity: bool = True
) -> Dict[str, Any]:
    """
    计算根因诊断准确率
    
    Args:
        predicted_root_cause: 系统预测的根因
        ground_truth_root_cause: 真实根因
        use_semantic_similarity: 是否使用语义相似度（需要embedding模型）
    
    Returns:
        准确率统计
    """
    # 简单的关键词匹配
    predicted_words = set(predicted_root_cause.lower().split())
    ground_truth_words = set(ground_truth_root_cause.lower().split())
    
    if not ground_truth_words:
        return {
            "exact_match": False,
            "keyword_overlap": 0.0,
            "jaccard_similarity": 0.0
        }
    
    # Jaccard相似度
    intersection = predicted_words & ground_truth_words
    union = predicted_words | ground_truth_words
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # 关键词重叠率
    keyword_overlap = len(intersection) / len(ground_truth_words) if ground_truth_words else 0.0
    
    # 精确匹配
    exact_match = predicted_root_cause.lower().strip() == ground_truth_root_cause.lower().strip()
    
    return {
        "exact_match": exact_match,
        "keyword_overlap": round(keyword_overlap, 4),
        "jaccard_similarity": round(jaccard, 4),
        "predicted": predicted_root_cause,
        "ground_truth": ground_truth_root_cause
    }


def calculate_processing_time_metrics(
    signal_detection_time: float,
    candidate_retrieval_time: float,
    rag_diagnosis_time: float,
    total_time: float
) -> Dict[str, Any]:
    """
    计算处理时间指标
    
    Args:
        signal_detection_time: 信号检测时间（秒）
        candidate_retrieval_time: 候选检索时间（秒）
        rag_diagnosis_time: RAG诊断时间（秒）
        total_time: 总时间（秒）
    
    Returns:
        时间统计
    """
    return {
        "signal_detection_time": round(signal_detection_time, 3),
        "candidate_retrieval_time": round(candidate_retrieval_time, 3),
        "rag_diagnosis_time": round(rag_diagnosis_time, 3),
        "total_time": round(total_time, 3),
        "signal_detection_percent": round((signal_detection_time / total_time) * 100, 2) if total_time > 0 else 0,
        "candidate_retrieval_percent": round((candidate_retrieval_time / total_time) * 100, 2) if total_time > 0 else 0,
        "rag_diagnosis_percent": round((rag_diagnosis_time / total_time) * 100, 2) if total_time > 0 else 0
    }


def generate_evaluation_summary(metrics: Dict[str, Any]) -> str:
    """
    生成评估摘要
    
    Args:
        metrics: 所有评估指标的字典
    
    Returns:
        格式化的摘要字符串
    """
    summary_lines = [
        "=" * 60,
        "EVALUATION SUMMARY",
        "=" * 60
    ]
    
    if "signal_detection" in metrics:
        sd = metrics["signal_detection"]
        summary_lines.append("\n📊 Signal Detection:")
        summary_lines.append(f"  - Precision: {sd.get('precision', 0):.2%}")
        summary_lines.append(f"  - Recall: {sd.get('recall', 0):.2%}")
        summary_lines.append(f"  - F1-Score: {sd.get('f1', 0):.2%}")
    
    if "noise_reduction" in metrics:
        nr = metrics["noise_reduction"]
        summary_lines.append("\n🔇 Noise Reduction:")
        summary_lines.append(f"  - Reduction Rate: {nr.get('noise_reduction_rate', 0):.2f}%")
        summary_lines.append(f"  - Filtered: {nr.get('filtered_logs', 0)}/{nr.get('total_logs', 0)} logs")
    
    if "cost_efficiency" in metrics:
        ce = metrics["cost_efficiency"]
        summary_lines.append("\n💰 Cost Efficiency:")
        summary_lines.append(f"  - Cost Reduction: {ce.get('cost_reduction_percent', 0):.2f}%")
        summary_lines.append(f"  - Tokens Used: {ce.get('llm_tokens_used', 0)}")
    
    if "root_cause" in metrics:
        rc = metrics["root_cause"]
        summary_lines.append("\n🎯 Root Cause Diagnosis:")
        summary_lines.append(f"  - Keyword Overlap: {rc.get('keyword_overlap', 0):.2%}")
        summary_lines.append(f"  - Jaccard Similarity: {rc.get('jaccard_similarity', 0):.2%}")
    
    if "processing_time" in metrics:
        pt = metrics["processing_time"]
        summary_lines.append("\n⏱️  Processing Time:")
        summary_lines.append(f"  - Total: {pt.get('total_time', 0):.3f}s")
        summary_lines.append(f"  - Signal Detection: {pt.get('signal_detection_time', 0):.3f}s ({pt.get('signal_detection_percent', 0):.1f}%)")
        summary_lines.append(f"  - RAG Diagnosis: {pt.get('rag_diagnosis_time', 0):.3f}s ({pt.get('rag_diagnosis_percent', 0):.1f}%)")
    
    summary_lines.append("\n" + "=" * 60)
    
    return "\n".join(summary_lines)
