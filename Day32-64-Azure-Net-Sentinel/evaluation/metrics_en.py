"""
Evaluation metrics calculation module
"""

from typing import List, Dict, Any


def calculate_signal_detection_accuracy(
    detected_signals: List[Dict], ground_truth_signals: List[Dict]
) -> Dict[str, float]:
    """
    Calculate signal detection accuracy metrics

    Args:
        detected_signals: List of anomaly signals detected by the system
        ground_truth_signals: List of true anomaly signals (ground truth)

    Returns:
        Dictionary containing precision, recall, F1 scores
    """
    if not ground_truth_signals:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "detected_count": len(detected_signals),
            "ground_truth_count": 0,
        }

    # Simple matching logic: check if detected signal time windows overlap with ground truth
    # This needs to be adjusted based on actual data structure
    true_positives = 0
    false_positives = 0

    for detected in detected_signals:
        detected_start = detected.get("window_start")
        # Check if matches any ground truth signal
        matched = False
        for gt in ground_truth_signals:
            gt_start = gt.get("window_start")
            # Simple time window matching (within 5-minute window)
            # Actual implementation should use more precise time matching logic
            if detected_start and gt_start:
                # Need to parse and compare time formats here
                matched = True
                break
        if matched:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth_signals) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detected_count": len(detected_signals),
        "ground_truth_count": len(ground_truth_signals),
    }


def calculate_noise_reduction_rate(
    total_logs: int, candidate_logs: int
) -> Dict[str, Any]:
    """
    Calculate noise reduction rate

    Args:
        total_logs: Total number of logs
        candidate_logs: Number of candidate logs (after signal detection)

    Returns:
        Noise reduction statistics
    """
    if total_logs == 0:
        return {
            "noise_reduction_rate": 0.0,
            "total_logs": 0,
            "candidate_logs": 0,
            "filtered_logs": 0,
        }

    filtered_logs = total_logs - candidate_logs
    reduction_rate = (filtered_logs / total_logs) * 100

    return {
        "noise_reduction_rate": round(reduction_rate, 2),
        "total_logs": total_logs,
        "candidate_logs": candidate_logs,
        "filtered_logs": filtered_logs,
        "reduction_ratio": f"{candidate_logs}/{total_logs}",
    }


def calculate_cost_efficiency(
    total_logs: int,
    candidate_logs: int,
    llm_tokens_used: int,
    baseline_tokens_used: int = None,
) -> Dict[str, Any]:
    """
    Calculate cost efficiency metrics

    Args:
        total_logs: Total number of logs
        candidate_logs: Number of candidate logs
        llm_tokens_used: Number of tokens used by LLM
        baseline_tokens_used: Number of tokens used by baseline method (optional)

    Returns:
        Cost efficiency statistics
    """
    # Estimate: how many tokens would be needed if processing all logs directly
    # Assume average 50 tokens per log
    estimated_baseline_tokens = (
        baseline_tokens_used if baseline_tokens_used else (total_logs * 50)
    )

    cost_reduction = (
        ((estimated_baseline_tokens - llm_tokens_used) / estimated_baseline_tokens)
        * 100
        if estimated_baseline_tokens > 0
        else 0.0
    )

    return {
        "llm_tokens_used": llm_tokens_used,
        "estimated_baseline_tokens": estimated_baseline_tokens,
        "cost_reduction_percent": round(cost_reduction, 2),
        "tokens_per_log": (
            round(llm_tokens_used / candidate_logs, 2) if candidate_logs > 0 else 0
        ),
        "efficiency_ratio": f"{llm_tokens_used}/{estimated_baseline_tokens}",
    }


def calculate_root_cause_accuracy(
    predicted_root_cause: str,
    ground_truth_root_cause: str,
    use_semantic_similarity: bool = True,
) -> Dict[str, Any]:
    """
    Calculate root cause diagnosis accuracy

    Args:
        predicted_root_cause: Root cause predicted by the system
        ground_truth_root_cause: True root cause
        use_semantic_similarity: Whether to use semantic similarity (requires embedding model)

    Returns:
        Accuracy statistics
    """
    # Simple keyword matching
    predicted_words = set(predicted_root_cause.lower().split())
    ground_truth_words = set(ground_truth_root_cause.lower().split())

    if not ground_truth_words:
        return {"exact_match": False, "keyword_overlap": 0.0, "jaccard_similarity": 0.0}

    # Jaccard similarity
    intersection = predicted_words & ground_truth_words
    union = predicted_words | ground_truth_words
    jaccard = len(intersection) / len(union) if union else 0.0

    # Keyword overlap rate
    keyword_overlap = (
        len(intersection) / len(ground_truth_words) if ground_truth_words else 0.0
    )

    # Exact match
    exact_match = (
        predicted_root_cause.lower().strip() == ground_truth_root_cause.lower().strip()
    )

    return {
        "exact_match": exact_match,
        "keyword_overlap": round(keyword_overlap, 4),
        "jaccard_similarity": round(jaccard, 4),
        "predicted": predicted_root_cause,
        "ground_truth": ground_truth_root_cause,
    }


def calculate_processing_time_metrics(
    signal_detection_time: float,
    candidate_retrieval_time: float,
    rag_diagnosis_time: float,
    total_time: float,
) -> Dict[str, Any]:
    """
    Calculate processing time metrics

    Args:
        signal_detection_time: Signal detection time (seconds)
        candidate_retrieval_time: Candidate retrieval time (seconds)
        rag_diagnosis_time: RAG diagnosis time (seconds)
        total_time: Total time (seconds)

    Returns:
        Time statistics
    """
    return {
        "signal_detection_time": round(signal_detection_time, 3),
        "candidate_retrieval_time": round(candidate_retrieval_time, 3),
        "rag_diagnosis_time": round(rag_diagnosis_time, 3),
        "total_time": round(total_time, 3),
        "signal_detection_percent": (
            round((signal_detection_time / total_time) * 100, 2)
            if total_time > 0
            else 0
        ),
        "candidate_retrieval_percent": (
            round((candidate_retrieval_time / total_time) * 100, 2)
            if total_time > 0
            else 0
        ),
        "rag_diagnosis_percent": (
            round((rag_diagnosis_time / total_time) * 100, 2) if total_time > 0 else 0
        ),
    }


def generate_evaluation_summary(metrics: Dict[str, Any]) -> str:
    """
    Generate evaluation summary

    Args:
        metrics: Dictionary of all evaluation metrics

    Returns:
        Formatted summary string
    """
    summary_lines = ["=" * 60, "EVALUATION SUMMARY", "=" * 60]

    if "signal_detection" in metrics:
        sd = metrics["signal_detection"]
        summary_lines.append("\n📊 Signal Detection:")
        summary_lines.append(f"  - Precision: {sd.get('precision', 0):.2%}")
        summary_lines.append(f"  - Recall: {sd.get('recall', 0):.2%}")
        summary_lines.append(f"  - F1-Score: {sd.get('f1', 0):.2%}")

    if "noise_reduction" in metrics:
        nr = metrics["noise_reduction"]
        summary_lines.append("\n🔇 Noise Reduction:")
        summary_lines.append(
            f"  - Reduction Rate: {nr.get('noise_reduction_rate', 0):.2f}%"
        )
        summary_lines.append(
            f"  - Filtered: {nr.get('filtered_logs', 0)}/{nr.get('total_logs', 0)} logs"
        )

    if "cost_efficiency" in metrics:
        ce = metrics["cost_efficiency"]
        summary_lines.append("\n💰 Cost Efficiency:")
        summary_lines.append(
            f"  - Cost Reduction: {ce.get('cost_reduction_percent', 0):.2f}%"
        )
        summary_lines.append(f"  - Tokens Used: {ce.get('llm_tokens_used', 0)}")

    if "root_cause" in metrics:
        rc = metrics["root_cause"]
        summary_lines.append("\n🎯 Root Cause Diagnosis:")
        summary_lines.append(f"  - Keyword Overlap: {rc.get('keyword_overlap', 0):.2%}")
        summary_lines.append(
            f"  - Jaccard Similarity: {rc.get('jaccard_similarity', 0):.2%}"
        )

    if "processing_time" in metrics:
        pt = metrics["processing_time"]
        summary_lines.append("\n⏱️  Processing Time:")
        summary_lines.append(f"  - Total: {pt.get('total_time', 0):.3f}s")
        summary_lines.append(
            f"  - Signal Detection: {pt.get('signal_detection_time', 0):.3f}s ({pt.get('signal_detection_percent', 0):.1f}%)"
        )
        summary_lines.append(
            f"  - RAG Diagnosis: {pt.get('rag_diagnosis_time', 0):.3f}s ({pt.get('rag_diagnosis_percent', 0):.1f}%)"
        )

    summary_lines.append("\n" + "=" * 60)

    return "\n".join(summary_lines)
