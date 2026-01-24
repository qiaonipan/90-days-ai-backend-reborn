# Evaluation Framework

This directory contains the evaluation framework for the anomaly diagnosis system.

## Structure

```
evaluation/
├── metrics.py          # Metrics calculation module (Chinese comments)
├── metrics_en.py       # Metrics calculation module (English version)
├── evaluate.py         # Main evaluation script (Chinese comments)
├── README.md           # This file
└── results/            # Evaluation results directory
```

## Purpose

The evaluation framework provides:

1. **Signal Detection Accuracy**: Precision, recall, F1-score for anomaly signal detection
2. **Noise Reduction Rate**: Percentage of logs filtered out before LLM processing
3. **Cost Efficiency**: Token usage comparison with baseline methods
4. **Root Cause Accuracy**: Keyword overlap and semantic similarity metrics
5. **Processing Time**: Performance metrics for each stage

## Usage

### Basic Evaluation

```bash
python evaluation/evaluate.py
```

This will:
- Load logs from `data/HDFS_2k.log`
- Load ground truth from `data/benchmark/hdfs_ground_truth.json`
- Run evaluation and generate metrics
- Save results to `evaluation/results/evaluation_results.json`

## Metrics

### Signal Detection
- **Precision**: Percentage of detected signals that are true anomalies
- **Recall**: Percentage of true anomalies that were detected
- **F1-Score**: Harmonic mean of precision and recall

### Noise Reduction
- **Reduction Rate**: Percentage of logs filtered out
- **Target**: 95%+ noise reduction

### Cost Efficiency
- **Token Usage**: Actual LLM tokens consumed
- **Cost Reduction**: Percentage saved compared to baseline (direct RAG)

### Root Cause Diagnosis
- **Keyword Overlap**: Percentage of keywords matching ground truth
- **Jaccard Similarity**: Set-based similarity metric

## Ground Truth Format

See `data/benchmark/README.md` for ground truth annotation format.

## Future Work

- [ ] Baseline comparison experiments
- [ ] Multi-scenario validation
- [ ] Performance benchmarking
- [ ] Visualization tools
