# Baseline Confidence Estimation

## Overview
This module establishes the foundational baseline for confidence estimation in the edge-to-cloud wildlife monitoring pipeline. It evaluates the inherent uncertainty metrics produced by a standard EfficientNet-B0 backbone trained with Cross-Entropy Loss.

## Supported Confidence Metrics
This baseline extracts confidence directly from the model's output logits using two primary functions:
1. **Maximum Softmax Probability (MaxProb / MSP):** The standard probability of the predicted class. While ubiquitous, it is often overconfident on Out-of-Distribution (OOD) data.
2. **Weight Difference (WDF):** The normalized difference between the top-1 and top-2 softmax probabilities, offering a slightly more nuanced view of the model's internal decision margin.

## Files
* `baseline_runner.py`: The execution script that dynamically locates the project root and launches `main.py` to train or evaluate the baseline models.
* `README.md`: This documentation file.

## Usage
To execute the baseline training or evaluation sweeps, run:
```bash
python baseline_runner.py
```