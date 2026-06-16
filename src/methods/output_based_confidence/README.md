# Output-Based Confidence Estimation

## Overview
This module contains the foundational, "free" confidence estimation metrics. Output-based confidence methods extract uncertainty directly from the standard classifier's outputs (raw logits or softmax probabilities). 

These methods are incredibly lightweight, making them highly suitable for TinyML edge devices. They require **zero extra parameters**, **zero architectural changes**, and **minimal computational overhead**. However, they are traditionally susceptible to overconfidence when presented with Out-of-Distribution (OOD) data.

## Implemented Metrics

The `conf_methods.py` script provides several distinct mathematical approaches to measure confidence from the logit distribution:

1. **Maximum Softmax Probability (MaxProb / MSP):**
   * *Concept:* Takes the highest softmax probability as the confidence score. 
   * *Pros:* Ubiquitous and extremely fast.
   * *Cons:* Notoriously overconfident on unknown classes.

2. **Negative Entropy:**
   * *Concept:* Calculates the Shannon entropy of the softmax distribution, normalizes it against the theoretical maximum entropy ($\ln(C)$), and inverts it.
   * *Behavior:* High confidence means a highly peaked distribution; low confidence means a flat, uniform distribution.

3. **Margin:**
   * *Concept:* The absolute difference between the Top-1 and Top-2 probabilities.
   * *Behavior:* Captures the model's decisiveness between its top two choices.

4. **Weight Difference (WDF):**
   * *Concept:* A normalized version of Margin. It divides the margin by the sum of the absolute values of the Top-1 and Top-2 probabilities.

5. **Kurtosis (KRT):**
   * *Concept:* Measures the "peakedness" of the raw logit distribution using the 4th statistical moment. 
   * *Note:* This is an experimental metric. It is mathematically unbounded and will fail gracefully on binary classification datasets. Do not use as the primary confidence metric without scaling.

## Files
* `conf_methods.py`: Contains the vectorized NumPy and PyTorch implementations for all output-based scoring functions.

## Usage
Because these methods do not require specialized training or architectural wrappers, they are typically passed directly into the evaluation pipeline as function pointers. 

Example usage during evaluation:
```python
from methods.output_based_confidence.conf_methods import maxprob_confidence, wdf_confidence
from src.evaluation import evaluate_model

# Pass the functions directly to the evaluator
evaluate_model(
    classifier=model,
    output_path="path/to/output",
    test_loader=test_loader,
    confidence_functions=[maxprob_confidence, wdf_confidence]
)
```