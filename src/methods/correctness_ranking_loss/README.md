# Correctness Ranking Loss (CRL)

## Overview
Correctness Ranking Loss (CRL) is an ordinal ranking approach that teaches the network to align its confidence scores with its historical accuracy. It forces the model to assign higher confidence to samples it frequently classifies correctly and lower confidence to samples it struggles with.

## Implementation Details
The `CRLLoss` module maintains a running dictionary tracking the historical correctness of every training sample across epochs. During a forward pass, it samples pairs of instances. If the historical accuracy of Sample A is higher than Sample B, the loss applies a penalty if the current predicted confidence for Sample B exceeds Sample A. 

Confidence ($\kappa$) during training is measured via the margin (Top-1 - Top-2 probability).

## Files
* `crl.py`: Contains the stateful `CRLLoss` class.
* `crl_runner.py`: The execution script to train models using the CRL methodology.

## Usage
Because CRL relies on historical tracking from the first epoch, it is trained from scratch.
```bash
python crl_runner.py
```