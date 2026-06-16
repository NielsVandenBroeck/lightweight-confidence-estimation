# Outlier Exposure (OE)

## Overview
Outlier Exposure (OE) is a regularization technique designed to improve a model's ability to detect Out-of-Distribution (OOD) samples. Rather than acting as a standalone confidence scoring method, OE modifies the training loss landscape.

During training, a proportion of the batches (e.g., 50%) are injected with known OOD samples labeled with a specific flag (e.g., `-1`). The OE loss function forces the network to output a flat, uniform distribution (maximum entropy) for these samples, teaching the model to express uncertainty when it encounters unfamiliar data.

## Integration
This module is natively integrated into the Baseline and Distance-Based training pipelines. When the `--oe` flag is passed with a value greater than `0.0`, the `oe_loss` function is engaged.

## Files
* `oe.py`: Contains the `oe_loss` PyTorch function, which routes ID samples to standard Cross-Entropy and OOD samples to a log-softmax penalization.