# Monte Carlo Dropout (MC-Dropout)

## Overview
Monte Carlo Dropout (MC-Dropout) offers a computationally efficient Bayesian approximation to capture predictive uncertainty. By leaving Dropout layers active during inference, the network becomes stochastic. 

## Implementation Details
The `MCDropoutWrapper` forces the network to retain Dropout functionality even when `eval()` mode is active for BatchNorm stability. 

A single image is passed through the network $N$ times. The confidence score is derived by computing the normalized **Entropy** of the averaged predictions. 
* **Low Entropy (High Confidence):** The network consistently predicts the same class despite different neurons being dropped.
* **High Entropy (Low Confidence):** The predictions vary wildly, indicating uncertainty.

## Files
* `mc_dropout.py`: Contains the `MCDropoutWrapper` and the entropy confidence scoring function.
* `mc_runner.py`: The execution script to evaluate the impact of different base dropout rates and the number of forward passes ($N$).

## Usage
```bash
python mc_runner.py
```