# Classification Confidence Network (ConfNet)

## Overview
ConfNet introduces a learnable, class-agnostic auxiliary network that maps the outputs of the main classifier to a calibrated scalar confidence score. It explicitly learns to predict the probability that the classifier's prediction is correct.

## Architecture
The `ClassificationConfidenceNetwork` supports multiple architectural ablations to suit edge constraints:
* **Inputs:** It can process either the sorted softmax probabilities (`logits`) or the high-dimensional penultimate layer (`features`, e.g., 1280 dimensions).
* **Depth:** It can be instantiated as a lightweight single linear layer or a multi-layer perceptron (e.g., `1280 -> 64 -> 1`).

## Training Losses
ConfNet supports two distinct training regimes:
1. **ConfNet Regularized Loss:** A confidence-weighted Cross-Entropy loss with an $\alpha$ penalty to prevent the network from collapsing to zero confidence.
2. **Binary Cross-Entropy (BCE):** Treats confidence estimation strictly as a binary classification task (1 for a correct prediction, 0 for an error).

## Files
* `confnet.py`: Contains the model architecture and the custom loss functions (`compute_confnet_loss`, `compute_confnet_bce_loss`).
* `confnet_runner.py`: The execution script for training/evaluating the ablation matrix.
* `confnet_loss_param_tuner.py`: An Optuna script to search for the optimal Learning Rate and $\alpha$ penalty.

## Usage
ConfNet typically requires a frozen, pre-trained backbone. Ensure the baseline is trained before running:
```bash
python confnet_runner.py
```