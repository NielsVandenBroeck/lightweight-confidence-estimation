# Distance-Based Confidence

## Overview
Distance-based confidence leverages the geometry of the penultimate feature space. Confidence is calculated based on the distance from a test sample's embedding to the known distributions of the training classes.

## Geometric Configurations
This module supports three evaluation methods:
1. **K-Means Centroids:** Computes confidence based on the distance to class centroids. Highly memory-efficient for TinyML edge devices.
2. **KD-Trees / BallTrees:** Computes local density by querying the exact $k$-nearest neighbors within the predicted class.
3. **Global Trees:** Compares the local class density to the global density across all data points.

## Contrastive Distance Loss
To improve geometric separation, models can be fine-tuned using `compute_distance_loss`. This applies a contrastive margin that pulls embeddings of the same class together while pushing different classes (and OOD samples) apart in the L2-normalized space.

## Files
* `distance_based.py`: Core logic for distance loss and confidence scoring (`kmeans_distance_score`, `tree_distance_score`).
* `group_data_trees.py`: Utilities for building Scikit-Learn geometries and generating PCA plots.
* `hooked_model.py`: A PyTorch wrapper to cleanly extract internal embedding vectors.
* `dist_runner.py`: Execution script for training and evaluating distance geometries.
* `distance_loss_param_tuner.py`: Optuna script to tune the contrastive margin and loss weight.

## Usage
```bash
python dist_runner.py