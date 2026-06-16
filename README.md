# Learning to Trust: Lightweight Confidence Estimation for Edge-to-Cloud Wildlife Monitoring

## Overview
Dual-tier IoT wildlife monitoring systems offload uncertain edge predictions to a high-accuracy cloud server, enabling energy- and bandwidth-efficient operation. The viability of this pipeline depends critically on well-calibrated confidence: a lightweight TinyML edge model (e.g., EfficientNet-B0) must output high confidence on correct predictions to minimize unnecessary transmissions, while reliably flagging Out-of-Distribution (OOD) species for cloud offloading.

This repository benchmarks and extends multiple confidence estimation paradigms across broad, narrow, and fine-grained classification configurations, evaluated on both the CIFAR-100 benchmark and a domain-specific Ohio Small Animals camera-trap dataset.

## Installation

```bash
# Clone the repository
git clone https://github.com/NielsVandenBroeck/lightweight-confidence-estimation.git
cd lightweight-confidence-estimation

# Install the required dependencies
pip install -r requirements.txt
```

---

## Implemented Confidence Methods

### 1. Baseline & Outlier Exposure (OE)
The baseline relies on a standard backbone trained with Cross-Entropy Loss, estimating uncertainty using Maximum Softmax Probability (MaxProb) or Weight Difference (WDF). To improve OOD detection, Outlier Exposure (OE) forces known OOD samples towards a flat, maximum-entropy distribution.

> **Reference:** Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich. 2019. "Deep Anomaly Detection with Outlier Exposure." Paper presented at the International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1812.04606.

* **Execution:** `python src/methods/baseline/baseline_runner.py`


### 2. ConfNet
ConfNet introduces a class-agnostic confidence estimator that transforms the distribution of class probabilities (or penultimate feature embeddings) into a scalar confidence score.

<pre>
          +------------+        +---------+                         +------+
input --> | classifier | --+--> | confnet | --> confidence score -->|      |
          +------------+   |    +---------+                         | loss |
                           +--------------------------------------->|      |
                                                                    +------+
</pre>

> **Reference:** Wan, Sheng, Tung-Yu Wu, Wing H. Wong, and Chen-Yi Lee. 2018. “Confnet: Predict with Confidence.” 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), April, 2921–25. https://doi.org/10.1109/ICASSP.2018.8461745.

* **Execution:** `python src/methods/confnet/confnet_runner.py`


### 3. Correctness Ranking Loss (CRL)
This method trains the model using an ordinal ranking loss function. It tracks historical accuracy during training and penalizes the model if it assigns a higher confidence score to a sample it frequently misclassifies compared to a sample it consistently classifies correctly.

> **Reference:** Moon, Jooyoung, Jihyo Kim, Younghak Shin, and Sangheum Hwang. 2020. “Confidence-Aware Learning for Deep Neural Networks.” Paper presented at International Conference on Machine Learning. Proceedings of Machine Learning Research. https://arxiv.org/abs/2007.01458.

* **Execution:** `python src/methods/crl/crl_runner.py`


### 4.Distance-based Confidence Score
Distance-based confidence relies on the geometric representation of the data. Embeddings are extracted from the penultimate layer of the classifier. Confidence is computed based on the distance between the test sample's embedding and the K-Means centroids (or KD-Tree local neighborhoods) of the training data.

> **Reference:** Mandelbaum, Amit, and Daphna Weinshall. 2017. “Distance-Based Confidence Score for Neural Network Classifiers.” arXiv:1709.09844. Preprint, arXiv, September 28. https://doi.org/10.48550/arXiv.1709.09844.

* **Execution:** `python src/methods/distance_based/dist_runner.py`


### 5. Monte Carlo Dropout (MC-Dropout)
A Bayesian approximation that captures predictive uncertainty by measuring the entropy across multiple stochastic forward passes. Dropout layers remain active during inference, and the variance in predictions indicates the model's uncertainty.

> **Reference:** Gal, Yarin, and Zoubin Ghahramani. 2016. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." In Proceedings of the 33rd International Conference on Machine Learning (ICML), 1050–1059. https://arxiv.org/abs/1506.02142.

* **Execution:** `python src/methods/monte_carlo_dropout/mc_runner.py`

---

## Evaluation Pipeline
Because traditional calibration metrics like Expected Calibration Error (ECE) can fail to accurately represent performance under heavy OOD conditions, this repository utilizes a Risk-Coverage framework.

**Key Metrics Extracted:**
* **ECE (Expected Calibration Error):** Measures the standard alignment between predicted confidence and empirical accuracy.
* **MEC (Mean Effective Confidence):** Evaluates how well the model assigns high confidence to correct predictions and low confidence to incorrect ones.
* **AUC (Area Under the ROC Curve):** Measures the ability of the confidence score to discriminate between correct predictions and errors (often driven by OOD samples).
* **E-AURC:** Area Under the Risk-Coverage Curve to evaluate the model's confidence ranking capability.
* **Coverage @ 10% Risk:** The proportion of samples the edge device can process without exceeding a 10% error rate.
* **FPR @ 95% TPR:** False Positive Rate at 95% True Positive Rate.
* **Hardware Overhead:** FLOPs, Parameter counts, and Memory (MB) profiling to ensure TinyML viability.

### Running the Global Evaluation
To evaluate all trained models across all configurations and compile a comprehensive LaTeX/CSV benchmark table:
```bash
python src/evaluate_all.py
```

## Trained Models & Evaluation Data
Because of the size of the model checkpoints across all configurations, the output data is hosted via GitHub Releases rather than directly in the git history. 

You can download the stripped checkpoints, evaluation logs, and visualization plots here:
* [Download CIFAR-100 Results](https://github.com/NielsVandenBroeck/lightweight-confidence-estimation/releases/download/v1.0-thesis/Cifar100_Results.zip)
* [Download Ohio Small Animals Results](https://github.com/NielsVandenBroeck/lightweight-confidence-estimation/releases/download/v1.0-thesis/Ohio_Results.zip)