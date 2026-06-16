import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from typing import Union

from src.utils import torch_tensor_to_numpy_logits


def maxprob_confidence(logits: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Takes the highest logit and returns it as a probability."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    probs = F.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return max_probs


def neg_entropy_confidence(logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Calculates confidence based on normalized negative entropy."""
    probs = softmax(torch_tensor_to_numpy_logits(logits), axis=1)

    # Calculate entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    # Theoretical maximum entropy for C classes is log(C)
    max_entropy = np.log(logits.shape[1] if isinstance(logits, torch.Tensor) else logits.shape[1])

    # Normalize entropy to [0, 1] and invert it:
    norm_conf = 1.0 - (entropy / max_entropy)
    return norm_conf


def margin_confidence(logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Calculates confidence as the difference between top-1 and top-2 probabilities."""
    probs = softmax(torch_tensor_to_numpy_logits(logits), axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]

    max1 = sorted_probs[:, 0]
    max2 = sorted_probs[:, 1]
    return max1 - max2


def wdf_confidence(logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    WDF confidence per sample. High difference means high confidence.
    conf = (max1 - max2) / (abs(max1) + abs(max2) + eps)
    """
    probs = softmax(torch_tensor_to_numpy_logits(logits), axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]

    max1 = sorted_probs[:, 0]
    max2 = sorted_probs[:, 1]

    denom = np.abs(max1 + max2) + 1e-12
    return (max1 - max2) / denom


def krt_confidence(logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Kurtosis confidence per sample.
    NOTE: Do not use for primary confidence (unbounded, fails on 2-class datasets).
    """
    L = torch_tensor_to_numpy_logits(logits)
    mu = L.mean(axis=1, keepdims=True)
    sigma = L.std(axis=1, keepdims=True)

    z = np.divide(L - mu, sigma, out=np.zeros_like(L), where=sigma != 0)

    # Standardized fourth moment (Kurtosis)
    return (z ** 4).mean(axis=1)