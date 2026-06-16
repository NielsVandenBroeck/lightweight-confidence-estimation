import numpy as np
import torch
from typing import Union, Callable, Any, Optional
from torch.utils.data import DataLoader


def get_num_classes_from_loader(loader: DataLoader) -> int:
    """Extracts the number of classes from a dataloader, handling wrappers."""
    try:
        ds = loader.dataset
        # handle common nested wrappers (Subset, etc.)
        if hasattr(ds, "dataset"):
            ds = ds.dataset
        if hasattr(ds, "classes"):
            return len(ds.classes)
    except Exception:
        pass

    # fallback: inspect first batch of labels
    for batch in loader:
        labels = batch[1]
        if labels is None:
            break
        if labels.ndim == 1:
            return int(labels.max().item() + 1)
        if labels.ndim == 2:
            return labels.size(1)
        break

    raise RuntimeError("Could not detect number of classes from loader")


def torch_tensor_to_numpy_logits(logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch tensor or numpy array logits to numpy array of shape (N, C)."""
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    logits = np.asarray(logits, dtype=float)

    # ensure 2D: (C,) -> (1, C)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    return logits


def call_confidence_fn(fn: Callable,
                       logits: Optional[torch.Tensor] = None,
                       embeddings: Optional[torch.Tensor] = None,
                       preds: Optional[torch.Tensor] = None,
                       **kwargs: Any) -> Any:
    """
    Helper to call confidence functions in a backward-compatible way.
    confidence_functions: list of callables for confidence.
    """
    if fn.__name__ == "distance_confidence":
        return fn(embeddings=embeddings, predictions=preds)
    elif fn.__name__ == "confnet_confidence":
        return fn(logits=logits, embeddings=embeddings)
    else:
        # baseline style: only logits
        return fn(logits=logits)