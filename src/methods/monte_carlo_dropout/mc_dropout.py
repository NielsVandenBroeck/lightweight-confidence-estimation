import numpy as np
import torch
import torch.nn as nn
from typing import Any, Tuple, Union, Optional


def mc_dropout_confidence(logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Extracts the MC Dropout confidence score attached to the pseudo-logits by the wrapper."""
    mean_probs = torch.softmax(logits, dim=1)

    epsilon = 1e-12
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=1)

    num_classes = mean_probs.shape[1]
    normalized_entropy = entropy / np.log(num_classes)

    return 1.0 - normalized_entropy


class MCDropoutWrapper(nn.Module):
    """Wraps a trained PyTorch model to perform Monte Carlo Dropout during inference."""

    def __init__(self, model: nn.Module, num_passes: int = 10):
        super().__init__()
        self.model = model
        self.num_passes = num_passes

    def enable_dropout(self) -> None:
        """Keeps BatchNorm in eval() mode, but turns Dropout back on."""
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Performs N forward passes and averages the predicted probabilities."""
        self.model.eval()
        self.enable_dropout()

        all_probs = []
        first_emb = None

        with torch.no_grad():
            for i in range(self.num_passes):
                if return_embeddings:
                    logits, emb = self.model(x, return_embeddings=True)
                    if i == 0:
                        first_emb = emb
                else:
                    logits = self.model(x)

                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        mean_probs = torch.stack(all_probs).mean(dim=0)

        # Convert mean probabilities back to pseudo-logits
        epsilon = 1e-12
        pseudo_logits = torch.log(mean_probs + epsilon)

        if return_embeddings:
            return pseudo_logits, first_emb
        return pseudo_logits