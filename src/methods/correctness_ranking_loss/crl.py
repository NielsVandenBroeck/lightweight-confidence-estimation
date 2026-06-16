r"""
For a pair (x_i, x_j), the loss is:

L_{CR}(x_i, x_j) = \max(0, -g(c_i, c_j)(\kappa_i - \kappa_j) + |\kappa_i - \kappa_j|)

Where:
- c_i = 1 if prediction is correct, 0 otherwise
- g(c_i, c_j) = \text{sign}(c_i - c_j) \in \{-1, 0, +1\}
- \kappa_i is a confidence function (we’ll use margin: top-1 - top-2 probability)

ref.:
Moon, Jooyoung, Jihyo Kim, Younghak Shin, and Sangheum Hwang. 2020.
“Confidence-Aware Learning for Deep Neural Networks.”
Paper presented at International Conference on Machine Learning.
Proceedings of Machine Learning Research. https://arxiv.org/abs/2007.01458.
"""

import torch
import torch.nn.functional as F
from typing import Any, Tuple, Union, Dict


class CRLLoss:
    """Stateful wrapper to compute Confidence Ranking Loss (CRL)."""

    def __init__(self, b: int = 32, crl_weight: float = 1.0, warmup_epochs: int = 1):
        self.b = b
        self.target_crl_weight = crl_weight
        self.current_crl_weight = 0.0
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.history: Dict[str, Dict[str, float]] = {}

    def step_epoch(self) -> None:
        """Called by train.py at the end of each epoch."""
        self.current_epoch += 1

        if self.current_epoch >= self.warmup_epochs:
            self.current_crl_weight = self.target_crl_weight

    def __call__(self,
                 logits: torch.Tensor,
                 labels: torch.Tensor,
                 return_components: bool = False,
                 **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        paths = kwargs.get('paths', None)
        B = logits.shape[0]

        # 1. Safely handle one-hot and OOD labels without destroying the -1 flags
        if labels.ndim == 2:
            is_ood = labels[:, 0] == -1
            labels = torch.argmax(labels, dim=1)
            labels[is_ood] = -1

        id_mask = labels != -1
        ood_mask = labels == -1

        # 2. Base Loss Computation
        ce_loss_per_sample = F.cross_entropy(logits, labels, reduction='none', ignore_index=-1)
        oe_loss_per_sample = -(logits.log_softmax(dim=-1).mean(dim=-1))
        base_ce_oe_loss = torch.where(ood_mask, oe_loss_per_sample, ce_loss_per_sample).mean()

        # 3. Differentiable Confidence (Kappa) Calculation
        probs = F.softmax(logits, dim=1)
        top2_probs, _ = torch.topk(probs, min(2, probs.shape[1]), dim=1)

        if probs.shape[1] > 1:
            kappa = top2_probs[:, 0] - top2_probs[:, 1]
        else:
            kappa = top2_probs[:, 0]

        preds = torch.argmax(probs, dim=1)
        current_correct = (preds == labels).float()

        # 4. Update History
        c_history = torch.zeros(B, device=logits.device)

        if paths is not None:
            for i, p in enumerate(paths):
                # Ignore OOD samples for history tracking
                if ood_mask[i]:
                    continue

                if p not in self.history:
                    self.history[p] = {"correct": 0.0, "seen": 0.0}

                self.history[p]["correct"] += current_correct[i].item()
                self.history[p]["seen"] += 1.0

                # Laplace Smoothing: (correct + 1) / (seen + 2)
                smoothed_c = (self.history[p]["correct"] + 1.0) / (self.history[p]["seen"] + 2.0)
                c_history[i] = smoothed_c

        # 5. Sample random pairs (Only among In-Distribution data)
        valid_indices = torch.nonzero(id_mask).squeeze()
        if valid_indices.numel() > 1:
            idx_i = valid_indices[torch.randint(0, valid_indices.numel(), (self.b,))]
            idx_j = valid_indices[torch.randint(0, valid_indices.numel(), (self.b,))]

            ci = c_history[idx_i]
            cj = c_history[idx_j]
            ki = kappa[idx_i]
            kj = kappa[idx_j]

            # 6. Calculate CRL Loss: L_CR = max(0, -g * (ki - kj) + |ci - cj|)
            g = torch.sign(ci - cj)
            delta_k = ki - kj
            abs_delta_c = torch.abs(ci - cj)

            raw_loss = F.relu(-g * delta_k + abs_delta_c)

            mask = (g != 0).float()
            valid_pairs_loss = raw_loss * mask
            num_valid_pairs = mask.sum()

            crl_loss = valid_pairs_loss.sum() / num_valid_pairs if num_valid_pairs > 0 else torch.tensor(0.0, device=logits.device)
        else:
            crl_loss = torch.tensor(0.0, device=logits.device)

        total_loss = base_ce_oe_loss + (self.current_crl_weight * crl_loss)

        if return_components:
            return total_loss, base_ce_oe_loss.detach(), crl_loss.detach()

        return total_loss